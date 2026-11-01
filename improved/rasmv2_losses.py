# -*- coding: utf-8 -*-
"""RASMv2 的改进损失项。

既有 losses.py 不做任何修改: L1_Vgg_losses (Charbonnier + VGG 感知损失) 被**包装**为基类，
新增的三项都是针对去阴影任务特性设计的任务损失:

  1. BoundaryLoss                半影区加权的内容损失
  2. NonShadowConsistencyLoss   非阴影区"去均值"一致性损失 (容忍 ISTD+ 的全局光照校正)
  3. ChromaticityLoss           色度一致性损失 (Finlayson 色度不变物理先验)
"""

import os
import sys

_RASM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RASM_ROOT not in sys.path:
    sys.path.insert(0, _RASM_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import L1_Vgg_losses   # 复用原损失(包装，不修改)


def _dilate(m, k):
    return F.max_pool2d(m, k, 1, k // 2)


def _erode(m, k):
    return -F.max_pool2d(-m, k, 1, k // 2)


class BoundaryLoss(nn.Module):
    """半影区(penumbra)加权的内容损失。

    阴影边界处的光照过渡最剧烈，也是残留阴影与伪影最容易出现的区域
    (论文 Related Work 亦指出传统方法在 penumbra 上容易失败)。
    本损失对"阴影膨胀带 = dilate(m) - erode(m)"额外加权，
    同时用 1/w 归一化保持与朴素 Charbonnier 相同的量级。
    """

    def __init__(self, band=9, w_shadow=0.5, w_band=1.0, eps=1e-3):
        super().__init__()
        self.band = band if band % 2 == 1 else band + 1
        self.w_shadow = w_shadow
        self.w_band = w_band
        self.eps = eps

    def forward(self, out, gt, m):
        band = (_dilate(m, self.band) - _erode(m, self.band)).clamp(0, 1)
        w = 1.0 + self.w_shadow * m + self.w_band * band
        ch = torch.sqrt((out - gt).pow(2) + self.eps * self.eps)
        # 权重图是单通道，须先 expand 到与图像同形状再加权，否则归一化会少算一个通道因子
        w = w.expand_as(ch)
        return (w * ch).sum() / w.sum().clamp(min=1.0)


class NonShadowConsistencyLoss(nn.Module):
    """非阴影区一致性损失 (容忍全局光照/色彩偏移)。

    ISTD+ 的 GT 非阴影区域是经线性回归"校正"过的，与输入非阴影区存在整体光照差异；
    直接对非阴影区做 L1 会逼着模型去拟合一个与任务无关的常量偏移。
    故本损失先对 (out - gt) 做大窗口均值滤波，只惩罚**局部不一致**部分。
    """

    def __init__(self, pool=31):
        super().__init__()
        self.pool = pool if pool % 2 == 1 else pool + 1

    def forward(self, out, gt, m):
        d = out - gt
        # 用 replicate 而非 0 填充做均值滤波: 否则图像边缘的均值被拉低，
        # 会把"常量偏移"误判成局部不一致
        p = self.pool // 2
        d = d - F.avg_pool2d(F.pad(d, (p, p, p, p), mode='replicate'), self.pool, 1, 0)
        ns = (1.0 - m).expand_as(d)
        return (d.abs() * ns).sum() / ns.sum().clamp(min=1.0)


class ChromaticityLoss(nn.Module):
    """色度一致性损失 —— 把 Finlayson 的"色度不变"物理先验写成正则项。

    阴影本质是光照衰减: I_s ≈ A·I_free (A<1 且逐通道近似一致)，因此强度改变而色度不变。
    直接监督归一化色度 c = I / ΣI 可以让去阴影结果不偏色；
    为避免暗区 (ΣI→0) 的数值不稳定，只在亮度足够的位置计算。
    """

    def __init__(self, min_lum=0.1, eps=1e-2):
        super().__init__()
        self.min_lum = min_lum
        self.eps = eps

    @staticmethod
    def _chroma(x, eps):
        return x / (x.sum(dim=1, keepdim=True) + eps)

    def forward(self, out, gt):
        co = self._chroma(out, self.eps)
        cg = self._chroma(gt, self.eps)
        valid = (gt.mean(dim=1, keepdim=True) > self.min_lum).float()
        diff = (co - cg).abs() * valid
        return diff.sum() / (valid.sum() * co.shape[1]).clamp(min=1.0)


class RASMv2Loss(nn.Module):
    """组合损失: 既有 L1+VGG 损失 + 三项任务损失。

    base 默认为原 losses.L1_Vgg_losses，因此也可传入任意其它损失作为基类。
    """

    def __init__(self, w_boundary=1.0, w_ns=0.5, w_chroma=0.2,
                 band=9, ns_pool=31, min_lum=0.1, base=None):
        super().__init__()
        self.base = base if base is not None else L1_Vgg_losses()
        self.w_boundary = w_boundary
        self.w_ns = w_ns
        self.w_chroma = w_chroma
        self.boundary = BoundaryLoss(band=band)
        self.ns = NonShadowConsistencyLoss(pool=ns_pool)
        self.chroma = ChromaticityLoss(min_lum=min_lum)

    def forward(self, out, gt, m=None):
        loss = self.base(out, gt)
        if m is None:
            return loss
        if self.w_boundary > 0:
            loss = loss + self.w_boundary * self.boundary(out, gt, m)
        if self.w_ns > 0:
            loss = loss + self.w_ns * self.ns(out, gt, m)
        if self.w_chroma > 0:
            loss = loss + self.w_chroma * self.chroma(out, gt)
        return loss
