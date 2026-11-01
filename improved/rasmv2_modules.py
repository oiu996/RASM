# -*- coding: utf-8 -*-
"""RASMv2 的改进模块集合 —— 对 RASM (Regional Attention for Shadow Removal, ACM MM'24) 的扩展。

遵循开闭原则 (OCP) 的三条硬约束:
  1. 本文件不修改 RASM/ 下任何既有文件，只新增模块；
  2. 通过 ``from model import ...`` 读写式复用既有基础件
     (CAB / LeFF / Downsample / Upsample / InputProj / OutputProj / CATransformerBlock)；
  3. 新模块的属性命名、拓扑结构与既有 RASM 保持一致，
     因此 ``preset='rasm'`` 时可直接加载官方发布的 RASM 权重做对照/续训。

包含的改进(均可独立开闭，便于消融):
  A. MGRA  掩码引导的区域注意力: RIB(光照关系偏置) + NS-Gate(非阴影 key 门控) + dilation 感知的 RPB
  B. NSRC  非阴影区域上下文: 可分离掩码空洞聚合，把"阴影 token 参考邻域非阴影 token"变成显式通路
  C. IAFM  掩码条件仿射调制 (Mask-FiLM): 让每一层都感知 mask
  D. 感受野尺度自适应: 不同分辨率层用不同 (kernel_size, dilation)
"""

import math
import os
import sys
import warnings

_RASM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RASM_ROOT not in sys.path:
    sys.path.insert(0, _RASM_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

# 复用既有基础件（只读，不改动）
from model import (CAB, LeFF, Mlp, Downsample, Upsample, InputProj, OutputProj,
                   CATransformerBlock)


def _inv_softplus(x):
    """求 p 使得 softplus(p) == x，用于把"初始抑制强度"翻译成参数初值。"""
    return math.log(math.expm1(x)) if x < 20.0 else x


# ==================================================================================
# A-1. 支持外部注意力偏置的邻域注意力 (纯 PyTorch 实现，可脱离 natten 运行)
# ==================================================================================
class NeighborhoodAttentionBias(nn.Module):
    """分块(chunk)实现的 Neighborhood Attention，额外支持外部加性偏置。

    与 RASM 原用的 ``natten.NeighborhoodAttention2D`` 在内部区域数值等价:
      * 采用 NA 论文所述的零填充处理边界 (``border_mode='zero'``)；
      * 相对位置偏置 RPB 改为 (Δy, Δx) 物理位移 -> MLP 的连续参数化 (DiNAT 风格)，
        因此**对 dilation 敏感**；natten 的表格是按 kernel 网格索引的，d=1 与 d=2 共用同一张表。

    之所以自己实现而不直接用 natten: natten 的 NA 是黑盒，无法在注意力 logits 上叠加
    "阴影/非阴影" 结构化先验(即 MGRA 的 RIB + NS-Gate)，而这两者正是本次改进的核心。
    代价是用 im2col 展开，故按行分块以控制显存峰值。

    输入输出均为 NHWC: (B, H, W, C)。
    """

    def __init__(self, dim, num_heads, kernel_size=11, dilation=2, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0., use_rpb=True,
                 rpb_hidden=64, border_mode='zero', chunk_elements=2_000_000):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size 必须为奇数'
        assert dim % num_heads == 0, 'dim 必须能被 num_heads 整除'
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.border_mode = border_mode
        self.chunk_elements = chunk_elements
        self.num_nbr = kernel_size * kernel_size
        self.pad = dilation * (kernel_size - 1) // 2

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if use_rpb:
            self.rpb = nn.Sequential(nn.Linear(2, rpb_hidden), nn.ReLU(inplace=True),
                                     nn.Linear(rpb_hidden, num_heads))
            nn.init.zeros_(self.rpb[2].weight)
            nn.init.zeros_(self.rpb[2].bias)
            c = (kernel_size - 1) // 2
            gy, gx = torch.meshgrid((torch.arange(kernel_size) - c) * dilation,
                                    (torch.arange(kernel_size) - c) * dilation,
                                    indexing='ij')
            pos = torch.stack([gy, gx], dim=-1).reshape(-1, 2).float()
            pos = pos / max(1.0, float(c * dilation))
            self.register_buffer('rpb_pos', pos, persistent=False)
        else:
            self.rpb = None

        self._border_cache = {}

    def extra_repr(self):
        return (f'dim={self.dim}, heads={self.num_heads}, kernel={self.kernel_size}, '
                f'dilation={self.dilation}, border={self.border_mode}')

    def _rpb_table(self):
        if self.rpb is None:
            return None
        t = self.rpb(self.rpb_pos)                       # (K*K, heads)
        return t.permute(1, 0).contiguous().view(1, self.num_heads, 1, self.num_nbr)

    def _border_mask(self, H, W, device):
        """border_mode='mask' 时，越界位置对应的 logits 置 -inf。"""
        key = (H, W, device)
        if key in self._border_cache:
            return self._border_cache[key]
        K, D = self.kernel_size, self.dilation
        c = (K - 1) // 2
        yy = torch.arange(H, device=device).view(H, 1, 1) + (torch.arange(K, device=device).view(1, 1, K) - c) * D
        xx = torch.arange(W, device=device).view(1, W, 1) + (torch.arange(K, device=device).view(1, 1, K) - c) * D
        valid = ((yy >= 0) & (yy < H)).float().view(H, 1, 1, K) * ((xx >= 0) & (xx < W)).float().view(1, W, K, 1)
        valid = valid.reshape(H * W, K * K)                                  # (L, J)
        mask = (1.0 - valid) * (-1e4)
        self._border_cache[key] = mask
        return mask

    def forward(self, x, attn_bias=None):
        """x: (B,H,W,C) -> y: (B,H,W,C)

        attn_bias: None | Tensor(B,heads,L,J) | callable(r0, r1) -> Tensor(B,heads,lc,J) | None
        """
        B, H, W, C = x.shape
        h, hd, K, D = self.num_heads, self.head_dim, self.kernel_size, self.dilation
        J = self.num_nbr
        span = D * (K - 1)
        pad = self.pad

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)                                        # (B,H,W,C)
        q = q.reshape(B, H * W, h, hd).permute(0, 2, 1, 3)                     # (B,h,L,hd)
        k = k.permute(0, 3, 1, 2).reshape(B * h, hd, H, W)
        v = v.permute(0, 3, 1, 2).reshape(B * h, hd, H, W)
        kp = F.pad(k, (pad, pad, pad, pad))
        vp = F.pad(v, (pad, pad, pad, pad))

        rpb = self._rpb_table()
        border = self._border_mask(H, W, x.device) if self.border_mode == 'mask' else None

        # 每个 chunk 的 logits 规模 = B*h*J*rows*W，据此自适应分块
        rows = max(1, int(self.chunk_elements // max(1, B * h * J * W)))
        outs = []
        for r0 in range(0, H, rows):
            r1 = min(r0 + rows, H)
            lc = (r1 - r0) * W

            ks = kp[:, :, r0:r1 + span, :]
            ku = F.unfold(ks, kernel_size=K, dilation=D)                       # (B*h, hd*J, lc)
            ku = ku.view(B * h, hd, J, lc).permute(0, 3, 1, 2)                  # (B*h, lc, hd, J)
            ku = ku.reshape(B * h * lc, hd, J).contiguous()                     # bmm 需要连续布局
            qc = q[:, :, r0 * W:r1 * W].reshape(B * h * lc, 1, hd)
            logits = torch.bmm(qc, ku).view(B, h, lc, J).mul_(self.scale)

            if rpb is not None:
                logits.add_(rpb)
            if attn_bias is not None:
                ext = attn_bias(r0, r1) if callable(attn_bias) else attn_bias[:, :, r0 * W:r1 * W]
                if ext is not None:
                    logits.add_(ext)
            if border is not None:
                logits.add_(border[r0 * W:r1 * W].view(1, 1, lc, J))

            attn = logits.softmax(dim=-1)
            attn = self.attn_drop(attn)

            vs = vp[:, :, r0:r1 + span, :]
            vu = F.unfold(vs, kernel_size=K, dilation=D)
            vu = vu.view(B * h, hd, J, lc).permute(0, 3, 2, 1)                  # (B*h, lc, J, hd)
            vu = vu.reshape(B * h * lc, J, hd).contiguous()
            o = torch.bmm(attn.reshape(B * h * lc, 1, J), vu).view(B, h, lc, hd)
            outs.append(o)

        y = torch.cat(outs, dim=2)                                              # (B, h, L, hd)
        y = y.permute(0, 2, 1, 3).reshape(B, H, W, C)
        y = self.proj_drop(self.proj(y))
        return y


# ==================================================================================
# A-2. 掩码引导的注意力偏置生成器: RIB + NS-Gate
# ==================================================================================
class MaskGuidedBias(nn.Module):
    """为区域注意力注入"阴影/非阴影"结构化先验的外部偏置生成器。

    对邻域内每个 (query, key) 对附加两项偏置:

    1) RIB (Relational Illumination Bias, 光照关系偏置)
       B_illu[h, 2*q_shadow + k_shadow]，即按 (query 是否在阴影, key 是否在阴影) 的
       4 种组合给每个 head 一个可学习标量。它把论文的核心假设
       "修复阴影所需的关键信息来自紧邻的非阴影区域" 直接写成注意力先验:
       训练中 (q=阴影, k=非阴影) 这一项若被抬高，阴影 token 就会更多地聚合非阴影信息。

    2) NS-Gate (非阴影 key 门控)
       对位于阴影内、且邻域内确实存在非阴影 key 的 query，把同为阴影的 key 的 logits
       统一减去一个可学习的非负强度; 邻域全被阴影覆盖时自动失效 (fallback)，
       避免极端情况下信息通路被切断。
       默认用软门控而非硬屏蔽: 阴影只衰减光照而保留纹理/色度，
       把阴影 key 全部屏蔽会丢掉纹理线索，故强度交给网络自己学。

    实现上利用了 softmax 的平移不变性: 只保留与 key 标签相关的部分，
    与 query 有关的常数项被省略，因此偏置张量规模与 logits 相同但无需额外显存放大。
    """

    def __init__(self, num_heads, kernel_size=11, dilation=2, use_illu=True,
                 use_gate=True, hard_gate=False, gate_init=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_illu = use_illu
        self.use_gate = use_gate
        self.hard_gate = hard_gate
        self.pad = dilation * (kernel_size - 1) // 2
        # (4, heads): 索引 2*q + k
        self.illu = nn.Parameter(torch.zeros(4, num_heads)) if use_illu else None
        if use_gate:
            self.gate_raw = nn.Parameter(torch.full((num_heads,), _inv_softplus(gate_init)))
        else:
            self.gate_raw = None
        self._m = None
        self._valid_cache = {}

    def set_mask(self, m):
        self._m = m

    def clear(self):
        self._m = None

    def extra_repr(self):
        return (f'heads={self.num_heads}, kernel={self.kernel_size}, '
                f'dilation={self.dilation}, illu={self.use_illu}, gate={self.use_gate}')

    def _valid_unfold(self, H, W, device):
        """邻域内"未越界"指示器的 unfold 缓存 (1, J, L)，用于剔除图外的伪非阴影 key。"""
        key = (H, W, device)
        if key in self._valid_cache:
            return self._valid_cache[key]
        v = F.unfold(F.pad(torch.ones(1, 1, H, W, device=device),
                           (self.pad, self.pad, self.pad, self.pad)),
                     kernel_size=self.kernel_size, dilation=self.dilation)   # (1, J, L)
        self._valid_cache[key] = v
        return v

    def __call__(self, r0, r1):
        """被 NeighborhoodAttentionBias 按 chunk 回调，返回 (B, heads, lc, J) 或 None。"""
        m = self._m
        if m is None:
            return None
        B, _, H, W = m.shape
        K, D = self.kernel_size, self.dilation
        mp = F.pad(m, (self.pad, self.pad, self.pad, self.pad))
        strip = mp[:, :, r0:r1 + D * (K - 1), :]
        mk = F.unfold(strip, kernel_size=K, dilation=D).transpose(1, 2)       # (B, lc, J) 1=阴影
        mq = m[:, :, r0:r1].reshape(B, -1, 1)                                # (B, lc, 1)
        mv = self._valid_unfold(H, W, m.device)[:, :, r0 * W:r1 * W].transpose(1, 2)  # (1, lc, J)

        coef = None
        if self.use_illu:
            # 只保留随 key 标签变化的分量: B[2q+k] = B[2q] + (B[2q+1]-B[2q])*k，常数项被 softmax 抵消
            b0 = self.illu[1] - self.illu[0]
            b1 = self.illu[3] - self.illu[2]
            c = (1.0 - mq) * b0.view(1, 1, -1) + mq * b1.view(1, 1, -1)      # (B, lc, heads)
            coef = c.permute(0, 2, 1)                                         # (B, heads, lc)
        if self.use_gate:
            has_ns = ((1.0 - mk) * mv).sum(-1, keepdim=True) > 0.5            # (B, lc, 1)
            if self.hard_gate:
                strength = torch.full((1,), 1e4, device=m.device, dtype=m.dtype)
            else:
                strength = F.softplus(self.gate_raw)
            c = (mq * has_ns.float()) * strength.view(1, 1, -1)               # (B, lc, heads)
            coef = -c.permute(0, 2, 1) if coef is None else coef - c.permute(0, 2, 1)
        if coef is None:
            return None
        return coef.unsqueeze(-1) * mk.unsqueeze(1)                           # (B, heads, lc, J)


# ==================================================================================
# B. 非阴影区域上下文聚合 (NSRC)
# ==================================================================================
class NonShadowRegionalContext(nn.Module):
    """非阴影区域上下文聚合 NSRC (Non-Shadow Regional Context)。

    动机: RASM 的区域注意力是内容驱动的，采样哪些 key 并不区分其是否处于阴影;
    而论文 Fig.3 的核心论点是"紧邻阴影的非阴影区域携带了修复所需的关键信息"。
    本模块用一个几乎免费的、可分离的一维掩码空洞聚合把这一先验显式化:
    对每个位置，沿水平/垂直两条膨胀线只对【非阴影】位置的特征做平均，
    再减去自身得到"外部非阴影线索"，经 1x1 卷积与通道门控后残差注入主分支。

    复杂度 O(4k)/token，比区域注意力的 O(k²) 低一个量级; 初始化为恒等 (输出 0)，
    因此可作为纯粹的增量改进单独消融。
    """

    def __init__(self, dim, kernel_size=11, dilation=2, reduction=4):
        super().__init__()
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.kernel_size = k
        self.dilation = dilation
        pad = (k // 2) * dilation
        self.pad_h, self.pad_v = (0, pad), (pad, 0)
        self.dil_h, self.dil_v = (1, dilation), (dilation, 1)

        kh = torch.ones(1, 1, 1, k)
        kh[0, 0, 0, k // 2] = 0.0                       # 中心置 0: 中心位置单独以 +x/+1 计入
        kv = torch.ones(1, 1, k, 1)
        kv[0, 0, k // 2, 0] = 0.0
        self.register_buffer('kh', kh, persistent=False)
        self.register_buffer('kv', kv, persistent=False)

        self.proj = nn.Conv2d(dim, dim, 1)
        hidden = max(1, 2 * dim // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, dim, 1), nn.Sigmoid())
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _masked_line_avg(x, m, kernel, pad, dilation):
        B, C, H, W = x.shape
        ns = 1.0 - m
        w = kernel.expand(C, 1, *kernel.shape[-2:])
        num = F.conv2d(x * ns, w, padding=pad, dilation=dilation, groups=C)
        den = F.conv2d(ns, kernel, padding=pad, dilation=dilation)
        return (num + x) / (den + 1.0)                  # 分母 +1 把自身计入，天然避免除零

    def forward(self, x, m):
        """x: (B,C,H,W) 特征;  m: (B,1,H,W) 阴影 mask (1=阴影)"""
        ah = self._masked_line_avg(x, m, self.kh, self.pad_h, self.dil_h)
        av = self._masked_line_avg(x, m, self.kv, self.pad_v, self.dil_v)
        ctx = 0.5 * (ah + av)
        res = ctx - x
        out = self.proj(res)
        g = self.gate(torch.cat([x, res], dim=1))
        return out * g


# ==================================================================================
# C. 掩码条件仿射调制 (Mask-FiLM)
# ==================================================================================
class MaskFiLM(nn.Module):
    """Mask-FiLM: 逐像素的掩码条件仿射 y = γ(m)·x + β(m)。

    RASM 只在输入端把 mask 与 RGB 拼成 4 通道过一次 InputProj，之后 mask 不再参与任何层
    (RALayer/CATransformerBlock 虽然接收 xm，但前向里并未使用)。
    本模块让每一层都感知 mask，等价于给阴影区与非阴影区两套统计量(条件归一化)，
    参数量仅 4C，初始化为恒等。
    """

    def __init__(self, dim):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(2, dim))
        self.bias = nn.Parameter(torch.zeros(2, dim))

    def forward(self, x, m):
        """x: (B,L,C); m: (B,L,1) ∈ {0,1}, 1=阴影"""
        g = self.gain[0] * (1.0 - m) + self.gain[1] * m
        b = self.bias[0] * (1.0 - m) + self.bias[1] * m
        return x * g + b


class CATransformerBlockV2(CATransformerBlock):
    """在原 CATransformerBlock 之后追加 Mask-FiLM (继承 = 对扩展开放，对修改封闭)。

    继承后子模块命名/权重前缀与原类完全一致，因此可直接加载官方权重。
    """

    def __init__(self, *args, use_film=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.milm = MaskFiLM(self.dim) if use_film else None

    def forward(self, x, xm, mask=None, img_size=(128, 128)):
        x = super().forward(x, xm, mask, img_size)
        if self.milm is not None and xm is not None:
            B, L, _ = x.shape
            m = (xm > 0.5).float().reshape(B, L, 1)
            x = self.milm(x, m)
        return x


# ==================================================================================
# A+B+C 组合: 掩码引导的区域注意力层 (替换原 RALayer)
# ==================================================================================
class MGRALayer(nn.Module):
    """Mask-Guided Regional Attention Layer。

    结构 (与 RASM 的 RALayer 一一对应，便于权重兼容):
        x = CA( MGRA(LN(x)) + NSRC(LN(x), m) ) + x
        x = GELU(MLP(LN(x))) + x
    其中 MGRA = 邻域注意力 + RPB + RIB + NS-Gate。
    """

    def __init__(self, dim, input_resolution, num_heads, kernel_size=11, dilation=2,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 token_projection='linear', token_mlp='leff', se_layer=False,
                 use_nsrc=True, nsrc_kernel=11, nsrc_dilation=2,
                 use_film=True, mask_bias=True, ns_gate=True, ns_gate_hard=False,
                 ns_gate_init=0.5, attn_impl='unfold', border_mode='zero',
                 chunk_elements=2_000_000):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.token_mlp = token_mlp
        self.attn_impl = attn_impl

        self.norm1 = norm_layer(dim)
        if attn_impl == 'natten':
            from natten import NeighborhoodAttention2D as NeighborhoodAttention
            if mask_bias or ns_gate:
                warnings.warn("attn_impl='natten' 无法在 logits 上叠加外部偏置，"
                              "RIB/NS-Gate 将被自动关闭（NSRC/FiLM 仍生效）")
                mask_bias = ns_gate = False
            self.attn = NeighborhoodAttention(dim, kernel_size=kernel_size, dilation=dilation,
                                              num_heads=num_heads, qkv_bias=qkv_bias,
                                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = NeighborhoodAttentionBias(dim, num_heads=num_heads, kernel_size=kernel_size,
                                                  dilation=dilation, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  attn_drop=attn_drop, proj_drop=drop,
                                                  border_mode=border_mode, chunk_elements=chunk_elements)
        self.mask_bias = mask_bias
        self.ns_gate = ns_gate
        if mask_bias or ns_gate:
            self.bias_gen = MaskGuidedBias(num_heads, kernel_size=kernel_size, dilation=dilation,
                                           use_illu=mask_bias, use_gate=ns_gate,
                                           hard_gate=ns_gate_hard, gate_init=ns_gate_init)
        else:
            self.bias_gen = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 与既有 RASM 的 RALayer 保持同名的 mlp 属性，使官方权重的键名可直接命中
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop) if token_mlp == 'ffn' \
            else LeFF(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.CAB = CAB(dim, kernel_size=3, reduction=4, bias=False, act=nn.PReLU())
        self.nsrc = NonShadowRegionalContext(dim, nsrc_kernel, nsrc_dilation) if use_nsrc else None
        self.film = MaskFiLM(dim) if use_film else None
        self._use_film = use_film

    def extra_repr(self):
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, heads={self.num_heads}, '
                f'kernel={self.kernel_size}, dilation={self.attn.dilation}, mlp={self.token_mlp}')

    def forward(self, x, xm, mask=None, img_size=(128, 128)):
        B, L, C = x.shape
        H, W = img_size
        if L != H * W:
            raise ValueError(f'token 数 {L} 与 img_size {H}x{W} 不匹配')

        shortcut = x
        xn = self.norm1(x)                                          # (B, L, C)

        m = None
        if xm is not None and (self.bias_gen is not None or self.nsrc is not None or self._use_film):
            m = (xm > 0.5).float()                                  # (B,1,H,W)
        if self.bias_gen is not None:
            self.bias_gen.set_mask(m if m is not None else torch.zeros(B, 1, H, W, device=x.device))

        x_map = xn.view(B, H, W, C)
        with torch.autocast(device_type='cuda', enabled=False):
            if self.attn_impl == 'natten':
                y = self.attn(x_map.float())
            else:
                y = self.attn(x_map.float(), attn_bias=self.bias_gen)
        if self.bias_gen is not None:
            self.bias_gen.clear()

        y = rearrange(y, 'b h w c -> b c h w')
        if self.nsrc is not None:
            mm = m if m is not None else torch.zeros(B, 1, H, W, device=x.device)
            xn_map = rearrange(xn, 'b (h w) c -> b c h w', h=H, w=W)
            y = y + self.nsrc(xn_map, mm)

        y = self.CAB(y)
        y = rearrange(y, 'b c h w -> b (h w) c')

        x = shortcut + self.drop_path(y)
        if self.film is not None:
            mm = m.reshape(B, L, 1) if m is not None else torch.zeros(B, L, 1, device=x.device)
            x = self.film(x, mm)
        # 属性名保持与既有 RASM 一致(mlp)，LeFF 需要 img_size 才能还原二维结构
        ff = self.mlp(self.norm2(x), img_size=img_size) if isinstance(self.mlp, LeFF) \
            else self.mlp(self.norm2(x))
        x = x + self.drop_path(ff)
        return x


# ==================================================================================
# 基础层: 与 BasicRASM 同构，但支持 MGRA / 逐块 dilation 调度 / FiLM
# ==================================================================================
class BasicRASMv2(nn.Module):
    """与既有 BasicRASM 同构的基础层。

    cab=True  -> 用 CATransformerBlockV2 (继承原 CATransformerBlock + Mask-FiLM)
    cab=False -> 用 MGRALayer，并按照传入的 dilations 列表为每层分配不同膨胀率
    """

    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', se_layer=False, cab=False,
                 kernel_size=11, dilations=None, block_kwargs=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cab = cab
        dilations = list(dilations) if dilations is not None else [2] * depth
        if len(dilations) == 1:
            dilations = dilations * depth
        block_kwargs = block_kwargs or {}

        if cab:
            self.blocks = nn.ModuleList([
                CATransformerBlockV2(dim=dim, input_resolution=input_resolution,
                                     num_heads=num_heads, win_size=win_size,
                                     shift_size=0 if (i % 2 == 0) else win_size // 2,
                                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                     norm_layer=norm_layer, token_projection=token_projection,
                                     token_mlp=token_mlp, se_layer=se_layer,
                                     use_film=block_kwargs.get('use_film', True))
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                MGRALayer(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                          kernel_size=kernel_size, dilation=dilations[i],
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop=drop, attn_drop=attn_drop,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer=norm_layer, token_projection=token_projection,
                          token_mlp=token_mlp, se_layer=se_layer, **block_kwargs)
                for i in range(depth)])

    def extra_repr(self):
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def forward(self, x, xm, mask=None, img_size=(128, 128)):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x, xm, mask, img_size)
            else:
                x = blk(x, xm, mask, img_size)
        return x
