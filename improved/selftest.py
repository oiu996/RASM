# -*- coding: utf-8 -*-
"""RASMv2 的 CPU 自检脚本（不需要 GPU 也能跑，用于验证改进模块的正确性）。

    cd /workspace/RASM && python improved/selftest.py

检查项:
  [1] 自定义邻域注意力 vs natten (内部区域数值等价)
  [2] 自定义邻域注意力 vs 朴素零填充参考实现 (含边界)
  [3] 掩码引导偏置 RIB/NS-Gate 的取值与 fallback 行为
  [4] RASMv2 各 preset 的构建 / 前向 / 反向
  [5] RASMv2(preset='rasm') 与官方权重 AISTD.pth 的键名兼容性
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# 容器/共享机上常见 CPU 配额限制(cgroup)，OMP 线程过多会被节流；自检统一用单线程
torch.set_num_threads(1)

from improved.rasmv2_modules import (NeighborhoodAttentionBias, MaskGuidedBias,
                                     NonShadowRegionalContext, MaskFiLM, MGRALayer)
from improved.rasmv2_model import RASMv2, load_weights

torch.manual_seed(1234)
OK = 'PASS'
FAIL = 'FAIL'


def naive_na(x, kernel_size, dilation):
    """零填充 + 无 RPB 的朴素邻域注意力(均匀权重)，用于校验 im2col 的 offset 对齐。"""
    B, H, W, C = x.shape
    K, D = kernel_size, dilation
    r = (K - 1) // 2
    pad = D * r
    xp = torch.zeros(B, H + 2 * pad, W + 2 * pad, C, dtype=x.dtype)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = torch.zeros_like(x)
    for i in range(H):
        for j in range(W):
            vals = [xp[:, i + pad + a * D, j + pad + b * D]
                    for a in range(-r, r + 1) for b in range(-r, r + 1)]
            out[:, i, j] = torch.stack(vals, dim=1).mean(1)
    return out


def test_attn_vs_natten():
    from natten import NeighborhoodAttention2D as NATTEN
    worst = 0.0
    for (C, heads, K, D) in ((16, 1, 3, 1), (32, 4, 5, 1), (32, 4, 7, 2), (16, 2, 11, 2)):
        H = W = 24
        x = torch.randn(1, H, W, C)
        a = NeighborhoodAttentionBias(C, num_heads=heads, kernel_size=K, dilation=D, use_rpb=True)
        b = NATTEN(C, kernel_size=K, dilation=D, num_heads=heads)
        with torch.no_grad():
            b.qkv.weight.copy_(a.qkv.weight)
            b.qkv.bias.copy_(a.qkv.bias)
            b.proj.weight.copy_(a.proj.weight)
            b.proj.bias.copy_(a.proj.bias)
            b.rpb.zero_()
            a.rpb[2].weight.zero_()
            a.rpb[2].bias.zero_()
        ya, yb = a(x), b(x)
        m = D * (K - 1) // 2 + 1
        sl = (slice(0, 1), slice(m, H - m), slice(m, W - m))
        worst = max(worst, (ya[sl] - yb[sl]).abs().max().item())
    print(f'  [1] 自定义 NA vs natten (内部区域) 最大误差 = {worst:.2e} -> '
          f'{OK if worst < 1e-4 else FAIL}')

    C, K, D = 8, 3, 2
    x = torch.randn(1, 6, 6, C)
    m = NeighborhoodAttentionBias(C, num_heads=1, kernel_size=K, dilation=D, use_rpb=False)
    with torch.no_grad():
        w = m.qkv.weight.view(3, C, C)
        w[0].zero_()
        w[1].zero_()
        w[2] = torch.eye(C)
        m.qkv.weight.copy_(w.reshape(m.qkv.weight.shape))
        m.qkv.bias.zero_()
        m.proj.weight.copy_(torch.eye(C))
        m.proj.bias.zero_()
    err = (m(x) - naive_na(x, K, D)).abs().max().item()
    print(f'  [1] 自定义 NA vs 朴素零填充参考 (含边界) 最大误差 = {err:.2e} -> '
          f'{OK if err < 1e-4 else FAIL}')


def test_mask_guided_bias():
    """左半为阴影、右半为非阴影；核 3x3 无膨胀。"""
    H = W = 12
    m = torch.zeros(1, 1, H, W)
    m[:, :, :, :6] = 1.0                                        # 左半 = 阴影
    gen = MaskGuidedBias(num_heads=1, kernel_size=3, dilation=1,
                         use_illu=False, use_gate=True, hard_gate=True, gate_init=1.0)
    gen.set_mask(m)
    bias = gen(0, H)                                            # (1, 1, L, 9)
    b = bias[0, 0]                                              # (L, 9)

    def idx_of(dy, dx):
        return (dy + 1) * 3 + (dx + 1)

    # a) 深处阴影: 邻域全为阴影 -> fallback，偏置应为 0
    deep = b[3 * W + 2]
    ok_a = deep.abs().max().item() < 1e-6
    # b) 紧邻边界的阴影像素(列 5): 邻域含非阴影(列 6) -> 只有阴影 key 被抑制
    near = b[3 * W + 5]
    sup = [idx_of(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0)]      # 列 4/5 均为阴影
    keep = [idx_of(dy, 1) for dy in (-1, 0, 1)]                        # 列 6 为非阴影
    ok_b = (all(near[i].item() < -1e3 for i in sup)
            and all(abs(near[i].item()) < 1e-6 for i in keep))
    # c) 非阴影像素: 不做任何抑制
    non = b[3 * W + 7]
    ok_c = non.abs().max().item() < 1e-6
    print(f'  [2] NS-Gate: 全阴影邻域 fallback={ok_a} | 边界阴影像素只抑制阴影 key={ok_b} '
          f'| 非阴影像素不抑制={ok_c} -> {OK if (ok_a and ok_b and ok_c) else FAIL}')
    print(f'      边界像素 9 个 key 的偏置 = {[round(v, 1) for v in near.tolist()]}')


@torch.no_grad()
def test_model():
    H = W = 48
    x = torch.randn(1, 3, H, W)
    xm = (torch.rand(1, 1, H, W) > 0.6).float()
    for preset in ('rasm', 'mgra', 'nsrc', 'film', 'rgate', 'full'):
        model = RASMv2(img_size=H, embed_dim=32, win_size=10, token_projection='linear',
                       token_mlp='leff', preset=preset, kernel_size=11, dilations=(2, 2, 2))
        model.eval()
        y = model(x, xm)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        shape_ok = tuple(y.shape) == (1, 3, H, W)
        finite = torch.isfinite(y).all().item()
        print(f'  [3] preset={preset:<5s} params={n:.2f}M shape_ok={shape_ok} finite={finite} -> '
              f'{OK if (shape_ok and finite) else FAIL}', flush=True)


def test_backward():
    # 用"中心块状阴影"而不是随机 mask: 保证下采样后每个尺度都同时存在阴影/非阴影 token，
    # 否则 NS-Gate 的 fallback、NSRC 的 res=0 会让这三项天然没有梯度(是数据问题不是代码问题)
    H = W = 32
    x = torch.randn(1, 3, H, W)
    xm = torch.zeros(1, 1, H, W)
    xm[:, :, H // 4:3 * H // 4, W // 4:3 * W // 4] = 1.0
    model = RASMv2(img_size=H, embed_dim=32, win_size=10, token_projection='linear',
                   token_mlp='leff', preset='full', kernel_size=11)
    y = model(x, xm)
    y.pow(2).mean().backward()
    targets = {
        'rpb(相对位置偏置)': 'attn.rpb.2.bias',
        'RIB(光照关系偏置)': 'bias_gen.illu',
        'NS-Gate 强度': 'bias_gen.gate_raw',
        'NSRC 投影': 'nsrc.proj.weight',
        'Mask-FiLM 增益': 'film.gain',
        '残差门控 beta': 'res_gate_raw',
    }
    bad = []
    for name, key in targets.items():
        hit = [n for n, p in model.named_parameters() if n.endswith(key)
               and p.grad is not None and p.grad.abs().sum() > 0]
        if not hit:
            bad.append(name)
    print(f'  [4] 新增参数梯度通路 {len(targets) - len(bad)}/{len(targets)} -> '
          f'{OK if not bad else "FAIL: " + str(bad)}', flush=True)
    print(f'      残差门控 beta = {F.softplus(model.res_gate_raw).item():.3f} '
          f'(初始化应≈0.1)', flush=True)


def test_load_official():
    ckpt = './log/RASM_istd_demo/models/AISTD.pth'
    if not os.path.isfile(ckpt):
        print('  [5] 跳过: 未找到官方权重', ckpt)
        return
    model = RASMv2(img_size=320, embed_dim=32, win_size=10, token_projection='linear',
                   token_mlp='leff', preset='rasm')
    out = load_weights(model, ckpt, strict=False)
    n_missing, n_unexp = len(out.missing_keys), len(out.unexpected_keys)
    # 允许缺失的只有 v2 新增的参数（rpb 等）；不允许出现"官方有但 v2 结构对不上"的冗余项
    new_only = all(any(t in k for t in ('rpb', 'illu', 'gate_raw', 'nsrc', 'film', 'res_gate'))
                   for k in out.missing_keys)
    ok = (n_unexp == 0) and new_only
    print(f'  [5] 加载官方权重 {os.path.basename(ckpt)}: 缺失 {n_missing} 项(均为 v2 新增={new_only}), '
          f'结构对不上的冗余项 {n_unexp} 项 -> {OK if ok else FAIL}', flush=True)


@torch.no_grad()
def test_losses():
    """验证三项新损失"是否真的做了它们宣称的事"。"""
    from improved.rasmv2_losses import (BoundaryLoss, NonShadowConsistencyLoss,
                                       ChromaticityLoss, _dilate, _erode)
    B, H, W = 1, 32, 32
    gt = torch.rand(B, 3, H, W).clamp(0.1, 0.9)
    m = torch.zeros(B, 1, H, W)
    m[:, :, 8:24, 8:24] = 1.0

    # (a) 非阴影一致性: 容忍全局常量偏移，但惩罚局部不一致
    ns = NonShadowConsistencyLoss(pool=9)
    l_const = ns(gt + 0.05, gt, m).item()                      # 纯常量偏移 -> 应≈0
    l_noise = ns(gt + 0.05 * torch.randn(B, 3, H, W), gt, m).item()
    ok_a = l_const < 0.02 and l_noise > l_const

    # (b) 色度一致性: 对纯光照衰减不敏感(尺度不变)，对偏色敏感
    ch = ChromaticityLoss(min_lum=0.05)
    l_dim = ch(gt * 0.5, gt).item()                            # 只变强度 -> 应≈0
    l_hue = ch(gt * torch.tensor([1.0, 0.7, 1.0]).view(1, 3, 1, 1), gt).item()
    ok_b = l_dim < 0.02 and l_hue > l_dim

    # (c) 半影区加权: 同等强度的误差，落在半影带上比落在平坦区上"更贵"；
    #     同时整体量级与朴素 L1 相当(权重已按 1/w̄ 归一化)
    bl = BoundaryLoss(band=9, w_shadow=0.0, w_band=1.0)
    band = (_dilate(m, 9) - _erode(m, 9)).clamp(0, 1)
    l_in = bl(gt + 0.1 * band, gt, m).item()
    l_out = bl(gt + 0.1 * (1 - band), gt, m).item()
    l_flat = bl(gt + 0.1 * torch.ones(B, 3, H, W), gt, torch.zeros(B, 1, H, W)).item()
    ok_c = l_in > l_out and abs(l_flat - 0.1) < 0.02
    print(f'  [6] 损失性质: 非阴影损失容忍常量偏移={ok_a} (常量 {l_const:.4f} vs 噪声 {l_noise:.4f}) '
          f'| 色度损失对减光不敏感={ok_b} (减光 {l_dim:.4f} vs 偏色 {l_hue:.4f}) '
          f'| 半影带内误差更贵={ok_c} (带内 {l_in:.4f} vs 带外 {l_out:.4f}, 量级 {l_flat:.4f}) -> '
          f'{OK if (ok_a and ok_b and ok_c) else FAIL}', flush=True)


if __name__ == '__main__':
    # 可只跑部分检查: python improved/selftest.py 3 6
    all_tests = {'1': test_attn_vs_natten, '2': test_mask_guided_bias, '3': test_model,
                 '4': test_backward, '5': test_load_official, '6': test_losses}
    sel = [a for a in sys.argv[1:] if a in all_tests] or list(all_tests)
    print('== RASMv2 自检 (%s) ==' % ','.join(sel), flush=True)
    for k in sel:
        all_tests[k]()
    print('== 完成 ==', flush=True)
