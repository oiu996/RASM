#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 专属单元测试：覆盖 natten 模型构建 / 前向 / AMP 反向。

这些测试在 CPU 环境会被自动跳过（skipif CUDA 不可用），因此：
  * 数据层 OK（见 test_dataset_cpu.py）与 GPU 训练链路 OK 是两个独立层面；
  * 切到 GPU 后跑 `python -m pytest check_script/ -v`，本文件才会真正执行。

模型 forward / natten 算子仅在 CUDA 编译后可用；本文件刻意把
`from model import RASM` 放进函数内部做惰性导入，避免 CPU 下 import 失败。

运行（在 RASM 目录下，GPU 环境）:
    python -m pytest check_script/test_model_gpu.py -v
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # RASM 根目录
sys.path.insert(0, ROOT)

require_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="模型依赖 natten(CUDA)，仅 GPU 环境运行；CPU 自动跳过",
)

IMG_SIZE = 320
EMBED_DIM = 32
WIN_SIZE = 10


def _build_model():
    """惰性导入 model（带来 natten），避免在 CPU 下 import 即失败。"""
    from model import RASM

    return RASM(
        img_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        win_size=WIN_SIZE,
        token_projection="linear",
        token_mlp="leff",
    ).cuda().eval()


@require_cuda
def test_model_build_and_param_count():
    """第一步就足以暴露 natten 能否正确编译/调用。"""
    model = _build_model()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    assert nparams > 0
    print(f"model ok, params={nparams:.2f}M")


@require_cuda
def test_forward_shape_and_finite():
    """覆盖 natten 实际算子调用，检查输出形状与数值有限性。"""
    model = _build_model()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).cuda()
    xm = (torch.randn(1, 1, IMG_SIZE, IMG_SIZE).cuda() > 0.5).float()
    with torch.no_grad():
        y = model(x, xm)
    assert y.shape == x.shape, f"输出形状 {tuple(y.shape)} 与输入 {tuple(x.shape)} 不一致"
    assert torch.isfinite(y).all(), "前向输出出现 NaN/Inf"


@require_cuda
def test_backward_with_amp():
    """最小一步训练：验证 natten + CUDA AMP 梯度链路可打通。"""
    model = _build_model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).cuda()
    xm = (torch.randn(2, 1, IMG_SIZE, IMG_SIZE).cuda() > 0.5).float()
    target = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).cuda()

    opt.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(x, xm)
        loss = torch.nn.functional.l1_loss(out, target)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).all(), "loss 出现 NaN/Inf"
    # 确认确实产生了梯度
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads), "未产生有效梯度"
