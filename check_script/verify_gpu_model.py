#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU 最小验证 (1/2)：模型构建 + 前向 + AMP 反向。
这一步足以暴露 natten 是否能正确编译/调用、AMP 是否在 GPU 上可用。

注意：本脚本必须在带 CUDA 的环境运行；CPU 上 natten 不可用会直接报错，属预期。
数据检查通过（check_script/test_dataset_cpu.py）只代表“数据层 OK”，与 GPU 训练链路是两回事。

运行（在 RASM 目录）:
    python check_script/verify_gpu_model.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # RASM 根目录
sys.path.insert(0, ROOT)

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from model import RASM

IMG_SIZE = 320
EMBED_DIM = 32
WIN_SIZE = 10


def main():
    assert torch.cuda.is_available(), "未检测到 CUDA，无法运行 GPU 验证"
    dev = torch.device("cuda", 0)

    print(
        ">> CUDA:",
        torch.version.cuda,
        "| torch:",
        torch.__version__,
        "| device:",
        torch.cuda.get_device_name(0),
    )

    # 1) 模型构建（最易暴露 natten 问题）
    model = RASM(
        img_size=IMG_SIZE,
        embed_dim=EMBED_DIM,
        win_size=WIN_SIZE,
        token_projection="linear",
        token_mlp="leff",
    ).to(dev)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f">> model ok, params={nparams:.2f}M")

    # 2) 前向（覆盖 natten 实际算子调用）
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)
    xm = (torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=dev) > 0.5).float()
    with torch.no_grad():
        y = model(x, xm)
    assert y.shape == x.shape, f"shape mismatch {tuple(y.shape)} vs {tuple(x.shape)}"
    assert torch.isfinite(y).all()
    print(f">> forward ok, out shape={tuple(y.shape)}")

    # 3) AMP 反向（覆盖 CUDA AMP 链路）
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = torch.randn_like(x)
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        out = model(x, xm)
        loss = torch.nn.functional.l1_loss(out, target)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).all()
    print(f">> backward ok, loss={loss.item():.4f}")

    print("ALL GPU MODEL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
