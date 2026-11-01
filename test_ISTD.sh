#!/bin/bash
# ============================================================
# RASM 测试脚本 —— ISTD+（含具体参数，GPU 版）
# 原版代码即为 GPU 版本，确保环境有可用 CUDA 即可直接运行。
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新（不再"卡住")
export PYTHONUNBUFFERED=1

# ---------- 路径 ----------
INPUT_DIR="datasets/ISTD_Dataset/test"
# 权重：官方预训练权重（已手动放置），目录与 train.py 硬编码的 ./log 保持一致
WEIGHTS="./log/RASM_istd_demo/models/AISTD.pth"
RESULT_DIR="./results_istd"
EX_NAME="RASM_istd_demo"

# ---------- 设备 ----------
GPU_DEV="0"                      # 使用第 0 号 GPU；多卡可写 '0,1' 等

# ---------- 模型结构参数（须与训练一致）----------
ARCH="RASM"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"
TILE=""                          # 留空=原分辨率推理；大图可设 720 做分块

python -u test.py \
    --arch ${ARCH} \
    --input_dir ${INPUT_DIR} \
    --weights ${WEIGHTS} \
    --result_dir ${RESULT_DIR} \
    --ex_name ${EX_NAME} \
    --save_images \
    --cal_metrics \
    --embed_dim ${EMBED_DIM} \
    --win_size ${WIN_SIZE} \
    --token_projection ${TOKEN_PROJECTION} \
    --token_mlp ${TOKEN_MLP} \
    ${TILE:+--tile ${TILE}} \
    --gpus "${GPU_DEV}"

# 输出说明：
#   --save_images  去阴影结果保存到 ./results_istd
#   --cal_metrics  输出 PSNR / SSIM / RMSE（全图、阴影区、非阴影区）
# 论文标准量化另见 evaluation/measure_shadow.m（需 MATLAB）。
