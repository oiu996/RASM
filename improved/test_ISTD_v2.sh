#!/bin/bash
# ============================================================
# RASMv2 测试脚本 —— ISTD+（在 RASM/ 目录下执行: bash improved/test_ISTD_v2.sh）
# preset / kernel_size / dilations 必须与训练时一致。
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新
export PYTHONUNBUFFERED=1
# ============================================================

cd "$(dirname "$0")/.." || exit 1

INPUT_DIR="datasets/ISTD_Dataset/test"
WEIGHTS="./log/RASMv2_istd/${RUN_NAME:-v2_istd}/models/model_best.pth"
RESULT_DIR="./results_istd_v2"
EX_NAME="RASMv2_istd"
GPU_DEV="0"

ARCH="RASMv2"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"

# ---------- 与训练保持一致 ----------
PRESET="full"
KERNEL_SIZE=11
DILATIONS="2 2 2"
TILE=""                          # 留空=原分辨率推理

python -u improved/test_v2.py \
    --arch ${ARCH} \
    --preset ${PRESET} \
    --kernel_size ${KERNEL_SIZE} \
    --dilations ${DILATIONS} \
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

# 用官方 RASM 权重跑 RASMv2 基线(preset=rasm)做对照:
#   PRESET="rasm" 且 WEIGHTS="./log/RASM_istd_demo/models/AISTD.pth"
