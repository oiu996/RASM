#!/bin/bash
# ============================================================
# RASMv2 训练脚本 —— SRD（在 RASM/ 目录下执行: bash improved/train_SRD_v2.sh）
# 与既有 train_SRD.sh 完全并列，只是换成 RASMv2 的改进模型/损失。
# 数据目录: datasets/SRD_ABC/{train,test}/{A,B,C}
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新
export PYTHONUNBUFFERED=1
# ============================================================

cd "$(dirname "$0")/.." || exit 1          # 切到 RASM/ 根目录，保证 import utils/model 可用

# ---------- 路径 ----------
TRAIN_DIR="datasets/SRD_ABC/train"
VAL_DIR="datasets/SRD_ABC/test"
ENV_NAME="_srd"
DATASET="SRD"
PRETRAIN_WEIGHTS="./log/RASM_srd_demo/models/SRD.pth"
# 固定默认 run_name：产物路径确定，test_SRD_v2.sh 的默认 WEIGHTS 才能直接对上
RUN_NAME="${RUN_NAME:-v2_srd}"

# ---------- 设备 ----------
GPU_DEV="0"

# ---------- 训练超参 ----------
TRAIN_PS=320
BATCH_SIZE=2                     # SRD 原图较大，按显存调整
NEPOCH=1000
LR_INITIAL=0.0004
WARMUP_EPOCHS=3
OPTIMIZER="adamw"
WEIGHT_DECAY=0.02

# ---------- 模型结构参数 ----------
ARCH="RASMv2"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"

# ---------- RASMv2 改进开关 ----------
PRESET="full"
KERNEL_SIZE=11
DILATIONS="2 2 2"
W_BOUNDARY=1.0
W_NS=0.5
W_CHROMA=0.2

python -u improved/train_v2.py \
    --arch ${ARCH} \
    --preset ${PRESET} \
    --kernel_size ${KERNEL_SIZE} \
    --dilations ${DILATIONS} \
    --w_boundary ${W_BOUNDARY} \
    --w_ns ${W_NS} \
    --w_chroma ${W_CHROMA} \
    --warmup \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
    --train_ps ${TRAIN_PS} \
    --batch_size ${BATCH_SIZE} \
    --nepoch ${NEPOCH} \
    --gpu "${GPU_DEV}" \
    --env ${ENV_NAME} \
    --train_dir ${TRAIN_DIR} \
    --val_dir ${VAL_DIR} \
    --lr_initial ${LR_INITIAL} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --optimizer ${OPTIMIZER} \
    --weight_decay ${WEIGHT_DECAY} \
    --embed_dim ${EMBED_DIM} \
    --win_size ${WIN_SIZE} \
    --token_projection ${TOKEN_PROJECTION} \
    --token_mlp ${TOKEN_MLP} \
    --dataset ${DATASET} \
    ${RUN_NAME:+--run_name ${RUN_NAME}}
