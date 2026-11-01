#!/bin/bash
# ============================================================
# RASMv2 训练脚本 —— ISTD+（在 RASM/ 目录下执行: bash improved/train_ISTD_v2.sh）
# 与既有 train_ISTD.sh 完全并列，只是换成 RASMv2 的改进模型/损失。
# 数据目录: datasets/ISTD_Dataset/{train,test}/{A,B,C}
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新
export PYTHONUNBUFFERED=1
# ============================================================

cd "$(dirname "$0")/.." || exit 1          # 切到 RASM/ 根目录，保证 import utils/model 可用

# ---------- 路径 ----------
TRAIN_DIR="datasets/ISTD_Dataset/train"
VAL_DIR="datasets/ISTD_Dataset/test"
ENV_NAME="_istd"                 # 输出目录: ./log/RASMv2_istd/<run_name>
DATASET="ISTD"
# 想直接在官方 RASM 权重上续训: 打开 --resume 并把 PRETRAIN_WEIGHTS 指向它
PRETRAIN_WEIGHTS="./log/RASM_istd_demo/models/AISTD.pth"
# 固定默认 run_name：产物路径确定，test_ISTD_v2.sh 的默认 WEIGHTS 才能直接对上
RUN_NAME="${RUN_NAME:-v2_istd}"

# ---------- 设备 ----------
GPU_DEV="0"

# ---------- 训练超参（沿用论文默认）----------
TRAIN_PS=320
BATCH_SIZE=4
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
PRESET="full"          # rasm(基线=论文RASM) / mgra / nsrc / film / rgate / full(全部)
KERNEL_SIZE=11         # 区域注意力邻域边长(论文默认 11)
DILATIONS="2 2 2"      # 三个区域注意力层膨胀率; 想做感受野尺度自适应可改 "2 1 2"
W_BOUNDARY=1.0         # 半影区加权损失权重
W_NS=0.5               # 非阴影一致性损失权重
W_CHROMA=0.2           # 色度一致性损失权重

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

# 续训(可直接吃官方 RASM 权重):
#   在命令末尾加 --resume，PRETRAIN_WEIGHTS 指向 ./log/RASM_istd_demo/models/AISTD.pth
#   缺失的只有 v2 新增的 RPB 等少量参数(随机初始化)，其余全部命中。
