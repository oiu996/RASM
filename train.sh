#!/bin/bash
# ============================================================
# RASM 训练脚本（含具体参数，GPU 版）
# RASM 原版代码即为 GPU 版本（.cuda() / CUDA AMP / natten 均依赖 CUDA），
# 确保环境已安装带 CUDA 的 PyTorch（如 torch==2.1.2+cu118）及对应 natten 轮子即可直接运行。
# 数据目录（仓库自带示例）: datasets/ISTD_Dataset/{train,test}/{A,B,C}
#   train_A / test_A = 阴影图, *_B = 阴影 mask, *_C = 无阴影 GT
# ============================================================

# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新
export PYTHONUNBUFFERED=1

# ---------- 路径 ----------
TRAIN_DIR="datasets/ISTD_Dataset/train"
VAL_DIR="datasets/ISTD_Dataset/test"
SAVE_DIR="./log"                 # 注：train.py 内部硬编码 ./log，--save_dir 实际被忽略
ENV_NAME="_istd_demo"            # 输出目录名: ./log/RASM_istd_demo

# ---------- 设备 ----------
GPU_DEV="0"                      # 使用第 0 号 GPU；多卡可写 '0,1' 等

# ---------- 训练超参（论文默认）----------
TRAIN_PS=320                     # 训练 patch 尺寸
BATCH_SIZE=4                     # 批大小（视显存调整）
NEPOCH=1000                      # 示例数据小，冒烟测试可先设 50~100
LR_INITIAL=0.0004
WARMUP_EPOCHS=3
OPTIMIZER="adamw"
WEIGHT_DECAY=0.02

# ---------- 模型结构参数 ----------
ARCH="RASM"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"

python -u train.py \
    --arch ${ARCH} \
    --warmup \
    --train_ps ${TRAIN_PS} \
    --batch_size ${BATCH_SIZE} \
    --nepoch ${NEPOCH} \
    --gpu "${GPU_DEV}" \
    --env ${ENV_NAME} \
    --save_dir ${SAVE_DIR} \
    --train_dir ${TRAIN_DIR} \
    --val_dir ${VAL_DIR} \
    --lr_initial ${LR_INITIAL} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --optimizer ${OPTIMIZER} \
    --weight_decay ${WEIGHT_DECAY} \
    --embed_dim ${EMBED_DIM} \
    --win_size ${WIN_SIZE} \
    --token_projection ${TOKEN_PROJECTION} \
    --token_mlp ${TOKEN_MLP}

# 训练产物: ./log/RASM_istd_demo/models/model_best.pth
