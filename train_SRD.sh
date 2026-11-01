#!/bin/bash
# ============================================================
# RASM 训练脚本 —— SRD 数据集（含具体参数，GPU 版）
# RASM 原版代码即为 GPU 版本（.cuda() / CUDA AMP / natten 依赖 CUDA），
# 确保环境已安装带 CUDA 的 PyTorch 及对应 natten 轮子即可直接运行。
# 数据目录（仓库自带示例）: datasets/SRD_ABC/{train,test}/{A,B,C}
#   train_A / test_A = 阴影图, *_B = 阴影 mask, *_C = 无阴影 GT
# 说明：SRD 原图分辨率普遍较大，可按显存把 BATCH_SIZE 调小。
# ============================================================

# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新
export PYTHONUNBUFFERED=1

# ---------- 路径 ----------
TRAIN_DIR="datasets/SRD_ABC/train"
VAL_DIR="datasets/SRD_ABC/test"
SAVE_DIR="./log"                 # 注：train.py 内部硬编码 ./log，--save_dir 实际被忽略
ENV_NAME="_srd_demo"             # 输出目录名: ./log/RASM_srd_demo
PRETRAIN_WEIGHTS="./log/RASM_srd_demo/models/SRD.pth"      # 官方预训练权重；仅当命令加 --resume 时生效
DATASET="SRD"                    # 归档权重的自描述命名前缀: RASM_SRD_e32_w10_linear_leff_...
RUN_NAME=""                      # 方案D: 留空=自动用时间戳新建 run 目录；填目录名=写回该目录(中断后续跑)

# ---------- 设备 ----------
GPU_DEV="0"                      # 使用第 0 号 GPU；多卡可写 '0,1' 等

# ---------- 训练超参（论文默认，SRD 可沿用）----------
TRAIN_PS=320                     # 训练 patch 尺寸
BATCH_SIZE=2                     # SRD 大图，按显存调整
NEPOCH=1000                      # 示例数据小，冒烟测试可先设 50~100
LR_INITIAL=0.0004
WARMUP_EPOCHS=3
OPTIMIZER="adamw"
WEIGHT_DECAY=0.02

# ---------- 模型结构参数（须与测试一致）----------
ARCH="RASM"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"

python -u train.py \
    --arch ${ARCH} \
    --warmup \
    --pretrain_weights ${PRETRAIN_WEIGHTS} \
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
    --token_mlp ${TOKEN_MLP} \
    --dataset ${DATASET} \
    ${RUN_NAME:+--run_name ${RUN_NAME}}

# 训练产物（方案D: 每个 run 一个独立目录，实验之间互不覆盖）:
#   ./log/RASM_srd_demo/<run_name>/models/model_best.pth       <- 固定指针，--resume / 测试脚本都用它
#   ./log/RASM_srd_demo/<run_name>/models/RASM_SRD_e32_w10_linear_leff_ep0785_valpsnr36.21.pth
#                                                              <- 自描述归档副本（同 run 内只留最新一份）
#   ./log/RASM_srd_demo/<run_name>/config.json                 <- 本次 run 的完整超参，便于复现
#   ./log/RASM_srd_demo/<run_name>/train.log                   <- 训练日志
#   <run_name> 留空时自动取时间戳，例如 20261027-143000
#
# 续训（方案D 的关键优势）:
#   1) 加 --resume，并把 PRETRAIN_WEIGHTS 指向【上一次 run】的 model_best.pth
#   2) 新产物会写入【新的】run 目录 -> 历史实验完全不会被覆盖
#   3) best_psnr 会从 ckpt 恢复，避免用更差的权重覆盖刚加载的最优权重
#   4) 若训练意外中断、想接着【同一个】run 目录继续跑，把 RUN_NAME 填成该目录名即可
