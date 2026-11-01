#!/bin/bash
# ============================================================
# RASM 训练“冒烟测试”脚本（GPU 版）
#
# 默认以极小配置跑通（NEPOCH=2, BATCH_SIZE=1），用于验证
# natten / 模型构建 / CUDA AMP 这套 GPU 专属训练链路能否打通。
# 能完整跑完一个 epoch 不报错，才算 GPU 训练链路打通。
#
# 覆盖变量即可复用于正式训练：
#   极小冒烟 : 默认（直接 bash check_script/train_smoke.sh）
#   正式训练 : NEPOCH=1000 BATCH_SIZE=2 bash check_script/train_smoke.sh
#
# 数据目录（仓库自带示例）: datasets/SRD_ABC/{train,test}/{A,B,C}
#   train_A / test_A = 阴影图, *_B = 阴影 mask, *_C = 无阴影 GT
# ============================================================

# ---------- 可被环境变量覆盖的默认值 ----------
TRAIN_DIR="${TRAIN_DIR:-datasets/SRD_ABC/train}"
VAL_DIR="${VAL_DIR:-datasets/SRD_ABC/test}"
SAVE_DIR="${SAVE_DIR:-./log}"
ENV_NAME="${ENV_NAME:-_srd_smoke}"
GPU_DEV="${GPU_DEV:-0}"

TRAIN_PS="${TRAIN_PS:-320}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NEPOCH="${NEPOCH:-2}"
LR_INITIAL="${LR_INITIAL:-0.0004}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"

ARCH="${ARCH:-RASM}"
EMBED_DIM="${EMBED_DIM:-32}"
WIN_SIZE="${WIN_SIZE:-10}"
TOKEN_PROJECTION="${TOKEN_PROJECTION:-linear}"
TOKEN_MLP="${TOKEN_MLP:-leff}"

# 切到 RASM 根目录（脚本位于 check_script/ 下），保证相对路径（datasets/、./log）正确
cd "$(dirname "$0")/.." || exit 1

python train.py \
    --arch "${ARCH}" \
    --warmup \
    --train_ps "${TRAIN_PS}" \
    --batch_size "${BATCH_SIZE}" \
    --nepoch "${NEPOCH}" \
    --gpu "${GPU_DEV}" \
    --env "${ENV_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --train_dir "${TRAIN_DIR}" \
    --val_dir "${VAL_DIR}" \
    --lr_initial "${LR_INITIAL}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --optimizer "${OPTIMIZER}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --embed_dim "${EMBED_DIM}" \
    --win_size "${WIN_SIZE}" \
    --token_projection "${TOKEN_PROJECTION}" \
    --token_mlp "${TOKEN_MLP}"

echo "训练冒烟测试完成（产物: ${SAVE_DIR}/RASM${ENV_NAME}/models/model_latest.pth）"
