#!/bin/bash
# ============================================================
# RASMv2 交叉测试脚本 —— 用 SRD 上训练的 v2 权重 测 ISTD 测试集
# （即：在 SRD 上训练的 RASMv2 权重，跑 datasets/ISTD_Dataset/test）
# 在 RASM/ 目录下执行: bash improved/test_SRD_on_ISTD_v2.sh
# preset / kernel_size / dilations 必须与训练时一致（默认 full / 11 / 2 2 2）。
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新。
# ============================================================
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")/.." || exit 1          # 切到 RASM/ 根目录，保证 import utils/model 可用

# ---------- 路径（交叉：SRD 上的 v2 权重 + ISTD 测试集）----------
INPUT_DIR="datasets/ISTD_Dataset/test"
# 默认指向 train_SRD_v2.sh 产出的固定 run 名 v2_srd；自定义 run 时用 RUN_NAME 覆盖
WEIGHTS="./log/RASMv2_srd/${RUN_NAME:-v2_srd}/models/model_best.pth"
RESULT_DIR="./results_srd_on_istd_v2"       # 跨测产物单独目录，避免与同测混淆
EX_NAME="RASMv2_srd_on_istd"

# ---------- 设备 ----------
GPU_DEV="0"                      # 使用第 0 号 GPU；多卡可写 '0,1' 等

# ---------- 模型结构参数（须与 SRD v2 训练时一致）----------
ARCH="RASMv2"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"

# ---------- RASMv2 改进开关（须与训练一致）----------
PRESET="full"          # rasm / mgra / nsrc / film / rgate / full
KERNEL_SIZE=11         # 区域注意力邻域边长
DILATIONS="2 2 2"      # 三个区域注意力层膨胀率
TILE=""                # 留空=原分辨率推理；ISTD 图较小一般无需分块

python -u improved/test_v2.py \
    --arch ${ARCH} \
    --dataset ISTD \
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

# 输出说明：
#   --save_images  去阴影结果保存到 ./results_srd_on_istd_v2
#   --cal_metrics  输出 PSNR / SSIM / RMSE（全图、阴影区、非阴影区）
#   结构化指标追加写入 ./results_srd_on_istd_v2/metrics_v2.jsonl
# 论文标准量化另见 evaluation/measure_shadow.m（需 MATLAB）。
