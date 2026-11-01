#!/bin/bash
# ============================================================
# RASM 交叉测试脚本 —— 用 ISTD+ 权重 测 SRD 数据集
# （即：在 ISTD+ 上预训练的 RASM 权重，跑 SRD_ABC/test）
# 在 RASM/ 目录下执行: bash test_ISTD_on_SRD.sh
# 结构参数必须与权重一致（RASM 四项），否则 state_dict 加载失败。
# 强制无缓冲输出，保证 tqdm 进度条/日志实时刷新。
# ============================================================
export PYTHONUNBUFFERED=1

# ---------- 路径（交叉：ISTD 权重 + SRD 测试集）----------
INPUT_DIR="datasets/SRD_ABC/test"
WEIGHTS="./log/RASM_istd_demo/models/AISTD.pth"   # ISTD+ 上预训练的权重
RESULT_DIR="./results_istd_on_srd"                # 跨测产物单独目录，避免与同测混淆
EX_NAME="RASM_istd_on_srd"

# ---------- 设备 ----------
GPU_DEV="0"                      # 使用第 0 号 GPU；多卡可写 '0,1' 等

# ---------- 模型结构参数（须与 ISTD 权重一致）----------
ARCH="RASM"
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"
TILE=""                          # SRD 原图较大，可分块推理，例如设 720；留空=原分辨率

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
#   --save_images  去阴影结果保存到 ./results_istd_on_srd
#   --cal_metrics  输出 PSNR / SSIM / RMSE（全图、阴影区、非阴影区）
#   结构化指标追加写入 ./results_istd_on_srd/metrics.jsonl
# 论文标准量化另见 evaluation/measure_shadow.m（需 MATLAB）。
