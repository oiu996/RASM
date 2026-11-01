#!/bin/bash
# ============================================================
# RASM 自定义图片盲测 —— 一键跑通（需要 GPU）
# 在 RASM/ 根目录执行：  bash showcase/custom/run_custom.sh
#
# 前提：把你自己带的图（jpg/png，命名随意，无需 mask / GT）放进
#       showcase/custom/input/   目录即可。
#
# 掩码来源（脚本自动判断，无需手动切换）：
#   - ORIGIN：无 mask 时用"亮度差"自动估一个粗 mask 仅驱动模型，
#             存 outputs/ORIGIN_masks/，图上明确标注"非真值"。
#   - DHAN  ：若 showcase/custom/outputs/DHAN_masks/ 存在且含掩码，
#             则用它作为掩码再跑一遍（DHAN 生成的阴影掩码）。
#   - UNION ：ORIGIN 与 DHAN 的并集（任一判为阴影即阴影），存 outputs/UNION_masks/。
#             （DHAN_masks 不存在时只跑 ORIGIN 一遍。）
#
# 产物（按掩码来源分三类）：
#   outputs/maskORIGIN/<权重>/...   outputs/maskDHAN/<权重>/...   outputs/maskUNION/<权重>/...
#   compare/maskORIGIN/  compare/maskDHAN/  compare/maskUNION/（各含 zoom/）
#   compare/zoomTOTAL/                       （各掩码来源 4 列一组，组间留空白）
# ============================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

# 自动切到仓库根目录（脚本可在任意位置执行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." && echo "工作目录: $(pwd)"

GPU_DEV="0"        # 用哪块卡
TILE=""            # 显存不足时改成 256（分块推理）

echo "===== 推理（需 GPU）====="
python -u showcase/custom/scripts/run_inference_custom.py --gpus "$GPU_DEV" ${TILE:+--tile $TILE}

echo "===== 拼图 ====="
python -u showcase/custom/scripts/make_figures_custom.py

echo "===== 完成 ====="
echo "对比图(ORIGIN): showcase/custom/compare/maskORIGIN/"
echo "对比图(DHAN)  : showcase/custom/compare/maskDHAN/   （仅当存在 DHAN_masks 时）"
echo "总览放大对比   : showcase/custom/compare/zoomTOTAL/  （仅当两种掩码都跑时）"
