#!/bin/bash
# ============================================================
# RASM 样例效果展示 —— 一键跑通（需要 GPU）
# 在 RASM/ 根目录执行：  bash showcase/run_demo.sh
#
# 做三件事：
#   1) 从测试集挑 10 张，复制 A(原图) / B(mask) / C(GT)          [CPU 也能跑]
#   2) 用两个权重分别推理，保存去阴影结果 + 指标                  [需要 GPU]
#   3) 拼成"输入 / mask / GT / 权重1 / 权重2"对比图 + 局部放大图  [CPU 也能跑]
# ============================================================
set -euo pipefail
export PYTHONUNBUFFERED=1

# 自动切到仓库根目录（脚本可以在任意位置执行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." && echo "工作目录: $(pwd)"

# ---------- 基本配置 ----------
GPU_DEV="0"        # 用哪块卡；多卡可写 '0,1'
NUM=10             # 挑选样例张数
TILE=""            # 留空 = 原分辨率（优先）；若显存不足，脚本会自动退回 TILE=256 分块推理，无需手动改
SKIP_SELECT=0      # 样例已选好、只想重跑推理时改成 1

# ---------- 两个"原模型权重"（可自行替换）----------
# 格式：名字=路径，名字会作为输出子目录名和图表上的列标题
WEIGHTS=(
  "srd=log/RASM_srd_demo/models/SRD.pth"        # SRD 上训练的官方权重
  "istd=log/RASM_istd_demo/models/AISTD.pth"    # ISTD+ 上训练的官方权重（跨数据集）
)

# ---------- 1. 挑样例 + 复制 A/B/C ----------
if [ "${SKIP_SELECT}" -eq 0 ]; then
  echo "===== [1/3] 挑选 ${NUM} 张样例 ====="
  python -u showcase/scripts/select_samples.py --num ${NUM}
fi

# ---------- 2. GPU 推理 ----------
echo "===== [2/3] 两个权重分别推理 ====="
python -u showcase/scripts/run_inference.py \
    --weights "${WEIGHTS[@]}" \
    ${TILE:+--tile ${TILE}} \
    --gpus "${GPU_DEV}"

# ---------- 3. 拼对比图 ----------
echo "===== [3/3] 生成对比图 ====="
python -u showcase/scripts/make_figures.py

echo "===== 完成 ====="
echo "逐样例对比图: showcase/compare/"
echo "局部放大图:   showcase/compare/zoom/"
echo "总览大图:     showcase/compare/all_samples_grid.png"
echo "指标表格:     showcase/metrics.md"
