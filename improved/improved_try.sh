#!/bin/bash
# ============================================================
# RASMv2 —— GPU 环境下的"官方权重续训 + 验证"一键脚本
#
# 在 /workspace/RASM 目录下执行（脚本内部会自动 cd 到 RASM/）：
#   bash improved/improved_try.sh                      # 默认: ISTD + preset=full + 300 epoch
#   bash improved/improved_try.sh istd full 300        # 显式指定 数据集/预设/epoch
#   bash improved/improved_try.sh srd mgra 200
#   NEPOCH=20 bash improved/improved_try.sh istd full  # 冒烟：先确认能跑通再放开 epoch
#   FROM_SCRATCH=1 bash improved/improved_try.sh istd full 300   # 不续训，从零训（lr 4e-4）
#   INIT_CKPT=./log/RASMv2_istd/<run>/models/model_best.pth NEPOCH=600 \
#       bash improved/improved_try.sh istd full        # 从"上一次 v2 run"断点续跑
#   WEIGHTS=./log/.../model_best.pth bash improved/improved_try.sh istd full  # 只测试，不训练
#   SKIP_BASELINE=1 bash improved/improved_try.sh istd full      # 跳过官方权重基线对照
#
# 流程：
#   [1] GPU 环境体检                [2] 官方权重剥离 optimizer -> 纯权重 init
#   [3] 加载体检（0 结构冲突）      [4] 续训                [5] 测试 best
#   [6] 官方权重基线对照（preset=rasm + AISTD.pth，用于确认基础设施没坏）
# ============================================================

set -eo pipefail
cd "$(dirname "$0")/.." || exit 1        # 切到 RASM/ 根目录，保证 import utils/model 可用

# ---------------- 入参与开关 ----------------
DS="$(echo "${1:-istd}" | tr 'A-Z' 'a-z')"       # istd | srd
PRESET="${2:-full}"                              # rasm | mgra | nsrc | film | rgate | full
NEPOCH_ARG="${3:-}"                              # 可选：覆盖默认 epoch
GPU_DEV="${GPU_DEV:-0}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"                # 1 = 不加载官方权重，从零训（基线对照用）
SKIP_BASELINE="${SKIP_BASELINE:-0}"              # 1 = 跳过 [6] 官方权重基线测试
CHECK_LOAD="${CHECK_LOAD:-1}"                    # 1 = 训练前做一次权重加载体检
INIT_CKPT="${INIT_CKPT:-}"                       # 可选：上次 v2 run 的 ckpt（带 epoch/optimizer）
WEIGHTS="${WEIGHTS:-}"                           # 可选：只测试，不训练

# ---------------- 数据集相关配置 ----------------
case "${DS}" in
  istd|ISTD)
    DS="istd"; DATASET="ISTD"
    TRAIN_DIR="datasets/ISTD_Dataset/train"
    VAL_DIR="datasets/ISTD_Dataset/test"
    TEST_DIR="datasets/ISTD_Dataset/test"
    OFFICIAL_CKPT="./log/RASM_istd_demo/models/AISTD.pth"
    ENV_NAME="_istd"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    DEF_NEPOCH=300
    TILE=""
    ;;
  srd|SRD)
    DS="srd"; DATASET="SRD"
    TRAIN_DIR="datasets/SRD_ABC/train"
    VAL_DIR="datasets/SRD_ABC/test"
    TEST_DIR="datasets/SRD_ABC/test"
    OFFICIAL_CKPT="./log/RASM_srd_demo/models/SRD.pth"
    ENV_NAME="_srd"
    BATCH_SIZE="${BATCH_SIZE:-2}"                # SRD 原图较大，按显存调整
    DEF_NEPOCH=200
    TILE="${TILE:-}"                             # 显存不够可设 720 走分块推理
    ;;
  *) echo "未知数据集: ${DS}（只支持 istd / srd）"; exit 1 ;;
esac

NEPOCH="${NEPOCH_ARG:-${NEPOCH:-${DEF_NEPOCH}}}"

# ---------------- 训练超参 ----------------
if [ "${FROM_SCRATCH}" = "1" ]; then
  LR="${LR:-0.0004}"                             # 从零训：论文默认
else
  LR="${LR:-0.0001}"                             # 续训：官方权重已收敛，用小学习率
fi
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
OPTIMIZER="adamw"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
TRAIN_PS="${TRAIN_PS:-320}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"

# ---------------- 模型结构（须与测试一致）----------------
EMBED_DIM=32
WIN_SIZE=10
TOKEN_PROJECTION="linear"
TOKEN_MLP="leff"
KERNEL_SIZE="${KERNEL_SIZE:-11}"
DILATIONS="${DILATIONS:-2 2 2}"                  # 想验证感受野尺度自适应可改 "2 1 2"
W_BOUNDARY="${W_BOUNDARY:-1.0}"
W_NS="${W_NS:-0.5}"
W_CHROMA="${W_CHROMA:-0.2}"

STAMP="$(date +%m%d-%H%M%S)"
RUN_NAME="${RUN_NAME:-ft_${PRESET}_ep${NEPOCH}_lr${LR}_${STAMP}}"
LOG_DIR="./log/RASMv2${ENV_NAME}/${RUN_NAME}"
MODELS_DIR="${LOG_DIR}/models"
INIT_DIR="./log/RASMv2${ENV_NAME}/_init"
INIT_WO="${INIT_DIR}/official_weights_only.pth"
RESULT_DIR="./results_${DS}_v2/${RUN_NAME}"

echo "================ RASMv2 续训 ================"
echo "数据集=${DATASET}  preset=${PRESET}  epoch=${NEPOCH}  lr=${LR}  batch=${BATCH_SIZE}  gpu=${GPU_DEV}"
echo "run 目录=${LOG_DIR}"
echo "============================================="

# ---------------- [1] GPU 环境体检 ----------------
echo "===== [1/6] 环境体检 ====="
python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit('未检测到可用 CUDA：本脚本面向 GPU 环境（train_v2.py 全程 .cuda() + AMP）。')
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('GPU0:', torch.cuda.get_device_name(0),
      '| 显存 %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
try:
    import natten
    print('natten', natten.__version__)
except Exception as e:                            # model.py 顶层 import natten，缺失则无法构建模型
    print('[WARN] natten 不可用:', e)
    print('       RASMv2 默认用 unfold 注意力，但 model.py 顶层仍 import natten；'
          '请安装与 torch/CUDA 版本匹配的 natten 轮子。')
PY

# ---------------- [2] 准备 init 权重 ----------------
# 官方 ckpt 里带 epoch(ISTD=900 / SRD=1000) 与 optimizer 状态：
#   * epoch 会让 start_epoch 直接跳到 901/1001 —— SRD 若 nepoch<=1000 则一个 epoch 都不跑；
#   * optimizer 的动量是按【官方模型的参数顺序】存的，RASMv2 多了 24 项 RPB 参数，
#     torch 会因 param group 长度不一致直接拒绝（或被错位加载）。
# 因此续训时统一先剥离成"纯权重 + epoch=0"，用本脚本自己的 lr/调度从头跑 NEPOCH 个 epoch。
mkdir -p "${INIT_DIR}"
if [ -n "${INIT_CKPT}" ] && [ -f "${INIT_CKPT}" ]; then
  echo "===== [2/6] 使用自定义 ckpt（保留 epoch/optimizer，真·断点续跑）: ${INIT_CKPT} ====="
  INIT_FILE="${INIT_CKPT}"
  python - "${INIT_CKPT}" "${NEPOCH}" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu')
ep = int(ck.get('epoch', 0))
if ep + 1 > int(sys.argv[2]):
    sys.exit(f'ckpt 已训练到 epoch {ep}，但 NEPOCH={sys.argv[2]}：训练循环会是空的，请调大 NEPOCH。')
print(f'ckpt epoch={ep} -> 将从 epoch {ep + 1} 续跑到 {sys.argv[2]}')
PY
elif [ "${FROM_SCRATCH}" = "1" ]; then
  echo "===== [2/6] FROM_SCRATCH=1：不加载任何权重 ====="
  INIT_FILE=""
else
  echo "===== [2/6] 剥离官方 ckpt 的 optimizer/epoch -> 纯权重 init ====="
  INIT_FILE="${INIT_WO}"
  python - "${OFFICIAL_CKPT}" "${INIT_WO}" <<'PY'
import os, sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu')
sd = ck.get('state_dict', ck.get('model', ck))
sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
torch.save({'epoch': 0, 'state_dict': sd}, dst)
print(f'官方 ckpt epoch={ck.get("epoch", "?")} -> 已生成 {dst}（{len(sd)} 项，epoch 归零、optimizer 丢弃）')
PY
fi

# ---------------- [3] 权重加载体检 ----------------
if [ "${CHECK_LOAD}" = "1" ] && [ -n "${INIT_FILE}" ]; then
  echo "===== [3/6] 加载体检（确认 0 项结构冲突）====="
  PRESET="${PRESET}" INIT_FILE="${INIT_FILE}" KERNEL_SIZE="${KERNEL_SIZE}" \
  DILATIONS="${DILATIONS}" python - <<'PY'
import os, sys
sys.path.insert(0, '.')
import torch
from improved.rasmv2_model import RASMv2, load_weights
preset = os.environ['PRESET']
k = int(os.environ['KERNEL_SIZE'])
dil = tuple(int(x) for x in os.environ['DILATIONS'].split())
m = RASMv2(img_size=320, embed_dim=32, win_size=10, token_projection='linear', token_mlp='leff',
           preset=preset, kernel_size=k, dilations=dil)
out = load_weights(m, os.environ['INIT_FILE'], strict=False)
assert len(out.unexpected_keys) == 0, f'出现结构冲突: {out.unexpected_keys[:5]}'
print(f'preset={preset}: 参数 {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M | '
      f'缺失(随机初始化) {len(out.missing_keys)} 项 | 结构冲突 0 项 -> PASS')
PY
else
  echo "===== [3/6] 跳过加载体检 ====="
fi

# ---------------- [4] 续训 ----------------
RESUME_ARGS=""
if [ "${FROM_SCRATCH}" != "1" ] && [ -n "${INIT_FILE}" ]; then
  RESUME_ARGS="--resume --pretrain_weights ${INIT_FILE}"
fi

if [ -n "${WEIGHTS}" ]; then
  echo "===== [4/6] 跳过训练（已指定 WEIGHTS=${WEIGHTS}）====="
  MODELS_DIR="$(dirname "${WEIGHTS}")"
elif [ "${FROM_SCRATCH}" = "1" ]; then
  echo "===== [4/6] 从零训练 preset=${PRESET}（${NEPOCH} epoch）====="
  python improved/train_v2.py \
      --arch RASMv2 \
      --preset ${PRESET} \
      --kernel_size ${KERNEL_SIZE} \
      --dilations ${DILATIONS} \
      --w_boundary ${W_BOUNDARY} \
      --w_ns ${W_NS} \
      --w_chroma ${W_CHROMA} \
      --warmup \
      --train_ps ${TRAIN_PS} \
      --batch_size ${BATCH_SIZE} \
      --nepoch ${NEPOCH} \
      --gpu "${GPU_DEV}" \
      --env ${ENV_NAME} \
      --train_dir ${TRAIN_DIR} \
      --val_dir ${VAL_DIR} \
      --lr_initial ${LR} \
      --warmup_epochs ${WARMUP_EPOCHS} \
      --optimizer ${OPTIMIZER} \
      --weight_decay ${WEIGHT_DECAY} \
      --train_workers ${TRAIN_WORKERS} \
      --eval_workers ${EVAL_WORKERS} \
      --embed_dim ${EMBED_DIM} \
      --win_size ${WIN_SIZE} \
      --token_projection ${TOKEN_PROJECTION} \
      --token_mlp ${TOKEN_MLP} \
      --dataset ${DATASET} \
      --run_name ${RUN_NAME}
else
  echo "===== [4/6] 在官方权重上续训 preset=${PRESET}（${NEPOCH} epoch, lr=${LR}）====="
  python improved/train_v2.py \
      --arch RASMv2 \
      --preset ${PRESET} \
      --kernel_size ${KERNEL_SIZE} \
      --dilations ${DILATIONS} \
      --w_boundary ${W_BOUNDARY} \
      --w_ns ${W_NS} \
      --w_chroma ${W_CHROMA} \
      --warmup \
      ${RESUME_ARGS} \
      --train_ps ${TRAIN_PS} \
      --batch_size ${BATCH_SIZE} \
      --nepoch ${NEPOCH} \
      --gpu "${GPU_DEV}" \
      --env ${ENV_NAME} \
      --train_dir ${TRAIN_DIR} \
      --val_dir ${VAL_DIR} \
      --lr_initial ${LR} \
      --warmup_epochs ${WARMUP_EPOCHS} \
      --optimizer ${OPTIMIZER} \
      --weight_decay ${WEIGHT_DECAY} \
      --train_workers ${TRAIN_WORKERS} \
      --eval_workers ${EVAL_WORKERS} \
      --embed_dim ${EMBED_DIM} \
      --win_size ${WIN_SIZE} \
      --token_projection ${TOKEN_PROJECTION} \
      --token_mlp ${TOKEN_MLP} \
      --dataset ${DATASET} \
      --run_name ${RUN_NAME}
fi

BEST_PTH="${WEIGHTS:-${MODELS_DIR}/model_best.pth}"

# ---------------- [5] 测试本次产物 ----------------
echo "===== [5/6] 测试 ${BEST_PTH} ====="
python improved/test_v2.py \
    --arch RASMv2 \
    --dataset ${DATASET} \
    --preset ${PRESET} \
    --kernel_size ${KERNEL_SIZE} \
    --dilations ${DILATIONS} \
    --train_ps ${TRAIN_PS} \
    --input_dir ${TEST_DIR} \
    --weights ${BEST_PTH} \
    --result_dir ${RESULT_DIR} \
    --ex_name "RASMv2_${DS}_${PRESET}_ft" \
    --save_images \
    --cal_metrics \
    --embed_dim ${EMBED_DIM} \
    --win_size ${WIN_SIZE} \
    --token_projection ${TOKEN_PROJECTION} \
    --token_mlp ${TOKEN_MLP} \
    ${TILE:+--tile ${TILE}} \
    --gpus "${GPU_DEV}"

# ---------------- [6] 官方权重基线对照 ----------------
# preset=rasm 与官方模型同构（RPB 末层零初始化 => 初始等价），可直接吃 AISTD.pth / SRD.pth；
# 这一步不训练，只是确认"基础设施没坏"并给续训结果一个锚点。
if [ "${SKIP_BASELINE}" != "1" ]; then
  echo "===== [6/6] 基线对照：preset=rasm + 官方权重（不训练）====="
  python improved/test_v2.py \
      --arch RASMv2 \
      --dataset ${DATASET} \
      --preset rasm \
      --kernel_size ${KERNEL_SIZE} \
      --dilations ${DILATIONS} \
      --train_ps ${TRAIN_PS} \
      --input_dir ${TEST_DIR} \
      --weights ${OFFICIAL_CKPT} \
      --result_dir "./results_${DS}_v2/rasm_official_baseline" \
      --ex_name "RASM_official_${DS}" \
      --cal_metrics \
      --embed_dim ${EMBED_DIM} \
      --win_size ${WIN_SIZE} \
      --token_projection ${TOKEN_PROJECTION} \
      --token_mlp ${TOKEN_MLP} \
      ${TILE:+--tile ${TILE}} \
      --gpus "${GPU_DEV}"
else
  echo "===== [6/6] 跳过基线对照 ====="
fi

echo
echo "===== 产物 ====="
echo "  训练 run : ${LOG_DIR}   (models/model_best.pth, metrics.jsonl, train.log, config.json)"
echo "  测试指标 : ${RESULT_DIR}/metrics_v2.jsonl"
echo "  基线指标 : ./results_${DS}_v2/rasm_official_baseline/metrics_v2.jsonl"
