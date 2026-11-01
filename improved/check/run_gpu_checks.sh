#!/bin/bash
# ============================================================
# RASMv2 训练/测试链路 —— GPU 环境一键体检
#
# 用法（任意目录执行，脚本内部会 cd 到 RASM/）:
#   bash improved/check/run_gpu_checks.sh                 # 全量（静态 + 自检 + 训练 + 测试）
#   bash improved/check/run_gpu_checks.sh static          # 只跑 CPU 静态检查
#   bash improved/check/run_gpu_checks.sh train           # 只跑训练链路
#   bash improved/check/run_gpu_checks.sh test            # 只跑测试链路
#   GPU_DEV=1 bash improved/check/run_gpu_checks.sh       # 指定显卡
#   CHECK_NEPOCH=41 CHECK_BATCH=4 bash improved/check/run_gpu_checks.sh
#
# 可调环境变量:
#   GPU_DEV          显卡号（默认 0）
#   CHECK_NEPOCH     端到端冒烟训练的 epoch（默认 21，>=21 才会触发一次验证）
#   CHECK_TRAIN_PS   端到端训练分辨率（默认 128）
#   CHECK_BATCH      端到端 batch（默认 2）
#   CHECK_ENV        log 子目录后缀（默认 _gpucheck -> log/RASMv2_gpucheck/）
#   CHECK_LOG_INTERVAL  训练心跳间隔（秒，默认 30；设 0 关闭。端到端训练会实时回显）
#   SKIP_SELFTEST=1  跳过 improved/selftest.py
#   PYTEST_ARGS      追加给 pytest 的参数，例如 "-x -k amp"
#
# 退出码非 0 表示有检查未通过。
# ============================================================

cd "$(dirname "$0")/../.." || exit 1          # -> RASM/
set -o pipefail

STAGE="${1:-all}"
GPU_DEV="${GPU_DEV:-0}"
CHECK_NEPOCH="${CHECK_NEPOCH:-21}"
CHECK_TRAIN_PS="${CHECK_TRAIN_PS:-128}"
CHECK_BATCH="${CHECK_BATCH:-2}"
CHECK_ENV="${CHECK_ENV:-_gpucheck}"
CHECK_LOG_INTERVAL="${CHECK_LOG_INTERVAL:-30}"
export GPU_DEV CHECK_NEPOCH CHECK_TRAIN_PS CHECK_BATCH CHECK_ENV CHECK_LOG_INTERVAL
export CUDA_VISIBLE_DEVICES="${GPU_DEV}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

FAILED=0
step() { echo; echo "==================== $* ===================="; }
run() { "$@"; local rc=$?; [ $rc -ne 0 ] && FAILED=1; return $rc; }
# 既有自检脚本：结果仅提示，不计入成败（它测的是改进模块本身，不是 train/test 链路）
run_soft() { "$@"; local rc=$?; [ $rc -ne 0 ] && echo "    (非阻塞) 该步骤未通过 rc=$rc"; return 0; }

echo "RASM 目录      : $(pwd)"
echo "显卡           : ${GPU_DEV}"
echo "端到端冒烟配置 : nepoch=${CHECK_NEPOCH} train_ps=${CHECK_TRAIN_PS} batch=${CHECK_BATCH}"

# ---------------------------------------------------------------- [1] 静态检查（CPU）
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "static" ]; then
  step "[1/4] 静态检查（CPU，不需要 GPU）"
  run python improved/check/check_static.py || echo ">>> 静态检查未通过"
fi

# ---------------------------------------------------------------- [2] 环境体检
if [ "${STAGE}" != "static" ]; then
  step "[2/4] GPU 环境体检"
  run python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit('未检测到 CUDA：train_v2.py / test_v2.py 全程 .cuda()，本检查必须在 GPU 环境跑')
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('GPU0:', torch.cuda.get_device_name(0),
      '| 显存 %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
try:
    import natten, timm
    print('natten', natten.__version__, '| timm', timm.__version__)
except Exception as e:
    print('[WARN] 依赖导入异常:', e)
PY
fi

# ---------------------------------------------------------------- [3] 模块自检（改进模块本身）
if [ "${STAGE}" = "all" ] && [ "${SKIP_SELFTEST:-0}" != "1" ]; then
  step "[3/4] 改进模块自检 improved/selftest.py（NA 等价 / 各 preset / 梯度 / 官方权重）"
  run_soft python -u improved/selftest.py
fi

# ---------------------------------------------------------------- [4] 训练 + 测试链路
if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "train" ]; then
  step "[4/4] 训练链路（单元 + 端到端 train_v2.py）"
  run python -m pytest improved/check/test_train_v2_gpu.py -v -s ${PYTEST_ARGS:-} \
      || echo ">>> 训练链路检查未通过"
fi

if [ "${STAGE}" = "all" ] || [ "${STAGE}" = "test" ]; then
  step "[4/4] 测试链路（单元 + 端到端 test_v2.py）"
  run python -m pytest improved/check/test_test_v2_gpu.py -v -s ${PYTEST_ARGS:-} \
      || echo ">>> 测试链路检查未通过"
fi

echo
echo "==================== 汇总 ===================="
if [ ${FAILED} -eq 0 ]; then
  echo "全部检查通过 ✅"
  echo "端到端产物: log/RASMv2${CHECK_ENV}/  (e2e_full = 训练, e2e_1ep = 单 epoch 对照)"
else
  echo "存在未通过的检查 ❌（见上方 FAIL / failed 输出）"
fi
exit ${FAILED}
