#!/bin/bash
# ============================================================
# 检查 PyTorch 与 natsort 版本（及 CUDA 可用性）。
# 在切到 GPU 环境后跑一次，确认依赖版本匹配（natten 轮子需与
# torch + CUDA 版本对应，见 setup.sh）。
# ============================================================
echo "== torch =="
python -c "import torch; print('version :', torch.__version__); print('cuda    :', torch.version.cuda); print('available:', torch.cuda.is_available())"

echo "== natsort =="
python -c "import natsort; print('version :', natsort.__version__)"

echo "== pip (Name/Version) =="
pip show torch natsort 2>/dev/null | grep -E '^(Name|Version):'
