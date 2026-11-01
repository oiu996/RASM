#!/bin/bash
# 1. 卸载旧版本
python -m pip uninstall -y torch torchvision torchaudio natten

# 2. 安装 PyTorch 2.1.2 + CUDA 11.8（PyTorch wheel 走上海交通大学镜像）
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
pip install torchtext==0.16.0 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html
python -m pip install natten==0.15.1+torch210cu118 -f https://whl.natten.org/old



python --version
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import natten; print(natten.__version__)"