#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU 版数据可用性检查脚本（不依赖 GPU / natten）。

用于验证 dataset.py 的“匹配键对齐”补丁是否让 ISTD 与 SRD 的
train / test 都能正确取出 A(阴影图)/B(mask)/C(GT) 三元组。

运行（在 RASM 目录下）：
    python check_dataset_cpu.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# dataset.py -> utils -> image_utils 会 import torch/cv2/numpy/skimage，
# 这些都是可在 CPU 上安装的库，与 GPU/natten 无关，因此本脚本可纯 CPU 运行。
from dataset import DataLoaderTrain, DataLoaderVal, _match_key  # noqa: E402

DATA_ROOT = os.path.join(HERE, "datasets")

# (数据集名, split, 目录, 使用哪种 loader)
CONFIGS = [
    ("ISTD", "train", os.path.join(DATA_ROOT, "ISTD_Dataset", "train"), "train"),
    ("ISTD", "test",  os.path.join(DATA_ROOT, "ISTD_Dataset", "test"),  "test"),
    ("SRD",  "train", os.path.join(DATA_ROOT, "SRD_ABC", "train"),      "train"),
    ("SRD",  "test",  os.path.join(DATA_ROOT, "SRD_ABC", "test"),       "test"),
]

N_SAMPLES = 3  # 每个 split 抽查的样本数


def check_one(name, split, root):
    print("=" * 64)
    print(f"[{name} / {split}]  root = {root}")
    if not os.path.isdir(root):
        print("  [FAIL] 目录不存在")
        return False

    try:
        if split == "train":
            loader = DataLoaderTrain(root, img_options={"patch_size": 320})
        else:
            loader = DataLoaderVal(root)
    except Exception as e:
        print(f"  [FAIL] 构造 DataLoader 异常: {e}")
        return False

    n = len(loader)
    print(f"  样本数: {n}")
    if n == 0:
        print("  [FAIL] 样本数为 0")
        return False

    ok = True
    for i in range(min(N_SAMPLES, n)):
        try:
            sample = loader[i]
        except Exception as e:
            print(f"  [FAIL] 第 {i} 个样本加载异常: {e}")
            ok = False
            continue

        # train 返回 (clean, noisy, mask, clean_fn, noisy_fn)
        # val   返回 (clean, noisy, mask, clean_fn, noisy_fn, mask_fn)
        clean, noisy, mask = sample[0], sample[1], sample[2]
        fns = sample[3:]
        # 验证 A/B/C 的“匹配键”一致 —— 这正是补丁要解决的 SRD 命名问题
        keys = [_match_key(os.path.basename(f)) for f in fns]
        aligned = len(set(keys)) == 1
        shapes = (f"clean{list(clean.shape)} "
                  f"noisy{list(noisy.shape)} "
                  f"mask{list(mask.shape)}")
        print(f"  sample[{i}] {shapes}  keys={keys}  aligned={aligned}")
        if not aligned:
            ok = False
    print(f"  => {'[PASS]' if ok else '[FAIL]'}")
    return ok


def main():
    results = []
    for name, split, root, _ in CONFIGS:
        results.append((name, split, check_one(name, split, root)))

    print("=" * 64)
    print("汇总:")
    all_ok = True
    for name, split, ok in results:
        print(f"  {name:5s} {split:5s} : {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print("=" * 64)
    print("全部通过 ✅" if all_ok else "存在失败 ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
