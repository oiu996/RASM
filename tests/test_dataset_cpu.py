#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU 版数据层单元测试（不依赖 GPU / natten）。

覆盖 dataset.py 的“匹配键对齐”补丁，确保 ISTD 与 SRD 的 train/test
都能正确取出 A(阴影图)/B(mask)/C(GT) 三元组，且形状/数值范围合理。

运行（在 RASM 目录下）:
    pip install pytest        # 仅需 CPU 版 torch/cv2/skimage/numpy（与 natten 无关）
    python -m pytest tests/test_dataset_cpu.py -v

这些测试全部可在纯 CPU 上执行，因为 dataset -> utils -> image_utils
只依赖 torch/cv2/skimage，不会触发 natten（natten 仅在使用 model 时引入）。
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import (  # noqa: E402
    DataLoaderTrain,
    DataLoaderVal,
    _match_key,
    _aligned_filenames,
)

DATA_ROOT = os.path.join(ROOT, "datasets")

ISTD_TRAIN = os.path.join(DATA_ROOT, "ISTD_Dataset", "train")
ISTD_TEST = os.path.join(DATA_ROOT, "ISTD_Dataset", "test")
SRD_TRAIN = os.path.join(DATA_ROOT, "SRD_ABC", "train")
SRD_TEST = os.path.join(DATA_ROOT, "SRD_ABC", "test")

PATCH = 320


# ---------------- _match_key：剥离 GT 后缀 ----------------
def test_match_key_plain_name():
    assert _match_key("10-10.png") == "10-10"


def test_match_key_no_shadow_suffix():
    assert _match_key("_MG_2374_no_shadow.jpg") == "_MG_2374"


def test_match_key_free_suffix():
    assert _match_key("_MG_2371_free.jpg") == "_MG_2371"


@pytest.mark.parametrize(
    "suffix", ["_shadow_free", "_nostadow", "_shadowfree", "_gt"]
)
def test_match_key_other_suffixes(suffix):
    assert _match_key("img" + suffix + ".png") == "img"


# ---------------- 数据集目录存在性 ----------------
@pytest.mark.parametrize(
    "root", [ISTD_TRAIN, ISTD_TEST, SRD_TRAIN, SRD_TEST]
)
def test_dataset_dir_exists(root):
    assert os.path.isdir(root), f"数据集目录缺失: {root}"


# ---------------- Loader 可构造且非空 ----------------
@pytest.mark.parametrize(
    "name,root,split",
    [
        ("ISTD", ISTD_TRAIN, "train"),
        ("ISTD", ISTD_TEST, "test"),
        ("SRD", SRD_TRAIN, "train"),
        ("SRD", SRD_TEST, "test"),
    ],
)
def test_loader_nonempty(name, root, split):
    loader = (
        DataLoaderTrain(root, img_options={"patch_size": PATCH})
        if split == "train"
        else DataLoaderVal(root)
    )
    assert len(loader) > 0, f"{name}/{split} 样本数为 0"


# ---------------- train 样本：形状 + A/B/C 匹配键一致 ----------------
@pytest.mark.parametrize(
    "name,root",
    [("ISTD", ISTD_TRAIN), ("SRD", SRD_TRAIN)],
)
def test_train_sample_shape_and_alignment(name, root):
    loader = DataLoaderTrain(root, img_options={"patch_size": PATCH})
    sample = loader[0]
    clean, noisy, mask = sample[0], sample[1], sample[2]
    fns = sample[3:]

    # 通道数
    assert clean.shape[0] == 3 and noisy.shape[0] == 3, "clean/noisy 应为 3 通道"
    assert mask.shape[0] == 1, "mask 应为单通道"
    # patch 尺寸
    assert clean.shape[1] == clean.shape[2] == PATCH, "patch 应为 320x320"
    assert clean.shape == noisy.shape, "clean 与 noisy 尺寸应一致"
    # A/B/C 匹配键一致（SRD 的命名差异正是在此被补丁修正）
    keys = [_match_key(os.path.basename(f)) for f in fns]
    assert len(set(keys)) == 1, f"{name} A/B/C 匹配键未对齐: {keys}"


# ---------------- val 样本：形状 + A/B/C 匹配键一致 ----------------
@pytest.mark.parametrize(
    "name,root", [("ISTD", ISTD_TEST), ("SRD", SRD_TEST)]
)
def test_val_sample_shape_and_alignment(name, root):
    loader = DataLoaderVal(root)
    sample = loader[0]
    clean, noisy, mask = sample[0], sample[1], sample[2]
    fns = sample[3:]
    assert clean.shape[0] == 3 and noisy.shape[0] == 3
    assert mask.shape[0] == 1
    keys = [_match_key(os.path.basename(f)) for f in fns]
    assert len(set(keys)) == 1, f"{name} val A/B/C 匹配键未对齐: {keys}"


# ---------------- 像素数值范围 ----------------
def test_pixel_range_in_unit_interval():
    loader = DataLoaderTrain(ISTD_TRAIN, img_options={"patch_size": PATCH})
    clean, noisy = loader[0][0], loader[0][1]
    assert float(clean.min()) >= 0.0 and float(clean.max()) <= 1.0
    assert float(noisy.min()) >= 0.0 and float(noisy.max()) <= 1.0


# ---------------- _aligned_filenames 退化兼容 ----------------
def test_aligned_filenames_empty_falls_back():
    # 三个空目录 -> 不应抛异常（按索引对齐的退化分支）
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for sub in ("gt", "inp", "mask"):
            os.makedirs(os.path.join(d, sub))
        clean, noisy, mask = _aligned_filenames(d, "gt", "inp", "mask")
        assert clean == noisy == mask == []
