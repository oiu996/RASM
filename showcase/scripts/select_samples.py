#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤 1：挑选样例并复制 A/B/C。

从 SRD（或 ISTD）测试集里挑 N 张"肉眼看得清差别"的图，把三件套复制出来：
    A = 带阴影原图（模型输入）
    B = 阴影 mask（模型输入）
    C = 无阴影 GT（正确答案）

挑选策略：按阴影面积占比（mask 覆盖率）排序后**等间隔**取样，
这样 10 张里既有"小面积阴影"也有"大面积阴影"，视觉上更有代表性。

本脚本只做文件 IO + mask 统计，**不需要 GPU**，可直接在 CPU 环境跑。

用法（在 RASM/ 根目录执行）：
    python showcase/scripts/select_samples.py                    # 默认 SRD 挑 10 张
    python showcase/scripts/select_samples.py --num 10 --seed 0
    python showcase/scripts/select_samples.py --data_root datasets/ISTD_Dataset/test
"""
import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RASM/ 根目录
sys.path.insert(0, ROOT)

from dataset import _aligned_filenames                  # noqa: E402
from utils import load_mask                             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='挑选展示样例并复制 A/B/C')
    p.add_argument('--data_root', default='datasets/SRD_ABC/test', type=str,
                   help='测试集根目录（下面应有 test_A / test_B / test_C）')
    p.add_argument('--out_dir', default='showcase/samples', type=str,
                   help='样例输出目录（A_input / B_mask / C_gt 三个子目录）')
    p.add_argument('--num', default=10, type=int, help='挑选张数')
    p.add_argument('--seed', default=0, type=int, help='随机种子（影响等间隔取样的起点）')
    p.add_argument('--min_ratio', default=0.03, type=float,
                   help='阴影面积占比下限（太小看不出效果）')
    p.add_argument('--max_ratio', default=0.45, type=float,
                   help='阴影面积占比上限（太大会整图发黑）')
    p.add_argument('--manifest', default='showcase/samples/manifest.json', type=str,
                   help='清单文件（记录 key / 原路径 / 阴影占比），后续脚本依赖它')
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(ROOT, args.data_root)
    assert os.path.isdir(data_root), f'数据集不存在: {data_root}'

    # 1) 按文件名对齐 A/B/C（复用训练/测试同一套对齐逻辑，避免错位）
    clean_paths, noisy_paths, mask_paths = _aligned_filenames(data_root, 'test_C', 'test_A', 'test_B')
    print(f'[select] 对齐到 {len(clean_paths)} 组 A/B/C：{data_root}')

    # 2) 统计每张的阴影面积占比
    stats = []
    for c, n, m in zip(clean_paths, noisy_paths, mask_paths):
        key = os.path.splitext(os.path.basename(c))[0]
        for suf in ('_free', '_no_shadow', '_shadow_free', '_gt'):
            if key.lower().endswith(suf):
                key = key[: -len(suf)]
                break
        mask = load_mask(m)
        ratio = float((mask > 0.5).mean())
        stats.append({'key': key, 'A': n, 'B': m, 'C': c, 'ratio': ratio})

    # 3) 按占比过滤 + 排序 + 等间隔取样（保证覆盖"小阴影→大阴影"）
    cand = sorted([s for s in stats if args.min_ratio <= s['ratio'] <= args.max_ratio],
                  key=lambda s: s['ratio'])
    if len(cand) < args.num:                       # 候选不够就放宽区间重来
        cand = sorted(stats, key=lambda s: s['ratio'])
    print(f'[select] 阴影占比在 [{args.min_ratio}, {args.max_ratio}] 的候选：{len(cand)} 张')

    step = max(1, len(cand) // args.num)
    start = rng.randint(0, max(1, len(cand) - step * (args.num - 1))) if len(cand) > args.num else 0
    picked = [cand[min(start + i * step, len(cand) - 1)] for i in range(args.num)]

    # 4) 复制三件套，统一命名为 01_xxx.jpg / 02_xxx.jpg ...（三个目录同名，便于配对）
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    sub = {'A': 'A_input', 'B': 'B_mask', 'C': 'C_gt'}
    for d in sub.values():
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    manifest = []
    for i, s in enumerate(picked, 1):
        safe = s['key'].replace(' ', '_').replace('.', '-')   # 避免文件名里出现多余的点号
        stem = f'{i:02d}_{safe}'
        item = {'id': f'{i:02d}', 'stem': stem, 'key': s['key'],
                'shadow_ratio': round(s['ratio'], 4)}
        for tag, dirname in sub.items():
            src = s[tag]
            dst = os.path.join(out_dir, dirname, stem + os.path.splitext(src)[1])
            shutil.copy2(src, dst)
            item[tag] = os.path.relpath(dst, ROOT)
        item['src'] = {t: os.path.relpath(s[t], ROOT) for t in ('A', 'B', 'C')}
        manifest.append(item)
        print(f'  [{i:02d}] {s["key"]:<16s} 阴影占比 {s["ratio"]*100:5.2f}%')

    manifest_path = args.manifest if os.path.isabs(args.manifest) else os.path.join(ROOT, args.manifest)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'data_root': os.path.relpath(data_root, ROOT),
                   'num': len(manifest),
                   'seed': args.seed,
                   'samples': manifest}, f, ensure_ascii=False, indent=2)
    print(f'\n[select] 样例已复制到: {out_dir}')
    print(f'[select] 清单已写入:   {manifest_path}')


if __name__ == '__main__':
    main()
