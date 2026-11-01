#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自定义图片盲测：只提供原图（无 B mask、无 C GT），用两个权重去阴影。

三种掩码来源各跑一遍：
  - ORIGIN：无 mask 时用"亮度差"自动估一个粗 mask 仅驱动模型，存 outputs/ORIGIN_masks/，
            图上明确标注"非真值"（不是真值，只是"模型把这里当阴影处理"）。
  - DHAN  ：若 outputs/DHAN_masks/ 存在且含掩码，则用它作为掩码再跑一遍（DHAN 生成的阴影掩码）。
  - UNION ：ORIGIN 与 DHAN 的并集（任一判为阴影即阴影），存 outputs/UNION_masks/。

结果按掩码来源分目录：
  outputs/maskORIGIN/<权重>/...
  outputs/maskDHAN/<权重>/...
  outputs/maskUNION/<权重>/...
对比图同理：compare/maskORIGIN/... 与 compare/maskDHAN/...，外加 compare/zoomTOTAL/。

用法（在 RASM/ 根目录，需 GPU）：
    python showcase/custom/scripts/run_inference_custom.py --gpus 0
    python showcase/custom/scripts/run_inference_custom.py --masks_dir showcase/custom/outputs/DHAN_masks
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # RASM/ 根目录
SCRIPTS = os.path.normpath(os.path.join(HERE, '..', '..', 'scripts'))   # showcase/scripts（主流程脚本）
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

import run_inference as RI                             # 复用 load_model / infer_tiled
from utils import load_img, load_mask, save_img        # noqa: E402


def pad_to_multiple(arr, multiple, is_mask=False):
    """把图像/mask pad 到 multiple 的倍数（向右/下补），返回 padded 数组。"""
    if arr.ndim == 3:
        H, W = arr.shape[:2]
    else:
        H, W = arr.shape
    Hp = int(np.ceil(H / multiple) * multiple)
    Wp = int(np.ceil(W / multiple) * multiple)
    if Hp == H and Wp == W:
        return arr
    if is_mask:
        return cv2.copyMakeBorder(arr, 0, Hp - H, 0, Wp - W, cv2.BORDER_CONSTANT, value=0)
    return cv2.copyMakeBorder(arr, 0, Hp - H, 0, Wp - W, cv2.BORDER_REFLECT_101)


def find_mask_file(d, fn):
    """在 d 目录下按文件名（或同 stem 的 png/jpg）找掩码，返回路径或 None。"""
    if not os.path.isdir(d):
        return None
    for c in (fn, os.path.splitext(fn)[0] + '.png', os.path.splitext(fn)[0] + '.jpg'):
        p = os.path.join(d, c)
        if os.path.isfile(p):
            return p
    return None


DEFAULT_WEIGHTS = RI.DEFAULT_WEIGHTS
EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')


def parse_args():
    p = argparse.ArgumentParser(description='自定义图片盲测推理（GPU）')
    p.add_argument('--input_dir', default='showcase/custom/input', type=str)
    p.add_argument('--out_root', default='showcase/custom/outputs', type=str)
    p.add_argument('--masks_dir', default=None, type=str,
                   help='DHAN 阴影掩码目录（默认: <out_root>/DHAN_masks）。存在且含掩码则多跑一遍 DHAN。')
    p.add_argument('--weights', nargs='+', default=DEFAULT_WEIGHTS)
    p.add_argument('--summary', default='showcase/custom/outputs/summary.json', type=str)
    p.add_argument('--embed_dim', type=int, default=32)
    p.add_argument('--win_size', type=int, default=10)
    p.add_argument('--token_projection', type=str, default='linear')
    p.add_argument('--token_mlp', type=str, default='leff')
    p.add_argument('--train_ps', type=int, default=320)
    p.add_argument('--gpus', default='0', type=str)
    p.add_argument('--tile', type=int, default=512,
                   help='分块推理尺寸，默认 512（已对齐 8 倍数）；设 0/负数强制全图')
    p.add_argument('--tile_overlap', type=int, default=64)
    p.add_argument('--mask_thresh', type=float, default=0.10,
                   help='自动估计 mask 的暗度阈值（0~1，越大阴影区域越小）')
    return p.parse_args()


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS)) if os.path.isdir(d) else []


def estimate_mask(rgb, thresh):
    """亮度差法粗估阴影 mask（仅用于驱动模型，不是真值）。"""
    H, W, _ = rgb.shape
    lab = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]                                   # 0~100
    s = max(8, min(H, W) // 16) | 1                     # 高斯核大小（奇数）
    blurred = cv2.GaussianBlur(L, (0, 0), sigmaX=s / 2.0)
    diff = blurred - L                                  # 阴影相对周围偏暗 -> 正值
    mask = (diff > thresh * 100).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask.astype(np.float32)


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if not torch.cuda.is_available():
        raise SystemExit('[error] 当前无可用 CUDA；本脚本为 GPU 设计，请切换 GPU 环境。')
    device = torch.device('cuda')
    print('[info] GPU:', torch.cuda.get_device_name(0))
    free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f'[info] 空闲显存约 {free / 1e9:.2f} GiB')
    if free < 1.5 * 1e9:
        print('[warn] 空闲显存 < 1.5 GiB，可能 OOM；请先释放占用进程或加 --tile 256。')

    input_dir = abspath(args.input_dir)
    imgs = list_images(input_dir)
    assert imgs, f'input 目录没有图片: {input_dir}'
    print(f'[info] 待推理图片: {len(imgs)} 张（来自 {input_dir}）')

    weights = []
    for w in args.weights:
        name, path = (w.split('=', 1) if '=' in w else (os.path.splitext(os.path.basename(w))[0], w))
        weights.append((name, abspath(path)))

    out_root = abspath(args.out_root)
    origin_mdir = os.path.join(out_root, 'ORIGIN_masks')
    os.makedirs(origin_mdir, exist_ok=True)
    dhan_dir = abspath(args.masks_dir) if args.masks_dir else os.path.join(out_root, 'DHAN_masks')
    # 只有"目录存在且确实含掩码"才认为要用 DHAN，否则只跑 ORIGIN（避免空目录误触发）
    has_dhan = os.path.isdir(dhan_dir) and any(find_mask_file(dhan_dir, fn) for fn in imgs)
    mask_sources = ['ORIGIN']
    if has_dhan:
        mask_sources += ['DHAN', 'UNION']
        print(f'[info] 检测到 DHAN 掩码，将额外跑 DHAN 与 UNION(ORIGIN∪DHAN) 掩码: {dhan_dir}')
    else:
        print(f'[info] 未检测到 DHAN 掩码（{dhan_dir} 不存在或为空），仅跑 ORIGIN 掩码。')

    # 预计算 ORIGIN 掩码（仅驱动模型，非真值）并保存；顺便备好 DHAN / UNION 掩码
    origin_masks, dhan_masks, union_masks = {}, {}, {}
    union_mdir = os.path.join(out_root, 'UNION_masks')
    if has_dhan:
        os.makedirs(union_mdir, exist_ok=True)
    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        A = load_img(os.path.join(input_dir, fn))
        m = estimate_mask(A, args.mask_thresh)
        origin_masks[stem] = m
        cv2.imwrite(os.path.join(origin_mdir, stem + '.png'), (m * 255).astype(np.uint8))
        if has_dhan:
            dp = find_mask_file(dhan_dir, fn)
            dm = load_mask(dp) if dp else m
            if dm.shape[:2] != A.shape[:2]:   # DHAN 掩码可能尺寸不同，先对齐到原图
                dm = cv2.resize(dm, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST)
            dhan_masks[stem] = dm
            um = ((m > 0.5) | (dm > 0.5)).astype(np.float32)   # 并集：ORIGIN ∪ DHAN
            union_masks[stem] = um
            cv2.imwrite(os.path.join(union_mdir, stem + '.png'), (um * 255).astype(np.uint8))

    summary = {'mask_sources': mask_sources, 'weights': {n: p for n, p in weights}, 'samples': []}

    for ms in mask_sources:
        for name, wpath in weights:
            assert os.path.isfile(wpath), f'权重不存在: {wpath}'
            print(f'\n=== 掩码 {ms} | 权重 {name} ({wpath}) ===')
            model = RI.load_model(args, wpath, device)
            odir = os.path.join(out_root, f'mask{ms}', name)
            os.makedirs(odir, exist_ok=True)
            for fn in imgs:
                stem = os.path.splitext(fn)[0]
                A = load_img(os.path.join(input_dir, fn))
                if ms == 'ORIGIN':
                    mask = origin_masks[stem]
                    mask_src = 'auto'
                elif ms == 'DHAN':
                    mask = dhan_masks[stem]
                    dp = find_mask_file(dhan_dir, fn)
                    mask_src = 'dhan' if dp else 'auto'
                else:  # UNION = ORIGIN ∪ DHAN
                    mask = union_masks[stem]
                    mask_src = 'union'
                if mask.shape[:2] != A.shape[:2]:
                    mask = cv2.resize(mask, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST)

                # RASM 有 3 次 /2 下采样 + 3 次 *2 上采样，输入须是 8 的倍数，否则 cat 维度错位
                H, W = A.shape[:2]
                A = pad_to_multiple(A, 8)
                mask = pad_to_multiple(mask, 8, is_mask=True)

                t_inp = torch.from_numpy(np.float32(A)).permute(2, 0, 1).unsqueeze(0).to(device)
                t_mask = torch.from_numpy(np.float32(mask)).unsqueeze(0).unsqueeze(0).to(device)
                t0 = time.time()
                try:
                    out = model(t_inp, t_mask) if (args.tile is None or args.tile <= 0) \
                        else RI.infer_tiled(model, t_inp, t_mask, args.tile, args.tile_overlap)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print('\n[CUDA OOM] 显存不足。释放占用进程或加 --tile 256 分块推理。')
                    sys.exit(1)
                out = out[:, :, :H, :W]                          # 裁回原尺寸
                out = torch.clamp(out.detach(), 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
                dt = time.time() - t0
                save_img(out * 255.0, os.path.join(odir, stem + '.jpg'))

                rec = next((x for x in summary['samples'] if x['stem'] == stem), None)
                if rec is None:
                    rec = {'stem': stem, 'file': fn, **{ms: {} for ms in mask_sources}}
                    summary['samples'].append(rec)
                rec[ms]['mask_src'] = mask_src
                rec[ms].setdefault('weights', {})[name] = {'seconds': round(dt, 2)}
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()

    with open(abspath(args.summary), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n[done] 结果: {out_root}；摘要: {abspath(args.summary)}')
    print('提示：本分支无真值，结果仅供肉眼对比，不能量化"是否偷看答案"。')


if __name__ == '__main__':
    main()
