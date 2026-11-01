#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把自定义盲测结果拼成对比图。

按掩码来源分三类：
  compare/maskORIGIN/  每张一组：输入 | 估计阴影(非真值) | 权重1输出 | 权重2输出  (+ zoom/)
  compare/maskDHAN/    同上，但用 DHAN 掩码
  compare/maskUNION/   同上，但用 ORIGIN∪DHAN 并集掩码
外加 compare/zoomTOTAL/：每组按掩码来源顺序排成若干"4 列一组"，组间留空白
（仅当跑了 ≥2 种掩码时生成；2 种=4+空白+4，3 种=4+空白+4+空白+4）

用法（在 RASM/ 根目录）：
    python showcase/custom/scripts/make_figures_custom.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # RASM/ 根目录
SCRIPTS = os.path.normpath(os.path.join(HERE, '..', '..', 'scripts'))   # showcase/scripts（主流程脚本）
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

import make_figures as MF                              # 复用 strip_row / read_rgb / resize_keep / get_font

EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')


def parse_args():
    p = argparse.ArgumentParser(description='生成自定义盲测对比图')
    p.add_argument('--input_dir', default='showcase/custom/input', type=str)
    p.add_argument('--out_root', default='showcase/custom/outputs', type=str)
    p.add_argument('--summary', default='showcase/custom/outputs/summary.json', type=str)
    p.add_argument('--compare_dir', default='showcase/custom/compare', type=str)
    p.add_argument('--tile_w', type=int, default=420)
    p.add_argument('--grid_scale', type=float, default=0.55)
    p.add_argument('--zoom_size', type=int, default=256)
    p.add_argument('--no_zoom', action='store_true')
    return p.parse_args()


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def find_mask_file(d, stem):
    for e in ('.png', '.jpg', '.jpeg', '.PNG'):
        p = os.path.join(d, stem + e)
        if os.path.isfile(p):
            return p
    return None


def load_mask_to_size(mp, H, W):
    """读灰度掩码并 resize 到 (H, W)（最近邻，保持二值），用于与输入图对齐。"""
    m = cv2.imread(mp, 0)
    if m is None:
        return None
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m


def main():
    args = parse_args()
    summary_path = abspath(args.summary)
    if not os.path.isfile(summary_path):
        raise SystemExit('[error] 先运行 run_inference_custom.py 生成推理结果。')
    summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    mask_sources = summary.get('mask_sources', ['ORIGIN'])
    weights = list(summary['weights'].keys())
    samples = summary['samples']
    input_dir = abspath(args.input_dir)
    out_root = abspath(args.out_root)
    cmp_root = abspath(args.compare_dir)
    os.makedirs(cmp_root, exist_ok=True)

    font_t, zh = MF.get_font(20)
    font_f, _ = MF.get_font(17)
    if zh:
        L = dict(a='A 输入（带阴影）',
                 m_origin='估计阴影(仅驱动模型)', m_dhan='DHAN 阴影掩码', m_union='并集掩码(ORIGIN∪DHAN)',
                 w='权重 {name} 输出', des='耗时 {s:.1f}s',
                 foots='非真值', foots_d='DHAN', foots_union='ORIGIN∪DHAN',
                 zooma='A 输入（局部）', zoomw='权重 {name}（局部）',
                 zoomm_origin='估计阴影(局部)', zoomm_dhan='DHAN 掩码(局部)', zoomm_union='并集掩码(局部)')
    else:
        L = dict(a='A Input (shadow)',
                 m_origin='Estimated mask (drive only)', m_dhan='DHAN mask', m_union='Union mask (ORIGIN∪DHAN)',
                 w='weight {name}', des='{s:.1f}s',
                 foots='not GT', foots_d='DHAN', foots_union='ORIGIN∪DHAN',
                 zooma='A Input (zoom)', zoomw='weight {name} (zoom)',
                 zoomm_origin='est mask (zoom)', zoomm_dhan='DHAN mask (zoom)', zoomm_union='union mask (zoom)')

    # 为某个样本 + 某个掩码来源，构造"输入 | mask | 权重1 | 权重2"的整图对比行
    def build_cells(sample, ms):
        stem = sample['stem']
        A = MF.read_rgb(os.path.join(input_dir, sample['file']))
        cell_h = int(round(args.tile_w * A.shape[0] / A.shape[1]))
        cells = [MF.resize_keep(A, args.tile_w)]
        titles = [L['a']]
        footers = ['你的原图']
        if ms == 'ORIGIN':
            mp, mt, mf = os.path.join(out_root, 'ORIGIN_masks', stem + '.png'), L['m_origin'], L['foots']
        elif ms == 'DHAN':
            mp, mt, mf = find_mask_file(os.path.join(out_root, 'DHAN_masks'), stem), L['m_dhan'], L['foots_d']
        else:  # UNION
            mp, mt, mf = find_mask_file(os.path.join(out_root, 'UNION_masks'), stem), L['m_union'], L['foots_union']
        m = load_mask_to_size(mp, A.shape[0], A.shape[1]) if mp and os.path.isfile(mp) else None
        if m is not None:
            cells.append(cv2.cvtColor(MF.resize_keep(m, args.tile_w), cv2.COLOR_GRAY2RGB))
        else:
            cells.append(np.zeros((cell_h, args.tile_w, 3), np.uint8))
        titles.append(mt)
        footers.append(mf)
        for n in weights:
            op = os.path.join(out_root, f'mask{ms}', n, stem + '.jpg')
            assert os.path.isfile(op), f'缺少结果: {op}'
            cells.append(MF.resize_keep(MF.read_rgb(op), args.tile_w))
            titles.append(L['w'].format(name=n))
            sec = sample[ms]['weights'].get(n, {}).get('seconds', 0)
            footers.append(L['des'].format(s=sec))
        return A, cells, titles, footers, cell_h

    # 为某个样本 + 某个掩码来源，构造"输入 | mask | 权重1 | 权重2"的局部放大行（按各自 mask 中心裁切）
    def build_zoom(sample, ms):
        stem = sample['stem']
        A = MF.read_rgb(os.path.join(input_dir, sample['file']))
        H, W = A.shape[:2]
        if ms == 'ORIGIN':
            mp, mt, mf = os.path.join(out_root, 'ORIGIN_masks', stem + '.png'), L['zoomm_origin'], L['foots']
        elif ms == 'DHAN':
            mp, mt, mf = find_mask_file(os.path.join(out_root, 'DHAN_masks'), stem), L['zoomm_dhan'], L['foots_d']
        else:  # UNION
            mp, mt, mf = find_mask_file(os.path.join(out_root, 'UNION_masks'), stem), L['zoomm_union'], L['foots_union']
        mask = load_mask_to_size(mp, A.shape[0], A.shape[1]) if (mp and os.path.isfile(mp)) else None
        if mask is not None and mask.sum() > 0:
            ys, xs = np.nonzero(mask > 127)
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = H // 2, W // 2
        half = args.zoom_size // 2
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        z = args.zoom_size * 2

        def crop(im):
            return cv2.resize(im[y0:y1, x0:x1], (z, z), interpolation=cv2.INTER_CUBIC)

        zcells = [crop(A)]
        zt = [L['zooma']]
        zf = ['阴影区域']
        mimg = crop(mask) if mask is not None else np.zeros((z, z), np.uint8)
        zcells.append(cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB) if mimg.ndim == 2 else mimg)
        zt.append(mt)
        zf.append(mf)
        for n in weights:
            zcells.append(crop(MF.read_rgb(os.path.join(out_root, f'mask{ms}', n, stem + '.jpg'))))
            zt.append(L['zoomw'].format(name=n))
            sec = sample[ms]['weights'].get(n, {}).get('seconds', 0)
            zf.append(L['des'].format(s=sec))
        return zcells, zt, zf, z

    # ---------- 各掩码来源各自的对比图 + 放大图 ----------
    for ms in mask_sources:
        cmp_dir = os.path.join(cmp_root, f'mask{ms}')
        zoom_dir = os.path.join(cmp_dir, 'zoom')
        os.makedirs(cmp_dir, exist_ok=True)
        os.makedirs(zoom_dir, exist_ok=True)
        rows = []
        for s in samples:
            A, cells, titles, footers, cell_h = build_cells(s, ms)
            img = MF.strip_row(cells, titles, footers, args.tile_w, cell_h, font_t, font_f)
            cv2.imwrite(os.path.join(cmp_dir, s['stem'] + '_compare.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows.append(img)
            if not args.no_zoom:
                zcells, zt, zf, z = build_zoom(s, ms)
                zimg = MF.strip_row(zcells, zt, zf, z, z, font_t, font_f)
                cv2.imwrite(os.path.join(zoom_dir, s['stem'] + '_zoom.png'),
                            cv2.cvtColor(zimg, cv2.COLOR_RGB2BGR))
        # 该来源总览大图
        if rows:
            Wmax = max(r.shape[1] for r in rows)
            pad = np.full((10, Wmax, 3), 255, np.uint8)
            stack = []
            for r in rows:
                if r.shape[1] < Wmax:
                    r = np.concatenate([r, np.full((r.shape[0], Wmax - r.shape[1], 3), 255, np.uint8)], 1)
                stack.append(r)
                stack.append(pad)
            grid = np.concatenate(stack, 0)
            if args.grid_scale != 1.0:
                grid = cv2.resize(grid, (int(Wmax * args.grid_scale), int(grid.shape[0] * args.grid_scale)),
                                  interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(cmp_dir, f'all_mask{ms}_grid.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # ---------- zoomTOTAL：各掩码来源一组"4 列"，组间留空白 ----------
    if len(mask_sources) >= 2 and not args.no_zoom:
        zt_dir = os.path.join(cmp_root, 'zoomTOTAL')
        os.makedirs(zt_dir, exist_ok=True)
        sep_w = 24
        for s in samples:
            groups = []
            for ms in mask_sources:
                zcells, zt, zf, z = build_zoom(s, ms)
                groups.append(MF.strip_row(zcells, zt, zf, z, z, font_t, font_f))
            blank = np.full((groups[0].shape[0], sep_w, 3), 255, np.uint8)
            total = groups[0]
            for g in groups[1:]:
                total = np.concatenate([total, blank, g], axis=1)
            cv2.imwrite(os.path.join(zt_dir, s['stem'] + '_zoomTOTAL.png'),
                        cv2.cvtColor(total, cv2.COLOR_RGB2BGR))
        print(f'[done] zoomTOTAL: {zt_dir}')

    print(f'[done] 对比图: {cmp_root}')


if __name__ == '__main__':
    main()
