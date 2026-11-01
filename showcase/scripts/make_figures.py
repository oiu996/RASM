#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤 3：把样例 + 两个权重的结果拼成"肉眼对比图"。

产物：
    showcase/compare/XX_key_compare.png   每张样例一图五行：A输入 / B阴影mask / C真值GT / 权重1输出 / 权重2输出
    showcase/compare/all_samples_grid.png 上面 10 张竖着拼成一张总览大图（汇报用）
    showcase/compare/zoom/XX_key_zoom.png 阴影区域局部放大对比（最能看出"阴影有没有被去掉"）

本脚本只读图 + 排版，**不需要 GPU**。

用法（在 RASM/ 根目录执行）：
    python showcase/scripts/make_figures.py
    python showcase/scripts/make_figures.py --tile_w 480 --grid_scale 0.6
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RASM/ 根目录
sys.path.insert(0, ROOT)

CJK_FONTS = [                                            # 有中文字体就用中文标签
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]
LATIN_FONTS = [                                          # 没有则退回英文标签
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
BG, FG, BAR = (255, 255, 255), (255, 255, 255), (30, 30, 30)
GAP = 6


def parse_args():
    p = argparse.ArgumentParser(description='生成样例对比图')
    p.add_argument('--samples_dir', default='showcase/samples', type=str)
    p.add_argument('--manifest', default='showcase/samples/manifest.json', type=str)
    p.add_argument('--out_root', default='showcase/outputs', type=str)
    p.add_argument('--metrics', default='showcase/metrics.json', type=str)
    p.add_argument('--compare_dir', default='showcase/compare', type=str)
    p.add_argument('--tile_w', default=420, type=int, help='每个格子的宽度（像素）')
    p.add_argument('--grid_scale', default=0.55, type=float, help='总览大图的缩放比例')
    p.add_argument('--zoom_size', default=256, type=int, help='局部放大裁剪边长（原图尺度）')
    p.add_argument('--no_zoom', action='store_true', help='跳过局部放大图')
    return p.parse_args()


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def get_font(size):
    """返回 (font, 是否支持中文)。优先 Noto CJK；没有中文字体时退回 DejaVu，标签自动切英文。"""
    for f in CJK_FONTS:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size), True
            except Exception:
                pass
    for f in LATIN_FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size), False
    return ImageFont.load_default(), False


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def resize_keep(img, w):
    h = int(round(w * img.shape[0] / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def strip_row(cells, titles, footers, cell_w, cell_h, font_t, font_f, top=34, bottom=30):
    """cells: 若干 RGB ndarray（已缩放到 cell_w x cell_h），横向拼一条，带标题条 / 脚注条。"""
    n = len(cells)
    W = n * cell_w + (n - 1) * GAP
    H = top + cell_h + bottom
    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)
    for i, (cell, t, f) in enumerate(zip(cells, titles, footers)):
        x = i * (cell_w + GAP)
        # 标题条
        d.rectangle([x, 0, x + cell_w, top], fill=BAR)
        tw = d.textbbox((0, 0), t, font=font_t)
        d.text((x + (cell_w - (tw[2] - tw[0])) / 2, (top - (tw[3] - tw[1])) / 2 - 2), t,
               font=font_t, fill=FG)
        # 图像
        canvas.paste(Image.fromarray(cell), (x, top))
        # 脚注条
        d.rectangle([x, top + cell_h, x + cell_w, H], fill=(245, 245, 245))
        fw = d.textbbox((0, 0), f, font=font_f)
        d.text((x + (cell_w - (fw[2] - fw[0])) / 2, top + cell_h + 5), f, font=font_f, fill=(20, 20, 20))
    return np.array(canvas)


def make_compare(weight_names, txt, paths, metrics, tile_w, fonts):
    a = read_rgb(paths['A'])
    cell_h = int(round(tile_w * a.shape[0] / a.shape[1]))
    cells = [resize_keep(a, tile_w),
             cv2.cvtColor(resize_keep(read_gray(paths['B']), tile_w), cv2.COLOR_GRAY2RGB),
             resize_keep(read_rgb(paths['C']), tile_w)]
    titles = [txt['a'], txt['b'], txt['c']]
    footers = [txt['in_'], txt['in_'], txt['ref']]
    for n in weight_names:
        cells.append(resize_keep(read_rgb(paths[n]), tile_w))
        titles.append(txt['out'].format(name=n))
        m = metrics[n]
        footers.append(f"PSNR {m['psnr']:.2f} / RMSE {m['rmse']:.2f}")
    return strip_row(cells, titles, footers, tile_w, cell_h, fonts['t'], fonts['f'])


def make_zoom(weight_names, txt, paths, metrics, zoom_size, fonts):
    """裁一块阴影最集中的区域放大对比：A / GT / 权重1 / 权重2。"""
    a = read_rgb(paths['A'])
    mask = read_gray(paths['B'])
    H, W = mask.shape[:2]
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = int(ys.mean()), int(xs.mean())
    half = zoom_size // 2
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)

    def crop(im):
        c = im[y0:y1, x0:x1]
        return cv2.resize(c, (zoom_size * 2, zoom_size * 2), interpolation=cv2.INTER_CUBIC)

    cells = [crop(a), crop(read_rgb(paths['C']))]
    titles = [txt['za'], txt['zc']]
    footers = [txt['shadow'], txt['free']]
    for n in weight_names:
        cells.append(crop(read_rgb(paths[n])))
        titles.append(txt['zout'].format(name=n))
        footers.append(f"PSNR {metrics[n]['psnr']:.2f}")
    return strip_row(cells, titles, footers, zoom_size * 2, zoom_size * 2, fonts['t'], fonts['f'])


def main():
    args = parse_args()
    manifest = json.load(open(abspath(args.manifest), 'r', encoding='utf-8'))
    metrics_path = abspath(args.metrics)
    if not os.path.isfile(metrics_path):
        raise SystemExit(f'[error] 找不到 {metrics_path}：请先运行 run_inference.py 生成推理结果和指标，'
                         f'再执行本拼图脚本。')
    metrics = json.load(open(metrics_path, 'r', encoding='utf-8'))
    weight_names = metrics['weight_names']
    samples = manifest['samples']

    cmp_dir = abspath(args.compare_dir)
    zoom_dir = os.path.join(cmp_dir, 'zoom')
    os.makedirs(cmp_dir, exist_ok=True)
    os.makedirs(zoom_dir, exist_ok=True)

    font_t, zh = get_font(20)
    font_f, _ = get_font(17)
    fonts = {'t': font_t, 'f': font_f}
    if zh:
        txt = dict(a='A 输入（带阴影）', b='B 阴影 mask', c='C 真值 GT（无阴影）',
                   in_='模型输入', ref='参考答案', out='权重 {name} 输出',
                   za='A 输入（局部放大）', zc='C 真值 GT（局部放大）',
                   zout='权重 {name}（局部）', shadow='有阴影', free='无阴影')
    else:
        txt = dict(a='A Input (shadow)', b='B Shadow mask', c='C Ground truth',
                   in_='model input', ref='reference', out='weight: {name}',
                   za='A Input (zoom)', zc='C GT (zoom)',
                   zout='weight: {name} (zoom)', shadow='shadow', free='shadow-free')
    print(f'[info] 标签语言: {"中文" if zh else "英文（未找到中文字体）"}')
    rows = []
    for s in samples:
        stem = s['stem']
        paths = {'A': abspath(s['A']), 'B': abspath(s['B']), 'C': abspath(s['C'])}
        for n in weight_names:
            p = os.path.join(abspath(args.out_root), n, stem + '.jpg')
            assert os.path.isfile(p), f'缺少推理结果（先跑 run_inference.py）: {p}'
            paths[n] = p
        m = metrics['samples'][stem]

        img = make_compare(weight_names, txt, paths, m, args.tile_w, fonts)
        cv2.imwrite(os.path.join(cmp_dir, stem + '_compare.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append(img)
        if not args.no_zoom:
            z = make_zoom(weight_names, txt, paths, m, args.zoom_size, fonts)
            cv2.imwrite(os.path.join(zoom_dir, stem + '_zoom.png'),
                        cv2.cvtColor(z, cv2.COLOR_RGB2BGR))

    # 总览大图：所有样例纵向堆叠
    W = max(r.shape[1] for r in rows)
    pad = np.full((10, W, 3), 255, dtype=np.uint8)
    stack = []
    for r in rows:
        if r.shape[1] < W:                       # 右侧补白，保持宽度一致
            r = np.concatenate([r, np.full((r.shape[0], W - r.shape[1], 3), 255, dtype=np.uint8)], axis=1)
        stack.append(r)
        stack.append(pad)
    grid = np.concatenate(stack, axis=0)
    if args.grid_scale != 1.0:
        grid = cv2.resize(grid, (int(W * args.grid_scale), int(grid.shape[0] * args.grid_scale)),
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(cmp_dir, 'all_samples_grid.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f'[done] 逐样例对比图: {cmp_dir}')
    print(f'[done] 局部放大图:   {zoom_dir}')
    print(f'[done] 总览大图:     {os.path.join(cmp_dir, "all_samples_grid.png")}  {grid.shape[1]}x{grid.shape[0]}')


if __name__ == '__main__':
    main()
