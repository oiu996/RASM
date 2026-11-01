#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 RASM 的 4 个跨数据集测试产物（results_*）整理成学术论文风格的 2×2 结果展示。

所谓「测距 2×2」：两个模型（在 ISTD 上训练 / 在 SRD 上训练） × 两个测试集
（ISTD / SRD），构成一个 2×2 泛化性评估矩阵：

                测试集 ISTD        测试集 SRD
    模型 ISTD    results_istd      results_istd_on_srd
    模型 SRD     results_srd_on_istd  results_srd

每个目录下的 metrics.jsonl 是「追加累积」的，这里取 num_images 最大（即完整评测）
的那一次作为代表记录。本脚本产出：

    results_academic/
    ├── table_2x2_psnr.md        # 2×2 PSNR 矩阵（Markdown）
    ├── table_2x2_rmse.md        # 2×2 RMSE 矩阵（Markdown）
    ├── table_full_metrics.md    # 4 个配置的全部指标明细（Markdown）
    ├── table_per_image_stats.md # 逐图统计（均值/标准差/最值）
    ├── table_full_metrics.tex   # 同上，LaTeX 三线表
    ├── metrics_summary.csv       # 汇总指标 CSV
    ├── fig_2x2_matrix.png        # 核心：2×2 矩阵热力图（PSNR / RMSE）
    ├── fig_metric_bars.png       # 4 项指标分组柱状图（域内 vs 跨域）
    ├── fig_box_psnr.png          # 各配置逐图 PSNR 箱线图
    └── fig_gap.png               # 泛化性落差（域内 - 跨域）

只依赖标准库；绘图时需要 matplotlib/numpy（缺失则跳过绘图并提示）。

在 RASM/ 根目录执行：
    python check_script/make_2x2_academic.py
"""

import os
import json
import glob
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # RASM/
OUT = os.path.join(ROOT, 'results_academic')
os.makedirs(OUT, exist_ok=True)


# ---------------- 读取产物 ----------------
def load_canonical(path):
    """读 metrics.jsonl，返回 num_images 最大的那条记录（完整评测）。"""
    best = None
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n = rec.get('num_images') or 0
        if best is None or n > best['_n']:
            rec['_n'] = n
            best = rec
    return best


# 2×2 结构定义：model（训练集） × testset（测试集）
CONFIGS = [
    dict(key='results_istd',        model='ISTD', test='ISTD', kind='in-domain'),
    dict(key='results_srd_on_istd', model='SRD',  test='ISTD', kind='cross'),
    dict(key='results_istd_on_srd', model='ISTD', test='SRD',  kind='cross'),
    dict(key='results_srd',          model='SRD',  test='SRD',  kind='in-domain'),
]

data = {}
for c in CONFIGS:
    fp = os.path.join(ROOT, c['key'], 'metrics.jsonl')
    if not os.path.exists(fp):
        print('[warn] 缺失: {}'.format(fp))
        continue
    rec = load_canonical(fp)
    s = rec.get('summary', {})
    per = rec.get('per_image', [])
    data[c['key']] = dict(cfg=c, summary=s, per_image=per,
                          ex_name=rec.get('ex_name'),
                          weights=os.path.basename(rec.get('weights') or ''),
                          num_images=rec.get('num_images'))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def stats(values):
    values = [f(v) for v in values]
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (mean, var ** 0.5, min(values), max(values))


# ---------------- Markdown 表格 ----------------
def md_table(rows):
    ncol = len(rows[0])
    widths = [max(len(str(r[i])) for r in rows) for i in range(ncol)]
    fmt = lambda r: '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |'
    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    return '\n'.join([fmt(rows[0]), sep] + [fmt(r) for r in rows[1:]])


def matrix_table(metric, title, fmt='{:.2f}'):
    """2×2 矩阵：行=模型，列=测试集，单元格=该指标。"""
    def cell(model, test):
        for k, v in data.items():
            c = v['cfg']
            if c['model'] == model and c['test'] == test:
                return fmt.format(f(v['summary'].get(metric)))
        return '-'
    rows = [
        ['模型 \\ 测试集', 'ISTD', 'SRD'],
        ['ISTD (训练于 ISTD)', cell('ISTD', 'ISTD'), cell('ISTD', 'SRD')],
        ['SRD  (训练于 SRD)',  cell('SRD', 'ISTD'),  cell('SRD', 'SRD')],
    ]
    out = '## {}\n\n'.format(title) + md_table(rows)
    out += '\n\n> 对角线为域内测试，反对角线为跨数据集泛化测试。\n'
    return out


# 1) 2×2 PSNR / RMSE 矩阵
open(os.path.join(OUT, 'table_2x2_psnr.md'), 'w', encoding='utf-8').write(
    matrix_table('psnr', '表 1. 2×2 跨数据集评估 —— 整体 PSNR (dB) ↑'))
open(os.path.join(OUT, 'table_2x2_rmse.md'), 'w', encoding='utf-8').write(
    matrix_table('rmse', '表 2. 2×2 跨数据集评估 —— 整体 RMSE ↓'))

# 2) 全部指标明细
rows = [['配置 (训练→测试)', '图数', 'PSNR↑', 'RMSE↓', 'S-PSNR↑', 'S-RMSE↓',
         'NS-PSNR↑', 'NS-RMSE↓', '权重']]
for k, v in data.items():
    c = v['cfg']
    s = v['summary']
    label = '{}→{}'.format(c['model'], c['test'])
    rows.append([label, str(v['num_images']),
                 '{:.4f}'.format(f(s.get('psnr'))),
                 '{:.4f}'.format(f(s.get('rmse'))),
                 '{:.4f}'.format(f(s.get('psnr_s'))),
                 '{:.4f}'.format(f(s.get('rmse_s'))),
                 '{:.4f}'.format(f(s.get('psnr_ns'))),
                 '{:.4f}'.format(f(s.get('rmse_ns'))),
                 v['weights']])
full_md = '## 表 3. 四个配置的全部指标明细\n\n' + md_table(rows)
full_md += '\n\n> S-* / NS-* 分别为阴影区 / 非阴影区指标；SSIM 在 test.py 中未启用（恒为 0），此处不列。\n'
open(os.path.join(OUT, 'table_full_metrics.md'), 'w', encoding='utf-8').write(full_md)

# 3) 逐图统计
rows = [['配置', 'PSNR 均值', 'PSNR 标准差', 'PSNR 最小', 'PSNR 最大',
         'RMSE 均值', 'RMSE 最小', 'RMSE 最大']]
for k, v in data.items():
    c = v['cfg']
    psnrs = [p.get('psnr') for p in v['per_image']]
    rmses = [p.get('rmse') for p in v['per_image']]
    pm, ps, pmn, pmx = stats(psnrs)
    rm, _, rmn, rmx = stats(rmses)
    rows.append(['{}→{}'.format(c['model'], c['test']),
                 '{:.3f}'.format(pm), '{:.3f}'.format(ps),
                 '{:.3f}'.format(pmn), '{:.3f}'.format(pmx),
                 '{:.3f}'.format(rm), '{:.3f}'.format(rmn), '{:.3f}'.format(rmx)])
open(os.path.join(OUT, 'table_per_image_stats.md'), 'w', encoding='utf-8').write(
    '## 表 4. 逐图统计\n\n' + md_table(rows))

# 4) LaTeX 三线表
tex_rows = ['\\begin{tabular}{lcccccccc}',
            '\\toprule',
            '配置 & 图数 & PSNR & RMSE & S-PSNR & S-RMSE & NS-PSNR & NS-RMSE \\\\',
            '\\midrule']
for k, v in data.items():
    c = v['cfg']; s = v['summary']
    tex_rows.append('{0}$\\rightarrow${1} & {2} & {3:.2f} & {4:.2f} & {5:.2f} & {6:.2f} & {7:.2f} & {8:.2f} \\\\'.format(
        c['model'], c['test'], v['num_images'],
        f(s.get('psnr')), f(s.get('rmse')), f(s.get('psnr_s')),
        f(s.get('rmse_s')), f(s.get('psnr_ns')), f(s.get('rmse_ns'))))
tex_rows += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(OUT, 'table_full_metrics.tex'), 'w', encoding='utf-8').write(
    '\n'.join(tex_rows))

# 5) CSV
with open(os.path.join(OUT, 'metrics_summary.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['config', 'trained_on', 'tested_on', 'kind', 'num_images',
                'psnr', 'rmse', 'psnr_s', 'rmse_s', 'psnr_ns', 'rmse_ns', 'weights'])
    for k, v in data.items():
        c = v['cfg']; s = v['summary']
        w.writerow([k, c['model'], c['test'], c['kind'], v['num_images'],
                    f(s.get('psnr')), f(s.get('rmse')), f(s.get('psnr_s')),
                    f(s.get('rmse_s')), f(s.get('psnr_ns')), f(s.get('rmse_ns')),
                    v['weights']])

print('[ok] 表格 / CSV / LaTeX 已写入:', OUT)


# ---------------- 绘图 ----------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print('[skip] 未安装 matplotlib/numpy，跳过绘图。安装: pip install matplotlib numpy')
    raise SystemExit(0)

IN_COLOR = '#2c7fb8'   # 域内
CROSS_COLOR = '#d95f0e'  # 跨域
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

labels = ['{}→{}'.format(v['cfg']['model'], v['cfg']['test']) for v in data.values()]
order = list(data.keys())

# 图 A：2×2 矩阵热力图（PSNR / RMSE），论文核心图
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric, title, cmap in [
        (axes[0], 'psnr', 'PSNR (dB) ↑', 'YlGnBu'),
        (axes[1], 'rmse', 'RMSE ↓', 'YlOrRd')]:
    grid = np.zeros((2, 2))
    grid[:] = np.nan
    for k, v in data.items():
        i = 0 if v['cfg']['model'] == 'ISTD' else 1
        j = 0 if v['cfg']['test'] == 'ISTD' else 1
        grid[i, j] = f(v['summary'].get(metric))
    im = ax.imshow(grid, cmap=cmap, aspect='auto')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['test ISTD', 'test SRD'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['train ISTD', 'train SRD'])
    for i in range(2):
        for j in range(2):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, '{:.2f}'.format(grid[i, j]), ha='center', va='center',
                        color='black', fontweight='bold')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle('2×2 Cross-dataset Evaluation (RASM)', fontsize=13, fontweight='bold')
fig.savefig(os.path.join(OUT, 'fig_2x2_matrix.png'))
plt.close(fig)

# 图 B：4 项指标分组柱状图（域内 vs 跨域）
metrics = [('psnr', 'PSNR (dB)'), ('rmse', 'RMSE'),
           ('psnr_s', 'S-PSNR'), ('psnr_ns', 'NS-PSNR')]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (m, title) in zip(axes.ravel(), metrics):
    vals = [f(data[k]['summary'].get(m)) for k in order]
    colors = [IN_COLOR if data[k]['cfg']['kind'] == 'in-domain' else CROSS_COLOR for k in order]
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_title(title)
    for b, val in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, val, '{:.2f}'.format(val),
                ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=IN_COLOR, label='in-domain'),
                    Patch(color=CROSS_COLOR, label='cross-domain')],
           loc='upper center', ncol=2, frameon=False)
fig.suptitle('Per-metric comparison across the 4 configurations', fontsize=13, fontweight='bold')
fig.savefig(os.path.join(OUT, 'fig_metric_bars.png'))
plt.close(fig)

# 图 C：逐图 PSNR 箱线图
fig, ax = plt.subplots(figsize=(9, 5))
boxes = [np.array([f(p.get('psnr')) for p in data[k]['per_image']]) for k in order]
bp = ax.boxplot(boxes, labels=labels, patch_artist=True, showmeans=True)
for patch, k in zip(bp['boxes'], order):
    c = IN_COLOR if data[k]['cfg']['kind'] == 'in-domain' else CROSS_COLOR
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel('PSNR (dB) per image')
ax.set_title('Per-image PSNR distribution')
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(OUT, 'fig_box_psnr.png'))
plt.close(fig)

# 图 D：泛化性落差（域内 - 跨域），按模型
fig, ax = plt.subplots(figsize=(7, 5))
models = ['ISTD', 'SRD']
gaps = []
for mdl in models:
    ind = next(k for k, v in data.items() if v['cfg']['model'] == mdl and v['cfg']['test'] == mdl)
    crs = next(k for k, v in data.items() if v['cfg']['model'] == mdl and v['cfg']['test'] != mdl)
    g = f(data[ind]['summary'].get('psnr')) - f(data[crs]['summary'].get('psnr'))
    gaps.append(g)
bars = ax.bar(models, gaps, color=['#2c7fb8', '#756bb1'])
for b, val in zip(bars, gaps):
    ax.text(b.get_x() + b.get_width() / 2, val, '{:.2f} dB'.format(val),
            ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('PSNR generalization gap (in-domain − cross-domain)')
ax.set_title('Cross-dataset generalization drop')
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(OUT, 'fig_gap.png'))
plt.close(fig)

print('[ok] 4 张图已写入:', OUT)
print('完成。产物位于', OUT)
