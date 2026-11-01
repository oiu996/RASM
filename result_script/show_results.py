#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 RASM 的训练 / 测试产物转成可直接看的结果展示。

在 RASM/ 根目录下执行：
    python check_script/show_results.py summary   # 所有训练 run 的汇总对比表
    python check_script/show_results.py curve     # 训练曲线（存成 PNG）
    python check_script/show_results.py test      # 所有测试记录的指标对比表
    python check_script/show_results.py all       # 以上全部

输出均为 Markdown 表格，可直接贴进报告 / README。
只依赖标准库；只有画曲线时才需要 matplotlib（缺失会自动跳过并提示）。
"""

import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # RASM/
LOG_ROOT = os.path.join(ROOT, 'log')


# ---------------- 基础工具 ----------------
def read_jsonl(path):
    """逐行读 jsonl；文件不存在返回空列表。"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def md_table(rows):
    """rows[0] 为表头，转成 Markdown 表格字符串。"""
    ncol = len(rows[0])
    widths = [max(len(str(r[i])) for r in rows) for i in range(ncol)]

    def fmt(r):
        return '| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |'

    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    return '\n'.join([fmt(rows[0]), sep] + [fmt(r) for r in rows[1:]])


# ---------------- 收集训练 run ----------------
def find_runs():
    """扫描 log/<arch><env>/<run_name>/ 下的所有 run，返回记录列表。"""
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(LOG_ROOT, '*', '*', 'config.json'))):
        run_dir = os.path.dirname(cfg_path)
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        inner = cfg.get('config', {})
        rec = {
            'run_dir'  : os.path.relpath(run_dir, ROOT),
            'started'  : (cfg.get('run_started_at') or '')[:19],
            'source'   : cfg.get('source_weights'),
            'resumed'  : cfg.get('resumed', False),
            'dataset'  : inner.get('dataset'),
            'arch'     : inner.get('arch'),
            'embed_dim': inner.get('embed_dim'),
            'win_size' : inner.get('win_size'),
            'proj'     : inner.get('token_projection'),
            'mlp'      : inner.get('token_mlp'),
            'nepoch'   : inner.get('nepoch'),
            'metrics'  : os.path.join(run_dir, 'metrics.jsonl'),
        }
        # summary.json 只有训练正常结束才会生成
        summary_path = os.path.join(run_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, encoding='utf-8') as f:
                s = json.load(f)
            rec['best_psnr']  = s.get('best_psnr')
            rec['best_epoch'] = s.get('best_epoch')
            rec['finished']   = (s.get('finished_at') or '')[:19]
        else:
            rec['best_psnr'] = rec['best_epoch'] = None
            rec['finished'] = ''
        runs.append(rec)
    return runs


# ---------------- 各子命令 ----------------
def cmd_summary():
    """所有训练 run 的横向对比表。"""
    runs = find_runs()
    if not runs:
        print('未找到任何 run —— 请确认在 RASM/ 根目录执行，且 log/ 下已有训练产物。')
        return

    rows = [['run 目录', '数据集', '最优 PSNR', '最优 epoch', '起始时间', '权重来源']]
    for r in runs:
        src = os.path.basename(r['source']) if r['source'] else '从零训练'
        psnr = '{:.4f}'.format(r['best_psnr']) if r['best_psnr'] is not None else '(训练未跑完)'
        rows.append([
            r['run_dir'], str(r['dataset']), psnr,
            str(r['best_epoch'] if r['best_epoch'] is not None else '-'),
            r['started'], src,
        ])
    print('## 训练 run 汇总\n')
    print(md_table(rows))

    done = [r for r in runs if r['best_psnr'] is not None]
    if done:
        b = max(done, key=lambda x: x['best_psnr'])
        print('\n**最优 run**: `{}` —— val PSNR = {:.4f} (epoch {})'.format(
            b['run_dir'], b['best_psnr'], b['best_epoch']))


def cmd_curve():
    """把所有 run 的验证 PSNR 画到一张图上。"""
    runs = find_runs()
    try:
        import matplotlib
        matplotlib.use('Agg')          # 无显示器环境必须
        import matplotlib.pyplot as plt
    except ImportError:
        print('未安装 matplotlib，跳过绘图。安装: pip install matplotlib')
        return

    plt.figure(figsize=(11, 6))
    plotted = 0
    for r in runs:
        recs = [x for x in read_jsonl(r['metrics']) if x.get('phase') == 'val']
        if not recs:
            continue
        xs = [x['epoch'] for x in recs]
        ys = [x['psnr'] for x in recs]
        best = '{:.2f}'.format(r['best_psnr']) if r['best_psnr'] is not None else '?'
        plt.plot(xs, ys, marker='o', markersize=3,
                 label='{} (best {})'.format(r['run_dir'], best))
        plotted += 1

    if not plotted:
        print('没有可绘制的评估记录（metrics.jsonl 不存在或为空）')
        return

    plt.xlabel('Epoch')
    plt.ylabel('Val PSNR (dB)')
    plt.title('Validation PSNR')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    out = os.path.join(ROOT, 'training_curve.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print('训练曲线已保存: {}'.format(out))


def cmd_test():
    """所有测试记录的指标对比表（每次测试一行，追加累积）。"""
    files = sorted(glob.glob(os.path.join(ROOT, 'results_*', 'metrics.jsonl')))
    if not files:
        print('未找到测试记录 —— 请先运行 bash test_ISTD.sh / test_SRD.sh')
        return

    rows = [['测试时间', '结果目录', '权重', 'PSNR', 'RMSE',
             'S-PSNR', 'S-RMSE', 'NS-PSNR', 'NS-RMSE', '图数']]
    for fp in files:
        for rec in read_jsonl(fp):
            s = rec.get('summary', {})
            rows.append([
                (rec.get('timestamp') or '')[:19],
                os.path.basename(rec.get('result_dir') or ''),
                os.path.basename(rec.get('weights') or ''),
                '{:.4f}'.format(s.get('psnr', 0)),
                '{:.4f}'.format(s.get('rmse', 0)),
                '{:.4f}'.format(s.get('psnr_s', 0)),
                '{:.4f}'.format(s.get('rmse_s', 0)),
                '{:.4f}'.format(s.get('psnr_ns', 0)),
                '{:.4f}'.format(s.get('rmse_ns', 0)),
                str(rec.get('num_images', '')),
            ])
    print('## 测试指标对比\n')
    print(md_table(rows))
    print('\n> 注: SSIM 相关字段在 test.py 源码里被注释掉，恒为 0，故此表未列出 SSIM。')
    print('> S-* 为阴影区指标，NS-* 为非阴影区指标。')


# ---------------- 入口 ----------------
def main():
    cmds = {'summary': cmd_summary, 'curve': cmd_curve, 'test': cmd_test}
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg not in cmds and arg != 'all':
        print(__doc__)
        sys.exit(1)
    if arg == 'all':
        for fn in (cmd_summary, cmd_curve, cmd_test):
            fn()
            print()
    else:
        cmds[arg]()


if __name__ == '__main__':
    main()
