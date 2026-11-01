#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步骤 2：GPU 推理 —— 用两个权重分别给样例去阴影，并算出逐图指标。

输入：showcase/samples 下的 A_input / B_mask / C_gt（由 select_samples.py 生成）
输出：
    showcase/outputs/<权重名>/XX_key.jpg   ← 去阴影结果图
    showcase/metrics.json                  ← 逐图 PSNR / RMSE（全图 / 阴影区 / 非阴影区）
    showcase/metrics.md                    ← 可直接贴进报告 / README 的表格

指标定义与 RASM 原版 test.py 完全一致，方便和全量测试结果横向对比。

用法（在 RASM/ 根目录执行，需 GPU）：
    python showcase/scripts/run_inference.py
    python showcase/scripts/run_inference.py --weights srd=log/RASM_srd_demo/models/SRD.pth \
                                             istd=log/RASM_istd_demo/models/AISTD.pth
    python showcase/scripts/run_inference.py --tile 720          # 显存不够时切分块推理
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # RASM/ 根目录
sys.path.insert(0, ROOT)

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


# 两个"原模型权重"的默认值（官方预训练）
DEFAULT_WEIGHTS = [
    'srd=log/RASM_srd_demo/models/SRD.pth',       # SRD 上训练的官方权重
    'istd=log/RASM_istd_demo/models/AISTD.pth',   # ISTD+ 上训练的官方权重（跨数据集）
]


def parse_args():
    p = argparse.ArgumentParser(description='样例去阴影推理（GPU）')
    p.add_argument('--samples_dir', default='showcase/samples', type=str)
    p.add_argument('--manifest', default='showcase/samples/manifest.json', type=str)
    p.add_argument('--out_root', default='showcase/outputs', type=str,
                   help='结果根目录，每个权重一个子目录')
    p.add_argument('--weights', nargs='+', default=DEFAULT_WEIGHTS,
                   help='权重列表，格式 名字=路径（也可只给路径，名字取文件名主干）')
    p.add_argument('--metrics_json', default='showcase/metrics.json', type=str)
    p.add_argument('--metrics_md', default='showcase/metrics.md', type=str)
    # ---- 模型结构参数（必须与训练一致，默认即 RASM 官方配置）----
    p.add_argument('--embed_dim', type=int, default=32)
    p.add_argument('--win_size', type=int, default=10)
    p.add_argument('--token_projection', type=str, default='linear')
    p.add_argument('--token_mlp', type=str, default='leff')
    p.add_argument('--train_ps', type=int, default=320)
    # ---- 设备 / 推理方式 ----
    p.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
    p.add_argument('--tile', type=int, default=None,
                   help='分块推理尺寸（如 720），显存不足时使用；默认原分辨率')
    p.add_argument('--tile_overlap', type=int, default=32)
    p.add_argument('--save_float_npy', action='store_true',
                   help='额外保存 float32 结果（npy），便于后续做误差分析')
    return p.parse_args()


def abspath(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_model(args, weight_path, device):
    """构建 RASM 并加载权重（兼容 DataParallel 的 module. 前缀 / 不同 ckpt 键名）。"""
    from model import RASM
    model = RASM(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=args.win_size,
                 token_projection=args.token_projection, token_mlp=args.token_mlp)
    ckpt = torch.load(weight_path, map_location=device)
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith('module.') else k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f'  [warn] 缺失权重 {len(missing)} 项（示例: {list(missing)[:3]}）')
    if unexpected:
        print(f'  [warn] 多余权重 {len(unexpected)} 项（示例: {list(unexpected)[:3]}）')
    model.to(device)
    model.eval()
    return model


def _infer(model, inp, mask, args):
    """按当前 args.tile 选择整图或分块推理。"""
    if args.tile is None:
        return model(inp, mask)
    return infer_tiled(model, inp, mask, args.tile, args.tile_overlap)


@torch.no_grad()
def infer_tiled(model, inp, mask, tile, overlap):
    """显存不够时按 tile 分块推理（与原版 test.py 逻辑一致）。"""
    b, c, h, w = inp.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, 'tile 需为 8 的倍数'
    stride = tile - overlap
    h_idx = list(range(0, h - tile, stride)) + [h - tile]
    w_idx = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(inp)
    W = torch.zeros_like(E)
    for i in h_idx:
        for j in w_idx:
            out = model(inp[..., i:i + tile, j:j + tile], mask[..., i:i + tile, j:j + tile])
            E[..., i:i + tile, j:j + tile].add_(out)
            W[..., i:i + tile, j:j + tile].add_(torch.ones_like(out))
    return E.div_(W)


def calc_metrics(out, gt, mask01):
    """与 test.py 完全一致的指标：全图 / 阴影区(S) / 非阴影区(NS) 的 PSNR 与 RMSE(LAB)。"""
    bm = np.expand_dims((mask01 > 0.5).astype(np.float32), axis=2)
    m = {
        'psnr': psnr_loss(out, gt),
        'psnr_s': psnr_loss(out * bm, gt * bm),
        'psnr_ns': psnr_loss(out * (1 - bm), gt * (1 - bm)),
    }
    lab_out, lab_gt = cv2.cvtColor(out, cv2.COLOR_RGB2LAB), cv2.cvtColor(gt, cv2.COLOR_RGB2LAB)
    m['rmse'] = float(np.abs(lab_out - lab_gt).mean() * 3)
    m['rmse_s'] = float(np.abs(cv2.cvtColor(out * bm, cv2.COLOR_RGB2LAB)
                               - cv2.cvtColor(gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum())
    m['rmse_ns'] = float(np.abs(cv2.cvtColor(out * (1 - bm), cv2.COLOR_RGB2LAB)
                                - cv2.cvtColor(gt * (1 - bm), cv2.COLOR_RGB2LAB)).sum() / (1 - bm).sum())
    return {k: round(float(v), 4) for k, v in m.items()}


def main():
    args = parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if not torch.cuda.is_available():
        raise SystemExit('[error] 当前环境没有可用 CUDA，本脚本设计为 GPU 运行；'
                         '请切换到 GPU 机器后再执行（或检查 CUDA_VISIBLE_DEVICES）。')
    device = torch.device('cuda')
    print(f'[info] GPU: {torch.cuda.get_device_name(0)}')
    # 注意：必须用 mem_get_info()，它返回驱动层面的真实空闲/总量（含其它进程占用）。
    # 旧的 "total_memory - memory_allocated" 只统计本进程已分配量，会漏掉其它进程占用的显存，
    # 从而把"被占满的卡"误报成"完全空闲"，掩盖真实的 OOM 风险。
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f'[info] 当前空闲显存约 {free_mem / 1e9:.2f} GiB '
          f'（总量 {total_mem / 1e9:.2f} GiB）')
    if free_mem < 3.0 * 1e9:
        print('[warn] 空闲显存不足 ~3.0 GiB（全分辨率推理约需 2.7 GiB），很可能 OOM；'
              '请先释放占用显存的其它进程（nvidia-smi 查 PID 后 kill），或换用空闲 GPU（--gpus 1）。'
              '脚本会在 OOM 时自动切换为分块推理。')

    manifest_path = abspath(args.manifest)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)['samples']
    print(f'[info] 样例数: {len(samples)}（清单 {manifest_path}）')

    # 解析权重列表
    weights = []
    for w in args.weights:
        if '=' in w:
            name, path = w.split('=', 1)
        else:
            path = w
            name = os.path.splitext(os.path.basename(w))[0]
        weights.append((name, abspath(path)))

    out_root = abspath(args.out_root)
    result = {'samples': {}, 'weights': {}}
    for name, wpath in weights:
        assert os.path.isfile(wpath), f'权重不存在: {wpath}'
        print(f'\n=== 权重: {name}  ({wpath}) ===')
        model = load_model(args, wpath, device)
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'  参数量: {n_param:.2f} M')
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        for s in tqdm(samples, desc=f'  infer[{name}]', ncols=100):
            gt = load_img(abspath(s['C']))
            inp = load_img(abspath(s['A']))
            mask = load_mask(abspath(s['B']))
            if mask.shape[:2] != inp.shape[:2]:              # 保险：mask 与输入对齐
                mask = cv2.resize(mask, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_NEAREST)

            # RASM 有 3 次 /2 下采样 + 3 次 *2 上采样，输入须是 8 的倍数，否则 cat 维度错位
            H, W = inp.shape[:2]
            inp = pad_to_multiple(inp, 8)
            mask = pad_to_multiple(mask, 8, is_mask=True)

            t_inp = torch.from_numpy(np.float32(inp)).permute(2, 0, 1).unsqueeze(0).to(device)
            t_mask = torch.from_numpy(np.float32(mask)).unsqueeze(0).unsqueeze(0).to(device)

            t0 = time.time()
            try:
                out = _infer(model, t_inp, t_mask, args)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if args.tile is None:
                    # 整图推理 OOM：自动退回分块推理，避免直接退出（可扛住"部分被占用"的卡）
                    args.tile = 256
                    print('\n[CUDA OOM] 整图推理显存不足，自动切换为分块推理 (TILE=256) 重试本图及后续图片。')
                    print('  若仍 OOM，请先释放占用显存的其它进程（nvidia-smi 查 PID 后 kill），'
                          '或换用空闲 GPU（--gpus 1）。')
                    try:
                        out = _infer(model, t_inp, t_mask, args)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print('[CUDA OOM] 分块推理仍显存不足，终止。请释放显存或换卡后重试。')
                        sys.exit(1)
                else:
                    print('\n[CUDA OOM] 分块推理仍显存不足，常见原因：GPU 被其它进程占满。')
                    print('  解决办法：')
                    print('   1) 查看占用： nvidia-smi    （记下占大头的 PID）')
                    print('   2) 释放占用： kill <PID>     （或向占用者确认能否释放）')
                    print('   3) 换空闲卡： bash showcase/run_demo.sh 前设 GPU_DEV=1（或 --gpus 1）')
                    sys.exit(1)
            out = out[:, :, :H, :W]                          # 裁回原尺寸
            out = torch.clamp(out.detach(), 0, 1).cpu().numpy().squeeze().transpose(1, 2, 0)
            dt = time.time() - t0

            save_img(out * 255.0, os.path.join(out_dir, s['stem'] + '.jpg'))
            if args.save_float_npy:
                np.save(os.path.join(out_dir, s['stem'] + '.npy'), out.astype(np.float32))

            rec = result['samples'].setdefault(s['stem'], {'id': s['id'], 'key': s['key'],
                                                           'shadow_ratio': s['shadow_ratio']})
            rec[name] = {**calc_metrics(out, gt, mask), 'seconds': round(dt, 2)}
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    names = [n for n, _ in weights]
    summary = {}
    for n in names:
        vals = [result['samples'][s['stem']][n] for s in samples]
        summary[n] = {k: round(float(np.mean([v[k] for v in vals])), 4)
                      for k in ('psnr', 'psnr_s', 'psnr_ns', 'rmse', 'rmse_s', 'rmse_ns')}
        summary[n]['seconds_per_image'] = round(float(np.mean([v['seconds'] for v in vals])), 2)
    result['weights'] = {n: abspath(p) for n, p in weights}
    result['weight_names'] = names
    result['summary'] = summary

    with open(abspath(args.metrics_json), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n[done] 指标 JSON: {abspath(args.metrics_json)}')

    # ---- Markdown 表格 ----
    lines = ['| # | 样例 | 阴影占比 | ' +
             ' | '.join(f'{n} PSNR↑' for n in names) +
             ' | ' + ' | '.join(f'{n} RMSE↓' for n in names) + ' |',
             '|---|------|---------|' + '---------|' * (2 * len(names))]
    for s in samples:
        r = result['samples'][s['stem']]
        cells = [f"{r[n]['psnr']:.2f}" for n in names] + [f"{r[n]['rmse']:.2f}" for n in names]
        lines.append(f"| {s['id']} | `{s['key']}` | {s['shadow_ratio']*100:.1f}% | " +
                     ' | '.join(cells) + ' |')
    avg_psnr = ' | '.join(f'**{summary[n]["psnr"]:.2f}**' for n in names)
    avg_rmse = ' | '.join(f'**{summary[n]["rmse"]:.2f}**' for n in names)
    lines.append(f'| — | **平均** | — | {avg_psnr} | {avg_rmse} |')
    lines.append('')
    lines.append('> PSNR 越高越好，RMSE(LAB) 越低越好；指标口径与原版 `test.py` 一致。')
    with open(abspath(args.metrics_md), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[done] 指标表格: {abspath(args.metrics_md)}')
    for n in names:
        print(f"  {n:<8s} PSNR={summary[n]['psnr']:.2f}  RMSE={summary[n]['rmse']:.2f}")


if __name__ == '__main__':
    main()
