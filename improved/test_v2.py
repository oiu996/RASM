# -*- coding: utf-8 -*-
"""RASMv2 的测试入口（新增文件，不改动既有 test.py）。

与既有 test.py 的差异:
  1. 模型为 RASMv2，权重加载用 load_weights(..., strict=False)（可加载官方 RASM 权重）；
  2. 结构与训练保持一致即可，preset/kernel_size/dilations 需与训练时相同；
  3. 指标落盘到 <result_dir>/metrics_v2.jsonl，不与既有 results_* 混用。

用法:
    cd /workspace/RASM && python improved/test_v2.py --input_dir ... --weights ... ...
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RASM_ROOT = os.path.dirname(_HERE)
for p in (_RASM_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

import utils
from utils.loader import get_validation_data
from improved.rasmv2_model import RASMv2, load_weights

parser = argparse.ArgumentParser(description='RASMv2 shadow removal evaluation')
# 与 train_v2.py / *.sh 的参数风格保持一致（测试时不使用，仅为了不吃掉未知参数）
parser.add_argument('--arch', default='RASMv2', type=str)
parser.add_argument('--dataset', default='ISTD', type=str)
parser.add_argument('--input_dir', default='datasets/ISTD_Dataset/test/', type=str)
parser.add_argument('--result_dir', default='./results_istd_v2/', type=str)
parser.add_argument('--weights', default='./log/RASMv2_istd/models/model_best.pth', type=str)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--save_images', action='store_true')
parser.add_argument('--cal_metrics', action='store_true')
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--win_size', type=int, default=10)
parser.add_argument('--token_projection', type=str, default='linear')
parser.add_argument('--token_mlp', type=str, default='leff')
parser.add_argument('--train_ps', type=int, default=320)
parser.add_argument('--tile', type=int, default=None)
parser.add_argument('--tile_overlap', type=int, default=32)
parser.add_argument('--ex_name', type=str, default='RASMv2')
# v2 专有
parser.add_argument('--preset', type=str, default='full',
                    choices=['rasm', 'mgra', 'nsrc', 'film', 'rgate', 'full'])
parser.add_argument('--kernel_size', type=int, default=11)
parser.add_argument('--dilations', type=int, nargs=3, default=(2, 2, 2))
parser.add_argument('--attn_impl', type=str, default=None, choices=['unfold', 'natten'])
parser.add_argument('--border_mode', type=str, default='zero', choices=['zero', 'mask'])

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8,
                         drop_last=False)

model_restoration = RASMv2(img_size=args.train_ps, embed_dim=args.embed_dim, win_size=args.win_size,
                           token_projection=args.token_projection, token_mlp=args.token_mlp,
                           preset=args.preset, kernel_size=args.kernel_size,
                           dilations=tuple(args.dilations),
                           attn_impl=args.attn_impl, border_mode=args.border_mode)
print('# model parameters: %.2f M (preset=%s)' % (
    sum(p.numel() for p in model_restoration.parameters()) / 1e6, args.preset))
load_weights(model_restoration, args.weights, strict=False)
print('===>Testing using weights: ', args.weights)
model_restoration.cuda()
model_restoration.eval()

with torch.no_grad():
    psnr_all, psnr_s, psnr_ns, rmse_all, rmse_s, rmse_ns, per_image, times = [], [], [], [], [], [], [], []
    pbar = tqdm(test_loader, file=sys.stdout, dynamic_ncols=True, ncols=100)
    for data_test in pbar:
        rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_noisy = data_test[1].cuda()
        mask = data_test[2].cuda()
        filenames = data_test[3]

        torch.cuda.synchronize()
        t0 = time.time()
        if args.tile is None:
            rgb_restored = model_restoration(rgb_noisy, mask)
        else:
            b, c, h, w = rgb_noisy.shape
            tile = min(args.tile, h, w)
            assert tile % 8 == 0, 'tile size should be multiple of 8'
            stride = tile - args.tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
            W = torch.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                    E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(
                        model_restoration(in_patch, mask_patch))
                    W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(torch.ones_like(E[..., :1, :1, :1]))
            rgb_restored = E.div_(W)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

        if args.cal_metrics:
            bm = np.expand_dims((mask > 0).float().cpu().numpy().squeeze(), axis=2)
            psnr_all.append(psnr_loss(rgb_restored, rgb_gt))
            psnr_ns.append(psnr_loss(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
            psnr_s.append(psnr_loss(rgb_restored * bm, rgb_gt * bm))
            lab = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
            rmse_all.append(np.abs(lab(rgb_restored) - lab(rgb_gt)).mean() * 3)
            rmse_s.append(np.abs(lab(rgb_restored * bm) - lab(rgb_gt * bm)).sum() / bm.sum())
            rmse_ns.append(np.abs(lab(rgb_restored * (1 - bm)) - lab(rgb_gt * (1 - bm))).sum()
                           / (1 - bm).sum())
            per_image.append({'filename': filenames[0],
                              'psnr': round(psnr_all[-1], 4), 'psnr_s': round(psnr_s[-1], 4),
                              'psnr_ns': round(psnr_ns[-1], 4), 'rmse': round(rmse_all[-1], 4),
                              'rmse_s': round(rmse_s[-1], 4), 'rmse_ns': round(rmse_ns[-1], 4)})
            pbar.set_postfix(file=os.path.basename(filenames[0])[:18],
                             psnr=f"{psnr_all[-1]:.2f}", refresh=True)

        if args.save_images:
            utils.save_img(rgb_restored * 255.0, os.path.join(args.result_dir, filenames[0]))

if args.cal_metrics:
    n = len(test_dataset)
    print(f'\n## {args.ex_name}')
    print('PSNR: %.4f, RMSE: %.4f, 平均推理 %.3fs' % (sum(psnr_all) / n, sum(rmse_all) / n,
                                                      sum(times) / len(times)))
    print('SPSNR: %.4f, SRMSE: %.4f' % (sum(psnr_s) / n, sum(rmse_s) / n))
    print('NSPSNR: %.4f, NSRMSE: %.4f' % (sum(psnr_ns) / n, sum(rmse_ns) / n))
    print('|%s|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|' % (args.ex_name, sum(psnr_all) / n, sum(rmse_all) / n,
                                                  sum(psnr_s) / n, sum(rmse_s) / n,
                                                  sum(psnr_ns) / n, sum(rmse_ns) / n))
    utils.append_jsonl(os.path.join(args.result_dir, 'metrics_v2.jsonl'),
                       dict(timestamp=utils.now_iso(), ex_name=args.ex_name,
                            weights=os.path.abspath(args.weights),
                            input_dir=os.path.abspath(args.input_dir), preset=args.preset,
                            kernel_size=args.kernel_size, dilations=tuple(args.dilations),
                            num_images=n,
                            summary={'psnr': round(sum(psnr_all) / n, 4),
                                     'rmse': round(sum(rmse_all) / n, 4),
                                     'psnr_s': round(sum(psnr_s) / n, 4),
                                     'rmse_s': round(sum(rmse_s) / n, 4),
                                     'psnr_ns': round(sum(psnr_ns) / n, 4),
                                     'rmse_ns': round(sum(rmse_ns) / n, 4),
                                     'avg_time': round(sum(times) / len(times), 4)},
                            per_image=per_image))
