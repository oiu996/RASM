# -*- coding: utf-8 -*-
"""RASMv2 的训练入口（新增文件，不改动既有 train.py）。

与既有 train.py 的差异:
  1. 模型改为 RASMv2 (build_rasmv2)，支持 preset / kernel_size / dilations 等 v2 参数；
  2. 损失改为 RASMv2Loss (既有 L1+VGG 损失 + 三项任务损失)，把 mask 一并传入；
  3. 权重加载用 load_weights(..., strict=False)，可直接在官方 RASM 权重上续训。

用法与既有脚本一致，只是 --arch 固定为 RASMv2 并多了若干 v2 开关:
    cd /workspace/RASM && python improved/train_v2.py --arch RASMv2 ...
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RASM_ROOT = os.path.dirname(_HERE)
for p in (_RASM_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import datetime
import glob
import random
import shutil
import time

import numpy as np
import options
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from timm.utils import NativeScaler

# ---------------- 参数: 复用既有 options，只在其上"追加"v2 专有项 ----------------
_parser = options.Options().init(argparse.ArgumentParser(description='RASMv2 shadow removal'))
_parser.add_argument('--preset', type=str, default='full',
                     choices=['rasm', 'mgra', 'nsrc', 'film', 'rgate', 'full'],
                     help='消融预设: 每套只开一类改进')
_parser.add_argument('--kernel_size', type=int, default=11, help='区域注意力的邻域边长(奇数)')
_parser.add_argument('--dilations', type=int, nargs=3, default=(2, 2, 2),
                     help='三个区域注意力层的膨胀率 (enc2, bottleneck, dec0)')
_parser.add_argument('--attn_impl', type=str, default=None, choices=['unfold', 'natten'])
_parser.add_argument('--border_mode', type=str, default='zero', choices=['zero', 'mask'])
_parser.add_argument('--w_boundary', type=float, default=1.0, help='半影区加权损失权重')
_parser.add_argument('--w_ns', type=float, default=0.5, help='非阴影一致性损失权重')
_parser.add_argument('--w_chroma', type=float, default=0.2, help='色度一致性损失权重')
_parser.add_argument('--log_interval', type=float, default=30.0,
                     help='控制台进度心跳间隔(秒): 每过这么久打印一次 epoch/iter/loss/lr/ETA；0 表示不打印')
opt = _parser.parse_args()
print(opt)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from utils.loader import get_training_data, get_validation_data
from improved.rasmv2_model import build_rasmv2, load_weights
from improved.rasmv2_losses import RASMv2Loss

######### Logs dir ###########
run_name = opt.run_name if opt.run_name else datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(_RASM_ROOT, 'log', 'RASMv2' + opt.env, run_name)
utils.mkdir(log_dir)
utils.mkdir(os.path.join(log_dir, 'results'))
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(model_dir)
logname = os.path.join(log_dir, 'train.log')
print('Run dir: ', log_dir)

######### Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.benchmark = True

######### Model ###########
model_restoration = build_rasmv2(opt)
n_params = sum(p.numel() for p in model_restoration.parameters()) / 1e6
print('# model parameters: %.2f M (preset=%s)' % (n_params, opt.preset))
with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
optimizer = (optim.Adam if opt.optimizer.lower() == 'adam' else optim.AdamW)(
    model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=opt.weight_decay)
model_restoration.cuda()

######### Resume (支持直接加载官方 RASM 权重) ###########
resume_best_psnr = 0.0
if opt.resume:
    load_weights(model_restoration, opt.pretrain_weights, strict=False)
    try:
        ck = torch.load(opt.pretrain_weights, map_location='cpu')
        start_epoch = ck.get('epoch', 0) + 1
        if 'optimizer' in ck:
            optimizer.load_state_dict(ck['optimizer'])
            for p in optimizer.param_groups:
                p['lr'] = opt.lr_initial
        resume_best_psnr = ck.get('best_psnr', 0.0)
    except Exception as e:
        print('==> 仅加载权重(未恢复优化器状态):', e)
    print('==> Resumed from :', os.path.abspath(opt.pretrain_weights))

######### Scheduler ###########
if opt.warmup:
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.nepoch - opt.warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1,
                                       total_epoch=opt.warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler.step()

######### Loss ###########
criterion = RASMv2Loss(w_boundary=opt.w_boundary, w_ns=opt.w_ns, w_chroma=opt.w_chroma).cuda()

######### DataLoader ###########
print('===> Loading datasets')
train_dataset = get_training_data(opt.train_dir, {'patch_size': opt.train_ps})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)
val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)
print('Sizeof training set: ', train_dataset.__len__(), ', sizeof validation set: ', val_dataset.__len__())

metrics_path = os.path.join(log_dir, 'metrics.jsonl')
# 注意: run_name 不放进 RUN_META，由各调用点显式传入（避免 dict(..., **RUN_META) 重复关键字）
RUN_META = dict(arch='RASMv2', preset=opt.preset, dataset=opt.dataset,
                embed_dim=opt.embed_dim, kernel_size=opt.kernel_size,
                dilations=tuple(opt.dilations), source_weights=os.path.abspath(opt.pretrain_weights)
                if opt.resume else None)
utils.write_json(os.path.join(log_dir, 'config.json'),
                 dict(run_started_at=utils.now_iso(), run_name=run_name,
                      config=vars(opt), **RUN_META))


def _archive_best(epoch, psnr):
    for f in glob.glob(os.path.join(model_dir, '*_ep*_valpsnr*.pth')):
        try:
            os.remove(f)
        except OSError:
            pass
    name = "RASMv2_{}_{}_e{}_k{}_d{}_{}_{}_ep{:04d}_valpsnr{:05.2f}.pth".format(
        opt.dataset, opt.preset, opt.embed_dim, opt.kernel_size, ''.join(map(str, opt.dilations)),
        opt.token_projection, opt.token_mlp, epoch, psnr)
    shutil.copy(os.path.join(model_dir, 'model_best.pth'), os.path.join(model_dir, name))
    return name


######### Train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr, best_epoch, best_iter = resume_best_psnr, 0, 0
eval_now = 20 * len(train_loader)
display_iter = 100
display_iter_loss = 0
loss_scaler = NativeScaler()
torch.cuda.empty_cache()
index = 0

# 心跳日志: 每 log_interval 秒向控制台 + train.log 打印一行进度 (0 = 关闭)
log_interval = float(getattr(opt, 'log_interval', 30.0))
_train_t0 = time.time()
_beat_t0 = _train_t0
_beat_loss, _beat_n = 0.0, 0
_total_iters = opt.nepoch * len(train_loader)

for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    pbar = tqdm(train_loader, file=sys.stdout, dynamic_ncols=True, ncols=100)
    for i, data in enumerate(pbar, 0):
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        if 5 < epoch < opt.nepoch - 100:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        with torch.cuda.amp.autocast():
            restored = torch.clamp(model_restoration(input_, mask), 0, 1)
        loss = criterion(restored, target, mask)
        loss_scaler(loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()
        display_iter_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", ep=f"{epoch}/{opt.nepoch}", refresh=True)

        # ---- 心跳: 每 log_interval 秒打印一次训练进度 ----
        _beat_loss += loss.item()
        _beat_n += 1
        if log_interval > 0 and (time.time() - _beat_t0) >= log_interval:
            _now = time.time()
            _speed = index / max(_now - _train_t0, 1e-6)
            _eta_h = max(_total_iters - index, 0) / max(_speed, 1e-6) / 3600.0
            _mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            _msg = ('[进度 %s] Ep %d/%d | iter %d/%d | 全局 %d/%d | loss %.4f | lr %.2e | '
                    '%.2f it/s | 剩余 %.2f h | 显存峰值 %.2f GB'
                    % (datetime.datetime.now().strftime('%H:%M:%S'), epoch, opt.nepoch,
                       i + 1, len(train_loader), index, _total_iters,
                       _beat_loss / max(_beat_n, 1), scheduler.get_lr()[0],
                       _speed, _eta_h, _mem))
            print(_msg, flush=True)
            with open(logname, 'a') as f:
                f.write(_msg + '\n')
            _beat_t0, _beat_loss, _beat_n = _now, 0.0, 0

        if (index + 1) % display_iter == 0:
            display_iter_loss = 0

        if (index + 1) % eval_now == 0 and i > 0:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for data_val in tqdm(val_loader, file=sys.stdout, dynamic_ncols=True, ncols=100):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    with torch.cuda.amp.autocast():
                        restored = torch.clamp(model_restoration(input_, mask), 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
                psnr_val_rgb = sum(psnr_val_rgb) / len(val_loader)

                is_best = psnr_val_rgb > best_psnr
                archived_name = None
                if is_best:
                    best_psnr, best_epoch, best_iter = psnr_val_rgb, epoch, i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_psnr': best_psnr},
                               os.path.join(model_dir, 'model_best.pth'))
                    archived_name = _archive_best(epoch, psnr_val_rgb)
                utils.append_jsonl(metrics_path, dict(timestamp=utils.now_iso(), phase='val',
                                                      run_name=run_name,
                                                      epoch=epoch, iter=i,
                                                      psnr=round(psnr_val_rgb, 4),
                                                      lr=scheduler.get_lr()[0], is_best=is_best,
                                                      best_psnr=round(best_psnr, 4),
                                                      archived_name=archived_name, **RUN_META))
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname, 'a') as f:
                    f.write("[Ep %d it %d\t PSNR: %.4f] ---- [best_Ep %d best_it %d Best_PSNR %.4f]\n"
                            % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            model_restoration.train()
            torch.cuda.empty_cache()
    scheduler.step()
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLR {:.6f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLR {:.6f}\n".format(
            epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr},
               os.path.join(model_dir, 'model_latest.pth'))
    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch, 'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr},
                   os.path.join(model_dir, 'model_epoch_{}.pth'.format(epoch)))

utils.write_json(os.path.join(log_dir, 'summary.json'),
                 dict(finished_at=utils.now_iso(), run_name=run_name,
                      best_epoch=best_epoch, best_iter=best_iter,
                      best_psnr=round(best_psnr, 4), final_epoch=opt.nepoch, **RUN_META))
print("==> Run dir   : ", log_dir)
print("==> Best PSNR : {:.4f} @ epoch {} (iter {})".format(best_psnr, best_epoch, best_iter))
