import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import shutil
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

from torch.utils.tensorboard import SummaryWriter


######### Logs dir ###########
# 方案D: 每个 run 落在独立子目录 log/<arch><env>/<run_name>/ 下，实验之间物理隔离、互不覆盖。
#   --run_name 不传 -> 自动用时间戳新建目录（推荐）
#   --run_name 传入 -> 写回该目录（训练意外中断后，接着同一个 run 继续跑）
run_name = opt.run_name if opt.run_name else datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env, run_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, 'train.log')
print("Now time is : ", datetime.datetime.now().isoformat())
print("Run dir: ", log_dir)
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# 注: config.json 在下方 Resume 之后统一写入 —— 因为要把「原有权重来源」一起记进去

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)



######### Model ###########
model_restoration = utils.get_arch(opt)
print('# model_restoration parameters: %.2f M'%(sum(param.numel() for param in model_restoration.parameters())/ 1e6))

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
# model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()


######### Resume ###########
source_weights = None    # 原有权重来源: 续训时为 ckpt 绝对路径，从零训练则为 None
resume_best_psnr = 0.0   # 从 ckpt 恢复的历史最优 PSNR（在下方 "Start Epoch" 处赋给 best_psnr）
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    source_weights = os.path.abspath(path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)
    # 修复: 原版 best_psnr 在 resume 之后才被初始化为 0，续训首个 epoch 的 PSNR 必然 > 0，
    #       会用更差的权重覆盖掉刚加载进来的最优权重。这里从 ckpt 恢复历史最优值。
    resume_best_psnr = utils.load_best_psnr(path_chk_rest)
    # 记录续训来源，保证可追溯
    with open(os.path.join(log_dir, 'resume_from.txt'), 'w') as f:
        f.write(os.path.abspath(path_chk_rest)+'\n')
    print("==> Resumed from : ", os.path.abspath(path_chk_rest))
    print("==> Restored best_psnr : {:.4f}".format(resume_best_psnr))

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### 运行元信息 / 结构化指标 ###########
# 本次 run 的身份信息 = 自描述命名里的那些字段 + 权重来源；每条指标记录都会带上
RUN_META = {
    'run_name'        : run_name,
    'arch'            : opt.arch,
    'dataset'         : opt.dataset,
    'embed_dim'       : opt.embed_dim,
    'win_size'        : opt.win_size,
    'token_projection': opt.token_projection,
    'token_mlp'       : opt.token_mlp,
    'source_weights'  : source_weights,   # 原有权重（续训来源），从零训练为 None
}

# config.json: 完整超参 + 起始时间 + 权重来源，用于事后复现
utils.write_json(os.path.join(log_dir, 'config.json'), {
    'run_started_at' : utils.now_iso(),
    'run_name'       : run_name,
    'run_dir'        : os.path.abspath(log_dir),
    'source_weights' : source_weights,
    'resumed'        : bool(opt.resume),
    'config'         : vars(opt),
})

# metrics.jsonl: 每个评估点追加一行（含时间戳 / 权重来源 / 指标 / 身份信息）
metrics_path = os.path.join(log_dir, 'metrics.jsonl')

# ######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()


######### Loss ###########
# criterion = CharbonnierLoss().cuda()
from losses import L1_Vgg_losses
criterion = L1_Vgg_losses().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

######## eval ################
with torch.no_grad():
    model_restoration.eval()
    for ii, data_val in enumerate(tqdm(val_loader, file=sys.stdout, dynamic_ncols=True, ncols=100), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        mask = data_val[2].cuda()
        filenames = data_val[3]
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_, mask)
        restored = torch.clamp(restored,0,1)
       
       
writer = SummaryWriter(comment=opt.boardname)

def _archive_best(model_dir, opt, epoch, psnr):
    """把当前最优权重另存一份『自描述』命名的归档副本:
       {arch}_{dataset}_e{embed_dim}_w{win_size}_{token_projection}_{token_mlp}_ep{epoch:04d}_valpsnr{psnr:05.2f}.pth
       同一 run 目录内只保留最新一份归档，避免每次刷新 best 都堆积一堆文件。
       注意: 固定名 model_best.pth 必须保留 —— --resume 与测试脚本都依赖它做稳定指针。
    """
    for f in glob.glob(os.path.join(model_dir, '*_ep*_valpsnr*.pth')):
        try:
            os.remove(f)
        except OSError:
            pass
    name = "{}_{}_e{}_w{}_{}_{}_ep{:04d}_valpsnr{:05.2f}.pth".format(
        opt.arch, opt.dataset, opt.embed_dim, opt.win_size,
        opt.token_projection, opt.token_mlp, epoch, psnr)
    shutil.copy(os.path.join(model_dir, "model_best.pth"),
                os.path.join(model_dir, name))
    return name

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = resume_best_psnr   # 续训时沿用 ckpt 里的历史最优，避免被更差的权重覆盖
best_epoch = 0
best_iter = 0
eval_now = 20 * len(train_loader)
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

display_iter = 100
display_iter_loss = 0
loss_scaler = NativeScaler()
torch.cuda.empty_cache()
ii=0
index = 0
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    epoch_ssim_loss = 0
    pbar = tqdm(train_loader, file=sys.stdout, dynamic_ncols=True, ncols=100)
    for i, data in enumerate(pbar, 0):
        # zero_grad
        index += 1
        optimizer.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()
        mask = data[2].cuda()
        if epoch > 5 and epoch < opt.nepoch-100:
            target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
        with torch.cuda.amp.autocast():
            restored = model_restoration(input_, mask)
            restored = torch.clamp(restored,0,1)
        loss = criterion(restored, target)
        
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        
        epoch_loss +=loss.item()
        display_iter_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", ep=epoch, refresh=True)
        
        #### tensorboard display ######
        if (index + 1) % display_iter == 0:
            writer.add_scalar('Loss', display_iter_loss, index)
            display_iter_loss = 0
            if epoch < 50 or epoch > opt.nepoch-30:
                writer.add_images('input', input_, index)
                writer.add_images('mask', mask, index)
                writer.add_images('GT', target, index)
                writer.add_images("result", restored, index)
        
        #### Evaluation ####
        if (index+1)%eval_now==0 and i>0:
            eval_shadow_rmse = 0
            eval_nonshadow_rmse = 0
            eval_rmse = 0
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, data_val in enumerate(tqdm(val_loader, file=sys.stdout, dynamic_ncols=True, ncols=100), 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    mask = data_val[2].cuda()
                    filenames = data_val[3]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_, mask)
                    restored = torch.clamp(restored,0,1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

                psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                writer.add_scalars("PSNR", {'val': psnr_val_rgb}, epoch)
                
                is_best = psnr_val_rgb > best_psnr
                archived_name = None
                if is_best:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'best_psnr': best_psnr,
                                }, os.path.join(model_dir,"model_best.pth"))
                    # 方案D: 另存一份「自描述」命名的归档副本（同一 run 内只留最新一份）
                    archived_name = _archive_best(model_dir, opt, epoch, psnr_val_rgb)

                # 结构化落盘: 每个评估点追加一行（时间戳 / 指标 / 是否刷新最优 / 身份信息）
                utils.append_jsonl(metrics_path, dict(
                    timestamp     = utils.now_iso(),
                    phase         = 'val',
                    epoch         = epoch,
                    iter          = i,
                    psnr          = round(psnr_val_rgb, 4),
                    lr            = scheduler.get_lr()[0],
                    is_best       = is_best,
                    best_psnr     = round(best_psnr, 4),
                    archived_name = archived_name,
                    **RUN_META,
                ))
                print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                
            with torch.no_grad():
                model_restoration.eval()
                psnr_train_rgb = []
                for ii, data_train in enumerate(tqdm(train_loader, file=sys.stdout, dynamic_ncols=True, ncols=100), 0):
                    target = data_train[0].cuda()
                    input_ = data_train[1].cuda()
                    mask = data_train[2].cuda()
                    filenames = data_train[3]
                    with torch.cuda.amp.autocast():
                        restored = model_restoration(input_, mask)
                    restored = torch.clamp(restored,0,1)
                    psnr_train_rgb.append(utils.batch_PSNR(restored, target, False).item())
                psnr_train_rgb = sum(psnr_train_rgb)/len(train_dataset)
                writer.add_scalars("PSNR", {'train': psnr_train_rgb}, epoch)
                utils.append_jsonl(metrics_path, dict(
                    timestamp = utils.now_iso(),
                    phase     = 'train',
                    epoch     = epoch,
                    iter      = i,
                    psnr      = round(psnr_train_rgb, 4),
                    lr        = scheduler.get_lr()[0],
                    **RUN_META,
                ))
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_psnr': best_psnr,
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
writer.close()

# 训练结束: 写一份汇总，便于快速查看本次 run 的最优结果与产物
utils.write_json(os.path.join(log_dir, 'summary.json'), dict(
    finished_at   = utils.now_iso(),
    best_epoch    = best_epoch,
    best_iter     = best_iter,
    best_psnr     = round(best_psnr, 4),
    final_epoch   = opt.nepoch,
    archived_best = sorted(f for f in os.listdir(model_dir)
                           if '_ep' in f and '_valpsnr' in f),
    **RUN_META,
))
print("==> Run dir   : ", log_dir)
print("==> Best PSNR : {:.4f} @ epoch {} (iter {})".format(best_psnr, best_epoch, best_iter))



