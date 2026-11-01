# -*- coding: utf-8 -*-
"""RASMv2 训练/测试链路检查的公共工具。

设计约束:
  * 本模块 import 时只依赖标准库（torch / cv2 / numpy 等一律在函数内部惰性导入），
    因此 CPU 上的静态检查 (check_static.py) 与 GPU 上的单元测试可以共用它；
  * 所有路径都以本文件位置推算，不依赖当前工作目录；
  * 只做"读"，不改动 RASM/ 下任何既有文件。

目录关系:
    improved/check/_rasmv2_common.py
    improved/                     <- IMPROVED_DIR
    RASM/                         <- RASM_ROOT（import model / utils / losses 的根）
"""

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # .../RASM/improved/check
IMPROVED_DIR = os.path.dirname(HERE)                    # .../RASM/improved
RASM_ROOT = os.path.dirname(IMPROVED_DIR)               # .../RASM

for _p in (HERE, RASM_ROOT, IMPROVED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATASET_DIRS = {
    'ISTD': (os.path.join(RASM_ROOT, 'datasets', 'ISTD_Dataset', 'train'),
             os.path.join(RASM_ROOT, 'datasets', 'ISTD_Dataset', 'test')),
    'SRD': (os.path.join(RASM_ROOT, 'datasets', 'SRD_ABC', 'train'),
            os.path.join(RASM_ROOT, 'datasets', 'SRD_ABC', 'test')),
}
OFFICIAL_CKPT = {
    'ISTD': os.path.join(RASM_ROOT, 'log', 'RASM_istd_demo', 'models', 'AISTD.pth'),
    'SRD': os.path.join(RASM_ROOT, 'log', 'RASM_srd_demo', 'models', 'SRD.pth'),
}
# losses.py 里 Vgg19 的默认本地权重路径（离线必需）
VGG_WEIGHTS = '/workspace/vgg/vgg19-dcbb9e9d.pth'

PRESETS = ('rasm', 'mgra', 'nsrc', 'film', 'rgate', 'full')
# 训练脚本与测试脚本必须完全一致的结构参数，任一不一致 => 权重加载静默降级
STRUCT_KEYS = ('PRESET', 'KERNEL_SIZE', 'DILATIONS', 'EMBED_DIM', 'WIN_SIZE',
               'TOKEN_PROJECTION', 'TOKEN_MLP')
# RASMv2 相对官方 RASM 新增的参数名（用于判定"缺失键"是否都是新增项）
# 注意 milm: CATransformerBlockV2 里的 Mask-FiLM 属性名是 milm，不是 film
V2_NEW_KEY_TOKENS = ('rpb', 'illu', 'gate_raw', 'nsrc', 'film', 'milm', 'res_gate', 'bias_gen')


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def build_model(preset='full', img_size=128, embed_dim=32, win_size=10,
                kernel_size=11, dilations=(2, 2, 2), **kwargs):
    """构造 RASMv2。img_size 必须是 8 的倍数（U-Net 有 3 次 1/2 下采样）。"""
    from improved.rasmv2_model import RASMv2
    return RASMv2(img_size=img_size, embed_dim=embed_dim, win_size=win_size,
                  token_projection='linear', token_mlp='leff', preset=preset,
                  kernel_size=kernel_size, dilations=tuple(dilations), **kwargs)


def block_mask(batch, size, device=None, lo=0.25, hi=0.75):
    """中心块状阴影：保证下采样后每个尺度都同时存在阴影/非阴影 token。

    随机 mask 会让 NS-Gate 的 fallback / NSRC 的 res=0 天然没有梯度，
    那是数据问题不是代码问题，会把"梯度通路"检查变成假阴性。
    """
    import torch
    m = torch.zeros(batch, 1, size, size, device=device)
    m[:, :, int(size * lo):int(size * hi), int(size * lo):int(size * hi)] = 1.0
    return m


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(root, n_train=8, n_test=4, size=128, seed=0,
                           shadow_lo=0.25, shadow_hi=0.75, shadow_scale=0.4):
    """生成 <root>/train/{train_A,train_B,train_C} + <root>/test/{test_A,test_B,test_C}。

    A = 输入（阴影区整体变暗）  B = 二值 mask(0/255)  C = GT。
    A/B/C 同名文件 => dataset.py 的 _aligned_filenames 能按名字对齐。
    目录层级与真实数据集一致，故 --train_dir <root>/train、--val_dir <root>/test 可直接用。
    """
    import cv2
    import numpy as np

    root = str(root)
    rng = np.random.RandomState(seed)
    for split, n in (('train', n_train), ('test', n_test)):
        sub_dirs = {}
        for sub in ('A', 'B', 'C'):
            d = os.path.join(root, split, '%s_%s' % (split, sub))
            os.makedirs(d, exist_ok=True)
            sub_dirs[sub] = d
        for i in range(n):
            gt = rng.rand(size, size, 3).astype(np.float32) * 0.5 + 0.25
            mask = np.zeros((size, size), np.uint8)
            a, b = int(size * shadow_lo), int(size * shadow_hi)
            mask[a:b, a:b] = 255
            inp = gt.copy()
            inp[mask > 0] *= shadow_scale
            name = 'img_%03d.png' % i
            for sub, arr in (('A', inp), ('C', gt)):
                bgr = cv2.cvtColor((np.clip(arr, 0, 1) * 255).astype(np.uint8),
                                   cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(sub_dirs[sub], name), bgr)
            cv2.imwrite(os.path.join(sub_dirs['B'], name), mask)
    return root


def list_images(d):
    exts = ('.png', '.jpg', '.jpeg')
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if os.path.splitext(f)[1].lower() in exts)


# --------------------------------------------------------------------------- #
# 子进程 / 落盘产物
# --------------------------------------------------------------------------- #
def run_py(args, timeout=3600, env=None, cwd=RASM_ROOT):
    """在 RASM/ 目录下运行 python，返回 CompletedProcess（stdout/stderr 已解码）。"""
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([sys.executable] + list(args), cwd=cwd, env=e,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=timeout)


def run_py_stream(args, timeout=3600, env=None, log_path=None, prefix='    | ', cwd=RASM_ROOT):
    """同 run_py，但把子进程输出实时回显到控制台（可选同步落盘）。

    端到端训练耗时较长，用它才能在 pytest 里看到 train_v2.py 的心跳日志，
    而不是等进程结束才一次性拿到输出。
    """
    e = dict(os.environ)
    if env:
        e.update(env)
    proc = subprocess.Popen([sys.executable] + list(args), cwd=cwd, env=e,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    fh = open(log_path, 'a', encoding='utf-8') if log_path else None
    try:
        for line in proc.stdout:
            lines.append(line)
            print(prefix + line.rstrip('\n'), flush=True)
            if fh:
                fh.write(line)
        proc.wait(timeout=timeout)
    finally:
        if fh:
            fh.close()
    return subprocess.CompletedProcess(list(args), proc.returncode, ''.join(lines), None)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fresh_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# 指标参考实现（独立于 test_v2.py，用于交叉验证其指标是否算对）
# --------------------------------------------------------------------------- #
def ref_metrics(pred01, gt01, mask01):
    """pred01/gt01: (H,W,3) float 0~1, RGB；mask01: (H,W,1) 或 (H,W)，1=阴影。

    复刻论文/官方 test.py 的定义: PSNR(全图/阴影区/非阴影区) + LAB 空间 RMSE。
    """
    import cv2
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr_loss

    m = mask01.reshape(*mask01.shape[:2], 1) if mask01.ndim == 3 else mask01[..., None]
    m = (m > 0).astype(np.float64)
    psnr_all = psnr_loss(gt01, pred01)
    psnr_s = psnr_loss(gt01 * m, pred01 * m)
    psnr_ns = psnr_loss(gt01 * (1 - m), pred01 * (1 - m))
    lab = lambda x: cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    rmse_all = np.abs(lab(pred01) - lab(gt01)).mean() * 3
    rmse_s = np.abs(lab(pred01 * m) - lab(gt01 * m)).sum() / max(m.sum(), 1e-6)
    rmse_ns = np.abs(lab(pred01 * (1 - m)) - lab(gt01 * (1 - m))).sum() / max((1 - m).sum(), 1e-6)
    return dict(psnr=psnr_all, psnr_s=psnr_s, psnr_ns=psnr_ns,
                rmse=rmse_all, rmse_s=rmse_s, rmse_ns=rmse_ns)


def load_rgb_uint8(path):
    """读回 test_v2.py 用 utils.save_img 存下的图（RGB, 0~1）。"""
    import cv2
    import numpy as np
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# train_v2.py / test_v2.py 的命令行拼装（端到端检查用）
# --------------------------------------------------------------------------- #
def train_v2_cmd(train_dir, val_dir, run_name, nepoch, preset='full', gpu='0',
                 train_ps=128, batch_size=2, env='_gpucheck', dataset='ISTD',
                 kernel_size=11, dilations=(2, 2, 2), embed_dim=32, win_size=10,
                 log_interval=30.0):
    cmd = (['improved/train_v2.py',
            '--arch', 'RASMv2', '--preset', preset,
            '--kernel_size', str(kernel_size), '--dilations']
           + [str(d) for d in dilations]
           + ['--w_boundary', '1.0', '--w_ns', '0.5', '--w_chroma', '0.2', '--warmup',
              '--train_ps', str(train_ps), '--batch_size', str(batch_size),
              '--nepoch', str(nepoch), '--gpu', str(gpu), '--env', env,
              '--train_dir', train_dir, '--val_dir', val_dir,
              '--lr_initial', '0.0004', '--warmup_epochs', '3',
              '--optimizer', 'adamw', '--weight_decay', '0.02',
              '--train_workers', '2', '--eval_workers', '2',
              '--embed_dim', str(embed_dim), '--win_size', str(win_size),
              '--token_projection', 'linear', '--token_mlp', 'leff',
              '--dataset', dataset, '--run_name', run_name,
              '--log_interval', str(log_interval)])
    return cmd


def env_cfg():
    """端到端检查的可调参数（由 run_gpu_checks.sh 通过环境变量注入）。"""
    return dict(
        gpu=os.environ.get('GPU_DEV', '0'),
        nepoch=int(os.environ.get('CHECK_NEPOCH', '21')),
        train_ps=int(os.environ.get('CHECK_TRAIN_PS', '128')),
        batch=int(os.environ.get('CHECK_BATCH', '2')),
        env=os.environ.get('CHECK_ENV', '_gpucheck'),
        log_interval=float(os.environ.get('CHECK_LOG_INTERVAL', '30')),
    )


def test_v2_cmd(input_dir, weights, result_dir, ex_name, preset='full', gpu='0',
                train_ps=128, tile=None, tile_overlap=32, cal_metrics=True,
                save_images=True, kernel_size=11, dilations=(2, 2, 2),
                embed_dim=32, win_size=10):
    cmd = (['improved/test_v2.py',
            '--arch', 'RASMv2', '--dataset', 'ISTD', '--preset', preset,
            '--kernel_size', str(kernel_size), '--dilations']
           + [str(d) for d in dilations]
           + ['--train_ps', str(train_ps),
              '--input_dir', input_dir, '--weights', weights,
              '--result_dir', result_dir, '--ex_name', ex_name,
              '--embed_dim', str(embed_dim), '--win_size', str(win_size),
              '--token_projection', 'linear', '--token_mlp', 'leff', '--gpus', str(gpu)])
    if cal_metrics:
        cmd.append('--cal_metrics')
    if save_images:
        cmd.append('--save_images')
    if tile:
        cmd += ['--tile', str(tile), '--tile_overlap', str(tile_overlap)]
    return cmd
