# -*- coding: utf-8 -*-
"""RASMv2 **训练链路** 的 GPU 单元测试。

针对的是"只有真正跑起来才会暴露"的那类 bug：
  * AMP(autocast) + NativeScaler 下前向/反向是否 NaN；
  * v2 新增参数（RPB / RIB / NS-Gate / NSRC / Mask-FiLM / 残差门控）梯度是否真的通；
  * 新模块的恒等初始化是否被破坏（破坏了消融实验就没有基线了）；
  * 优化器能否把 loss 降下去（梯度断裂 / 学习率链路坏掉会立刻现形）；
  * ckpt 存→取是否严格一致（训练产物能否被测试端无损加载）；
  * train_v2.py 端到端能否跑完、产物是否齐全。

运行（RASM/ 目录下，GPU 环境）:
    GPU_DEV=0 python -m pytest improved/check/test_train_v2_gpu.py -v
"""

import json
import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pytest
import torch
import torch.nn as nn

from _rasmv2_common import (DATASET_DIRS, OFFICIAL_CKPT, PRESETS, V2_NEW_KEY_TOKENS,
                            block_mask, build_model, read_jsonl)

require_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason='需要 CUDA：RASMv2 训练全程 .cuda() + AMP')


# --------------------------------------------------------------------------- #
# 1. 每个 preset：构建 / AMP 前向 / NativeScaler 反向
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('preset', PRESETS)
@require_cuda
def test_amp_forward_backward(preset):
    torch.manual_seed(0)
    dev = torch.device('cuda:0')
    from timm.utils import NativeScaler

    model = build_model(preset=preset, img_size=64).to(dev).train()
    x = torch.rand(2, 3, 64, 64, device=dev)
    m = block_mask(2, 64, device=dev)
    tgt = torch.rand(2, 3, 64, 64, device=dev)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = NativeScaler()                       # train_v2.py 用的就是它
    with torch.cuda.amp.autocast():
        out = torch.clamp(model(x, m), 0, 1)
    assert tuple(out.shape) == (2, 3, 64, 64), '输出形状不对: %s' % (tuple(out.shape),)
    loss = nn.L1Loss()(out, tgt)
    scaler(loss, opt, parameters=model.parameters())

    assert torch.isfinite(loss).item(), 'loss 出现 NaN/Inf'
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, '没有任何参数拿到梯度'
    for g in grads:
        assert torch.isfinite(g).all(), '梯度出现 NaN/Inf'


# --------------------------------------------------------------------------- #
# 2. v2 新增参数的梯度通路
# --------------------------------------------------------------------------- #
@require_cuda
def test_new_params_have_gradients():
    dev = torch.device('cuda:0')
    torch.manual_seed(0)
    model = build_model(preset='full', img_size=64).to(dev).train()
    x = torch.rand(2, 3, 64, 64, device=dev)
    m = block_mask(2, 64, device=dev)             # 中心块状阴影：各尺度都同时有两类 token
    tgt = torch.rand(2, 3, 64, 64, device=dev)
    out = torch.clamp(model(x, m), 0, 1)
    nn.L1Loss()(out, tgt).backward()

    targets = {
        'RPB 相对位置偏置': 'attn.rpb.2.bias',
        'RIB 光照关系偏置': 'bias_gen.illu',
        'NS-Gate 强度': 'bias_gen.gate_raw',
        'NSRC 投影': 'nsrc.proj.weight',
        'Mask-FiLM 增益': 'gain',                 # 同时覆盖 MGRALayer.film 与 CATransformerBlockV2.milm
        '残差门控 beta': 'res_gate_raw',
    }
    dead = []
    for name, key in targets.items():
        hit = [n for n, p in model.named_parameters()
               if n.endswith(key) and p.grad is not None and p.grad.abs().sum() > 0]
        if not hit:
            dead.append(name)
    assert not dead, '以下新增参数没有梯度（代码 bug 或数据退化）: %s' % dead


# --------------------------------------------------------------------------- #
# 3. 新模块的初始化必须是"恒等"——否则消融基线不成立
# --------------------------------------------------------------------------- #
@require_cuda
def test_new_modules_init_identity():
    import torch.nn.functional as F

    dev = torch.device('cuda:0')
    model = build_model(preset='full', img_size=64).to(dev).eval()
    layer = model.encoderlayer_2.blocks[0]        # 区域注意力层（含 NSRC / FiLM / bias_gen）

    # NSRC: proj 零初始化 -> 模块输出恒为 0
    x = torch.randn(1, layer.dim, 16, 16, device=dev)
    m = block_mask(1, 16, device=dev)
    nsrc_out = layer.nsrc(x, m)
    assert torch.allclose(nsrc_out, torch.zeros_like(nsrc_out), atol=1e-6), \
        'NSRC 初始不是恒等（最大偏差 %.2e）' % nsrc_out.abs().max().item()

    # Mask-FiLM: gain=1 / bias=0 -> 恒等
    xl = torch.randn(1, 4, layer.dim, device=dev)
    ml = (torch.rand(1, 4, 1, device=dev) > 0.5).float()
    assert torch.allclose(layer.film(xl, ml), xl, atol=1e-6), 'Mask-FiLM 初始不是恒等'

    # NS-Gate 强度初值 0.5 / 残差门控 beta 初值 0.1 / RIB 初值 0
    gate = F.softplus(layer.bias_gen.gate_raw).mean().item()
    beta = F.softplus(model.res_gate_raw).item()
    illu = layer.bias_gen.illu.abs().max().item()
    assert abs(gate - 0.5) < 1e-3, 'NS-Gate 初值 %.4f != 0.5' % gate
    assert abs(beta - 0.1) < 1e-3, '残差门控 beta 初值 %.4f != 0.1' % beta
    assert illu == 0.0, 'RIB 初值应为 0，实际 %.4f' % illu


# --------------------------------------------------------------------------- #
# 4. 优化链路：必须能把 loss 降下去
# --------------------------------------------------------------------------- #
@require_cuda
def test_overfit_small_batch():
    torch.manual_seed(0)
    dev = torch.device('cuda:0')
    model = build_model(preset='full', img_size=64).to(dev).train()
    x = torch.rand(2, 3, 64, 64, device=dev)
    m = block_mask(2, 64, device=dev)
    tgt = torch.rand(2, 3, 64, 64, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    first = last = None
    for i in range(150):
        opt.zero_grad(set_to_none=True)
        out = torch.clamp(model(x, m), 0, 1)
        loss = nn.L1Loss()(out, tgt)
        loss.backward()
        opt.step()
        last = loss.item()
        if i == 0:
            first = last
    print('过拟合冒烟: loss %.4f -> %.4f' % (first, last))
    assert math.isfinite(last), '训练发散（NaN/Inf）'
    assert last < first * 0.75, '150 步后 loss 没有明显下降: %.4f -> %.4f' % (first, last)


# --------------------------------------------------------------------------- #
# 5. ckpt 存->取 严格一致（训练产物必须能被测试端无损加载）
# --------------------------------------------------------------------------- #
@require_cuda
def test_ckpt_roundtrip_strict(tmp_path):
    from improved.rasmv2_model import load_weights

    dev = torch.device('cuda:0')
    torch.manual_seed(0)
    model = build_model(preset='full', img_size=64).to(dev).eval()
    x = torch.rand(1, 3, 64, 64, device=dev)
    m = block_mask(1, 64, device=dev)
    with torch.no_grad():
        y0 = model(x, m)

    path = str(tmp_path / 'ckpt.pth')
    torch.save({'epoch': 3, 'state_dict': model.state_dict(),
                'optimizer': {}, 'best_psnr': 12.3}, path)

    model2 = build_model(preset='full', img_size=64).to(dev).eval()
    out = load_weights(model2, path, strict=True)
    assert not out.missing_keys, '缺失键 %s' % out.missing_keys[:5]
    assert not out.unexpected_keys, '冗余键 %s' % out.unexpected_keys[:5]
    with torch.no_grad():
        y1 = model2(x, m)
    assert torch.allclose(y0, y1, atol=1e-6), \
        '重载后输出不一致（最大偏差 %.2e）' % (y0 - y1).abs().max().item()


# --------------------------------------------------------------------------- #
# 6. 官方 RASM 权重能否吃下（0 项结构冲突）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ds', ['ISTD', 'SRD'])
@require_cuda
def test_official_ckpt_structural_compat(ds):
    from improved.rasmv2_model import load_weights

    ck = OFFICIAL_CKPT[ds]
    if not os.path.isfile(ck):
        pytest.skip('未找到官方权重 %s' % ck)
    model = build_model(preset='rasm', img_size=320).cuda()
    out = load_weights(model, ck, strict=False)
    new_only = all(any(t in k for t in V2_NEW_KEY_TOKENS) for k in out.missing_keys)
    assert len(out.unexpected_keys) == 0, '出现结构冲突: %s' % out.unexpected_keys[:5]
    assert new_only, '缺失项里出现非 v2 新增参数: %s' % [
        k for k in out.missing_keys if not any(t in k for t in V2_NEW_KEY_TOKENS)][:5]


# --------------------------------------------------------------------------- #
# 7. preset 不一致必须"可被检测"——训练用 full、测试用 rasm 会静默丢参数
# --------------------------------------------------------------------------- #
@require_cuda
def test_preset_mismatch_is_detectable(tmp_path):
    from improved.rasmv2_model import load_weights

    # 方向一: full 训练的权重 -> rasm 模型。full 是 rasm 的超集，因此表现为"多余键":
    # (v2 独有参数被静默丢弃)，而不是缺失键。
    src = build_model(preset='full', img_size=64).cuda()
    path = str(tmp_path / 'full.pth')
    torch.save({'epoch': 1, 'state_dict': src.state_dict()}, path)

    dst = build_model(preset='rasm', img_size=64).cuda()
    out_full2rasm = load_weights(dst, path, strict=False)
    assert len(out_full2rasm.unexpected_keys) > 0, \
        'full->rasm: v2 独有参数没被报告为多余键，preset 不一致将无法察觉'
    assert not out_full2rasm.missing_keys, \
        'full->rasm 不应有缺失键（full 的键是 rasm 的超集）'

    # 方向二: rasm 权重 -> full 模型。v2 新增参数无从加载，表现为"缺失键"。
    src2 = build_model(preset='rasm', img_size=64).cuda()
    path2 = str(tmp_path / 'rasm.pth')
    torch.save({'epoch': 1, 'state_dict': src2.state_dict()}, path2)

    dst2 = build_model(preset='full', img_size=64).cuda()
    out_rasm2full = load_weights(dst2, path2, strict=False)
    assert len(out_rasm2full.missing_keys) > 0, \
        'rasm->full: v2 新增参数没被报告为缺失键，检查逻辑失效'

    warnings.warn('确认过的风险: 训练/测试 preset 不一致时 load_weights(strict=False) 仍会静默通过'
                  '（full->rasm 丢 %d 个参数, rasm->full 缺 %d 个参数），必须人工保证 preset 一致'
                  % (len(out_full2rasm.unexpected_keys), len(out_rasm2full.missing_keys)),
                  RuntimeWarning)


# --------------------------------------------------------------------------- #
# 8. 训练期数据增强与 DataLoader
# --------------------------------------------------------------------------- #
@require_cuda
def test_mixup_aug_on_cuda():
    from utils import MixUp_AUG

    gt = torch.rand(4, 3, 64, 64, device='cuda')
    inp = torch.rand(4, 3, 64, 64, device='cuda')
    m = torch.ones(4, 1, 64, 64, device='cuda')
    g2, i2, m2 = MixUp_AUG().aug(gt, inp, m)
    for t in (g2, i2, m2):
        assert t.shape[0] == 4 and tuple(t.shape[2:]) == (64, 64), 'MixUp 输出形状异常: %s' % (
            tuple(t.shape),)
        assert torch.isfinite(t).all()
    assert float(m2.min()) >= 0.0 and float(m2.max()) <= 1.0


@require_cuda
def test_train_dataloader_batch():
    from utils.loader import get_training_data

    train_dir = DATASET_DIRS['ISTD'][0]
    if not os.path.isdir(train_dir):
        pytest.skip('未找到训练集 %s' % train_dir)
    ds = get_training_data(train_dir, {'patch_size': 128})
    item = ds[0]
    assert len(item) >= 3, 'DataLoaderTrain 应至少返回 (clean, noisy, mask)'
    clean, noisy, mask = item[0], item[1], item[2]
    assert tuple(clean.shape) == (3, 128, 128), str(tuple(clean.shape))
    assert tuple(mask.shape) == (1, 128, 128), str(tuple(mask.shape))
    for name, t in (('clean', clean), ('noisy', noisy)):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0, '%s 超出 [0,1]' % name
    uniq = set(torch.unique(mask).tolist())
    assert uniq <= {0.0, 1.0}, 'mask 不是二值: %s' % sorted(uniq)


# --------------------------------------------------------------------------- #
# 9. 真实损失（L1 + VGG 感知 + 三项任务损失）在 GPU 上可反传
# --------------------------------------------------------------------------- #
@require_cuda
def test_real_loss_finite_and_backward():
    from improved.rasmv2_losses import RASMv2Loss

    crit = RASMv2Loss().cuda()                    # 默认 base = L1 + VGG(需要 /workspace/vgg 权重)
    out = torch.rand(2, 3, 128, 128, device='cuda', requires_grad=True)
    tgt = torch.rand(2, 3, 128, 128, device='cuda')
    m = block_mask(2, 128, device='cuda')
    loss = crit(out, tgt, m)
    assert torch.isfinite(loss).item(), '总损失 NaN/Inf（VGG 分支多半是权重没加载）'
    loss.backward()
    assert out.grad is not None and torch.isfinite(out.grad).all()


# --------------------------------------------------------------------------- #
# 10. lr 调度链路（warmup + cosine）
# --------------------------------------------------------------------------- #
@require_cuda
def test_warmup_cosine_scheduler():
    from warmup_scheduler import GradualWarmupScheduler

    model = build_model(preset='rasm', img_size=64).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    nepoch, warm = 20, 3
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nepoch - warm, eta_min=1e-6)
    sch = GradualWarmupScheduler(opt, multiplier=1, total_epoch=warm, after_scheduler=cosine)
    sch.step()
    lrs = []
    for _ in range(nepoch):
        lrs.append(sch.get_lr()[0])               # train_v2.py 每 epoch 都会调它写日志
        sch.step()
    assert all(math.isfinite(l) for l in lrs), '学习率出现 NaN/Inf: %s' % lrs
    assert max(lrs) <= 1e-4 + 1e-12, 'warmup 阶段学习率超过设定值: %s' % lrs
    assert lrs[-1] < lrs[0], '余弦退火没有生效: %s' % lrs


# --------------------------------------------------------------------------- #
# 11. 端到端：train_v2.py 真的能跑完并产出完整产物
# --------------------------------------------------------------------------- #
@require_cuda
def test_e2e_training_artifacts(trained_run):
    from improved.rasmv2_model import load_weights

    log_dir, models = trained_run['log_dir'], trained_run['models']
    latest = os.path.join(models, 'model_latest.pth')
    best = os.path.join(models, 'model_best.pth')
    cfg_path = os.path.join(log_dir, 'config.json')
    metrics_path = os.path.join(log_dir, 'metrics.jsonl')

    assert os.path.isfile(cfg_path), '缺少 config.json'
    assert os.path.isfile(latest), '缺少 model_latest.pth'
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    assert cfg.get('preset') == trained_run['preset'], 'config.json 里 preset 与实际不符'

    records = read_jsonl(metrics_path)
    val_records = [r for r in records if r.get('phase') == 'val']
    assert val_records, ('train_v2.py 全程没有触发验证 —— eval_now = 20*len(train_loader)，'
                         '数据集太小或 epoch 太少时一次都不会评估')
    assert os.path.isfile(best), '触发过验证却没写出 model_best.pth'

    # 产物必须能被"结构完全一致"的模型严格加载
    model = build_model(preset=trained_run['preset'], img_size=trained_run['train_ps']).cuda()
    out = load_weights(model, best, strict=True)
    assert not out.missing_keys and not out.unexpected_keys, \
        '训练产物无法被同 preset 模型严格加载: 缺失 %d / 冗余 %d' % (
            len(out.missing_keys), len(out.unexpected_keys))


@require_cuda
def test_one_epoch_run_known_limitation(one_epoch_run):
    """短训练（NEPOCH < 21）不会产出 model_best.pth —— 这是 train_v2.py 的已知限制。

    improved_try.sh 第 [5] 步直接拿 model_best.pth 去测试，冒烟时(NEPOCH=20)会 FileNotFound。
    """
    latest = os.path.join(one_epoch_run['models'], 'model_latest.pth')
    best = os.path.join(one_epoch_run['models'], 'model_best.pth')
    assert os.path.isfile(latest), '单 epoch 训练应至少产出 model_latest.pth'
    if not os.path.isfile(best):
        warnings.warn('已知限制确认: 只跑 1 个 epoch 时没有 model_best.pth'
                      '（eval_now = 20*len(train_loader)，评估点还没到）。'
                      '冒烟测试请用 model_latest.pth 兜底，或把 NEPOCH 提到 21 以上。',
                      RuntimeWarning)
