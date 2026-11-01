# -*- coding: utf-8 -*-
"""RASMv2 **测试链路** 的 GPU 单元测试。

针对的是 test_v2.py 这一侧容易静默出错的地方：
  * 验证集 DataLoader 的三元组(GT/输入/mask)是否对齐、取值范围是否 [0,1]；
  * 训练分辨率(320)下构建的模型能否直接在原分辨率(如 ISTD 640x480)上推理；
  * test_v2.py 端到端跑通的产物：metrics_v2.jsonl 字段、存图数量与尺寸；
  * 指标是否算对：用独立实现从"存下来的图"重算一遍做交叉验证；
  * 阴影/非阴影分区指标是否真的分开了（mask 用错会表现为两者完全相等）；
  * --tile 分块推理与整图推理是否一致（分块索引写错会表现为明显偏移）；
  * 官方权重 + preset=rasm 能否正常推理（基线对照前置条件）。

运行（RASM/ 目录下，GPU 环境）:
    GPU_DEV=0 python -m pytest improved/check/test_test_v2_gpu.py -v
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pytest
import torch

# 注意: 不能把 test_v2_cmd 直接导入成同名名字，pytest 会把它当成测试用例收集
from _rasmv2_common import (DATASET_DIRS, OFFICIAL_CKPT, build_model, env_cfg, list_images,
                            load_rgb_uint8, ref_metrics, run_py, test_v2_cmd as make_test_cmd)

require_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason='需要 CUDA：test_v2.py 全程在 GPU 上推理')
CFG = env_cfg()
# 与 test_v2.py 对齐的量级阈值（存图经过 uint8 量化，故留 0.5dB / 0.5 的容差）
PSNR_TOL_DB = 0.5
RMSE_TOL = 0.5


# --------------------------------------------------------------------------- #
# 1. 验证集数据
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('ds', ['ISTD', 'SRD'])
@require_cuda
def test_val_dataset_shapes(ds):
    from utils.loader import get_validation_data

    d = DATASET_DIRS[ds][1]
    if not os.path.isdir(d):
        pytest.skip('未找到验证集 %s' % d)
    dataset = get_validation_data(d)
    item = dataset[0]
    assert len(item) == 6, 'DataLoaderVal 应返回 (clean, noisy, mask, fn, fn, fn)，实际 %d 项' % len(item)
    clean, noisy, mask = item[0], item[1], item[2]
    assert clean.dim() == 3 and clean.shape[0] == 3, 'GT 形状异常: %s' % (tuple(clean.shape),)
    assert tuple(noisy.shape) == tuple(clean.shape), '输入与 GT 形状不一致'
    assert mask.shape[0] == 1 and tuple(mask.shape[1:]) == tuple(clean.shape[1:]), \
        'mask 与图像不对齐: %s' % (tuple(mask.shape),)
    for name, t in (('clean', clean), ('noisy', noisy), ('mask', mask)):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0, '%s 超出 [0,1]' % name


# --------------------------------------------------------------------------- #
# 2. 训练分辨率下构建的模型，能否直接在原分辨率上推理
# --------------------------------------------------------------------------- #
@require_cuda
def test_forward_full_resolution():
    from utils.loader import get_validation_data

    d = DATASET_DIRS['ISTD'][1]
    if not os.path.isdir(d):
        pytest.skip('未找到验证集 %s' % d)
    noisy, mask = get_validation_data(d)[0][1:3]
    noisy = noisy.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()

    torch.manual_seed(0)
    model = build_model(preset='full', img_size=320).cuda().eval()
    with torch.no_grad():
        out = model(noisy, mask)
    assert tuple(out.shape) == tuple(noisy.shape), '输出 %s != 输入 %s' % (
        tuple(out.shape), tuple(noisy.shape))
    assert torch.isfinite(out).all(), '原分辨率推理出现 NaN/Inf'


# --------------------------------------------------------------------------- #
# 3. test_v2.py 端到端：指标落盘
# --------------------------------------------------------------------------- #
@require_cuda
def test_e2e_metrics_record(e2e_test_result):
    rec = e2e_test_result['record']
    n_img = len(list_images(os.path.join(e2e_test_result['dataset'], 'test', 'test_C')))
    assert rec['num_images'] == n_img, 'num_images=%s 与数据集 %d 不符' % (rec['num_images'], n_img)
    assert len(rec['per_image']) == n_img, 'per_image 少了图: %d' % len(rec['per_image'])
    summary = rec['summary']
    for k in ('psnr', 'rmse', 'psnr_s', 'rmse_s', 'psnr_ns', 'rmse_ns'):
        assert k in summary, 'summary 缺少字段 %s' % k
        assert math.isfinite(float(summary[k])), '指标 %s = %s（NaN/Inf 说明 mask 退化或算错）' % (
            k, summary[k])
    assert 5.0 < float(summary['psnr']) < 60.0, 'PSNR=%s 不在合理区间' % summary['psnr']
    assert 0.0 <= float(summary['rmse']) < 100.0, 'RMSE=%s 不在合理区间' % summary['rmse']


# --------------------------------------------------------------------------- #
# 4. 存图：数量与尺寸
# --------------------------------------------------------------------------- #
@require_cuda
def test_e2e_saved_images(e2e_test_result):
    import cv2

    gt_dir = os.path.join(e2e_test_result['dataset'], 'test', 'test_C')
    result_dir = e2e_test_result['result_dir']
    gts = list_images(gt_dir)
    saved = sorted(f for f in os.listdir(result_dir) if f.lower().endswith('.png'))
    assert len(saved) == len(gts), '存图 %d 张 != 数据集 %d 张' % (len(saved), len(gts))
    for name in saved:
        gt = cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_COLOR)
        out = cv2.imread(os.path.join(result_dir, name), cv2.IMREAD_COLOR)
        assert out is not None, '结果图打不开: %s' % name
        assert out.shape == gt.shape, '%s 尺寸 %s != GT %s' % (name, out.shape, gt.shape)


# --------------------------------------------------------------------------- #
# 5. 指标交叉验证：用独立实现从存图重算一遍
# --------------------------------------------------------------------------- #
@require_cuda
def test_metrics_match_reference(e2e_test_result):
    import cv2
    import numpy as np

    ds = e2e_test_result['dataset']
    result_dir = e2e_test_result['result_dir']
    gt_dir = os.path.join(ds, 'test', 'test_C')
    mask_dir = os.path.join(ds, 'test', 'test_B')

    for rec in e2e_test_result['record']['per_image']:
        name = rec['filename']
        pred = load_rgb_uint8(os.path.join(result_dir, name))
        gt = cv2.cvtColor(cv2.imread(os.path.join(gt_dir, name), cv2.IMREAD_COLOR),
                          cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        msk = cv2.imread(os.path.join(mask_dir, name), 0).astype(np.float32) / 255.0
        ref = ref_metrics(pred, gt, msk)
        assert abs(ref['psnr'] - float(rec['psnr'])) < PSNR_TOL_DB, \
            '%s PSNR 不一致: 参考 %.4f vs 脚本 %.4f' % (name, ref['psnr'], float(rec['psnr']))
        assert abs(ref['rmse'] - float(rec['rmse'])) < RMSE_TOL, \
            '%s RMSE 不一致: 参考 %.4f vs 脚本 %.4f' % (name, ref['rmse'], float(rec['rmse']))


# --------------------------------------------------------------------------- #
# 6. 阴影/非阴影分区指标必须真的分开（mask 用错 => 两者会完全相等）
# --------------------------------------------------------------------------- #
@require_cuda
def test_shadow_split_is_meaningful(e2e_test_result):
    for rec in e2e_test_result['record']['per_image']:
        assert abs(float(rec['psnr_s']) - float(rec['psnr_ns'])) > 1e-3, \
            '%s 的 psnr_s 与 psnr_ns 完全相同，mask 分区可能没生效' % rec['filename']
        assert abs(float(rec['rmse_s']) - float(rec['rmse_ns'])) > 1e-3, \
            '%s 的 rmse_s 与 rmse_ns 完全相同，mask 分区可能没生效' % rec['filename']


# --------------------------------------------------------------------------- #
# 7. --tile 分块推理 vs 整图推理
# --------------------------------------------------------------------------- #
@require_cuda
def test_tile_vs_full_consistent(trained_run, e2e_test_result, tmp_path_factory):
    import numpy as np

    res_tile = str(tmp_path_factory.mktemp('rasmv2_res_tile'))
    size = trained_run['train_ps']
    proc = run_py(make_test_cmd(os.path.join(trained_run['dataset'], 'test'),
                                e2e_test_result['weights'], res_tile, ex_name='check_tile',
                                preset=e2e_test_result['preset'], gpu=CFG['gpu'],
                                train_ps=size, tile=max(8, size // 2), tile_overlap=32),
                  timeout=1800)
    assert proc.returncode == 0, 'tile 推理失败（尾部输出）:\n' + proc.stdout[-4000:]

    diffs = []
    for name in sorted(f for f in os.listdir(res_tile) if f.lower().endswith('.png')):
        a = load_rgb_uint8(os.path.join(e2e_test_result['result_dir'], name))
        b = load_rgb_uint8(os.path.join(res_tile, name))
        assert a.shape == b.shape, '%s 尺寸不一致' % name
        diffs.append(float(np.abs(a - b).mean()))
    assert diffs, '没有可比对的存图'
    worst = max(diffs)
    print('tile vs 整图: 平均 MAE %.5f, 最差 %.5f' % (float(np.mean(diffs)), worst))
    assert worst < 0.05, '分块推理与整图推理差异过大（最差 MAE %.4f），分块索引/E,W 累加可能写错' % worst


# --------------------------------------------------------------------------- #
# 8. 官方权重 + preset=rasm 的基线推理（不训练，只确认能跑且输出不崩）
# --------------------------------------------------------------------------- #
@require_cuda
def test_official_weights_rasm_forward():
    from improved.rasmv2_model import load_weights
    from utils.loader import get_validation_data

    ck = OFFICIAL_CKPT['ISTD']
    if not os.path.isfile(ck):
        pytest.skip('未找到官方权重 %s' % ck)
    d = DATASET_DIRS['ISTD'][1]
    if not os.path.isdir(d):
        pytest.skip('未找到验证集 %s' % d)

    model = build_model(preset='rasm', img_size=320).cuda().eval()
    out = load_weights(model, ck, strict=False)
    assert len(out.unexpected_keys) == 0, '官方权重出现结构冲突: %s' % out.unexpected_keys[:5]

    noisy, mask = get_validation_data(d)[0][1:3]
    noisy = noisy.unsqueeze(0).cuda()
    mask = mask.unsqueeze(0).cuda()
    with torch.no_grad():
        y = torch.clamp(model(noisy, mask), 0, 1)
    assert torch.isfinite(y).all(), '官方权重推理出现 NaN/Inf'
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0, '输出超出 [0,1]'
    mse = float(((y - noisy) ** 2).mean().clamp(min=1e-12))
    psnr = 10 * math.log10(1.0 / mse)
    print('官方权重(AISTD.pth) 输出 vs 输入 PSNR = %.2f dB' % psnr)
    assert psnr > 10.0, '输出相对输入偏离过大（%.2f dB），权重或前向有问题' % psnr


# --------------------------------------------------------------------------- #
# 9. 训练/测试 preset 不一致的风险确认
# --------------------------------------------------------------------------- #
@require_cuda
def test_preset_mismatch_hazard_on_real_ckpt(trained_run):
    from improved.rasmv2_model import load_weights

    best = os.path.join(trained_run['models'], 'model_best.pth')
    if not os.path.isfile(best):
        pytest.skip('没有 model_best.pth，跳过')
    # full 是 rasm 的超集: 把 full 的权重喂给 rasm 模型时，rasm 的每个键都能命中，
    # 因此表现为"多余键"(v2 独有参数被静默丢弃)，而不是缺失键。
    wrong = build_model(preset='rasm', img_size=trained_run['train_ps']).cuda()
    out = load_weights(wrong, best, strict=False)
    assert len(out.unexpected_keys) > 0, \
        'preset 不一致（full 训练 -> rasm 测试）却没有任何多余键，风险将无从察觉'
    print('确认: 训练 preset=full 的权重喂给 preset=rasm 会丢掉 %d 个参数（必须避免）'
          % len(out.unexpected_keys))
