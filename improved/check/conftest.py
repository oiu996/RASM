# -*- coding: utf-8 -*-
"""pytest 共享夹具：合成数据集 + 端到端(子进程)跑一次 train_v2.py。

只有真正需要它们的测试才会触发，因此跑单个单元测试时不会额外耗时。
端到端夹具用 session 作用域，多个测试复用同一次训练产物。

可用环境变量调节（见 run_gpu_checks.sh）:
    GPU_DEV / CHECK_NEPOCH / CHECK_TRAIN_PS / CHECK_BATCH / CHECK_ENV
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pytest
import torch

from _rasmv2_common import (RASM_ROOT, env_cfg, fresh_dir, make_synthetic_dataset,
                            read_jsonl, run_py, run_py_stream,
                            test_v2_cmd as make_test_cmd,
                            train_v2_cmd as make_train_cmd)   # noqa: F401

CFG = env_cfg()
GPU_DEV = CFG['gpu']
NEPOCH = CFG['nepoch']
TRAIN_PS = CFG['train_ps']
BATCH = CFG['batch']
LOG_ENV = CFG['env']
LOG_INTERVAL = CFG['log_interval']

require_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason='需要 CUDA：RASMv2 训练/测试链路全程 .cuda()')


@pytest.fixture(scope='session')
def syn_dataset(tmp_path_factory):
    """8 张训练 + 4 张测试的最小数据集（A/B/C 三分支、阴影为居中方块）。"""
    root = str(tmp_path_factory.mktemp('rasmv2_syn') / 'data')
    make_synthetic_dataset(root, n_train=8, n_test=4, size=TRAIN_PS)
    return root


@pytest.fixture(scope='session')
def trained_run(syn_dataset):
    """端到端跑一次 train_v2.py（默认 21 epoch）。

    epoch 数取 21 是有意的：train_v2.py 里 eval_now = 20 * len(train_loader)，
    本数据集 len(train_loader)=4，即第 80 次迭代（epoch 20 的第 3 个 batch）才会
    触发一次验证 —— 只有发生过验证，才会产出 model_best.pth 与 metrics.jsonl。
    """
    if not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    run_name = 'e2e_full'
    log_dir = os.path.join(RASM_ROOT, 'log', 'RASMv2' + LOG_ENV, run_name)
    fresh_dir(log_dir)
    proc = run_py_stream(make_train_cmd(os.path.join(syn_dataset, 'train'),
                                        os.path.join(syn_dataset, 'test'),
                                        run_name, NEPOCH, preset='full', gpu=GPU_DEV,
                                        train_ps=TRAIN_PS, batch_size=BATCH, env=LOG_ENV,
                                        log_interval=LOG_INTERVAL),
                         timeout=7200,
                         log_path=os.path.join(log_dir, 'e2e_stdout.log'))
    assert proc.returncode == 0, 'train_v2.py 端到端训练失败（尾部输出）:\n' + proc.stdout[-4000:]
    return dict(run_name=run_name, log_dir=log_dir,
                models=os.path.join(log_dir, 'models'),
                dataset=syn_dataset, train_ps=TRAIN_PS, batch=BATCH,
                nepoch=NEPOCH, preset='full', stdout=proc.stdout)


@pytest.fixture(scope='session')
def e2e_test_result(trained_run, tmp_path_factory):
    """用训练产物跑一次 test_v2.py（整图推理 + 落盘指标 + 存图）。"""
    if not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    result_dir = str(tmp_path_factory.mktemp('rasmv2_res_full'))
    weights = os.path.join(trained_run['models'], 'model_best.pth')
    assert os.path.isfile(weights), '训练集产物缺少 model_best.pth: %s' % weights
    proc = run_py(make_test_cmd(os.path.join(trained_run['dataset'], 'test'), weights,
                                result_dir, ex_name='check_full',
                                preset=trained_run['preset'], gpu=GPU_DEV,
                                train_ps=trained_run['train_ps']),
                  timeout=1800)
    assert proc.returncode == 0, 'test_v2.py 端到端推理失败（尾部输出）:\n' + proc.stdout[-4000:]
    records = read_jsonl(os.path.join(result_dir, 'metrics_v2.jsonl'))
    assert records, 'test_v2.py 没有写出 metrics_v2.jsonl'
    return dict(result_dir=result_dir, weights=weights, record=records[-1],
                dataset=trained_run['dataset'], preset=trained_run['preset'],
                train_ps=trained_run['train_ps'], stdout=proc.stdout)


@pytest.fixture(scope='session')
def one_epoch_run(syn_dataset):
    """只跑 1 个 epoch：用于确认"短训练不产出 model_best.pth"这一已知限制。"""
    if not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    run_name = 'e2e_1ep'
    log_dir = os.path.join(RASM_ROOT, 'log', 'RASMv2' + LOG_ENV, run_name)
    fresh_dir(log_dir)
    proc = run_py_stream(make_train_cmd(os.path.join(syn_dataset, 'train'),
                                        os.path.join(syn_dataset, 'test'),
                                        run_name, 1, preset='full', gpu=GPU_DEV,
                                        train_ps=TRAIN_PS, batch_size=BATCH, env=LOG_ENV,
                                        log_interval=LOG_INTERVAL),
                         timeout=1800,
                         log_path=os.path.join(log_dir, 'e2e_stdout.log'))
    assert proc.returncode == 0, 'train_v2.py 单 epoch 训练失败:\n' + proc.stdout[-4000:]
    return dict(models=os.path.join(log_dir, 'models'), stdout=proc.stdout)
