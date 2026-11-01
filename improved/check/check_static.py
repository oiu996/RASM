 # -*- coding: utf-8 -*-
"""RASMv2 训练/测试链路的**静态**检查（CPU 即可运行，不需要 GPU）。

它回答的是"切到 GPU 之前就能发现的那类 bug"：
  1. 语法：improved/ 与 improved/check/ 下所有 .py 能否编译；
  2. 参数：shell 脚本传给 python 的每个 --flag 是否真的存在于对应 argparse；
         训练脚本与测试脚本的结构参数(preset/kernel_size/dilations/...)是否一致；
  3. 路径：脚本里写死的数据集/权重/VGG 感知损失权重是否真实存在；
  4. 依赖：train_v2.py / test_v2.py 用到的 utils API 是否都还在；
  5. 权重兼容：RASMv2(preset=rasm) 能否吃下官方 AISTD.pth / SRD.pth（0 项结构冲突）；
  6. 损失性质：三项新损失"宣称的性质"是否成立（纯 CPU 张量运算，秒级）。

用法（在 RASM/ 目录下）:
    python improved/check/check_static.py
退出码非 0 表示有 FAIL。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ast
import py_compile
import re

from _rasmv2_common import (DATASET_DIRS, IMPROVED_DIR, OFFICIAL_CKPT, PRESETS,
                            RASM_ROOT, STRUCT_KEYS, V2_NEW_KEY_TOKENS, VGG_WEIGHTS)

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((bool(ok), name, detail))
    print('[%s] %-52s %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    return bool(ok)


# --------------------------------------------------------------------------- #
def iter_py(directory):
    for f in sorted(os.listdir(directory)):
        if f.endswith('.py'):
            yield os.path.join(directory, f)


def check_compile():
    files = list(iter_py(IMPROVED_DIR)) + list(iter_py(HERE))
    bad = []
    for f in files:
        try:
            # 只做语法/字节码编译，不落 .pyc（避免污染工作区）
            compile(open(f, encoding='utf-8').read(), f, 'exec')
        except Exception as e:                                  # noqa: BLE001
            bad.append('%s: %s' % (os.path.relpath(f, RASM_ROOT), e))
    return check('语法编译 improved/*.py + check/*.py', not bad,
                 '%d 个文件' % len(files) if not bad else '; '.join(bad))


# --------------------------------------------------------------------------- #
def py_flags(path):
    """从 .py 里 AST 提取所有 add_argument('--xxx')。"""
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == 'add_argument' and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str) \
                    and a0.value.startswith('--'):
                out.add(a0.value)
    return out


def sh_flags(path):
    txt = open(path, encoding='utf-8').read()
    txt = re.sub(r'#[^\n]*', '', txt)                           # 去掉注释
    return set(re.findall(r'--[A-Za-z_][A-Za-z0-9_-]*', txt))


def sh_vars(path):
    """提取 sh 里的 KEY=value / KEY="value" 赋值。"""
    txt = open(path, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'^\s*([A-Z][A-Z0-9_]*)=("([^"]*)"|\'([^\']*)\'|([^\s#\'"]+))',
                         txt, re.M):
        val = next((g for g in m.groups()[2:] if g is not None), '')
        out[m.group(1)] = val.strip()
    return out


def check_cli_flags():
    p_option = os.path.join(RASM_ROOT, 'options.py')
    p_train = os.path.join(RASM_ROOT, 'improved', 'train_v2.py')
    p_test = os.path.join(RASM_ROOT, 'improved', 'test_v2.py')
    flags_train = py_flags(p_option) | py_flags(p_train)
    flags_test = py_flags(p_test)

    sh_dir = IMPROVED_DIR
    mapping = {
        'train_ISTD_v2.sh': flags_train,
        'train_SRD_v2.sh': flags_train,
        'test_ISTD_v2.sh': flags_test,
        'test_SRD_v2.sh': flags_test,
        'improved_try.sh': flags_train | flags_test,
    }
    ok = True
    for sh, allowed in mapping.items():
        path = os.path.join(sh_dir, sh)
        if not os.path.isfile(path):
            check('shell 脚本存在: %s' % sh, False, '未找到')
            ok = False
            continue
        unknown = sorted(sh_flags(path) - allowed)
        ok &= check('参数名合法: %s' % sh, not unknown,
                    '未知参数 %s' % unknown if unknown else '%d 个参数全部命中' % len(sh_flags(path)))
    return ok


def check_struct_consistency():
    ok = True
    for tag in ('ISTD', 'SRD'):
        tr = os.path.join(IMPROVED_DIR, 'train_%s_v2.sh' % tag)
        te = os.path.join(IMPROVED_DIR, 'test_%s_v2.sh' % tag)
        if not (os.path.isfile(tr) and os.path.isfile(te)):
            continue
        vt, ve = sh_vars(tr), sh_vars(te)
        diff = [k for k in STRUCT_KEYS
                if k in vt and k in ve and vt[k].replace('"', '') != ve[k].replace('"', '')]
        ok &= check('训练/测试结构参数一致: %s' % tag, not diff,
                    '不一致: %s' % diff if diff else 'preset/kernel/dilations/embed/win/proj/mlp 一致')
    return ok


# --------------------------------------------------------------------------- #
PATH_HINT = ('训练产物实际在 log/RASMv2<env>/<run_name>/models/model_best.pth，'
             '脚本默认路径少了 <run_name> 这一层')


def check_paths():
    ok = True
    for ds, (tr, te) in DATASET_DIRS.items():
        for tag, p in (('train', tr), ('test', te)):
            ok &= check('数据集存在: %s %s' % (ds, tag), os.path.isdir(p), os.path.relpath(p, RASM_ROOT))
    for ds, p in OFFICIAL_CKPT.items():
        ok &= check('官方权重存在: %s' % ds, os.path.isfile(p), os.path.relpath(p, RASM_ROOT))
    ok &= check('VGG19 感知损失权重存在', os.path.isfile(VGG_WEIGHTS), VGG_WEIGHTS)

    must = {'TRAIN_DIR', 'VAL_DIR', 'INPUT_DIR', 'PRETRAIN_WEIGHTS', 'OFFICIAL_CKPT'}
    for sh in sorted(f for f in os.listdir(IMPROVED_DIR) if f.endswith('.sh')):
        vals = sh_vars(os.path.join(IMPROVED_DIR, sh))
        for key in sorted(must & set(vals)):
            v = vals[key]
            if not v or v.startswith('$'):
                continue
            p = v if os.path.isabs(v) else os.path.join(RASM_ROOT, v)
            ok &= check('%s: %s 指向的路径存在' % (sh, key), os.path.exists(p), v)
        # 测试脚本的默认权重：缺了它，跑完训练直接测试会 FileNotFound
        # 只看 test_*_v2.sh（continue.sh / improved_try.sh 里的 WEIGHTS 是占位/环境变量）
        if sh.startswith('test_') and sh.endswith('_v2.sh') and 'WEIGHTS' in vals:
            v = vals['WEIGHTS']
            # 展开脚本里最常见的 ${VAR:-default} 默认值写法
            v = re.sub(r'\$\{(\w+):-([^}]*)\}', lambda m: os.environ.get(m.group(1), m.group(2)), v)
            p = v if os.path.isabs(v) else os.path.join(RASM_ROOT, v)
            # 产物要等训练跑完才出现，所以这里只校验"路径形态"必须带 <run_name> 这一层；
            # 文件不存在只提示，不算失败（否则没训练过的干净环境必然误报）。
            rel = os.path.relpath(p, RASM_ROOT)
            parts = rel.split(os.sep)
            # 正确形态: log/RASMv2<env>/<run_name>/models/model_best.pth  (5 段)
            shaped = (len(parts) == 5 and parts[0] == 'log'
                      and parts[-1] == 'model_best.pth' and parts[-2] == 'models')
            ok &= check('%s: 默认 WEIGHTS 路径形态' % sh, shaped,
                        rel if shaped else '%s -> %s' % (rel, PATH_HINT))
            if shaped and not os.path.isfile(p):
                print('[INFO] %s: %s 尚未产生（训练跑完才会出现），非错误' % (sh, rel))
    return ok


# --------------------------------------------------------------------------- #
def _help_flags(script):
    """跑 `python <script> --help` 拿到 argparse 真正接受的参数集合。

    --help 会在 parse_args 时直接 SystemExit(0)，不会执行训练/推理，故 CPU 上安全。
    """
    import subprocess
    proc = subprocess.run([sys.executable, script, '--help'], cwd=RASM_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=900)
    if proc.returncode != 0:
        return None, (proc.stdout or '')[-500:]
    return set(re.findall(r'--[A-Za-z_][A-Za-z0-9_-]*', proc.stdout)), ''


def check_e2e_cmd_flags():
    """端到端检查会真实执行的命令行，必须与 argparse 完全对齐。"""
    from _rasmv2_common import test_v2_cmd as make_test_cmd
    from _rasmv2_common import train_v2_cmd as make_train_cmd

    ok = True
    cases = [
        ('improved/train_v2.py', make_train_cmd('TRAIN', 'VAL', 'RUN', 1)),
        ('improved/test_v2.py',
         make_test_cmd('INPUT', 'W', 'RES', 'EX', tile=64)),
    ]
    for script, cmd in cases:
        flags, err = _help_flags(script)
        if flags is None:
            ok &= check('%s 可解析(--help)' % script, False, err)
            continue
        used = {t for t in cmd if t.startswith('--')}
        unknown = sorted(used - flags)
        ok &= check('端到端命令参数合法: %s' % script, not unknown,
                    '未知参数 %s' % unknown if unknown else '%d 个参数全部命中' % len(used))
    return ok


# --------------------------------------------------------------------------- #
def check_utils_api():
    import utils
    from utils.loader import get_training_data, get_validation_data
    need = ['mkdir', 'save_img', 'batch_PSNR', 'MixUp_AUG', 'write_json', 'append_jsonl', 'now_iso']
    miss = [n for n in need if not hasattr(utils, n)]
    ok = check('utils API 齐全', not miss, '缺失 %s' % miss if miss else ','.join(need))
    ok &= check('utils.loader 可用', callable(get_training_data) and callable(get_validation_data), '')
    return ok


# --------------------------------------------------------------------------- #
def check_official_weights_compat():
    """CPU 上即可做：RASMv2(preset=rasm) 与官方权重的键名兼容。

    这是"能在官方权重上续训"的前提：允许缺失 v2 新增参数，
    但不允许出现"官方有、v2 结构对不上"的冗余项。
    """
    try:
        import torch
        from improved.rasmv2_model import RASMv2, load_weights
    except Exception as e:                                      # noqa: BLE001
        return check('官方权重键名兼容(CPU)', False, '导入失败: %s' % e)

    torch.set_num_threads(1)
    ok = True
    for preset in ('rasm', 'full'):
        try:
            model = RASMv2(img_size=320, embed_dim=32, win_size=10, token_projection='linear',
                           token_mlp='leff', preset=preset)
        except Exception as e:                                  # noqa: BLE001
            ok &= check('构建 RASMv2(preset=%s)' % preset, False, str(e))
            continue
        for ds, ck in OFFICIAL_CKPT.items():
            if not os.path.isfile(ck):
                continue
            out = load_weights(model, ck, strict=False)
            new_only = all(any(t in k for t in V2_NEW_KEY_TOKENS) for k in out.missing_keys)
            good = (len(out.unexpected_keys) == 0) and new_only
            detail = '缺失 %d 项(均为 v2 新增=%s), 结构冲突 %d 项' % (
                len(out.missing_keys), new_only, len(out.unexpected_keys))
            if not new_only:
                bad = [k for k in out.missing_keys
                       if not any(t in k for t in V2_NEW_KEY_TOKENS)]
                detail += ' | 非预期缺失: %s' % bad[:3]
            ok &= check('官方权重兼容: preset=%s + %s' % (preset, ds), good, detail)
    return ok


# --------------------------------------------------------------------------- #
def check_losses_cpu():
    """三项新损失的性质（纯 CPU 运算，秒级）。"""
    import torch
    from improved.rasmv2_losses import (BoundaryLoss, ChromaticityLoss,
                                        NonShadowConsistencyLoss, RASMv2Loss,
                                        _dilate, _erode)

    torch.manual_seed(0)
    gt = torch.rand(1, 3, 32, 32).clamp(0.1, 0.9)
    m = torch.zeros(1, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1.0

    ns = NonShadowConsistencyLoss(pool=9)
    l_const = ns(gt + 0.05, gt, m).item()
    l_noise = ns(gt + 0.05 * torch.randn(1, 3, 32, 32), gt, m).item()
    ok = check('NonShadowConsistency: 容忍常量偏移、惩罚局部噪声',
               l_const < 0.02 and l_noise > l_const,
               '常量 %.4f vs 噪声 %.4f' % (l_const, l_noise))

    ch = ChromaticityLoss(min_lum=0.05)
    l_dim = ch(gt * 0.5, gt).item()
    l_hue = ch(gt * torch.tensor([1.0, 0.7, 1.0]).view(1, 3, 1, 1), gt).item()
    ok &= check('Chromaticity: 对纯减光不敏感、对偏色敏感',
                l_dim < 0.02 and l_hue > l_dim, '减光 %.4f vs 偏色 %.4f' % (l_dim, l_hue))

    bl = BoundaryLoss(band=9, w_shadow=0.0, w_band=1.0)
    band = (_dilate(m, 9) - _erode(m, 9)).clamp(0, 1)
    l_in = bl(gt + 0.1 * band, gt, m).item()
    l_out = bl(gt + 0.1 * (1 - band), gt, m).item()
    ok &= check('Boundary: 半影带内误差更贵', l_in > l_out,
                '带内 %.4f vs 带外 %.4f' % (l_in, l_out))

    comb = RASMv2Loss(w_boundary=1.0, w_ns=0.5, w_chroma=0.2, base=torch.nn.L1Loss())
    total = comb(gt * 0.9, gt, m).item()
    ok &= check('RASMv2Loss(base=L1) 可组合且为有限值', total == total and total < float('inf'),
                'total=%.4f' % total)
    return ok


# --------------------------------------------------------------------------- #
def main():
    print('== RASMv2 静态检查（CPU，无需 GPU）==', flush=True)
    print('RASM_ROOT = %s' % RASM_ROOT, flush=True)
    check_compile()
    check_cli_flags()
    check_struct_consistency()
    check_e2e_cmd_flags()
    check_paths()
    check_utils_api()
    check_official_weights_compat()
    check_losses_cpu()

    n_fail = sum(1 for ok_, _, _ in RESULTS if not ok_)
    print('\n== 合计 %d 项, 失败 %d 项 ==' % (len(RESULTS), n_fail))
    if n_fail:
        print('以下检查未通过（这些在切到 GPU 之前就应该修掉）:')
        for ok_, name, detail in RESULTS:
            if not ok_:
                print('  - %s | %s' % (name, detail))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
