# RASM 训练 / 测试产物路径速查

> 所有相对路径均相对于 **`RASM/` 根目录**（即 `train.py` / `test.py` 所在目录）。
> 运行脚本时请在该目录下执行（如 `cd RASM && bash train_SRD.sh`），
> 否则 `datasets/`、`log/`、`results_*` 等相对路径会错位。

---

## 一、训练产物（`train_ISTD.sh` / `train_SRD.sh` → `train.py`）

`train.py` 内部硬编码：

```python
# 方案D: 每个 run 一个独立子目录
run_name = opt.run_name if opt.run_name else datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env, run_name)
```

其中 `dir_name` = `RASM/` 根目录，`opt.arch="RASM"`。

**重要**：`--save_dir` 参数实际被忽略（脚本注释已注明），所有产物都在 `./log/RASM_<env>/<run_name>/` 下。
`<run_name>` 不传时自动取时间戳（如 `20261027-143000`），故每次启动都是一个全新 run 目录，
实验之间物理隔离、互不覆盖；显式传 `--run_name` 则写回指定目录（用于训练中断后续跑同一个 run）。

| 产物 | ISTD 具体路径 | SRD 具体路径 |
|---|---|---|
| 最佳权重（固定指针） | `RASM/log/RASM_istd_demo/<run_name>/models/model_best.pth` | `RASM/log/RASM_srd_demo/<run_name>/models/model_best.pth` |
| 最佳权重（自描述归档副本） | `.../models/RASM_ISTD_e32_w10_linear_leff_ep0785_valpsnr36.21.pth` | `.../models/RASM_SRD_e32_w10_linear_leff_ep0785_valpsnr36.21.pth` |
| 最新权重 | `.../models/model_latest.pth` | 同左 |
| 按 epoch 保存的权重 | `.../models/model_epoch_{N}.pth` | 同左 |
| 训练日志 | `.../train.log` | 同左 |
| 本次 run 完整超参 + 起始时间 + 权重来源 | `.../config.json` | 同左 |
| **逐 epoch 结构化指标** | `.../metrics.jsonl`（每个评估点一行，含时间戳/权重来源/指标/是否刷新最优） | 同左 |
| **run 结束汇总** | `.../summary.json`（最优 epoch / best_psnr / 归档文件名） | 同左 |
| 续训来源记录 | `.../resume_from.txt`（仅 `--resume` 时生成） | 同左 |
| TensorBoard event 文件 | `RASM/runs/<时间戳> exp`（默认 `SummaryWriter(comment=...)`，不进 `log/`） | 同左 |
| `results/` 目录 | `.../results/`（已创建，但 `train.py` 未落盘图片，基本为空） | 同左 |

环境变量对应关系：
- `train_ISTD.sh`：`ENV_NAME="_istd_demo"` → `RASM/log/RASM_istd_demo`
- `train_SRD.sh`：`ENV_NAME="_srd_demo"` → `RASM/log/RASM_srd_demo`

---

## 二、测试产物（`test_ISTD.sh` / `test_SRD.sh` → `test.py`）

`test.py` 用 `--result_dir` 作为**平铺输出目录**（不会与 `--ex_name` 拼接；
`--ex_name` 仅用于终端打印指标表头）。

| 产物 | ISTD 具体路径 | SRD 具体路径 |
|---|---|---|
| 去阴影结果图（需 `--save_images`，脚本已带） | `RASM/results_istd/<原文件名>` | `RASM/results_srd/<原文件名>` |
| **指标（结构化落盘）** | `RASM/results_istd/metrics.jsonl`（一次测试一行，含时间戳/权重/汇总/逐图指标） | `RASM/results_srd/metrics.jsonl` |
| 指标（终端打印） | 仍打印到 stdout，可 `tee` 重定向留存 | 同左 |
| 读取的权重 | `RASM/log/RASM_istd_demo/models/AISTD.pth` | `RASM/log/RASM_srd_demo/models/SRD.pth` |

`test.py` 关键落盘代码：

```python
utils.mkdir(args.result_dir)                                   # 创建 ./results_*
...
utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]))  # 保存去阴影图
```

---

## 三、冒烟 / 验证脚本产物（`check_script/`）

| 脚本 | 产物 |
|---|---|
| `check_script/train_smoke.sh`（默认 `ENV_NAME="_srd_smoke"`） | `RASM/log/RASM_srd_smoke/models/model_latest.pth` 等 |
| `check_script/verify_gpu_model.py` | 无文件产物，仅在终端打印 `params=...` / `forward ok` / `backward ok` |
| `check_script/check_env.sh` | 无文件产物，仅打印 torch / natsort 版本与 CUDA 可用性 |
| `check_script/test_dataset_cpu.py` | 无文件产物（pytest 断言，可在 CPU 跑） |
| `check_script/test_model_gpu.py` | 无文件产物（pytest 断言，GPU 才真正执行） |

---

## 四、常见注意点

1. **`--save_dir` 无效**：训练产物位置由 `train.py` 的 `log_dir` 决定，由 `--arch` + `--env` + `--run_name`
   三级拼成，想换目录改脚本里的 `ENV_NAME`（实验组）或 `RUN_NAME`（单次 run）。
2. **相对路径依赖 CWD**：`train_SRD.sh` 等原版脚本假定你直接 `cd RASM && bash train_SRD.sh`；
   我写的 `check_script/train_smoke.sh` 已用 `cd "$(dirname "$0")/.."` 自动回到 `RASM/` 根目录。
3. **测试权重必须存在**：`test.py` 通过 `utils.load_checkpoint` 加载 `--weights`，缺失会报错；
   需先跑完对应训练（或下载官方预训练权重并填好路径）。
4. **测试指标已结构化落盘**：每次测试向 `results_*/metrics.jsonl` **追加**一行，多次测试会累积成历史，
   可直接横向对比不同权重。转成表格：
   `python -c "import pandas as pd; pd.read_json('results_istd/metrics.jsonl', lines=True).to_csv('m.csv', index=False)"`
   （终端 stdout 仍会打印，想留存可 `bash test_SRD.sh | tee results_srd/metrics.txt`）
5. **续训（方案D）**：加 `--resume` 并把 `--pretrain_weights` 指向**上一次 run** 的 `model_best.pth`；
   新产物写入**新的** run 目录，历史实验不会被覆盖。`best_psnr` 会从 ckpt 的 `best_psnr` 字段恢复
   （老权重无该字段则回退为 0），避免续训首个 epoch 用更差的权重覆盖已加载的最优权重。
   想接着**同一个** run 目录跑，显式传 `--run_name <目录名>`。
6. **归档副本只留最新一份**：每次刷新 best 时，会先删掉本 run 目录内上一份 `*_ep*_valpsnr*.pth` 再写新的，
   避免堆积；固定名 `model_best.pth` 始终保留，供 `--resume` 与测试脚本引用。
7. **时间戳都记在哪儿**（四处，粒度由粗到细）：
   | 位置 | 内容 | 说明 |
   |---|---|---|
   | run 目录名 | `log/RASM_istd_demo/20261027-143000/` | 最粗粒度，一眼区分不同次实验 |
   | `config.json` 的 `run_started_at` | run 启动时刻 | 配合 `source_weights` 说明"从哪来、何时开始" |
   | `metrics.jsonl` 每行的 `timestamp` | 每个评估点的时刻 | 可算耗时、看训练曲线的时间轴 |
   | `summary.json` 的 `finished_at` | run 结束时刻 | 与 `run_started_at` 相减得总耗时 |
   测试侧同理：`results_*/metrics.jsonl` 每行带 `timestamp`，记录该次测试的时刻。
