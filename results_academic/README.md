# RASM 2×2 跨数据集评估 —— 学术数据展示产物

> 生成脚本：`result_script/make_2x2_academic.py`
> 运行：`cd RASM && python result_script/make_2x2_academic.py`
> 输出目录：`RASM/results_academic/`
>
> RASMv2 变体（读 `results_*_v2/metrics_v2.jsonl`）：`result_script/make_2x2_academic_v2.py` → 输出 `RASM/results_academic_v2/`

## 1. 「测距 2×2」的含义

指 **2×2 跨数据集泛化评估矩阵**：两个训练模型（ISTD 上训 / SRD 上训）× 两个测试集（ISTD / SRD），
对应仓库 4 个 `results_*` 测试产物目录。

| 模型＼测试集 | ISTD 测试 | SRD 测试 |
|---|---|---|
| **ISTD 训** | `results_istd` | `results_istd_on_srd` |
| **SRD 训** | `results_srd_on_istd` | `results_srd` |

对角线 = 域内测试；反对角线 = 跨数据集泛化测试。RMSE 等距离型误差即「距」。

## 2. 产物清单

| 文件 | 内容 |
|---|---|
| `table_2x2_psnr.md` | 2×2 整体 PSNR (dB) 矩阵 |
| `table_2x2_rmse.md` | 2×2 整体 RMSE 矩阵 |
| `table_full_metrics.md` | 4 个配置的全部指标（PSNR/RMSE/S-/NS- 细分） |
| `table_per_image_stats.md` | 逐图 PSNR/RMSE 的均值/标准差/最值 |
| `table_full_metrics.tex` | 上述明细的 LaTeX 三线表 |
| `metrics_summary.csv` | 汇总指标 CSV |
| `fig_2x2_matrix.png` | 核心图：2×2 热力图（PSNR / RMSE 两幅） |
| `fig_metric_bars.png` | 4 项指标分组柱状图（域内蓝 / 跨域橙） |
| `fig_box_psnr.png` | 各配置逐图 PSNR 箱线图 |
| `fig_gap.png` | 泛化性落差（域内 − 跨域 PSNR） |

## 3. 指标说明

- 每个 `results_*/metrics.jsonl` 是「追加累积」的；脚本取 `num_images` 最大（完整评测）的那次记录。
- S-* / NS-* 分别为**阴影区 / 非阴影区**指标。
- SSIM 在 `test.py` 中未启用（恒为 0），故所有表格不列 SSIM。
- 若要与论文口径严格对齐，RMSE/PSNR 可用；若要 SSIM，请改用 `evaluation/measure_shadow.m`（MATLAB）。

## 4. 已确认的全量（393 图）参考值

| 配置 | PSNR↑ | RMSE↓ | S-PSNR↑ | NS-PSNR↑ |
|---|---|---|---|---|
| SRD→SRD（域内） | 31.33 | 4.19 | 35.48 | 34.83 |
| ISTD→SRD（跨域） | 24.85 | 7.86 | 27.71 | 29.44 |

（ISTD→ISTD、SRD→ISTD 两格由脚本从上述 `metrics.jsonl` 自动读取后填入。）
