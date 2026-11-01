# result_script —— 结果后处理脚本

本目录专门存放**把训练/测试产物（metrics.jsonl / summary.json）汇总成可展示结果**的脚本，
与 `check_script/`（环境/数据/模型自检）职责分离。

> 约定：所有脚本都通过 `ROOT = dirname(dirname(abspath(__file__)))` 定位到 `RASM/` 根目录，
> 因此无论从哪个目录调用、放在哪个子目录，路径都自动正确。必须在 `RASM/` 根目录下运行。

| 脚本 | 作用 | 读取 | 输出 |
|---|---|---|---|
| `show_results.py` | 训练 run 汇总 / 训练曲线 / 测试指标对比（Markdown） | `log/*/*/config.json` + `results_*/metrics.jsonl` | 终端表格 + `training_curve.png` |
| `make_2x2_academic.py` | 原模型 2×2 跨数据集学术表格/图 | `results_{istd,srd,istd_on_srd,srd_on_istd}/metrics.jsonl` | `results_academic/` |
| `make_2x2_academic_v2.py` | RASMv2 维度 2×2 跨数据集学术表格/图 | `results_*_v2/metrics_v2.jsonl` | `results_academic_v2/` |

运行示例：

```bash
cd /workspace/RASM
python result_script/show_results.py all
python result_script/make_2x2_academic.py
python result_script/make_2x2_academic_v2.py
```
