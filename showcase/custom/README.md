# 自定义图片盲测（无 mask / 无 GT 的肉眼展示）

前面 `showcase/` 主流程用的是 SRD 测试集里**已知 A/B/C** 的样例，能算指标、能对照真值。
但那有个隐患：这些数据模型训练/评测时都见过，**没法排除"权重里偷看了答案"的嫌疑**。

所以这里单独开一个分支：你**自己提供图片**，只有"原图"，没有 B（mask）、也没有 C（GT）。
两个权重各自去阴影，直接肉眼对比输出，不评任何指标——这就是一个**盲测 / 可信度展示**分支。

---

## 1. 怎么用

1. 把你的图丢进 `showcase/custom/input/`（jpg/png，命名随意，几张都行）。
2. 在 `RASM/` 根目录执行：

```bash
bash showcase/custom/run_custom.sh
```

得到：

- `outputs/srd/`、`outputs/istd/` —— 两个权重的去阴影结果
- `outputs/_masks/` —— 自动估计的粗阴影 mask（仅用于驱动模型，**不是真值**）
- `compare/XX_compare.png` —— 每张图：输入 | (估计阴影) | 权重1输出 | 权重2输出
- `compare/zoom/` —— 阴影区域局部放大（最能看出差别）
- `compare/all_custom_grid.png` —— 总览大图

想多一列"估计阴影"：把 `run_custom.sh` 里 `SHOW_MASK=0` 改成 `1`。

---

## 2. 关于"没有 mask 怎么推理"

RASM 是 **mask-guided** 模型，推理时必须有阴影 mask 输入。但你只给了原图，所以：

- **默认**：脚本用"亮度差"自动估一个粗 mask（周围比当前亮 → 当前判为阴影），喂给模型。
  它只是个**驱动信号**，不代表真实阴影边界，也不是用来打分依据。
- **自带 mask**（可选）：若你有更准确的 mask，放进 `showcase/custom/masks/` 且与原图同名，
  推理时会被自动采用。

> 无论哪种，本分支**不计算 PSNR/RMSE**——因为没有真值，也正因为如此，
> 它才适合用来检验"换张没见过的图，模型到底还去不去得动阴影"。

---

## 3. 代码复用说明

本分支刻意只放两套轻量脚本，核心逻辑全部复用主流程，避免重复维护：

| 主流程（showcase/scripts/） | 本分支复用方式 |
|---|---|
| `run_inference.py` 的 `load_model` / `infer_tiled` | `run_inference_custom.py` 直接 `import` 复用 |
| `make_figures.py` 的 `strip_row` / `resize_keep` / `read_rgb` / `get_font` | `make_figures_custom.py` 直接 `import` 复用 |

本分支独有逻辑只有两块：
- `estimate_mask()`（亮度差自动估 mask）
- 不读 GT、不写指标、只标推理耗时的拼图流程。

---

## 4. 换权重 / 换数据

- 换权重：改 `run_custom.sh` 里没有用到的 `--weights`，或运行时手动传：
  ```bash
  python showcase/custom/scripts/run_inference_custom.py \
      --weights "official=log/RASM_srd_demo/models/SRD.pth" \
                "mine=log/your_run/models/model_best.pth" --gpus 0
  ```
- 换数据：直接替换 `input/` 里的图即可，无需动任何代码。
