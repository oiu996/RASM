# RASM 样例效果展示（肉眼对比）

指标表（PSNR / RMSE）只能说明"数字变好了"，**看不出模型到底做了什么**。
这里挑 10 张测试图，把**输入 → 模型输出 → 真值 GT** 摆在一张图上，让观众一眼看懂：
模型把阴影区域提亮成"本来该有的样子"，同时不动非阴影区域。

---

## 1. 一键运行（GPU 环境，在 `RASM/` 根目录）

```bash
bash showcase/run_demo.sh
```

分三步，也可以单独跑：

| 步骤 | 命令 | 是否需要 GPU |
|---|---|---|
| ① 挑样例、复制 A/B/C | `python showcase/scripts/select_samples.py --num 10` | 否 |
| ② 两个权重推理 + 算指标 | `python showcase/scripts/run_inference.py --gpus 0` | **是** |
| ③ 拼对比图 | `python showcase/scripts/make_figures.py` | 否 |

> 样例已在本仓库里选好（`showcase/samples/`），若不想重新挑图，设 `SKIP_SELECT=1` 再跑 `run_demo.sh`。
> 显存不够时把 `run_demo.sh` 里的 `TILE=""` 改成 `TILE="720"`（分块推理）。

---

## 2. 目录结构

```
showcase/
├── README.md
├── run_demo.sh                 一键脚本（GPU）
├── scripts/
│   ├── select_samples.py       ① 挑 10 张并复制 A/B/C（按阴影面积等间隔取样）
│   ├── run_inference.py        ② 两个权重推理 + 逐图指标（PSNR / RMSE）
│   └── make_figures.py         ③ 拼对比图 / 局部放大图 / 总览大图
├── samples/                    复制出来的"题目"
│   ├── manifest.json           清单：key、阴影占比、原图路径
│   ├── A_input/                带阴影原图（模型输入）
│   ├── B_mask/                 阴影 mask（模型输入）
│   └── C_gt/                   无阴影 GT（正确答案）
├── outputs/                    模型"答卷"
│   ├── srd/                    SRD.pth   的去阴影结果
│   └── istd/                   AISTD.pth 的去阴影结果
├── compare/                    最终给观众看的图
│   ├── 01_xxx_compare.png      每张一图五列：A输入 | B mask | C GT | 权重1 | 权重2
│   ├── zoom/01_xxx_zoom.png    阴影区域局部放大（最能看出差别）
│   └── all_samples_grid.png    10 张竖拼总览大图（汇报直接贴）
├── metrics.json                逐图 + 平均指标（机器可读）
└── metrics.md                  指标表格（可直接贴报告）
```

---

## 3. 本次对比的两个权重

| 列名 | 权重 | 说明 |
|---|---|---|
| `srd` | `log/RASM_srd_demo/models/SRD.pth` | SRD 数据集上训练的**官方权重**（同数据集，效果最好） |
| `istd` | `log/RASM_istd_demo/models/AISTD.pth` | ISTD+ 数据集上训练的**官方权重**（拿到 SRD 上跑 = 跨数据集） |

输入取自 SRD 测试集，所以两个权重的差距，本质是**"专业对口" vs "跨场景硬套"**的直观对照：
同数据集权重阴影去得干净、边界自然；跨数据集权重往往残留暗影或整片发灰。

想换成别的权重（例如自己训出来的 vs 官方），改一行即可：

```bash
python showcase/scripts/run_inference.py \
    --weights "official=log/RASM_srd_demo/models/SRD.pth" \
              "mine=log/RASM_srd_demo/srd_scratch_e1000/models/model_best.pth"
```

---

## 4. 图怎么看（给观众的 3 句话）

1. **看第 1 列 vs 第 3 列**：输入图里那块阴影，在 GT 里根本不存在——这就是模型要还原的目标。
2. **看第 4/5 列 vs 第 3 列**：模型输出越接近 GT，说明去阴影越成功；重点看**阴影边界有没有残留**、**颜色是不是和周围接得上**。
3. **看 `zoom/` 局部放大**：整图缩小后差别不明显，放大后能看清"暗影被提亮"和"纹理有没有被抹掉"。

指标口径与 RASM 原版 `test.py` 一致：PSNR 越高越好，RMSE(LAB) 越低越好；
`_s` / `_ns` 后缀分别是只统计阴影区 / 非阴影区（详见 `metrics.md`）。

---

## 5. 已选样例

| # | 样例 | 阴影面积占比 |
|---|---|---|
| 01 | IMG_1_5476 | 3.0% |
| 02 | _MG_2439 | 6.1% |
| 03 | IMG_6563 | 7.9% |
| 04 | IMG_2454 | 11.0% |
| 05 | _MG_2442 | 13.7% |
| 06 | IMG_2430 | 16.9% |
| 07 | _MG_5599 | 21.6% |
| 08 | _MG_3048 | 26.5% |
| 09 | IMG_6591 | 30.9% |
| 10 | IMG_6900 | 35.7% |

按阴影面积从小到大**等间隔**取样：既有"一小条阴影"也有"半张图都是阴影"，展示更有代表性。
重新挑图：`python showcase/scripts/select_samples.py --seed 1`（换种子换一批）。
