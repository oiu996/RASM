# RASMv2：对 RASM (Regional Attention for Shadow Removal, MM'24) 的改进

> 目录内所有内容均为**新增**，`RASM/` 下的任何既有文件（model.py / losses.py / train.py / test.py / utils/\*）都**没有被改动**，
> 符合开闭原则（OCP）：对扩展开放、对修改封闭。

```
RASM/
├── model.py                # 原 RASM 实现（未改动）
├── losses.py               # 原损失（未改动）
├── train.py / test.py      # 原训练/测试入口（未改动）
└── improved/               # ★ 本次全部新增
    ├── rasmv2_modules.py   # 改进模块：MGRA / NSRC / Mask-FiLM / 自定义邻域注意力
    ├── rasmv2_model.py     # RASMv2 模型 + 消融预设 + 权重加载工具
    ├── rasmv2_losses.py    # 改进损失：半影加权 / 非阴影一致性 / 色度一致性
    ├── train_v2.py  test_v2.py
    ├── train_ISTD_v2.sh  train_SRD_v2.sh  test_ISTD_v2.sh  test_SRD_v2.sh
    └── selftest.py         # CPU 自检（无需 GPU）
```

---

## 一、动机：论文留下的四个"没做完"

通读论文 + 对照 `model.py` 后，可以定位到四处改进空间：

| # | 观察到的现象 | 位置 |
|---|---|---|
| 1 | **mask 在注意力里完全没被用上**。`RALayer.forward(self, x, xm, mask, img_size)` 里 `xm = xm.permute(...)` 之后再无引用；mask 只在输入端 `torch.cat((x,xm),1)` 拼成 4 通道过一次 `InputProj`，之后再不参与任何层 | `model.py:434-458`、`model.py:675` |
| 2 | **区域注意力是"内容驱动"的，不区分 key 是否在阴影里**。论文 Fig.3/Fig.6 的核心论点是"阴影 token 应参考紧邻的非阴影 token"，但这条假设只能靠 NA 隐式学到，没有任何结构化先验保证 | 论文 §3.2.2 |
| 3 | **固定 (r=11, d=2) 贯穿三个不同分辨率的层**。Tab.5 只做了单尺度网格搜索，未考虑 1/4 与 1/8 分辨率上同一 (r,d) 对应的输入坐标系感受野相差一倍 | 论文 §3.2.3 / Tab.5 |
| 4 | **损失只约束像素与感知，没有利用去阴影的物理先验**。阴影本质是光照衰减（Related Work 里被反复引用的色度不变/梯度一致先验），且 ISTD+ 的 GT 非阴影区经线性回归校正过，直接做 L1 会让模型去拟合与任务无关的常量偏移 | 论文 §3.3 |

下面 5 项改进逐条对应这 4 点（第 4 点对应 E 与损失函数两组）。

---

## 二、改进清单

### A. MGRA：掩码引导的区域注意力（核心）

把"阴影 token 应该去看非阴影 token"从**隐式期望**变成**显式结构先验**，直接作用在注意力 logits 上：

| 组件 | 做法 | 参数量 |
|---|---|---|
| RIB（光照关系偏置） | 按 (query 是否在阴影, key 是否在阴影) 的 4 种组合，给每个 head 一个可学习标量 `B_illu[2q+k]` | 4·heads |
| NS-Gate（非阴影 key 门控） | 对阴影内 query，把邻域内同为阴影的 key 的 logits 减去可学习强度 `softplus(γ)`；**邻域全被阴影覆盖时自动失效**（fallback），避免信息通路被切断 | heads |
| RPB（相对位置偏置） | (Δy, Δx) **物理位移** → MLP → 每 head 标量 | 2·64+64+64·heads+heads |

为什么用软门控而不是硬屏蔽：**阴影只衰减光照，纹理与色度仍然保留**。把阴影 key 全部屏蔽会丢掉纹理线索，所以强度交给网络自己学（初值 0.5）。

为了让上述偏置能注入 logits，实现了 `NeighborhoodAttentionBias`（纯 PyTorch，见下）。

### A-0. 顺带的工程改进：可注入偏置的邻域注意力

natten 的 NA 是 CUDA 黑盒，无法叠加外部偏置，因此用 `F.unfold` + 分块 bmm 重新实现了等价的 NA：

* 内部区域与 natten **数值误差 ~1e-7**（`selftest.py` 第 1 项，4 组 (C,heads,K,d) 配置全部 PASS）；
* 与朴素零填充参考实现（含边界）误差 ~1e-7（第 2 项）；
* 按行分块（`chunk_elements` 自适应），避免 im2col 一次性展开导致显存爆炸；
* **不再依赖 natten**（natten 的 CUDA 轮子向来是复现的最大障碍），但仍可用 `--attn_impl natten` 切回官方实现；
* RPB 改用物理位移参数化，因此对 dilation 敏感（natten 的表是按 kernel 网格索引，d=1/d=2 共用一张表）。

代价：im2col 版比 fused kernel 慢一些。若只想要 NSRC/FiLM/损失三项而不要 MGRA，用 `--attn_impl natten` 即可（此时 RIB/NS-Gate 自动关闭）。

### B. NSRC：非阴影区域上下文聚合

用**可分离的一维掩码空洞聚合**把"A 的假设"再显式实现一遍，作为区域注意力的并行通路：

```
对每条位置 i：沿 1×k 与 k×1 两条膨胀线，只对【非阴影】位置做平均 → 邻域非阴影上下文 ctx
res = ctx - x                    # 只含"来自外部非阴影"的信息
out = Conv1x1(res) * 门控([x, res])
```

* 复杂度 O(4k)/token，比区域注意力 O(k²) 低一个量级；
* 分子分母都"中心置 0 再 +x/+1"，天然避免除零，且全阴影邻域时输出恒为 0（不引入伪信号）；
* 1×1 卷积零初始化 → **初始为恒等**，可单独消融。

### C. IAFM：逐层 Mask-FiLM

`y = γ(m)·x + β(m)`，γ/β 由"阴影/非阴影"两套参数按 m 线性混合，等价于让网络在两类区域用不同统计量（条件归一化）。
参数 4C/层，初始化为恒等。原 RASM 的 `CATransformerBlock` 通过**继承**复用（`CATransformerBlockV2`），权重键名不变。

### D. 感受野尺度自适应

同一 (r=11, d=2) 在 1/4 与 1/8 分辨率上对应的输入坐标系跨度相差一倍。提供 `--dilations` 逐层配置：

* 默认 `2 2 2`（与论文一致，保证可复现）；
* 可选 `2 1 2`（感受野尺度自适应：让三个区域注意力层在原图坐标系下的跨度都约等于半个图）；
* FLOPs 完全不变，只是把算力重新分配。

### E. 输出残差门控

`y = x + res ⊙ (m_膨胀 + β(1-m_膨胀))`，β=softplus(可学习, 初值 0.1)。

含义：**去阴影只需修改阴影区（含半影过渡带），非阴影区只允许常值级的光照校正**。
这正好对应论文 §4.2.1 自己承认的 SRD 上界问题，也兼容 ISTD+ 的 GT 做过线性回归校正这一数据集特性（否则强迫非阴影区逐像素相等会有偏）。

---

## 三、损失函数：把物理先验写进目标

`RASMv2Loss` = 包装既有 `L1_Vgg_losses`（不改动它）+ 三项任务损失：

| 损失 | 做法 | 针对的问题 |
|---|---|---|
| **半影区加权** `BoundaryLoss` | 对 `dilate(m)-erode(m)` 的半影带额外加权，用 1/w̄ 归一化保持量级 | 伪影/残留阴影主要出现在阴影边界 |
| **非阴影一致性** `NonShadowConsistencyLoss` | 对 (out−gt) 先做大窗口均值滤波，只惩罚**局部不一致** | ISTD+ 的 GT 非阴影区是线性回归校正过的，直接 L1 会逼模型拟合一个与任务无关的常量偏移 |
| **色度一致性** `ChromaticityLoss` | 监督归一化色度 `c = I/ΣI`（只在亮度足够处） | 阴影是光照衰减而非材质改变：改变强度、不改变色度（Finlayson 色度不变先验），可抑制偏色 |

权重默认 `w_boundary=1.0, w_ns=0.5, w_chroma=0.2`，均可在脚本里调。

---

## 四、代价

| preset | 参数量 (embed_dim=32) | 说明 |
|---|---|---|
| `rasm` | 5.16M | 等价论文 RASM（少了 natten 的 RPB 表，换成本实现的小 MLP） |
| `mgra` | 5.16M | +~0.01M（RIB/NS-Gate/FiLM 关闭时几乎无新增参数） |
| `film` | 5.17M | +0.01M |
| `nsrc` | 5.90M | +0.74M 都是 1/4、1/8 分辨率上的 1×1 卷积，**FLOPs 几乎不变** |
| `full` | 5.91M | 三项全开 |

FLOPs：MGRA 的偏置是逐 (query,key) 标量加法，NSRC 是 O(4k)，两者都远小于区域注意力本身的 O(k²)；新增损失不涉及前向。
因此相对原 RASM（5.2M）的代价主要是 NSRC 的 +0.7M 参数，若在意体量可关掉 NSRC（`--preset mgra` 或 `film`）。

---

## 五、用法

```bash
cd /workspace/RASM

# 1) 自检（CPU 即可，验证与 natten 等价 / 各 preset / 梯度通路 / 官方权重兼容）
OMP_NUM_THREADS=1 python -u improved/selftest.py

# 2) 训练（与既有脚本并列，参数见脚本顶部）
bash improved/train_ISTD_v2.sh
bash improved/train_SRD_v2.sh

# 3) 测试
bash improved/test_ISTD_v2.sh
bash improved/test_SRD_v2.sh

# 4) GPU 环境：在官方权重上续训 + 验证（一键脚本）
bash improved/improved_try.sh                 # 默认 ISTD / preset=full / 300 epoch
bash improved/improved_try.sh srd mgra 200
NEPOCH=20 bash improved/improved_try.sh istd full          # 冒烟
INIT_CKPT=./log/RASMv2_istd/<run>/models/model_best.pth NEPOCH=600 bash improved/improved_try.sh istd full
```

### 消融

`--preset` 每套只开一类改进，可直接与 `rasm`（= 论文 RASM）对照：

```bash
--preset rasm   # 基线：等价于论文 RASM，且可直接加载官方 AISTD.pth / SRD.pth
--preset mgra   # 只用 A 掩码引导区域注意力
--preset nsrc   # 只用 B 非阴影上下文
--preset film   # 只用 C Mask-FiLM
--preset rgate  # 只用 E 残差门控
--preset full   # 全开（默认）
```

### 在官方权重上继续训练

RASMv2 的拓扑与属性命名与既有 RASM 完全一致，因此可直接加载官方权重：

```
[load_weights] 缺失(将随机初始化) 24 项, 如: ['encoderlayer_2.blocks.0.attn.rpb.0.weight', ...]
[load_weights] 结构对不上的冗余项 0 项 -> PASS
```

缺失的 24 项全部是 v2 新增的 RPB 参数（6 个 MGRA block × 4），其余全部命中。
用法：训练脚本里加 `--resume`，并把 `PRETRAIN_WEIGHTS` 指向 `./log/RASM_istd_demo/models/AISTD.pth`。

> **续训的两个坑**（`improved_try.sh` 已自动规避）：
> 1. 官方 ckpt 自带 `epoch`（ISTD=900 / SRD=1000）→ `--resume` 会把 `start_epoch` 顶到 901/1001，
>    若 `NEPOCH<=1000`（SRD 就会中招）训练循环直接是空的；
> 2. 官方 ckpt 自带 optimizer 动量，按【官方模型的参数顺序】存储，而 RASMv2 参数数量不同
>    → `optimizer.load_state_dict` 会因 param group 长度不一致报错（错位加载更糟）。
>
> 因此续训时先把官方 ckpt 剥离成"纯权重 + epoch=0"，再用自己的 lr / 调度跑满 `NEPOCH` 个 epoch。
> 缺失项虽然更多（`preset=full` 是 101 项，含 NSRC/FiLM/残差门控），但它们全部是**零初始化/恒等初始化**，
> 起点行为与官方一致；`preset=rasm` 只缺 24 项 RPB，且 RPB 末层零初始化 → 与官方**完全等价**，可直接当基线。

---

## 六、建议的验证顺序（如要做完整实验）

1. `preset=rasm` + 官方权重 → 应复现论文数值（先确认基础设施没坏）；
2. `preset=rasm` 从零训练 → 得到本次环境下的基线；
3. 逐个 `mgra / nsrc / film / rgate` → 看单项贡献（尤其 SRD 上 `rgate` 对非阴影 RMSE 的影响）；
4. `dilations 2 1 2` vs `2 2 2` → 感受野尺度自适应是否真的有帮助；
5. `full` 与最优单项组合对比，检查是否有叠加收益；
6. 可视化 `bias_gen.illu` 的 4 个值：若 (q=阴影, k=非阴影) 显著高于 (q=阴影, k=阴影)，
   即证明"每个阴影区域聚合其邻域非阴影信息"这一假设被网络真正学会了
   （与论文 Fig.6 的注意力可视化形成互证）。

## 七、已知限制

* 本实现的 NA 用 im2col，训练速度慢于 natten 的 fused kernel；若瓶颈明显，
  可用 `--attn_impl natten` 退回官方实现（会同时失去 RIB/NS-Gate）。
* NSRC 的十字形聚合是 1-D 可分离的，是完整 k×k 邻域的近似（换取 O(4k) 复杂度）。
* 上述超参（NS-Gate 初值 0.5、损失权重、β=0.1）均为经验值，需要经过真实训练验证后再调。
