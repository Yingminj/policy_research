# 为什么动作模型几乎都用 ResNet18？——三个假设的检验

> 撰写日期：2026-08-17
> 起因：[`vita-arch-ablation-2026-08.md`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §1
> 只查清了"LeRobot 里哪些主干可换"，没有回答"为什么大家都不换"。
> 本文是文献侧的补齐，实测数字全部来自本目录已有报告，标注了出处；外部数字标注了来源与核实状态。

---

## 0. 一页结论

你的三个猜想，命中率是 **0 / 1 / 2**。

**猜想一「因为要高频推理」——基本不成立。** 这是流传最广也最容易证伪的解释。
`patch-policy` 实测（Table 3）：ResNet-18 VQ-BeT **5.79 ms（173 Hz）**、ACT **8.63 ms（116 Hz）**、
冻结 DINOv2 ViT-S patch **10.99 ms（91 Hz）**、ViT-L/WebSSL **21.43 ms（47 Hz）**。
只有 OpenVLA-OFT 的 61.71 ms（16 Hz）真正掉出反应式控制区间，而那是 **7.61B 的 VLM**，
不是 ViT 本身的代价。你自己的实测同样指向这里：
[`ACT-layer-depth-analysis`](./experiment_report/act/ACT-layer-depth-analysis.md) §6 结论是
"在 4090 上层数怎么调都不是实时性瓶颈"，真正的延迟量级是 `M·N·d`（150 ms–2.5 s），
比 `T_infer` 高一到两个数量级。**动作分块（action chunking）本身就是为掩盖推理延迟设计的**，
它把延迟预算从"每帧"放宽到"每块"，因此把 8 ms 换成 11 ms 在架构上毫无意义。

**猜想二「因为视觉信息量少」——错误，而且方向反了。** 见 §2。
真实情况是：策略**表现出**对视觉的依赖很低，因为本体感觉（proprioception）是一条更好走的捷径；
而当你堵住这条捷径，视觉的信息量立刻显形（state-free 策略在高度泛化上 0% → 85%）。
另外你现有的度量方式会**系统性低估**视觉损伤——见 §2.3。

**猜想三「ViT 有结构劣势」——成立，但必须加一个限定词才准确。**
劣势不在 ViT 架构本身，而在 **「ViT × 从头训练 × 一百条演示」这个组合**。
Diffusion Policy 自己的消融（Table 5，robomimic square）把这件事说得再清楚不过：

| 主干 | 从头训 | 冻结预训练 | 微调预训练 |
|---|---:|---:|---:|
| ResNet-18 (IN-21k) | **0.94** | 0.58 | 0.92 |
| ResNet-34 (IN-21k) | 0.92 | 0.40 | 0.94 |
| ViT-B/16 (CLIP) | **0.22** | 0.70 | **0.98** |

**ViT 从头训 0.22（全表最差），ViT 微调 0.98（全表最好）。**
所以"动作模型用 ResNet18"这个现象的准确表述不是"ResNet 比 ViT 好"，而是：

> 这一代动作模型选择了「视觉编码器与策略头端到端联合训练」这一范式，
> 而 **ResNet-18 是在一百条演示的数据量下唯一还能被从头训起来的架构**。
> ResNet18 不是最优解，是那个范式约束下的**唯一可行解**。

后面三节分别展开：为什么是这个范式（§1）、视觉信息量到底有多少（§2）、
ViT 的结构劣势具体是哪几条（§3）、以及反方证据与决策边界（§4–§5）。

---

## 1. 为什么是"端到端联合训练"这个范式

### 1.1 历史路径：一条从 2016 年拉到今天的直线

```
Levine et al. 2016  end-to-end visuomotor policies
      └─ spatial softmax（把特征图压成关键点坐标，保空间不保外观）
             ↓
robomimic (Mandlekar 2021)  ResNet-18 + spatial softmax，每相机一份，随机裁剪增广
             ↓
        ┌────┴─────────────────────┐
Diffusion Policy 2023          ACT / ALOHA 2023
ResNet-18 无预训练              ResNet-18（ImageNet 预训练）
GlobalAvgPool → SpatialSoftmax  保留 15×20 特征图当 token（DETR 血统）
BatchNorm → GroupNorm（为兼容 EMA）  每相机 300 token，4 相机 1200 token
             ↓
        LeRobot 把两者都收编，五个 policy 共用一行
        `if not vision_backbone.startswith("resnet"): raise`
```

这条线上的每一次选择都有当时的理由，但**理由从未被重新审计过**。
LeRobot 的那行 `startswith("resnet")` 校验器（act / act_delta / diffusion / vqbet / vita 五处逐字相同）
就是这条路径依赖的化石——[`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §1.3
实测过：ResNeXt / WideResNet 在两条构造路径上**都能跑通**，只是被那一行字符串挡住。
换句话说，连"必须是 ResNet"这个约束本身都不是设计决策，是**没人去改**。

### 1.2 范式约束：编码器必须可训练

有人会问：那为什么不干脆冻结一个大预训练模型？
两条独立证据说明**冻结在这个范式里是行不通的**：

1. **Diffusion Policy Table 5 的中间那列**：冻结 ResNet-18 掉到 0.58、冻结 ResNet-34 掉到 0.40。
   论文自己的措辞是"diffusion policy prefers different vision representation than what is offered
   in popular pretraining methods"。
2. **[Feature Extractor or Decision Maker (arXiv 2409.20248, ICRA)](https://arxiv.org/abs/2409.20248)**
   做得更彻底：提出 Visual Alignment Testing，证明在端到端训练的策略里，
   **视觉编码器本身在参与决策**，而不是纯粹做特征提取；
   用 OOD 预训练编码器（编码器丧失这个能力）的模型平均掉 **42%**。

这两条合起来给出了范式的硬约束：**编码器必须接受动作监督的梯度**。
一旦接受这个约束，架构选择就被"能不能在 100 条演示上从头训起来"筛选，
ViT 在这一关直接出局（0.22）。

### 1.3 数据量：差三到五个数量级

[X-Distill (arXiv 2601.11269)](https://arxiv.org/abs/2601.11269) 把这件事量化得最直白：
学界典型的手采数据是"tens to a few hundred manipulation trajectories"，
而 ViT 的归纳偏置缺失需要"exposure to massive datasets"来补。
ImageNet-21k 是 1400 万张图，你的 `express` 数据集是 **51 条 episode**。
中间隔着五个数量级。

这也解释了为什么**大规模 VLA 反而全用 ViT**——OpenVLA 的视觉塔是 DINOv2(~300M) + SigLIP(~400M)，
训练数据是 Open X-Embodiment 的 **97 万条 episode**、64×A100×15 天。
数据量过了阈值，归纳偏置就从资产变成负债（ViT 论文自己的原话是
"convolutional inductive bias is useful for smaller datasets, but for larger ones
learning the relevant patterns directly from data is sufficient, even beneficial"）。

**所以"动作模型用 ResNet18"和"VLA 用 ViT"不是矛盾，是同一条 scaling 曲线的两端。**
你的项目（51–186 条 episode / 任务）稳稳落在左端。

---

## 2. "视觉信息量少"这个猜想为什么是反的

### 2.1 观察到的低依赖是本体感觉捷径造成的

[Do You Need Proprioceptive States in Visuomotor Policies? (arXiv 2509.18644)](https://arxiv.org/abs/2509.18644)
的核心论证：本体感觉输入提供了精确的机器人构型，策略会**直接把绝对关节角关联到示教轨迹**，
形成捷径，从而过拟合训练轨迹、空间泛化崩溃。
移除本体输入（State-free Policy，只用双腕广角相机）后的真机结果：

| 泛化轴 | 有 state | State-free |
|---|---:|---:|
| 高度泛化 | 0% | **85%** |
| 水平泛化 | 6% | **64%** |

0% → 85%。这不是"视觉信息少"，这是"视觉信息在被一条捷径挤掉"。

**这条对你的 VITA 有直接含义。** [`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md)
§3.0 已经指出：`obs_encoder.projection` 是 `Linear(1552, 512)`，其中 state 占 16 维（约 1%），
视觉占 1536 维。听起来视觉主导，但那 16 维是**低噪声、与目标近乎恒等**的输入
（该报告 §3.2 实测 `action[t] − state[t]` 只有 0.0203 rad），
而 1536 维视觉是高噪声、需要学习的。在梯度意义上，1% 的宽度可以拿走 99% 的解释力。
S2（相对动作）之所以是对的方向，正是因为它把这条捷径**显式化**成结构，而不是让模型偷偷用。

### 2.2 视觉编码器不只是编码，它在做决策

§1.2 引的 2409.20248 是这条的正面证据。加上你自己的 §3.0 参数预算：
ResNet-18 主干 11.2M 参数在推理期真实运行，而 12.9M 的 flow net 是无条件的确定性映射。
**视觉通路是 VITA 推理期唯一携带外部信息的通路**，说它信息量少在因果上讲不通。

### 2.3 你现有的度量会低估视觉损伤（重要）

[`literature-review.md`](./literature-review.md) §1 用 **DINO CLS 余弦相似度**做压缩损伤的代理指标，
测得 PSNR ↔ CLS 余弦 Spearman ρ = 0.927。
[`patch-policy`](./patch-policy-2607.18236.md) 的 Table 4（Push-T 空间压缩阶梯）给了这个做法一记警告：

| patch 数 | 性能 |
|---:|---:|
| 256 | 0.69 |
| 64 | 0.52 |
| 16 | 0.53 |
| 4 | 0.51 |
| **1（≈CLS/全局池化）** | **0.48** |

**256 → 64 就掉掉了绝大部分，64 → 1 只再掉 0.04。** 也就是说 CLS 特征所在的位置，
正是这条曲线**已经跌到谷底之后**的平台。用一个位于平台上的量去测量损伤，
它对高频细节退化天然不敏感——而 JPEG / H.264 的损伤恰恰全集中在高频。

> **可执行的建议**（与 patch-policy 报告末节一致）：在 CLS 余弦之外补一个 **patch 级**指标
> ——逐 patch 特征余弦的**分布**（而非均值），或 patch 特征空间自相关的变化。
> 如果 patch 级指标对 CRF 的敏感度显著高于 CLS 级，就直接证明了当前代理指标偏乐观。
> 这本身是一个可发表的结论。

### 2.4 一个必须澄清的误读

[`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §2.4 第 5 条
用"去趋势残差的跨关节相关 ≈ 0（mean r = +0.029）"把视觉主干排除出**抖动**消融。
**那条结论是对的，但它的射程只到"视觉不是抖动的来源"**，
不能外推成"视觉信息量少"或"换主干没价值"。同一份报告的 §3.4 自己也写清楚了：
`AdaptiveAvgPool2d((1,1))` 把 7×7×512 压成 512、丢掉"在哪里"，
它推不动抖动，但**有望推动任务成功率——一个报告至今没有测量过的轴**。
这两件事不要混。

---

## 3. ViT 的结构性劣势：逐条拆

排序按"对机器人操作的杀伤力"，不是按知名度。

### 3.1 缺归纳偏置 → 低数据下直接崩溃（决定性）

局部性（locality）和平移等变性（translation equivariance）是卷积**写死在结构里**的先验，
ViT 必须从数据里学。在 100 条演示的量级上学不出来。
这是 DP Table 5 里 0.22 那个数字的全部解释，也是 X-Distill 整篇论文的出发点。

**代价的量级**：X-Distill 用 DINOv2 ViT-L 教师（304M）离线蒸馏到 ResNet-18 学生（11M，28× 更小），
在 34 个仿真基准上：

| 方法 | 仿真平均成功率 | 真机 5 任务平均 |
|---|---:|---:|
| **X-Distill (蒸馏后的 ResNet-18)** | **87.2%** | **75.6%** |
| ResNet-18 从头训 | 64.1% | 41.9% |
| DINOv2 微调 | 66.2% | 31.4% |
| PointNet / DP3（3D，特权点云） | 84.0% | — |
| π₀ (VLA) | — | 26.7% |

注意第二行和第三行：**在同一套小数据任务上，直接微调 DINOv2 ViT-L 反而不如从头训 ResNet-18**
（真机 31.4% vs 41.9%）。这是"ViT 结构劣势"最干净的一次实证——
不是因为 ViT 表征差（它是教师），而是因为**ViT 的适配过程在小数据上不稳**。

### 3.2 Patch 化引入离散相位不稳定 → 亚 patch 定位精度

ViT 把图像切成固定网格。图像整体平移几个像素，**token 归属就变了**——
同一个场景被表示成不同的 token 集合。
[Phase Marginalization for Patch-Grid Instability in ViTs (arXiv 2606.08132)](https://arxiv.org/pdf/2606.08132)
把这件事说得很准："dense prediction tasks require coordinate-stable outputs,
as pixel-level labels should not depend sensitively on arbitrary details of how an image is
partitioned before inference."

对分类，这是噪声；对**2 mm 容差的插线任务**，这是系统误差。
卷积的平移等变性在这里不是"锦上添花的先验"，是**任务本身要求的性质**。
DINOv2 ViT-B 的 patch 是 14×14 像素——在 224×224 输入下，
一个 patch 就是画面的 1/16 边长，末端执行器的精细位移完全落在单个 patch 内部。

### 3.3 全局池化 / CLS 是一道陡崖，不是一个旋钮

§2.3 的 Table 4 已经给了。补一句结构上的解释：
**这不是 ViT 独有的问题，但 ViT 生态默认这么用。**
社区消费预训练 ViT 的标准接口是 CLS token 或 avg pool，
于是"拿到了预训练权重却扔掉了它最值钱的部分"。
ACT 反而没有这个问题——它保留 15×20 特征图当 token；
Diffusion Policy 用 SpatialSoftmax；两者都在**主动避免**全局池化。
而你的 **VITA 恰恰在犯这个错**：`AdaptiveAvgPool2d((1,1))`，7×7×512 → 512/相机
（[`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §3.4）。

> 这条推论很硬：**在换主干之前，先把 VITA 的全局池化换掉**。
> 换成 ViT 但仍然取 CLS，等于从平台的一端走到平台的另一端，Table 4 说那只值 0.03。

### 3.4 O(N²) 在"高分辨率 × 多相机"下爆炸

ViT 自注意力对 patch 数是平方复杂度（O(H²W²) vs 卷积的 O(HW)）。
你的配置是 **3 相机 × 480×640**，这是最坏的一格。
[`ACT-layer-depth-analysis`](./experiment_report/act/ACT-layer-depth-analysis.md) §4.3 实测过同构的现象：
obs encoder 层的代价随 token 数**超线性**增长（分辨率减半时降 4.3×，含 N² 项），
"高分辨率 + 深 encoder 是最贵的组合"。
patch-policy 自己承认的最大局限也在这里：`O((T·P)²)`，T=10、P=256 时序列长 2560，
**多相机场景下的可扩展性未被验证**——而多相机正是真实操作平台的常态。

### 3.5 高范数 artifact token 污染稠密特征

[Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/pdf/2309.16588)：
CLIP / DINOv2 等模型在推理时会在**低信息量的背景区域**产生高范数 outlier token，
被模型挪用去存全局信息，导致注意力图不规则、**下游稠密任务退化**。
后续的 [Vision Transformers Don't Need Trained Registers (arXiv 2506.08010)](https://arxiv.org/abs/2506.08010)
给了免训练的缓解办法。

对分类无所谓（CLS 反而受益）；对"策略要从 patch 特征里读出物体在哪"是直接损伤。
**如果你走 patch-policy 路线，务必用带 register 的 DINOv2 权重**，这是一个免费的正确选择。

### 3.6 训练配方敏感 + 归一化层与 EMA 冲突

- ViT 需要 AdamW + 强增广 + warmup + layer scale 的整套配方，且对每一项都敏感。
  你的 ACT 仓库现状是**无 scheduler、无 warmup、无 grad clip、无 AMP、post-norm**
  （[`ACT-improvement-proposals`](./experiment_report/act/ACT-improvement-proposals-2026.md) §3）。
  在这个配方下换 ViT，大概率测到的是配方缺失而不是架构差异。
  这也是那份提案里 #3 排在 #9 前面的原因。
- Diffusion Policy 把 BatchNorm 换成 GroupNorm，是因为 **BN 的 running stats 与 DDPM 的 EMA 不兼容**。
  ViT 用 LayerNorm 天然没有这个问题——**这一条是 ViT 的优势**，为公平起见记在这里。
  但注意 LeRobot 里 `use_group_norm=True` 与预训练权重是**互斥的**（显式 raise），
  且分组数用 `num_features // 16`，对任意通道数不保证安全
  （[`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §1.4）。

### 3.7 部署侧：不是"慢"，是"跑不到加速器上"

这条比延迟本身更实在。**截至 JetPack 6.2，Jetson 的 DLA 不支持 Transformer / attention 层**；
YOLO / MobileNet / ResNet 这类经典 CNN 基本都 DLA 兼容。
同 FLOP 下 ViT 的实测延迟也普遍高于 CNN（有测量报告 5.00 GFLOP 的 ViT 是 4.95 GFLOP CNN 的 **1.75×**），
因为 attention 是访存受限而非算力受限。

对你目前的 4090 部署这条不适用；对将来上边缘平台，这是**换主干时必须先查的一格**。

### 3.8 域不匹配：腕部相机是 ViT 预训练分布之外

[StereoPolicy (arXiv 2605.09989)](https://arxiv.org/html/2605.09989v1) 的细节结论：
拼接冻结 DINOv2 的特征能提升性能，但**收益依赖视角——外部视角受益，wrist 视角反而变差**，
推测原因是 DINOv2 的预训练分布里没有近距离手腕视角。

这条对你是最直接可操作的一条：你的三相机是 **top + wrist_L + wrist_R**，
两个腕部相机在 ViT 预训练分布之外，一个外部视角在分布之内。
**分视角决策，不要一刀切**——这正是
[`ACT-improvement-proposals`](./experiment_report/act/ACT-improvement-proposals-2026.md) #9 已经写下的建议。

### 3.9 时间纠缠（Temporal Entanglement）

[The Temporal Trap / When PVRs Fall Short (arXiv 2502.03270, ICRA 2026)](https://arxiv.org/abs/2502.03270)：
时不变的预训练视觉表征在序列决策里会**把时间维压塌**——
静态抓取期的帧因为像素变化小而聚成一团，上升/下降运动产生近乎相同的特征，
造成"同一个观测对应不同动作"的歧义。论文实测策略成功率与
"latent space 能否捕捉任务进度线索"强相关。

**这一条与你 VITA 的断言 A 是同一个家族的病理。**
[`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §2.0 断言 A：
读出残差沿时间轴是白的（lag-1 −0.080 vs 训练集 +0.973）。
该报告把它归因于读出头缺乏时间结构（S1 带限读出头），这是对的；
但 Temporal Trap 提示了**第二条并列的解释路径**：如果视觉隐向量本身在时间上是纠缠的，
那么读出头再平滑也只是在给一个已经无时间结构的隐向量涂脂抹粉。
§2.2 里的 `n_obs_steps: 1 → 2` 那一臂正好能部分区分这两者——**这条轴的优先级应该往上提**。

---

## 4. 反方证据：ResNet18 也在付代价

上面九条不构成"应该继续用 ResNet18"。把反方证据摆齐：

| 来源 | 设置 | 结论 |
|---|---|---|
| Diffusion Policy Table 5 | robomimic square | **微调 CLIP ViT-B/16 = 0.98，全表最高**，且只训 50 epoch |
| [patch-policy](./patch-policy-2607.18236.md) Table 1 | BlockPush | ResNet-18 Patch ACT **0.15** vs WebSSL Patch VQ-BeT **1.68** |
| patch-policy Table 2 | 真机 Cable Insertion（2 mm 容差） | DINOv2 Patch **0.70** vs ResNet-18 Patch ACT **0.35** |
| X-Distill | 34 仿真 + 5 真机 | 蒸馏后 ResNet-18 **87.2% / 75.6%** vs 从头训 **64.1% / 41.9%** |
| [Open-TeleVision](https://arxiv.org/pdf/2407.01512)（*未逐条核对原文*） | 真机 pick / place | DINOv2 **92% / 88%** vs ResNet18 **74% / 58%** |
| [DINOv3-Diffusion Policy](https://arxiv.org/pdf/2509.17684)（*未逐条核对*） | PushT | 微调 DINOv3 **0.84**，相对 ResNet18 最多 +10 个百分点 |
| StereoPolicy（*未逐条核对*） | <100 条演示 | 低数据下 OpenCLIP-B/16、SigLIP-SO400M/14 **反而**优于 ResNet18 |

最后一行和 §3.1 表面矛盾，其实是同一条规律的两侧，区别只在**预训练权重的有无**：

```
从头训练：        ResNet18  ≫  ViT           （DP: 0.94 vs 0.22）
微调预训练：      ViT       >  ResNet18      （DP: 0.98 vs 0.92）
冻结预训练：      两者都差，ViT 相对好一点     （DP: 0.70 vs 0.58）
微调大 ViT + 小数据 + 弱配方：反而崩            （X-Distill 真机: 31.4% vs 41.9%）
```

**第四行是关键，也是最容易被忽略的一行。** 它说明"用预训练 ViT"不是一个可以随便打开的开关：
你既要有预训练权重，又要有能稳住它的训练配方和数据量，三者缺一，
结果可能比老老实实的 ResNet18 更差。X-Distill 正是为了绕开这个陷阱而生的——
**把 ViT 的知识离线搬进 CNN 的骨架里，只在 ImageNet 上做，不碰机器人数据**。

另有一条要记住的警告：
[Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale (arXiv 2606.14153)](https://arxiv.org/pdf/2606.14153)
用冻结主干嫁接诊断证明：**在小主干上胜出的编码器，换到大主干上排名会翻**。
所以任何"某编码器更好"的结论都必须绑定主干规模来读，包括本文引的每一个数字。

---

## 5. 决策边界：什么时候该换、换成什么

| 你的处境 | 该用 | 理由 |
|---|---|---|
| < 100 条演示，端到端联合训练，无配方改造预算 | **ResNet-18 从头 / ImageNet 微调** | DP Table 5 第一行；这仍是最稳的默认值 |
| 100–500 条，愿意补 warmup/cosine/clip/AMP | **X-Distill 式蒸馏 ResNet-18**，或微调 ViT-S | 保 CNN 骨架，拿 ViT 先验；不改延迟、不改部署链路 |
| 任务是高精度（≤ 5 mm 容差）或多物体空间推理 | **冻结 ViT 的 patch token + block-causal** | patch-policy Table 2/4：精度税减免最明显的场景 |
| 外部视角 + 腕部视角混合 | **分视角**：外部用冻结 DINOv2，腕部保留可训 ResNet18 | StereoPolicy 的实证结论 |
| 目标是边缘部署（Orin 级） | **ResNet-18，别动** | DLA 不支持 attention；ACT-layer §6 的外推表 |
| > 10⁴ 条 episode / 多具身预训练 | **ViT（SigLIP + DINOv2）** | OpenVLA / π₀ 那一端，你不在这里 |

**换主干之前必须先做完的三件事**（否则测到的是别的东西）：

1. **修相机身份嵌入。** [`ACT-layer-depth-analysis`](./experiment_report/act/ACT-layer-depth-analysis.md) §2.2
   实测三路相机的位置编码**逐字节相同**（`cam0 block == cam1 block: True`）。
   模型分不清哪 300 个 token 来自哪个相机。在这个状态下加强视觉主干，
   边际收益必然受限——先补 `nn.Embedding(n_cams, hidden_dim)`，2 小时。
2. **VITA 去掉 `AdaptiveAvgPool2d((1,1))`。** §3.3。换 ViT 却仍取全局特征等于白换。
3. **补齐训练配方**（grad clip / warmup+cosine / AMP）。§3.6。
   ViT 对配方的敏感度远高于 ResNet，缺配方去换 ViT 是在自找一个假阴性。

---

## 6. 落到你的项目上

**成本侧的实测（本目录已有，不需要重测）：**

| | 单帧推理 P50 | 参数 | 出处 |
|---|---:|---:|---|
| ACT + resnet18（3×480×640, 4090） | **8.99 ms** | 83.93M | [ACT-layer §4.2](./experiment_report/act/ACT-layer-depth-analysis.md) |
| ACT + resnet34 | 12.44 ms | 94.03M | 同上 |
| ACT + resnet50 | 14.41 ms | 97.00M | 同上（且 checkpoint 与 r18 不兼容） |
| VQ-BeT + 冻结 DINOv2 ViT-S patch | 10.99 ms（91 Hz） | 51.55M（可训 29.49M） | [patch-policy Table 3](./patch-policy-2607.18236.md) |
| VQ-BeT + 冻结 WebSSL ViT-L patch | 21.43 ms（47 Hz） | 334M（可训 30.34M） | 同上 |

**读法**：在你的 480×640×3 配置下，**resnet18 → resnet34 就要 +3.45 ms，
相当于加 8.6 层 decoder 的时间代价**（ACT-layer §4.2 原话）。
而 patch-policy 的 224×224 单相机 DINOv2 只要 10.99 ms。
**分辨率和相机数才是主导项，主干架构不是。** 如果真要上 ViT，
先把输入降到 224×224 再谈——这一步的收益比换架构大。

**建议的优先级**（与
[`ACT-improvement-proposals`](./experiment_report/act/ACT-improvement-proposals-2026.md)
的排序合并后）：

1. 免费/近免费，先做：相机身份嵌入（#4）、训练配方（#3）、
   VITA 的 adaLN-zero 修复与 S1a 带限读出头、去全局池化。
2. 一次重训能测的：相对动作（#2，先过 `prepare_relative_stats` 闸门，`express` 上 0.661 落在不确定带）。
3. **视觉主干（#9）排在这两者之后**，且第一步不是"换 ViT"，
   而是 **X-Distill 式蒸馏**——它保留 ResNet-18 的骨架、不改延迟、不改 checkpoint 形状、
   不碰部署链路，是本清单里风险/收益比最好的一条。
   若要走 patch 路线，用 patch-policy 的 block-causal mask + **带 register 的 DINOv2**，
   且**只对 top 相机用**，腕部保留可训 ResNet18。
4. 无论走哪条，**都必须离线评估**：
   [`vita-arch-ablation`](./experiment_report/vita/vita-arch-ablation-2026-08.md) §0 的约束仍然有效——
   P0-1 / P0-2 修好之前，100% 的 chunk 在下发前被换成零速 Hermite S 曲线，
   真机对任何两个变体给出的是同一条执行轨迹。

**顺带一条会影响论文的**：视觉主干这条轴上，
`literature-review` 关心的"压缩 → 动作质量"和 patch-policy 的 Table 4 有一个尚未被人做过的交叉——
**压缩损伤对 patch 级特征的影响是否显著大于对 CLS 级的影响**。
如果是（§2.3 预期如此），那么"现有压缩评估方法系统性低估了对稠密视觉策略的损伤"
本身就是一个独立结论，而且你手上已经有全部素材：九腿压缩网格 + 三个 backbone + 一个 100k ACT checkpoint。

---

## 7. 一句话回答你的三个问题

- **"是因为要更快的推理频率吗"** —— 不是。冻结 ViT-S patch 是 91 Hz，动作分块本来就为掩盖延迟而设计。
  唯一真实的部署顾虑是边缘平台的 **DLA 不支持 attention**，不是 FLOP。
- **"是因为视觉特征信息量少吗"** —— 不是，反了。视觉的表观低依赖来自本体感觉捷径
  （堵住后高度泛化 0% → 85%），而你当前的 CLS 级度量位于信息陡崖之下，**系统性低估**了视觉的作用。
- **"ViT 有什么结构劣势"** —— 九条（§3），但真正致命的只有第一条：
  **缺归纳偏置 × 一百条演示 × 从头训练 = 0.22**。
  其余八条是这条主因在不同侧面的展开。
  而这一条的解不是"回避 ViT"，是 **X-Distill / patch-policy 那种把 ViT 先验搬进可训小骨架的路线**。

---

## 参考文献

**核心消融与对照**
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/html/2303.04137v5) — Table 5 视觉编码器 3×3 消融（已直接核对）
- [X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning](https://arxiv.org/abs/2601.11269) — 2026-01，DINOv2→ResNet18 蒸馏，34 仿真 + 5 真机
- [Feature Extractor or Decision Maker: Rethinking the Role of Visual Encoders in Visuomotor Policies](https://arxiv.org/abs/2409.20248) — 编码器参与决策，OOD 预训练 −42%
- [Patch Policy: Efficient Embodied Control via Dense Visual Representations](https://arxiv.org/abs/2607.18236) — 本目录已有[阅读笔记](./patch-policy-2607.18236.md)

**ViT 结构性质**
- [Vision Transformers Need Registers](https://arxiv.org/pdf/2309.16588) — ICLR 2024，高范数 artifact token
- [Vision Transformers Don't Need Trained Registers](https://arxiv.org/abs/2506.08010) — 免训练缓解
- [Phase Marginalization for Patch-Grid Instability in Vision Transformers](https://arxiv.org/pdf/2606.08132) — patch 化的离散相位不稳定
- [An Image is Worth 16x16 Words](https://arxiv.org/pdf/2010.11929) — 归纳偏置 vs 数据规模的原始论述

**视觉表征在策略里的失效模式**
- [The Temporal Trap / When Pre-trained Visual Representations Fall Short](https://arxiv.org/abs/2502.03270) — ICRA 2026，时间纠缠
- [Do You Need Proprioceptive States in Visuomotor Policies?](https://arxiv.org/abs/2509.18644) — 本体感觉捷径，0%→85%
- [StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception](https://arxiv.org/html/2605.09989v1) — 分视角结论：外部视角受益，wrist 反而变差
- [DINOv3-Diffusion Policy](https://arxiv.org/pdf/2509.17684)
- [Encoder Winners Do Not Reliably Transfer Across VLA Backbone Scale](https://arxiv.org/pdf/2606.14153) — 编码器排名随主干规模翻转

**大规模一端（对照）**
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/html/2406.09246v2) — DINOv2+SigLIP ~700M 视觉塔，97 万 episode
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT/ALOHA)](https://arxiv.org/pdf/2304.13705)

**本目录内的关联报告**
- [`experiment_report/vita/vita-arch-ablation-2026-08.md`](./experiment_report/vita/vita-arch-ablation-2026-08.md) — §1 可替换主干矩阵、§3.0 参数预算、§3.4 全局池化瓶颈
- [`experiment_report/act/ACT-layer-depth-analysis.md`](./experiment_report/act/ACT-layer-depth-analysis.md) — §2.2 相机位置编码逐字节相同、§4.2 backbone 延迟扫描、§4.3 FLOP 分解
- [`experiment_report/act/ACT-improvement-proposals-2026.md`](./experiment_report/act/ACT-improvement-proposals-2026.md) — #3 训练配方、#4 相机嵌入、#9 视觉主干
- [`patch-policy-2607.18236.md`](./patch-policy-2607.18236.md) — Table 4 空间压缩陡崖
- [`literature-review.md`](./literature-review.md) — CLS 余弦作为压缩损伤代理指标

> **核实状态**：Diffusion Policy Table 5、X-Distill 主结果、patch-policy 全部表格已直接取自原文渲染。
> Open-TeleVision / DINOv3-DP / StereoPolicy 的具体数字来自检索摘要，**未逐条比对原文 PDF，引用前请复核**。
