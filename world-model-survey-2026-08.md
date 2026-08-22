# 世界模型（World Model）技术全景：主线、关键节点、与「动作」相关的应用

- **日期**：2026-08-21
- **范围**：世界模型的定义演变 → 五条技术主线 → 关键研究节点时间线 → **以 action 为落点的应用**（重点）→ 可复现的开源仓库清单 → 对本目录既有工作的接口
- **配套阅读**：[`rl-after-sft-four-papers-2026-08.md`](./rl-after-sft-four-papers-2026-08.md)、
  [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md)、
  [`no-embodiment-to-real-robot-2026-08.md`](./no-embodiment-to-real-robot-2026-08.md)、
  [`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md)、
  [`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md)

> **证据分级**（沿用本目录既有约定）：
> - **[✓]** = 本次直接抓取 arXiv abs 页面核实标题/作者/日期/摘要及其中的数字；
> - **[✓c]** = 代码仓库侧实抓（README / 模型卡 / 许可证原文），**未在本地安装或跑通**；
> - **[~]** = 仅来自检索结果元数据或二手摘要，**未逐字核实**，引用其数字前请自行复核。
>
> **两条必须先说的话**：
> 1. 本领域 2026 年的论文产出速度已经超过任何人的核实速度。本文只对 **10 篇论文 + 4 个仓库**做了逐字核实（标 [✓]/[✓c]），
>    其余数十条全部标 [~]。**凡是要写进你自己论文的数字，请只用 [✓] 那批，或自己重抓一遍。**
> 2. 「世界模型」这个词在 2026 年已经被稀释到几乎无意义——它同时指代 latent RL 动力学模型、视频生成器、
>    可交互游戏引擎、以及 VLA 里的一个辅助 loss。**本文强制采用一个操作性定义**（见 §1），
>    不满足这个定义的工作一律不算，否则整篇报告会退化成论文列表。

---

## 0. 一页结论

**世界模型这条线，2026 年的真实状态是：作为「数据引擎」和「评测器」已经产业化落地；作为「策略本身」仍在争论；作为「真机 RL 的替身环境」刚刚跑通但还不稳。**

| # | 定位（世界模型拿来干什么） | 成熟度 | 代表工作 | 开源可用性 |
|---|---|---|---|---|
| A | **数据引擎**：生成合成轨迹去喂模仿学习 | **已产业化** | DreamGen / GR00T-Dreams [~]、GigaBrain-0 [~] | 高（NVIDIA/GigaAI 均放权重+代码） |
| B | **评测器**：替代真机 rollout 给策略排序 | **可用，但只敢用来排序** | WorldEval [~]、Ctrl-World [✓]、dWorldEval [~] | 中 |
| C | **替身环境**：在想象里跑 RL 后训练 VLA | **刚跑通，幻觉是主要瓶颈** | WMPO [~]、RAW-Dream [✓]、GigaBrain-0.5M [✓] | 中（WMPO 有官方实现） |
| D | **策略本体**：预测未来 + 直接出动作 | **争论中**（是否必须生成视频？） | Cosmos Policy [✓]、GE-Act [✓]、UWM [~] | 高 |
| E | **规划器**：latent 空间 MPC/CEM | **小场景稳，长程弱** | DINO-WM [~]、V-JEPA 2-AC [~]、TD-MPC2 [~] | 高（全部有官方代码） |

**三句话总结这一代的共同发现**：

1. **「世界模型」的价值目前主要不在预测本身，而在于它把「昂贵的真机交互」换成了「便宜的 GPU 算力」。**
   DreamGen 那条线的说法最直白：范式从「扩规模的人类遥操作数据」转向「扩规模的 GPU 算力」[~]。
   A/B/C 三个定位本质上是同一件事在数据端、评测端、训练端的三次投影。

2. **动作条件（action-conditioned）是分水岭，不是修饰词。**
   一个不吃动作的视频生成模型是「视频生成模型」，不是世界模型。2026 年那篇 manipulation 综述把世界模型
   **操作性地定义为 "action-conditioned predictive system"** [✓ 2606.00113]，这是目前最干净的切法，本文采用它。

3. **最大的开放问题是闭环误差累积与幻觉**，且这个问题在 C（替身环境）里最致命：
   策略会主动去找世界模型的漏洞（对抗性利用），于是 RAW-Dream 要加**双噪声验证**过滤不可信 rollout [✓ 2605.12334]，
   WoVR 干脆把「模拟器可靠性」本身当成中心瓶颈来做 [~]。

**并且必须泼一盆冷水（针对本项目）**：
你在 [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md) 里的核心痛点是**没有闭环评测**，
而 [`rl-after-sft-four-papers-2026-08.md`](./rl-after-sft-four-papers-2026-08.md) 的结论是**四篇 RL 论文全部要求闭环**。
世界模型看起来像是「不用建真机闭环也能拿到闭环」的捷径——**它不是**。
B 类（评测器）需要先有一批真机成功率去校准排序质量；C 类（替身环境）需要一个奖励模型，而奖励模型仍然要真机标注。
**世界模型降低的是闭环的边际成本，不是闭环的启动成本。** 详见 §8。

---

## 1. 定义：什么算世界模型，什么不算

本文强制采用 2026 年 manipulation 综述的操作性定义 [✓ 2606.00113]：

> **世界模型 = 一个动作条件的预测系统（action-conditioned predictive system）**：
> 给定当前观测 `o_t` 和一段动作 `a_{t:t+H}`，预测未来的某种表征 `ẑ_{t+1:t+H}`。

该综述用三个问题来组织整个文献 [✓]：

| 轴 | 问题 | 取值谱系 |
|---|---|---|
| **What** | 预测什么表征？ | 像素 → latent embedding → 深度/法向（3D） → 4D 点云 → 显式物理状态 |
| **How** | 预测怎么连到动作？ | 隐式（同一个网络里出动作）↔ 显式规划（MPC/CEM 在预测上搜索） |
| **When** | 在 pipeline 的哪个阶段用？ | 预训练（表征）→ 训练（辅助 loss / 合成数据）→ 适应期（RL 后训练 / 评测） |

另一篇 2026 综述 [✓ 2605.00080] 用的切法略有不同但兼容：世界模型与策略的耦合方式、
世界模型作为 RL 与评测的学习式模拟器、以及机器人视频世界模型从「想象式生成」到「可控、结构化、基础模型规模」的演进。

**被排除在外的**（避免概念稀释）：
- 纯 text-to-video 生成模型（Sora 类）——不吃动作，属于 §3.2 的**上游 backbone**，不是世界模型本身；
- 只做单帧 affordance 预测的模型——没有时间维度；
- 只在训练时加一个「预测下一帧」辅助 loss 的 VLA——这是 §3.3 的边缘情形，本文归入「隐式世界建模」并单独标注。

---

## 2. 关键研究节点时间线

按「概念突破」而非「SOTA 刷新」筛选。

| 年份 | 节点 | 为什么是节点 | 证据 |
|---|---|---|---|
| 2018 | **Ha & Schmidhuber, World Models** | VAE + MDN-RNN + 微型 controller，**controller 完全在梦里训练**，然后部署回真环境有效。奠定「在想象里学策略」的范式 | [~] |
| 2019 | **PlaNet（RSSM）** | 确定性记忆 + 随机不确定性的混合状态空间，解决了 latent 动力学的表征问题，成为之后数年的默认骨架 | [~] |
| 2020–2023 | **Dreamer V1→V3** | V3 用**一套固定超参**横跨 150+ 任务，Minecraft 从零挖钻石；发表于 Nature | [~] |
| 2022 | **DayDreamer** | 世界模型第一次在**物理机器人**上直接在线学（locomotion + manipulation），把「想象训练」从仿真拽到真机 | [~] |
| 2024.05 | **iVideoGPT** | 把视觉/动作/奖励统一成 token 序列做 next-token 预测，autoregressive 世界模型的可扩展形式；配套 MBPO 式 model-based RL | [~] |
| 2024.10 | **LAPA（Latent Action Pretraining）** | **第一个无需真值动作标签**的 VLA 预训练方法：VQ-VAE 从相邻帧学离散 latent action，再微调映射到真实动作。打通「互联网视频 → 动作」 | [~] |
| 2024.11 | **DINO-WM** | 不重建像素，直接在 **DINOv2 patch 特征**上建动力学；测试时 MPC 零样本解任务，无需专家演示/奖励/逆动力学。ICML 2025 | [~] |
| 2025.01 | **NVIDIA Cosmos（Predict/Transfer/Reason）** | 世界模型第一次以「基础模型平台」形态开源，permissive 许可 + 完整后训练文档 | [✓c] |
| 2025.04 | **UWM（Unified World Model）** | 用**分离的扩散时间步**统一 policy / forward dynamics / inverse dynamics / video prediction 四件事，可在 DROID 上预训练。RSS 2025 | [~] |
| 2025.04 | **TesserAct** | 第一个开源通用 **4D** 世界模型：联合生成 RGB-D-Normal 视频并重建 4D 场景。把「预测什么」从像素推到几何。ICCV 2025 | [~] |
| 2025.05 | **DreamGen / GR00T-Dreams** | 用视频世界模型合成 **neural trajectories**，单任务遥操作数据 → 22 个新行为的零样本泛化；数据量放大约 10× | [~] |
| 2025.06 | **V-JEPA 2 / V-JEPA 2-AC** | 100 万小时视频自监督预训练 + **62 小时机器人数据**做动作条件后训练，在两个从未采集过数据的实验室 Franka 上零样本 CEM 规划抓放 | [~] |
| 2025.08 | **Genie 3（DeepMind）** | 24 fps / 720p / 数分钟一致性的实时可交互世界；被 DeepMind 定位为通向 AGI 的踏脚石。闭源 | [~] |
| 2025.08 | **Genie Envisioner（AgiBot）** | 首个**动作驱动**的开源机器人世界模型平台：GE-Base（指令条件视频扩散）+ GE-Act（flow-matching 动作解码器）+ GE-Sim（动作条件神经模拟器）+ EWMBench | [✓] |
| 2025.10 | **Ctrl-World** | 位姿条件记忆检索 + **帧级动作条件**，DROID 95k 轨迹/564 场景训练，20 秒以上一致 rollout；用它合成数据做 SFT 使策略成功率提升 **44.7%** | [✓] |
| 2025.11 | **WMPO** | 在**像素空间**世界模型里做 **on-policy GRPO**，不碰真环境；出现自我纠错等涌现行为 | [~] |
| 2026.01 | **Cosmos Policy** | 视频模型「单阶段后训练」直接变策略，**零架构改动**：动作当 latent frame 生成。LIBERO **98.5%**、RoboCasa **67.1%** | [✓] |
| 2026.01 | **LingBot-World** | Apache 2.0 开源，分钟级长记忆 + 16 fps 亚秒延迟实时交互，含 **Act 控制版本** | [✓c] |
| 2026.02 | **DreamDojo（NVIDIA）** | **44k 小时第一人称人类视频**训练的通用机器人世界模型，用**连续 latent action 作为统一代理动作**跨本体；支持实时遥操作/策略评测/model-based 规划 | [✓] |
| 2026.05 | **World Action Models 综述** | 正式提出 **WAM** 作为统一范式：目标是 **未来状态与动作的联合分布**，而非只建模动作。给出 Cascaded / Joint 分类学 | [✓] |
| 2026.06 | **Cosmos 3** | 首个全开放**全模态**（语言/图像/视频/音频/**动作**）世界基础模型，two-tower Mixture-of-Transformers，一举把 VLM + 视频生成 + 世界模拟器 + WAM 收进一个架构。OpenMDW-1.1 许可 | [✓] |

---

## 3. 五条技术主线

### 3.1 主线一：Latent 动力学 + 规划（最老、最稳、最不「性感」）

**思路**：不生成像素，只在紧凑 latent 空间预测；策略要么在想象轨迹上做 actor-critic，要么测试时用 MPC/CEM 搜索动作。

| 工作 | 关键点 | 仓库 |
|---|---|---|
| **DreamerV3** [~] | 世界模型把观测编码成**离散分类表征**，预测未来表征与奖励；actor-critic 完全在想象轨迹上训练；固定超参跨域 | `danijar/dreamerv3` |
| **TD-MPC2** [~] | 104 个连续控制任务一套超参；**开源 300+ checkpoint**，含多任务离线 RL | `nicklashansen/tdmpc2` |
| **DINO-WM** [~] | 在冻结的 DINOv2 特征上建 ViT 动力学；**测试时优化**（MPC）零样本解任务，6 个环境无需专家演示 | `dino-wm.github.io` |
| **V-JEPA 2-AC** [~] | 冻结图像编码器 + 动作条件预测器（block-causal attention）；CEM 采样优化能量函数；跨实验室零样本 65–80% 成功率、约 16 s/动作 | Meta 开源 |

**为什么这条线仍然重要**：它是唯一一条**不依赖视频生成质量**的路线。视频生成模型的计算代价与幻觉问题在这里都不存在。
代价是：latent 空间不可解释，长程漂移难以诊断，且难以吃互联网视频。

**与本项目的关联**：DINO-WM 与 V-JEPA 2-AC 的共同信号是——**冻结的通用视觉表征 + 一个轻量动作条件预测头**就够了。
这和你在 [`vision-backbone-why-resnet18-2026-08.md`](./vision-backbone-why-resnet18-2026-08.md) 里讨论的 backbone 选择直接相关：
如果预测器建在冻结特征上，backbone 的选择就从「能不能学动作」变成「特征够不够几何」。

---

### 3.2 主线二：视频世界模型（生成式，动作条件）

**思路**：把视频扩散/自回归模型改造成吃动作的、因果的、可交互的模型。

**改造的三个技术难点**（Vid2World 把这些讲得最清楚 [~]）：
1. **因果化**：预训练视频扩散是双向全序列注意力，必须改成因果自回归才能交互；
2. **动作注入**：帧级动作条件（Ctrl-World 的做法 [✓]）vs 段级；帧级才能做精确控制；
3. **长程记忆**：Ctrl-World 用**位姿条件记忆检索** [✓]，LingBot-World 用 KV cache 分块推理达到分钟级 [✓c]。

| 工作 | 定位 | 开源状态 |
|---|---|---|
| **Cosmos Predict 2.5 / Transfer 2.5** | 2B/14B，text/image/video 条件统一；**含 robot 变体支持 action-conditioned 推理与训练**，2026.02 加入动作条件模型蒸馏 | [✓c] 代码 Apache-2.0，权重 NVIDIA Open Model License |
| **Cosmos 3** | 全模态 MoT，动作是**一等模态**；开放 code + checkpoint + 合成数据集 + 评测基准 | [✓] OpenMDW-1.1 |
| **LingBot-World** | Base(Cam) / Base(Act) / Fast / 4-bit 量化；480P & 720P；961 帧（1 分钟@16fps）；动作串形式如 `"w-10,a-10"` | [✓c] Apache 2.0 |
| **Genie 3** | 24fps/720p/数分钟一致性，实时交互 | [~] 闭源，Project Genie 需 AI Ultra 订阅 |
| **GE-Base（Genie Envisioner）** | 指令条件视频扩散，捕捉真实机器人交互的时空语义动力学 | [✓] `AgibotTech/Genie-Envisioner` |
| **Vid2World** | 通用「视频扩散 → 交互世界模型」改造方法，ICLR 2026 | [~] `thuml/Vid2World` |
| **DreamDojo** | 44k 小时第一人称人类视频；**连续 latent action 做统一代理动作**；实时（RTX 5090 + PICO VR 遥操作演示） | [✓] 论文；[~] 权重/训练代码/评测基准全开源 |

**一个必须知道的反例**：有 2026 工作直接质问「WAM 真的需要视频生成吗，还是图像编辑就够了」（ImageWAM [~]）。
这说明**视频生成的高成本是否必要，学界尚无共识**。如果你只需要下一个关键帧而非连续视频，成本可以降一个数量级。

---

### 3.3 主线三：World Action Model（WAM）——动作与预测的联合建模 ★重点

这是 2026 年**与 action 最直接相关**的主线，也是 OpenMOSS 那篇综述正式命名的范式 [✓ 2605.12090]：

> WAM 是**统一预测性状态建模与动作生成**的具身基础模型，目标是**未来状态与动作的联合分布**，而非只建模动作。

#### 3.3.1 Cascaded WAM（级联式：先预测，再提动作）

世界建模与动作生成**分离**。子类 [✓c OpenMOSS 分类学]：
- **像素空间**：先生成未来视频/子目标图，再用学习式或几何式方法提取动作
  - 代表：**UniPi**、**AVDC**、**Gen2Act**、**Dreamitate**、**SuSIE**、**VLP**、**TesserAct**、**Vidar**、**RIGVid**
  - 提取方式：逆动力学模型（IDM）、光流、或 3D 配准
- **隐式规划**：在 latent 表征上规划

**优点**：模块解耦，视频生成器可以独立用互联网数据扩规模。
**致命弱点**：**误差两次累积**（生成误差 + IDM 误差），且生成的视频可能在物理上不可执行。

#### 3.3.2 Joint WAM（联合式：一个模型同时出未来和动作）★当前主流

- **自回归类**：**GR-1 / GR-2**、**CoT-VLA**、**WorldVLA**、**F1**、**RynnVLA-002**、**VLA-JEPA** [✓c]
- **扩散类**：**PAD**、**VideoVLA**、**UWM**、**Cosmos Policy**、**DreamZero**、**FLARE**、**UD-VLA**、**Motus**、**CoVAR** [✓c]
- **MoE/MoT 类**（视频专家与动作专家分离但交叉注意力交互）：**GE-Act**、**Motus**、**LingBot-VA**、**BagelVLA** [~ NTUMARS 分类]

**Cosmos Policy 是这条线目前最干净的一击** [✓ 2601.16163]：

> 「在目标平台采集的机器人演示数据上做**单阶段后训练**，**无任何架构修改**。」
> 动作被当作 latent frame 生成，同时预测未来状态和 value 用于规划。
> LIBERO **98.5%**、RoboCasa **67.1%**，真机双臂任务上超过从零训练的强 diffusion policy、
> 基于视频模型的策略、以及 SOTA VLA。代码、模型、训练数据全部发布。

**这一条对本项目的意义最大**：它说明「预训练视频模型 → 策略」**不需要你改架构**，
只需要把动作塞进 latent frame 的位置。这与你在
[`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md) 里的 image-to-action 路线是同一族问题的两个解。

#### 3.3.3 跨本体的关键技巧：latent action

WAM 面临的最大数据问题是——**互联网视频没有动作标签**。三代解法：

1. **LAPA (2024)** [~]：VQ-VAE 从相邻帧学**离散** latent action → 预训练 latent VLA → 少量真机数据微调映射到真实动作；
2. **Moto** [~]：把 latent 抽象从「通用动作码」换成「**运动 token**」，当作硬件无关的运动语言；
3. **DreamDojo (2026)** [✓]：改用**连续 latent action 作为统一代理动作**，跨本体后再做机器人特定微调。

**趋势判断**：离散 → 连续，通用码 → 运动语义。连续 latent action 是目前跨本体迁移的主流答案。

---

### 3.4 主线四：世界模型作为 RL 替身环境（真机 RL 的绕道方案）★与本项目最相关

**问题设定**：RL 能突破模仿学习天花板，但需要海量真机交互，无法直接部署到物理机器人。
**解法**：用世界模型当低成本可控虚拟环境，在**想象 rollout** 里做后训练。

| 工作 | 核心机制 | 核实 |
|---|---|---|
| **WMPO** | **像素空间**（不是 latent）世界模型，与 web-scale 预训练的 VLA 特征对齐；改造 OpenSora 架构；轻量奖励模型给二元成败信号；**on-policy GRPO** + 动态采样。涌现自我纠错、终身学习 | [~] `WM-PO/WMPO` |
| **RAW-Dream** | **任务无关**世界模型（在 diverse task-free behaviors 上预训练）+ 现成 VLM 当奖励模型；**双噪声验证机制**过滤不可信 rollout。仿真与真机均验证 | [✓ 2605.12334] |
| **GigaBrain-0.5M** | **RAMP**（Reinforcement leArning via world Model-conditioned Policy）+ 10,000+ 小时机器人数据；相对 RECAP 基线在叠衣服/装箱/做浓缩咖啡等困难任务上提升**约 30%** | [✓ 2602.12099] |
| **WoVR** | 把**模拟器可靠性**当成中心瓶颈：可控动作条件视频建模 + 关键帧初始化 rollout + **世界模型与策略共演化** | [~] |
| **World-Env / VLA-RFT / World4RL / DiWA** | 同族早期工作 | [~ NTUMARS 列表] |

**这条线的三个已知失效模式**（必须知道，否则会白烧算力）：

1. **幻觉与长程误差累积** —— 闭环想象 rollout 必然漂移；RAW-Dream 的双噪声验证、WoVR 的共演化都是在打这个补丁 [✓/~]；
2. **策略对抗性利用世界模型** —— on-policy RL 会主动搜索世界模型的漏洞；WMPO 选像素空间而非 latent 空间，一部分理由就是像素空间的漏洞更容易被奖励模型识破 [~]；
3. **奖励模型仍需真机标注** —— 这是本文 §0 泼冷水的技术根据。世界模型消除了 rollout 成本，**没有**消除奖励定义成本。

**与 [`rl-after-sft-four-papers-2026-08.md`](./rl-after-sft-four-papers-2026-08.md) 的接口**：
那篇的四种解法（DSRL 绕过 / SAC Flow 修架构 / RL-Co 换目标 / OPD 蒸馏）解决的是「生成式策略怎么算 log-prob」；
**这一节解决的是完全正交的另一半——「环境从哪来」**。两者可以叠加：
例如 DSRL 的噪声空间 RL + WMPO 的想象环境，理论上可以在完全不碰真机的前提下做后训练。**我没有看到有人做过这个组合，这是一个空位。**

---

### 3.5 主线五：世界模型作为策略评测器 ★与本项目最相关

**问题设定**：真机评测慢、贵、不可复现；仿真评测有 sim-real gap。
**解法**：用世界模型在想象里跑策略，看成败。

| 工作 | 关键结论 | 核实 |
|---|---|---|
| **WorldEval** | 世界模型可作为真机策略评测的**可扩展、可复现、可靠代理**；能正确**排序**不同策略以及同一策略的不同 checkpoint；还能当**安全检测器**拦截危险动作 | [~ 2505.19017] |
| **Ctrl-World** | 无需真机测试即可准确**排序**策略性能；进一步用它合成数据做 SFT，新指令成功率 **38.7% → 83.4%**（另一处表述为提升 44.7 个百分点） | [✓ 2510.10125] |
| **dWorldEval** | **动作中心的离散扩散**世界模型：视觉观测、语言指令、动作块映射到统一 token 空间，**动作作为一等 token**。ICML 2026 | [~] |
| **GE-Sim** | 动作条件神经模拟器，产出高保真 rollout 供闭环策略开发 | [✓ 2508.05635] |

**必须注意的措辞**：上述工作反复用的词是 **rank（排序）**，不是 **predict absolute success rate（预测绝对成功率）**。
**这是这条线目前的能力边界**：它能告诉你 checkpoint A 比 B 好，**不能**告诉你 A 上真机会是 62% 还是 78%。

**对 [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md) 的直接建议**：
你缺的是「训练 loss 不预测真机成功率」的替代指标。世界模型评测器**正好**填这个位置——
它给的是**序**而不是**值**，而选 checkpoint 恰恰只需要序。**这可能是本报告对本项目最可执行的一条。** 落地路径见 §8.2。

---

### 3.6 旁支：3D/4D 与物理接地

纯 2D 视频世界模型不知道深度、接触、质量。三种补法：

1. **几何增强的生成**：**TesserAct** 联合生成 RGB-D-Normal 并重建 4D 场景，提供操作所需的深度与位姿 [~]；**EnerVerse** 把具身环境扩成可计算的 4D 世界模型 [~]；
2. **回到显式物理引擎**：**Genesis**（开源多物理引擎 + Nyx 渲染器，单张 RTX 4090 上 43M FPS [~]）、
   **Newton**（NVIDIA + Google DeepMind + Disney Research，基于 Warp、MuJoCo Warp 后端，**支持可微物理**，已交由 Linux Foundation 治理 [~]）；
3. **混合**：Cosmos Transfer 系列做 sim→real 的「世界到世界」迁移，弥补仿真的感知鸿沟 [✓c]。

**判断**：神经世界模型与物理引擎不是替代关系。2026 的实际配方是
**物理引擎保证动力学正确 → Cosmos Transfer 类模型补上真实感 → 视频世界模型补上多样性**。

---

## 4. 与「动作」直接相关的五种应用模式（含可复制的落地形态）

这一节是本报告的重点，按**你能不能今天就用**排序。

### 4.1 模式一：世界模型当数据引擎（成熟度最高）

**做法**：单张图 + 语言指令 → 视频世界模型生成「机器人完成新任务」的视频 → 用 IDM 或 latent action 反解出动作 → 得到 **neural trajectory** → 混进模仿学习训练集。

**代表**：DreamGen / GR00T-Dreams 的 4 阶段流水线 [~]。
- 关键数字：只用**单个 pick-and-place 任务、单个环境**的遥操作数据，实现 **22 个新行为**在已见与未见环境的零样本泛化 [~]；
- GR00T N1.5 的「数据金字塔」：真机数据 + 合成数据（含 neural trajectory 与仿真）+ web/人类视频，有效训练数据放大约 **10×** [~]。

**开源**：`NVIDIA/GR00T-Dreams`。底座是 Cosmos-Predict2 [~]。

**对本项目的可行性**：**高**。这条不需要闭环、不需要奖励模型、不需要真机 RL——
它只是给你的模仿学习加数据。与 [`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md)
关注的「失败数据」是互补的：一个补正样本多样性，一个补负样本信息。

### 4.2 模式二：世界模型当策略（WAM / Cosmos Policy 式）

**做法**：拿预训练视频模型，在你自己的演示数据上**单阶段后训练**，动作以 latent frame 形式生成。
**关键卖点**：**零架构修改** [✓ 2601.16163]。

**可复制形态**：

```
预训练视频 WFM (Cosmos-Predict2.5 / GE-Base / LingBot-World-Act)
        │
        └─ 单阶段后训练（你的演示数据）
                 │
                 ├─ 输出：未来帧（可视化调试用）
                 ├─ 输出：动作 latent frame → 动作
                 └─ 输出：value → 规划 / 重排序
```

**注意**：value 头是 Cosmos Policy 相对普通 VLA 的额外收益——它让你可以做 test-time 规划而不只是开环执行 [✓]。

### 4.3 模式三：想象里做 RL 后训练（§3.4）

**门槛**：需要奖励模型。三种拿法，按成本升序：
1. 现成 VLM 当奖励模型（RAW-Dream 的做法 [✓]）——最便宜，但 VLM 判成败的噪声会直接进 RL；
2. 在真机轨迹上训一个轻量二元成败奖励模型（WMPO 的做法 [~]）——需要真机标注，但一次投入长期复用；
3. 人工设计密集奖励——不推荐，回到了传统 RL 的奖励工程泥潭。

### 4.4 模式四：世界模型当评测器（§3.5）

**最小可行形态**（这也是我给本项目的第一优先级建议）：
用 Ctrl-World 类模型（DROID 上预训练，帧级动作条件）在你的场景上微调，
然后对每个候选 checkpoint 跑 N 次想象 rollout，用 VLM 判成败，得到一个**排序**。
**只信序，不信值。**

### 4.5 模式五：世界模型当遥操作 / 预测显示

**较冷门但工程价值高**：世界模型实时预测远端机器人的执行结果，抵消网络延迟（generative predictive display）。
- **RynnWorld-Teleop**：动作条件世界模型用于数字遥操作，单张 H100 上流式自回归解码达 **40+ FPS** [~]；
- **DreamDojo**：PICO VR 手柄 + 单机 RTX 5090 实时遥操作虚拟机器人 [~]；
- **LingBot-World**：16 FPS、端到端交互延迟 <1 秒 [✓c]。

---

## 5. 开源仓库清单（按可用性排序）

> 这一节回应「最好有开源仓库支撑」的要求。**[✓c] 表示我实抓了仓库页面；[~] 表示只从检索结果得到链接，未逐字核实内容。**

### 5.1 世界模型基础平台（有权重、有许可证、有后训练文档）

| 仓库 | 内容 | 许可证 | 核实 |
|---|---|---|---|
| `nvidia-cosmos/cosmos-predict2.5` | 2B / 14B；pretrained / post-trained / distilled；**robot 变体支持 action-only 条件与 action+image policy 模型**；2026.02 加入动作条件模型蒸馏 | 代码 Apache-2.0；权重 NVIDIA Open Model License | [✓c] |
| `nvidia-cosmos/cosmos-transfer2.5` | 建在 Predict2.5 之上，多空间控制输入条件的世界模拟（sim→real 用） | 同上 | [~] |
| `nvidia-cosmos/cosmos-predict1` / `cosmos-transfer1` | 上一代，文档更成熟 | 同上 | [~] |
| **Cosmos 3** | 全模态（含动作）MoT；code + checkpoint + **合成数据集** + 评测基准 | **OpenMDW-1.1（Linux Foundation）** | [✓ 论文] |
| `Robbyant/lingbot-world` | Base(Cam) / **Base(Act)** / Fast / 4-bit；480P & 720P；961 帧长程；动作串控制 | **Apache 2.0** | [✓c] |
| `AgibotTech/Genie-Envisioner`（及 `-V1`） | GE-Base + GE-Act + GE-Sim + EWMBench 全栈 | [~] | [✓ 论文] |
| DreamDojo | 全部权重 + 训练代码 + 评测基准；支持自有机器人数据后训练 | CC BY 4.0（论文） | [✓ 论文]，[~] 仓库 |

### 5.2 世界模型策略 / WAM

| 仓库 | 内容 | 核实 |
|---|---|---|
| Cosmos Policy（NVIDIA research page） | 代码、模型、**训练数据**全放 | [✓] |
| `WEIRDLabUW/unified-world-model` | UWM 官方 PyTorch 实现；DROID 预训练 → 下游微调 | [~] |
| `open-gigaai/giga-brain-0` + HF `open-gigaai/GigaBrain-0-3.5B-Base` / `GigaBrain-0.1-3.5B-Base` | 世界模型驱动的 VLA，RGBD 输入 + 具身 CoT 监督 | [~] |
| `OpenDriveLab/WholebodyVLA` | 全身 loco-manipulation 的统一 latent VLA，ICLR 2026 | [~] |

### 5.3 Latent 动力学 / 规划

| 仓库 | 内容 | 核实 |
|---|---|---|
| `danijar/dreamerv3` | 官方实现 | [~] |
| `nicklashansen/tdmpc2` | 单任务在线 + 多任务离线；**300+ checkpoint** | [~] |
| `dino-wm.github.io` | DINO-WM，ICML 2025 | [~] |
| `thuml/iVideoGPT`（`thuml.github.io/iVideoGPT`） | autoregressive 世界模型 + MBPO 式 model-based RL | [~] |
| `thuml/Vid2World` | 视频扩散 → 交互世界模型的通用改造，ICLR 2026 | [~] |
| `UMass-Embodied-AGI/TesserAct` | 4D 世界模型，ICCV 2025 | [~] |

### 5.4 RL / 评测

| 仓库 | 内容 | 核实 |
|---|---|---|
| `WM-PO/WMPO` | 像素空间世界模型 + on-policy GRPO 的官方实现 | [~] |
| `dworldeval.github.io` / WorldEval | 世界模型当真机策略评测器 | [~] |
| `tsinghua-fib-lab/WorldArena` | 统一 benchmark：同时评感知与**功能效用** | [~] |

### 5.5 物理引擎（世界模型的对照组 / 互补品）

| 仓库 | 内容 | 核实 |
|---|---|---|
| `Genesis-Embodied-AI/genesis-world` | 统一多物理引擎 + Nyx 渲染 + Quadrants 编译；单卡 RTX 4090 43M FPS | [~] |
| **Newton**（NVIDIA + DeepMind + Disney，Linux Foundation） | 基于 Warp，MuJoCo Warp 后端，**可微物理**，支持端到端梯度回传 | [~] |
| Isaac Lab | GPU 加速多模态机器人学习框架 | [~] |

### 5.6 综述与 Awesome 列表（入口，强烈推荐从这里开始）

| 仓库 | 内容 | 核实 |
|---|---|---|
| `OpenMOSS/Awesome-WAM` | WAM 综述配套；**Cascaded / Joint 分类学**、逐篇阅读博客、benchmark leaderboard 与性能趋势可视化；2026-07-01 破 1000 star | [✓c] |
| `NTUMARS/Awesome-World-Model-for-Robotics-Policy` | 按 **World Model as Policy / as Simulator（RL/评测）/ for Video Generation / Benchmarks / Datasets** 组织；757 star | [✓c] |
| `LMD0311/Awesome-World-Model` | 偏自动驾驶 + 机器人 | [~] |
| `opendilab/awesome-model-based-RL` | model-based RL 的经典入口 | [~] |
| `DravenALG/awesome-vla-wam`、`Li-Zn-H/AwesomeWorldModels`、`liujiuming123/Awesome-Interactive-World-Model` | 其他角度的列表 | [~] |

**推荐起手式**：先读 `NTUMARS` 那个（分类学最贴近「拿来干什么」），
再用 `OpenMOSS` 的 leaderboard 定位当前 SOTA，两个列表覆盖面互补。

---

## 6. 数据与评测基准

### 6.1 数据来源的四个层次（WAM 综述的划分 [✓ 2605.12090]）

| 层 | 来源 | 规模 | 动作标签 |
|---|---|---|---|
| 1 | 机器人遥操作 | 小（千~万小时级） | ✅ 真值 |
| 2 | 便携式人类演示装置 | 中 | ⚠️ 需重定向 |
| 3 | 仿真 | 大（算力上限） | ✅ 但有 sim-real gap |
| 4 | 互联网级第一人称视频 | **极大**（DreamDojo 用了 44k 小时 [✓]） | ❌ 需 latent action |

**常用具体数据集**：Open-X Embodiment、DROID（Ctrl-World 用了其中 95k 轨迹 / 564 场景 [✓]）、
BridgeData v2、AgiBot-World（EWMBench 的数据源 [~]）、RT-1/1X、CALVIN、RoboMimic、MetaWorld [~ NTUMARS 列表]。
2026 那篇 manipulation 综述系统梳理了 **34 个** manipulation 数据集 [✓ 2606.00113]。

### 6.2 评测基准

| Benchmark | 评什么 | 核实 |
|---|---|---|
| **EWMBench** | 场景（scene）/ 运动（motion）/ 语义（semantics）三维度；数据源 AgiBot-World，每任务 4–10 个原子动作的多步序列 | [~] |
| **WorldModelBench** | 指令遵循（0–3）+ 常识（帧质量/时序质量）+ **物理遵守**（牛顿定律、质量守恒、流体、穿模、重力五类违规）；350 实例 / 7 域 / 56 子域 | [~] |
| **WorldArena** | 统一评**感知**与**功能效用**两个维度 | [~] |
| **RoboArena** | Cosmos 3 用来报告 policy 性能的榜 | [✓ 提及] |
| **SC3-Eval** | 用自洽视频生成评机器人基础模型 | [~] |

**一条方法论警告**：有 2026 立场论文直接问「世界模型到底该怎么为**具身决策**评测」[~ 2606.15032]，
论点是**视觉保真度指标（FVD/PSNR）与下游控制性能不相关**。
如果你要在论文里报世界模型质量，**光报 FVD 会被审稿人打**——必须报下游任务成功率或至少报排序相关性。

---

## 7. 未解决问题（也就是可发论文的空位）

按我判断的「有空位 × 可做性」排序：

1. **闭环幻觉与长程漂移** —— 全领域第一痛点。当前补丁（双噪声验证 [✓]、共演化 [~]）都是启发式，**没有原理性方案**。
2. **接触与力的建模** —— 2026 manipulation 综述明确把 contact modeling 列为新兴挑战 [✓ 2606.00113]。
   视频世界模型看不见力，而 manipulation 的成败大量取决于接触。这是 §3.6（物理引擎）存在的根本理由。
3. **动作对齐（action alignment）** —— 生成的视频「看起来对」但对应的动作序列在运动学上不可执行 [✓ 2606.00113]。
4. **视频生成是否必要** —— ImageWAM 那类质疑 [~]。如果关键帧就够，整个领域的算力预算可以砍一个数量级。**这个问题目前是开放的，且做出来影响很大。**
5. **闭环评测基准的缺失** —— 综述自己承认 closed-loop evaluation benchmarking 是空白 [✓ 2606.00113]。
6. **世界模型 RL 与生成式策略 RL 的组合** —— §3.4 末尾指出的空位：DSRL 式噪声空间 RL + 想象环境，**我没找到有人做过**。

---

## 8. 对本项目的落地建议

> 基于本目录既有笔记的现状（`env_eval_freq=0`、无 rollout 成功率记录、真机成功率 60–70%）。

### 8.1 不要做的事

- **不要**上来就搞世界模型 RL（§3.4）。它需要奖励模型，而奖励模型需要真机标注——你现在连成功率记录都没有。
- **不要**自己从零训世界模型。2026 年有至少 4 个 Apache-2.0 / OpenMDW 级别的开源底座（§5.1），从零训是纯浪费。
- **不要**用 FVD 之类的生成质量指标汇报世界模型好坏（§6.2 的警告）。

### 8.2 第一优先级：世界模型当评测器（模式四）

**理由**：它是唯一一个**不需要奖励模型、不需要真机闭环就能立刻产生价值**的用法，且正好命中
[`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md) 的核心痛点。

最小路径：
1. 取一个帧级动作条件的开源世界模型（Ctrl-World 类，或 LingBot-World-Act / GE-Sim）；
2. 在你的场景数据上微调（这一步不需要成功率标注，只需要 `(obs, action, next_obs)` 三元组——**你已经有了**）；
3. 每个 checkpoint 跑 N 次想象 rollout，用现成 VLM 判成败；
4. **只用来排序 checkpoint**，不报绝对成功率；
5. 用少量真机 rollout（20–30 次）校准这个排序的可信度——这是必须付的一次性成本。

**预期收益**：把「训练 loss 不预测真机成功率」换成「想象成功率排序 vs 真机排序的 Spearman 相关」，
这是一个可以写进论文的诊断指标。

### 8.3 第二优先级：世界模型当数据引擎（模式一）

**理由**：不需要闭环，纯粹给模仿学习加数据，与 [`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md) 互补。
用 `NVIDIA/GR00T-Dreams` 的流水线，底座换成 Cosmos-Predict2.5 的 robot 变体。

**风险**：生成轨迹的动作可能不可执行（§7 问题 3）。**必须**加一道运动学可行性过滤，否则会污染训练集。

### 8.4 第三优先级（论文角度）：§7.6 那个空位

DSRL（噪声空间 RL，基座当黑盒）+ WMPO（像素空间想象环境）的组合。
两边都有官方开源实现，两边都在你已经精读过的范围内
（DSRL 见 [`rl-after-sft-four-papers-2026-08.md`](./rl-after-sft-four-papers-2026-08.md)）。
**这是本报告里唯一一条我认为可以直接变成一篇论文的建议。**

---

## 9. 参考文献

### 9.1 已逐字核实 [✓]

1. **World Action Models: The Next Frontier in Embodied AI** — Siyin Wang, Junhao Shi, Zhaoyang Fu, et al. arXiv:2605.12090, 2026-05-12. https://arxiv.org/abs/2605.12090
2. **World Model for Robot Learning: A Comprehensive Survey** — Bohan Hou, Gen Li, ..., Pieter Abbeel, Jitendra Malik, Yilun Du, Jianfei Yang. arXiv:2605.00080, 2026-04-30. https://arxiv.org/abs/2605.00080
3. **World Models for Robotic Manipulation: A Survey** — Fangyuan Wang, Ziyuan Wang, et al. arXiv:2606.00113, 2026-05-27. https://arxiv.org/abs/2606.00113
4. **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning** — Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, ..., Chelsea Finn, Jinwei Gu. arXiv:2601.16163, 2026-01-22. https://arxiv.org/abs/2601.16163
5. **Cosmos 3: Omnimodal World Models for Physical AI** — NVIDIA（294 作者）. arXiv:2606.02800, 2026-06-01（v4 2026-06-23）. https://arxiv.org/abs/2606.02800
6. **Reinforcing VLAs in Task-Agnostic World Models (RAW-Dream)** — Yucen Wang, Rui Yu, Fengming Zhang, et al. arXiv:2605.12334, 2026-05-12. https://arxiv.org/abs/2605.12334
7. **DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos** — Shenyuan Gao, ..., Jitendra Malik, Pieter Abbeel, Ming-Yu Liu, Yuke Zhu, Joel Jang, Linxi Fan. arXiv:2602.06949, 2026-02-06. https://arxiv.org/abs/2602.06949
8. **GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning** — GigaBrain Team (Boyuan Wang et al.). arXiv:2602.12099, 2026-02-12. https://arxiv.org/abs/2602.12099
9. **Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation** — Yue Liao, Pengfei Zhou, Siyuan Huang, et al. arXiv:2508.05635, 2025-08-07（v3 2025-11-04）. https://arxiv.org/abs/2508.05635
10. **Ctrl-World: A Controllable Generative World Model for Robot Manipulation** — Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, Chelsea Finn. arXiv:2510.10125, 2025-10-11（rev. 2026-03-01）. https://arxiv.org/abs/2510.10125

### 9.2 仓库侧已实抓 [✓c]

11. **OpenMOSS/Awesome-WAM** — https://github.com/OpenMOSS/Awesome-WAM
12. **NTUMARS/Awesome-World-Model-for-Robotics-Policy** — https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
13. **Robbyant/lingbot-world** — https://github.com/Robbyant/lingbot-world
14. **nvidia-cosmos/cosmos-predict2.5** — https://github.com/nvidia-cosmos/cosmos-predict2.5

### 9.3 未逐字核实 [~]（引用前请复核）

15. Advancing Open-source World Models (LingBot-World). arXiv:2601.20540 — https://arxiv.org/abs/2601.20540
16. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985
17. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv:2411.04983（ICML 2025）
18. Latent Action Pretraining from Videos (LAPA). arXiv:2410.11758（ICLR 2025）
19. TesserAct: Learning 4D Embodied World Models. arXiv:2504.20995（ICCV 2025）
20. DreamGen: Unlocking Generalization in Robot Learning through Video World Models. arXiv:2505.12705
21. iVideoGPT: Interactive VideoGPTs are Scalable World Models. arXiv:2405.15223
22. Vid2World: Crafting Video Diffusion Models to Interactive World Models. arXiv:2505.14357（ICLR 2026）
23. WMPO: World Model-based Policy Optimization for VLA Models. arXiv:2511.09515
24. WorldEval: World Model as Real-World Robot Policies Evaluator. arXiv:2505.19017
25. GigaBrain-0: A World Model-Powered Vision-Language-Action Model. arXiv:2510.19430
26. EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models. arXiv:2505.09694
27. WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models. arXiv:2602.08971
28. How Should World Models Be Evaluated for Embodied Decision-Making? arXiv:2606.15032
29. GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. arXiv:2503.20523（Wayve）
30. Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets. RSS 2025 — https://weirdlabuw.github.io/uwm/
31. DayDreamer: World Models for Physical Robot Learning. arXiv:2206.14176（CoRL 2022）
32. Ha & Schmidhuber, World Models, 2018
33. Genie 3: A new frontier for world models — DeepMind blog, 2025-08
34. RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation. arXiv:2607.06558
35. Genesis — https://github.com/Genesis-Embodied-AI/genesis-world ；Newton（NVIDIA/DeepMind/Disney, Linux Foundation）
36. NVIDIA/GR00T-Dreams — https://github.com/nvidia/gr00t-dreams
37. WEIRDLabUW/unified-world-model — https://github.com/WEIRDLabUW/unified-world-model
38. WM-PO/WMPO — https://github.com/WM-PO/WMPO
39. open-gigaai/giga-brain-0 — https://github.com/open-gigaai/giga-brain-0
40. AgibotTech/Genie-Envisioner — https://github.com/AgibotTech/Genie-Envisioner

---

*本报告的 [~] 条目来自检索结果摘要，其中 2026 年的论文（26xx.xxxxx 系列）我只核实了 §9.1 列出的 10 篇。
2026 年该领域论文产出极快，检索摘要中的数字与我未核实的条目存在错误风险，写入正式论文前请逐条重抓。*
