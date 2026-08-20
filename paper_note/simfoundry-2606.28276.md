# SimFoundry：把一段手机视频变成可评测、可造数据的仿真场景

> 阅读笔记，撰写于 2026-08-20。
> 依据 arXiv v4（2026-08-05；v1 为 2026-06-26）的 LaTeXML HTML 全文，含全部附录 A–L。
> 代码部分依据 `NVlabs/SimFoundry` main 分支（最后 push 2026-08-19，首次开源 2026-08-14，184 stars）实抓的 `README.md` / `docs/INSTALL.md` / `scripts/pipeline/README.md` / `scripts/cfg/real2sim_cfg.yaml` 与 GitHub tree API（291 个文件）。**未在本地实际安装或跑通**，凡涉及运行时行为处均标注为文档/代码陈述。
> 本笔记按用户要求，额外展开两节：**§7 对「无本体研究」的适用性**、**§8 对「真机评测」的适用性**，以及 **§9 代码仓库实现难度评估**。

---

## Metadata

- **Title**: SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation
- **Authors**: Nadun Ranawaka, Josiah Wong, Wei-Lin Pai, Wei-Teng Chu, Tianyuan Dai, Masoud Moghani, Hang Yin, Yunfan Jiang, Wesley Durbano, Brandon Huynh, Yu Fang, Danfei Xu, Ruohan Zhang, Li Fei-Fei, Linxi Fan, Bowen Wen, Ajay Mandlekar, Yuke Zhu
- **Affiliations**: NVIDIA GEAR（通讯 amandlekar@nvidia.com）+ Georgia Tech（一作 nadun.ranawaka@gatech.edu）+ Stanford / UT Austin 体系
- **Venue / year**: arXiv:2606.28276v4（2026-06 首发，cs.RO），未见投稿会议信息；许可 CC BY-SA 4.0
- **Links**:
  - paper: https://arxiv.org/abs/2606.28276
  - project: https://research.nvidia.com/labs/gear/simfoundry/
  - code: https://github.com/NVlabs/SimFoundry （Apache-2.0，V0 已放出重建+增广+评测加载，**策略训练代码未放**）
- **Tags**: real2sim, sim2real, digital cousins, 场景重建, 策略评测, MimicGen, OmniGibson, VLA, π0/π0.5, GR00T

---

## 一句话总结

针对「真机训练与真机评测都贵到无法规模化」这一痛点，本文把一整条 real-to-sim 流水线（13 阶段、全部由现成基础模型拼装）做成了工程系统 SimFoundry：从**一段手机视频**零样本重建出物理可交互的数字孪生，再自动衍生出物体 / 布局 / 任务三个轴的 digital cousins；在 7 个任务 × 5 类策略上，仿真评测与真机评测的**平均 Pearson r = 0.911、MMRV = 0.018**（PolaRiS 基线低 0.59 以上），仿真训练的策略零样本上真机最高到 99–100% 成功率。**但它预测的是策略之间的排序与线性趋势，不是绝对成功率**——这是全文最需要被正确理解的一点（见「关键发现」与 §8）。

---

## 研究背景与动机

### 领域现状

机器人基础模型（π0、π0.5、GR00T N1.x、DreamZero）已经能覆盖相当宽的操作任务，但两侧同时卡住：

- **训练侧**：真机遥操作采集以月/年为单位（OXE、DROID 都是这个量级）。
- **评测侧**：作者引 [4] 指出，要做严谨的模型比较，跨任务需要**数千次 trial**。

仿真是公认的出路，且两条线各自已有成果：自动数据生成（MimicGen、RoboCasa 一系）与仿真评测相关性（SIMPLER）。**卡点不在仿真本身，而在「手工搭建与真实场景在视觉/几何/动力学上对齐的仿真环境」这件事本身不可规模化。**

### 现有痛点

作者在 Table 1 的系统对比里把痛点讲得很直白——现有 real-to-sim 系统各缺一块：

- **只做场景生成的**（DRAWER、R2R2R、以及 SAM3D 这类重建方法）：缺物理交互、缺任务定义、缺数据生成机制，闭不上 sim-to-real 的环。
- **只做仿真评测的**（SIMPLER、PolaRiS）：假设场景是人工调好的，聚焦短程原子操作，不支持自动生成多样物体/场景/任务用于训练。
- **ACDC（digital cousins 原始工作）**：有 cousin 思想，但不是端到端自动、不覆盖铰接体与背景。

### 核心矛盾

**要让仿真评测可信，仿真必须贴近某一个具体真实场景（digital twin）；要让仿真训练有用，仿真必须偏离那个场景以提供多样性（digital cousins）。** 这两个需求方向相反，过去被两拨系统分别处理。SimFoundry 的立论就是：先精确重建出 twin，再以 twin 为锚**受控地**偏离出 cousins，二者共用一套资产与任务表示，从而在同一个系统里同时服务评测与训练。

### 本文目标

- 输入约束到**单段 RGB 视频**（手机拍即可），输出是可交互、带物理参数、可被策略 rollout 的仿真场景。
- 成功标准三条：(a) 重建保真度超过 SAM3D；(b) 仿真评测与真机评测强相关且优于 PolaRiS；(c) 纯仿真数据训出的策略能零样本上真机。

### 切入角度

**系统/工程驱动，而非算法驱动。** 全文没有提出新的感知模型、新的策略架构、新的训练目标。贡献在于「把十余个现成基础模型按正确顺序接起来，并把每个接缝上的工程问题（尺度、位姿、穿模、碰撞几何、物理参数、任务谓词）都补上」。作者自己在 §4 明确说模块化的意义是「基础模型变强，流水线自动变强」。

### 核心 idea

关键观察是：**2026 年的开放基础模型已经强到可以把 real-to-sim 的每一个子问题分别解掉**（深度→DepthAnything3、分割→SAM3、图生 3D→Hunyuan3D-2.1/TRELLIS.2、位姿→FoundationPose、语义与物理参数→Gemini-Pro-3、网格分割→P3-SAM、凸分解→CoACD），过去缺的是把它们串起来并处理接缝的系统。一旦串起来，「重建一个场景」的成本降到**约 5 分钟/物体**，于是「为每个真实场景造一个专属仿真环境」从不可能变成可行——这才是能同时喂饱评测和训练的前提。

---

## 方法详解

### 整体框架

三阶段，共 13 个重建 stage（代码里就是 `stages/1b…13`）：

```
单段 RGB 视频
  │
  ├─ Extraction（提取）
  │    1b 抽帧 → 3 选代表帧 + 拟合支撑平面 → 4 统一世界系
  │    2  深度：DepthAnything3（单目/视频）或 FoundationStereo（双目）→ 点云 P_s
  │    5  Gemini 列物体 → SAM3 迭代分割 → 抠出后 RGB + 深度双 inpaint → 循环到无前景
  │    输出：每物体 RGB-D crop + mask
  │
  ├─ Generation（生成）
  │    6  Gemini-Image 超分/清理 crop
  │    7  Hunyuan3D-2.1（默认）或 TRELLIS.2 → 视觉网格 M_i
  │    8  位姿：点云对齐 + FoundationPose 精修 → p_i, s_i
  │    8b（可选）铰接：多视角渲染 → Gemini 列可动部件与关节类型 → P3-SAM 网格分割
  │        → Gemini 写调用 URDF API 的代码生成关节 → 仿真渲染运动视频 → critic VLM 打分
  │        → 迭代直到过阈值 → Gemini 估计 link 质量/关节摩擦/阻尼
  │    9/10 CoACD 生成碰撞几何；Gemini 估计质量与摩擦；打包 URDF
  │    11 PyBullet 里 spawn 全场景，每步强制清零速度以防解穿模爆炸，步进到静止 → 缓存 settled poses
  │    12/13 导入 USD → 输出 OmniGibson 场景 JSON
  │
  └─ Augmentation（增广，Pipeline B）
       object cousins: VLM 在图像空间改物体（几何/拓扑/外观）→ 重跑图生 3D → sim-ready
       scene cousins:  用语义空间谓词（OnTop / RightOf / Inside / Between / …）重排布局 + 从资产库加干扰物
       task cousins:   VLM 基于场景里的物体与 affordance 提出新任务 → 编译成 task YAML（谓词形式）
```

**背景（可选）**：3D Gaussian Splat 两条路线。自动路线复用同一段视频，SAM2 传播前景 mask + VOID 两遍 chunked video inpainting → 度量深度与位姿 → 深度监督 splat → 刚体变换桥接到仿真世界系。手动路线要求**再拍一段把前景物体物理搬走的视频**，然后交互式对齐。

### 关键设计

**1. 迭代式「分割—抠除—inpaint」分解（stage 5）。**
解决的是杂乱场景下的遮挡：每提取一个物体就把它从 RGB 和深度里 inpaint 掉，让下一轮分割面对一个「更空」的场景。代价是**误差累积**——附录 C 承认 Gemini image inpaint 偶发不稳定，会产出退化或重复的物体。我的判断是这是整条链路最脆的一环，因为它是串行的、且错误不可逆。

**2. 全部尺度/物理参数靠 VLM 猜，靠物理引擎兜底。**
质量、摩擦由 Gemini 查询给出；穿模由 PyBullet「步进到静止」解决。这是很聪明的偷懒：不追求物理参数正确，只追求**场景在初始化时物理稳定**。代价在 FAQ #2 里写明了——物体会相对拟合位姿漂移。对精细操作任务这是隐性误差源。

**3. 单一支撑平面假设。**
stage 3 只拟合一个支撑平面，这条假设一路贯穿到 stage 11 的稳定化。直接后果：**只支持 tabletop 场景**，多层货架、非平面环境不支持。作者在 §6 与附录 C 都承认了。代码文档也把它写进了拍摄须知（"One flat surface"，"其他台面不要入镜"）。

**4. Digital cousins 的三轴分解，是本文最可复用的概念资产。**
把「数据多样性」显式拆成 object / scene / task 三个正交轴，并对每个轴给出可自动化的生成算子（图像空间编辑 / 空间谓词重排 / VLM 任务提议）。作者明确说这三者不互斥，只表示主导变化轴。**这个 taxonomy 比这篇论文的任何单项技术都更有迁移价值。**

**5. 铰接体的 actor-critic 生成。**
关节参数由 VLM 写代码调 URDF API 生成，然后在仿真里动起来、渲染视频、交给另一个 critic VLM 打分并反馈，迭代到过阈值。这是把「VLM 生成 + 可执行验证 + 自我批判」这套范式落到了物理资产生成上，值得借鉴（对照本仓库 `no-embodiment-to-real-robot-2026-08.md` 里 S4 的可行性过滤思路——同样是「生成后用仿真验证」）。

### 数据生成与训练配方（附录 H，这是全文最容易被忽略但对复现最关键的一节）

- **每个任务先由人类用 JoyLo 遥操作在仿真里采 ~10–15 条演示。** 不是零人力。
- 然后用 **MimicGen** 增广：轨迹数量多样性 + 视觉域随机化（材质随机化、相机位姿随机化，DROID 上还有桌高随机化）。
- DROID：微调 DROID 预训练的 joint-position 版 π0 / π0.5，bs=256，lr=1e-5，10k steps，每 1k steps 在仿真里评一次，**选仿真最优 checkpoint 去真机评**。
- YAM：从零训 flow-matching 策略，观测 = 关节本体感 + 顶视固定相机 + 双腕相机，动作 = N-DoF 关节位置 + 归一化夹爪开合，40k steps，同样按仿真评测选 checkpoint。

---

## 实验关键数据

### 主实验 1：real-to-sim 评测相关性（Table G.1 / G.2）

协议（附录 J.1）：**每任务每策略 25 次 rollout**；每个物体的重置分布划成 5×5 网格共 25 个位置，每次 rollout 每物体独立无放回采一个位置并采一个旋转；同一任务下所有 checkpoint 用同一组位置。仿真与真机的**位置范围匹配但具体位置不逐一对应**（作者说是为了拿到分布级对应、避免相关性过拟合到本体感状态）。

| 任务 | 策略数 | 真机成功率 (%) | SimFoundry 仿真 (%) | Pearson r | MMRV |
|---|---|---|---|---|---|
| Stack Dishware（微调） | 3 | 100 / 100 / 40 | 34 / 64 / 0 | 0.883 | 0.000 |
| Store Marker（微调） | 3 | 48 / 60 / 32 | 4 / 20 / 0 | 0.915 | 0.000 |
| Throw Away Trash（微调） | 3 | 20 / 48 / 0 | 0 / 4 / 0 | 0.910 | 0.067 |
| Serve Fruits（零样本） | 5 | 0 / 72 / 4 / 40 / 8 | 4 / 80 / 20 / 32 / 12 | 0.960 | 0.016 |
| Cup in Bowl（零样本） | 5 | 88 / 100 / 68 / 92 / 100 | 56 / 92 / 40 / 92 / 92 | 0.907 | 0.016 |
| Marker in Cup（零样本） | 5 | 40 / 92 / 28 / 88 / 88 | 40 / 88 / 28 / 88 / 80 | 0.995 | 0.008 |
| Clear Table（零样本） | 5 | 0 / 40 / 0 / 8 / 16 | 12 / 36 / 0 / 28 / 28 | 0.810 | 0.016 |

（策略顺序：π0 / π0.5 / GR00T N1.6 / GR00T N1.7 / DreamZero）

PolaRiS 基线同协议同 checkpoint：Stack Dishware r=0.500，Store Marker 0.822，Throw Away Trash **无定义**（仿真全 0，方差为零），Serve Fruits 0.480，**Cup in Bowl −0.396**，Marker in Cup 0.512，**Clear Table −0.037**。SimFoundry 的优势是真实且巨大的。

**子任务评测**（附录 G.1.1）：利用仿真可任意重置状态的能力，从「前若干子任务已完成」的状态起评（例如 Store Marker 直接从抽屉已打开开始），与真机端到端成功率对照。三个微调任务的平均 Pearson 从 **0.902 → 0.951**。

### 主实验 2：sim-to-real 训练（Table G.4 / Figure G.2）

YAM 上物体 cousin 消融（Twin only vs Twin + 9 cousins）：

| 任务 | Sim Twin | Sim Cousins | **Real Twin** | **Real Cousins** |
|---|---|---|---|---|
| Stack Dishware（Twin → +9 Cousins） | 83 → 92 | 43 → 66 | **39 → 43** | **21 → 42** |
| Pot on Stove（Twin → +9 Cousins） | 85 → 100 | 17 → 93 | **91 → 99** | **14 → 64** |

Pot on Stove 的 Real Cousins 14 → 64 就是摘要里那个「50 点提升」。

多任务泛化（Table 2，13 个有数据任务 + 7 个 held-out 任务）：

| | π0.5-DROID | π0.5-FT | π0.5-DROID-FT |
|---|---|---|---|
| Sim | 30 | 51 | 61 |
| Sim – held out | 37 | 45 | 33 |
| Real | 28 | 45 | 46 |
| Real – held out | 26 | 29 | 26 |

co-training：π0.5 真机 Store Marker 60% → 92%；π0 仿真 Throw Away Trash +36%。

FAQ #6 给了很有说服力的对照：Store Marker 与 Throw Away Trash 上，π0-DROID 与 π0.5-DROID 未微调时**都是 0%**；Stack Dishware 上分别 52% / 48%，**纯仿真数据微调后到 100%**。

### 主实验 3：重建保真度（Table L.2，12 个 YCB tabletop 场景，按遮挡分 Easy/Med/Hard）

| 难度 | 指标 | SAM3D 零样本 | SimFoundry 零样本 | SimFoundry 调优（3min/物体） |
|---|---|---|---|---|
| Easy | F1↑ / Chamfer(m)↓ / 位置误差(m)↓ | 0.71 / 0.0081 / 0.016 | 0.92 / 0.0042 / 0.0060 | 0.99 / 0.0026 / 0.0041 |
| Med | 同上 | 0.66 / 0.0087 / 0.018 | 0.87 / 0.0047 / 0.0076 | 0.97 / 0.0033 / 0.0057 |
| Hard | 同上 | 0.68 / 0.0088 / 0.022 | **0.81 / 0.0091 / 0.018** | 0.93 / 0.0039 / 0.0073 |

**注意 Hard 一行**：零样本下 SimFoundry 的 Chamfer（0.0091）和位置误差（0.018）与 SAM3D（0.0088 / 0.022）几乎持平甚至更差，标准差还大得多（±0.0076 / ±0.018）。**强遮挡下「零样本超越 SOTA」这个说法只在 F1 上成立。** 3 分钟人工调优后才拉开差距——也就是说，**强遮挡场景下人在回路不是可选项。**

运行时（Table L.4，单张 RTX 3090 24GB）：2 物体 20 分钟、4 物体 34 分钟、9–11 物体 41–67 分钟，摊下来约 5 分钟/物体。自动背景 3DGS 流程**额外约 90 分钟/场景**。

### 关键发现

- **相关性高 ≠ 绝对值准。** Stack Dishware 真机 100% 对应仿真 34%，Store Marker 真机 60% 对应仿真 20%。仿真系统性地**大幅低估**微调策略的成功率（三个微调任务无一例外），却仍然保住了排序与线性趋势。这说明 sim-real gap 在这些任务上近似是一个**单调压缩变换**而非随机噪声。这是本文能成立的真正机制，也是它的使用边界。
- **子任务评测的收益来自「解耦瓶颈」。** 长程任务的端到端成功率被少数难子任务卡死，导致仿真侧集体趋零、方差消失、相关性不可靠。从中间状态起评把方差还回来了。**这是一个不依赖 SimFoundry、任何有仿真的团队都能立刻抄的协议改进。**
- **cousin 的收益高度集中在「held-out 物体/布局」上，而不是 twin 上。** Pot on Stove：Real Twin 91→99（+8），Real Cousins 14→64（+50）。也就是说 cousin 主要买的是**泛化保险**，不是原场景性能。
- **纯仿真数据对「预训练 checkpoint 完全不会的任务」价值最大**（FAQ #6 的 0% → 高分），对本来就会一点的任务是锦上添花。
- **重建质量随遮挡快速退化，且方差爆炸。** Hard 组零样本位置误差 0.018±0.018 m——标准差等于均值，意味着有些物体是完全放错的。

---

## 亮点与洞察

值得借鉴的东西，按可迁移性排序：

1. **三轴 cousin taxonomy（object / scene / task）**——一个清晰的数据多样性设计框架，与是否使用 SimFoundry 无关。
2. **子任务评测协议**——用仿真可任意重置的能力，把长程任务的评测方差救回来。对本仓库 `real-world-policy-evaluation-2026-08.md` 结论 3（「缺的不是指标而是统计」）是直接的、廉价的解法。
3. **5×5 网格无放回采样的评测协议**——把「初始条件」这个最大混淆变量显式控制住，且跨 checkpoint 固定。这个协议本身不需要仿真，真机上照样能做。
4. **「VLM 生成 → 仿真执行 → critic VLM 判定 → 迭代」的资产生成闭环**——铰接体生成用的这套，可以推广到任何「生成物需要物理验证」的场景。
5. **物理参数不求准、只求稳的务实取舍**——PyBullet settle 到静止再缓存位姿，用一个廉价步骤消掉整类初始化 bug。
6. **模块化本身作为贡献**——把每个 V_* 做成可换插槽，论文写作上把「依赖现成模型」这个弱点转化成了「随基础模型演进」的卖点。这个论证结构值得学。

---

## 局限与展望

### 作者承认的局限（§6 + 附录 C）

- 重度依赖现成基础模型，继承其全部失效模式。
- 所有 VLM 走**第三方远程服务**，同样输入非确定性输出；实际观察到 Gemini image inpaint 偶发产生退化/重复物体。
- 保真度受推断点云上界约束；单目输入下尺度与形状可能与真实不符。
- 铰接依赖 3D 网格分割，对图生 3D 的网格与有内部遮挡结构的物体很难。
- **单一平面假设 → 只支持 tabletop。**
- 自动背景流程省了二次拍摄，代价是约 90 分钟/场景的两遍视频 inpaint。

### 读者应警惕的局限（我的判断）

1. **相关性的统计强度被 Pearson 这个数字严重高估。** 三个微调任务的 r 是在 **N=3 个策略**上算的——三个点拟合一条直线，r=0.9 很容易；MMRV=0.000 在 N=3 时更是接近平凡（只需不发生排序反转）。零样本任务 N=5，好一些但依然很小。作者报的是**任务级 r 的均值**，不是在全部 policy-task 对上算的单一 r，也**没有任何置信区间或显著性检验**。这与本仓库 `real-world-policy-evaluation-2026-08.md` 结论 3 完全同构：问题不在指标选得对不对，在样本量。
2. **25 次 rollout 仍然不足以支撑这些结论。** 25 trials 下观测成功率 40% 的 95% Clopper–Pearson 区间约为 [21%, 61%]，宽 40 个百分点。Table G.1 里大量 4%、12%、20% 的差异都在噪声内。论文的相关性结论建立在**一层噪声很大的成功率估计**之上，全文未报 CI。
3. **摘要口径与正文不一致。** 摘要写「当在真实世界零样本评测 sim-trained 策略时，object/scene/task cousins 分别带来平均 17% / 21% / 40% 的提升」。但正文与 Figure G.2 caption 里，**task cousins 的 40% 明确是 "in simulation"**（13 个 task cousins 使 Throw Away Trash +60%、Store Marker +40%，均为仿真），scene cousins 的 21% 是「twin 场景 ~13% 与 cousin 场景 ~29%」的混合（且混了 sim 与 real）。只有 object cousins 的 17% 是明确的零样本 sim-to-real 数字。**引用这三个数字时必须回正文核对口径。**
4. **主实验用的背景不是论文主推的自动 3DGS 流程。** §4 脚注与附录 C 都说明：机器人实验里用的是 **Scaniverse（手机 app）扫出的 mesh 背景**，原因是生成的 3DGS 有近场裁剪问题、且渲染延迟高。也就是说 **0.911 这个数字不是端到端全自动流程跑出来的**，中间有一段是手机扫描 + 手动。
5. **「fully automated」的口径需要打折。** FAQ #3 承认机器人实验的场景是「自动流程 + 人工快速交互调优，几分钟迭代」；Table L.2 的强遮挡组显示零样本几何精度并不比 SAM3D 好；而代码里 `1_eval_policy_og_scene.py` 的开发者注释（见 §9.5）列了一整页手工调参项。**自动的是资产生成，不是「让仿真评测与真机对上」这件事。**
6. **重建保真度的评测有利于自己。** L.1 用的是 YCB 物体，**quasi-ground-truth 位姿本身是用 FoundationPose + 已知 CAD 网格估出来的**——而 FoundationPose 正是 SimFoundry 位姿模块用的模型（stage 8）。这不是严格的循环论证（GT 侧用了真 CAD 且从无遮挡视角估），但两侧共享同一位姿估计器的系统性偏差，会让 SimFoundry 的位置误差看起来偏低。SAM3D 侧还额外承受了坐标系/尺度对齐的转换损失。
7. **相关性只在 SimFoundry 自己重建的 7 个任务上验证。** 没有交叉验证：没有「在 A 场景标定的相关性能否外推到 B 场景」的实验。这意味着**每接一个新场景，相关性都是未经验证的假设**。
8. **策略侧全是 joint-position 控制、动作频率 15 Hz、DROID/YAM 两个本体。** 换本体、换控制模式（末端位姿、力控、柔顺）、换频率的结论完全未知。附录 I 还提到「仿真里的关节控制器增益调得更高以最小化跟踪误差」——**仿真里的执行链路被有意做得比真机更理想**，这一点对本仓库关心的执行链路问题（Hermite S 曲线、gripper 尺度）是个直接警告：SimFoundry 的仿真评测不会替你发现部署侧的执行失配。

### 可做的下一步

- 把 25 → 更大 rollout 数，或改用序贯检验；报置信区间。
- 把相关性从「任务级 r 的均值」改成在全部 policy-task 对上做混合效应模型，并报外推到新场景的相关性。
- 把单平面假设放开到多层/非平面。
- 让 cousin 生成受目标本体的运动学可行性约束（当前 cousin 只保 affordance，不保「机器人够得着」）。

---

## 相关工作与启发

| 维度 | ACDC | SIMPLER | PolaRiS | R2R2R / DRAWER | **SimFoundry** |
|---|---|---|---|---|---|
| 输入 | 单图 | 人工搭建 | 浏览器编辑器 + 外部重建工具 | 视频/扫描 | 单段视频 |
| 自动化 | 部分 | 无（手工调） | 需用户用 COLMAP+2DGS 自己重建再导入 | 场景生成自动 | 13 stage 全自动 + 可选人工微调 |
| 铰接体 | 否 | 否 | 否 | 部分 | 是（VLM actor-critic） |
| 用于评测 | 否 | 是（视觉匹配） | 是（需浅层微调策略才有相关性） | 否 | 是（**零样本，不微调策略**） |
| 用于训练 | 是（cousins） | 否 | 有限 | 有限 | 是（object/scene/task cousins + MimicGen） |

**vs PolaRiS（最直接的竞品）**：PolaRiS 的相关性依赖「用 10% PolaRiS 仿真数据 + 90% DROID 数据浅层共训策略」，SimFoundry 不动策略、完全零样本评测。作者把这个 co-trained checkpoint 从对比里剔除了，理由正当（协议不同），但这也意味着**两者的对比是「零样本 vs 零样本」，而 PolaRiS 在零样本下本来就不是设计工况**。这个对比对 SimFoundry 有利。

**vs ACDC**：digital cousins 概念来自 ACDC（代码也部分派生自 `cremebrule/digital-cousins`，README 明确致谢并标注 Apache-2.0 派生）。SimFoundry 的增量是把 cousin 从「物体实例随机化」扩展到 scene 与 task 两个轴，并全自动化。

---

## 7. 对「无本体研究」的适用性

按本仓库 `no-embodiment-to-real-robot-2026-08.md` §0 的三级分类：

**SimFoundry 属于 L-C（无真机采集，但本体模型在场），而且是 L-C 里最完整的一套。** 它不是 L-A——训练数据不是从人类视频来的，是从**仿真里的机器人遥操作**来的。

### 它确实解决了什么

1. **把采集环节从真机上彻底摘下来。** 附录 H 的配方是：人用 JoyLo / spacemouse / 键盘 / Oculus 在**仿真里**遥操作采 10–15 条演示 → MimicGen 增广 → 训练。全链路不需要真机在场。代码里 `s14_teleop` 支持 `device: [keyboard, spacemouse, oculus]`，一个几百元的 spacemouse 就能替代一整套真机遥操作台。
2. **绕开了 L-A/L-B 路线最痛的那三个 gap。** 无本体手持采集（UMI 系）的三大痛点是视觉 gap、运动学 gap、动力学 gap（见 `no-embodiment-to-real-robot-2026-08.md` 结论 3）。SimFoundry 全部规避：因为演示是**用目标机器人的 URDF 在物理引擎里采的**，关节限位、自碰撞、可达性天然满足；`5_generate_demos.py` 还用 **CuRobo 做自由空间运动规划**，生成的轨迹是运动规划器产出的、本身就可执行。**FeasibleCap 报告的「83% 帧不可行」这类问题在这条路线下不存在。** 这是 L-C 相对 L-A/L-B 的结构性优势。
3. **场景侧的无本体也解决了。** 过去 L-C 的最大成本是「搭仿真场景」，SimFoundry 把它降到 5 分钟/物体 + 一段手机视频。
4. **物体/布局多样性可以无限造。** object/scene cousins 是纯软件成本，这是真机采集永远做不到的。

### 它没有解决什么（对无本体研究者的真实约束）

1. **你必须有目标机器人的高质量 URDF/USD，并且它必须能在 OmniGibson 里跑起来。** 代码里机器人来自 `omnigibson.robots.REGISTERED_ROBOTS`，配置里硬编码了 `controllable__frankapanda__robot0/panda_link0/external_cam0_opposite` 这样的 prim path。**接入一个自定义本体（例如本仓库的 Marvin）不是配置项，是要往 OmniGibson 里加一个 robot class + USD 资产 + 控制器 + CuRobo 运动学配置的工程量。** 这是把「无真机」换成了「无真机但要有一份高质量本体模型 + 一个 Isaac Sim 环境」。
2. **动力学 gap 被搬家了，没被消除。** 附录 I 明确写：仿真里关节控制器增益**调得比真机更高**以最小化跟踪误差。也就是说仿真里的执行是被刻意理想化的。抓取默认走 OmniGibson 的 assisted grasping 抽象（代码注释里有 "Change assisted → physical grasping" 这一条调参项）。**接触力、柔顺、滑移、控制器带宽这些真机上真正杀人的东西，SimFoundry 不给你答案。**
3. **仍然需要人类遥操作。** 10–15 条/任务不多，但不是零；而且仿真遥操作有自己的技能门槛（无力反馈、视角受限）。
4. **依赖真机做一次性标定。** 见 §8。

### 对本仓库的具体含义

- 如果目标是「在没有真机的窗口期继续推进策略研究」，SimFoundry 这条路比手持夹爪 / 人类视频路线**技术风险低得多**——因为它把不确定性集中在「仿真是否像真机」这一个可测量的问题上，而不是分散在采集保真度、重定向、可行性过滤等一长串不可测环节上。
- 但它的准入门槛也高得多：你需要 Isaac Sim 级别的仿真栈 + 一个能跑的本体模型。仓库里已有 `marvin_description/{mjcf,urdf}` 与 `marvin_hardware_sim`，这是好起点，但 MJCF/URDF → OmniGibson robot class 的移植是**周级而非天级**的工作。
- **最务实的用法不是全盘采用，而是只取 Pipeline A/B 当资产工厂**：用它把真实场景变成一批带物理参数的 USD/URDF 资产，然后导入自己已有的仿真栈（`lerobot_vlahost/envs/`、`marvin_hardware_sim`），跳过 OmniGibson 依赖。资产格式是标准 USD + URDF，这条路可行。

---

## 8. 对「真机评测」的适用性

这是全文与本仓库最相关的部分，但结论需要非常小心。

### 它确实解决了什么

1. **把评测成本从「每次都烧真机」改成「一次性标定 + 后续在仿真里迭代」。** 这是范式上的正确方向，也与 `real-world-policy-evaluation-2026-08.md` 方案 6（real-to-sim）指向一致。
2. **它把「仿真评测能否预测真机」这个问题从信念变成了可测量的量**（Pearson r + MMRV），并给出了一套可复制的测量协议（附录 J.1）。
3. **相对 PolaRiS 是压倒性的改进。** PolaRiS 在两个任务上相关性是**负的**（−0.396、−0.037）——即在那些任务上，仿真评测比抛硬币还差。SimFoundry 全部任务 r ≥ 0.81。
4. **子任务评测这个协议是真正的增量**，且**不需要 SimFoundry 也能用**（只要你的仿真能重置到中间状态）。0.902 → 0.951 的提升来自把长程任务的方差救回来。
5. **它给出了 actionable 的诊断**，不只是一个分数：论文举例 GR00T N1.7 在精确抓取上强、π0.5 在语言跟随上强。这正是本仓库缺的「失效归因」。

### 它没有解决什么

1. **它不能替代真机评测，只能摊薄真机评测。** 相关性系数**本身就是用真机数据算出来的**——要建立 r=0.911 这个数字，作者做了 7 任务 × 3–5 策略 × 25 次的**真机** rollout，量级约 700+ 次真机试验。**没有真机，就没有相关性；没有相关性，仿真数字的可信度未知。** 对完全无真机的团队，SimFoundry 提供的是「一个大概率有用但无法验证的代理指标」。
2. **绝对值不可用，只有排序可用。** 再强调一次：Stack Dishware 真机 100% ↔ 仿真 34%。**任何形如「仿真成功率达到 X% 就可以上真机」的决策规则都会失效。** 能用的只有「A 策略在仿真里比 B 好，所以 A 在真机上大概率也比 B 好」。这限制了它的用途：**它是 checkpoint 选择器和消融裁判，不是验收工具。**
3. **统计强度不足以支撑「相关性已被证明」这个说法。** N=3~5 个策略、25 次 rollout、无置信区间。按本仓库 `real-world-policy-evaluation-2026-08.md` 结论 3 的标准，这篇论文自己也落在「19.8% 显著性」那个统计问题里。**不要把 0.911 当成一个已经落定的常数去引用。**
4. **相关性的可迁移性完全没验证。** 每个新场景、新任务、新本体，相关性都需要重新标定。论文没有做「跨场景外推」实验。
5. **它测不到部署链路的问题。** 仿真里的控制器是理想化的（增益调高、assisted grasping）。本仓库已确认的两个链路缺陷——下发合成 Hermite 零速 S 曲线而非策略原始输出、gripper 状态尺度约为训练范围的 1.25 倍——**SimFoundry 的仿真评测一个都发现不了**，因为它评的是策略输出，不是部署后处理之后的实际执行。这与 `real-world-policy-evaluation-2026-08.md` 方案 2 的定位是互补而非替代关系。

### 给本仓库的建议

**可以立刻抄、不需要 SimFoundry 的三样东西**（成本最低、杠杆最高）：

1. **5×5 网格无放回初始条件采样协议**（附录 J.1）。真机、现有仿真都能用。它解决的是「每次评测初始条件不可比」这个最大混淆源。
2. **子任务评测协议**（附录 G.1.1）。仓库已有 `marvin_hardware_sim`，能重置到中间状态就能做。这是目前唯一被量化验证过能提升长程任务评测可信度的手段。
3. **MMRV 指标**。比 Pearson 更贴近实际用途（我们要的是「选对 checkpoint」，不是「预测准成功率」），且对绝对值偏移不敏感——**正好绕开了 SimFoundry 绝对值不准这个问题**。

**需要投入才能用的**：Pipeline A/B 当资产工厂（见 §7）。

**明确不要做的**：不要指望用 SimFoundry 的仿真成功率替代真机验收，也不要在没做过自己的相关性标定前引用 0.911。

---

## 9. 代码仓库实现难度评估

仓库：https://github.com/NVlabs/SimFoundry ，Apache-2.0，2026-08-14 首次开源（V0：刚体 + 铰接生成），291 个文件（151 py / 51 yaml / 22 sh / 9 patch），0 open issues。

### 9.1 放出了什么 / 没放出什么

| 组件 | 状态 |
|---|---|
| Pipeline A（13 阶段重建） | ✅ 完整 |
| Pipeline B（object/scene/task cousins + 任务提议） | ✅ 完整 |
| Pipeline C（OmniGibson 加载、smoke test、策略评测、遥操作、标注、waypoint 提取、**demo 生成**、replay） | ✅ 代码在（`stages/0…6`、`sweep_eval.py`） |
| 策略客户端（openpi / gr00t / dreamzero） | ✅ `simfoundry/policies/` |
| 域随机化（材质/光照） | ✅ `simfoundry/domain_randomization/` |
| 论文用的场景与资产 | ❌ "Coming Soon" |
| 自动背景生成（3DGS） | ⚠️ 代码在 `auto_bg_reconstruction/`，README 却标 "Coming Soon" |
| **策略训练代码** | ❌ 明确未放 |

**README 里「Data generation and policy training code is not included」这句话是不准确的**——`C_application/stages/{2,3,3b,4,5,6}` 加 `simfoundry/utils/data_gen_utils.py`（CuRobo 运动规划 + waypoint 回放 + IK）就是数据生成。真正缺的是**策略训练代码**和**论文实验的具体配置/资产**。所以「照着论文复现 sim-to-real 数字」做不到，但「用它生成自己的数据然后接自己的训练栈」是可行的。

### 9.2 安装成本（`docs/INSTALL.md`）

**这是我见过的机器人开源项目里安装负担最重的一档之一。**

| 项 | 要求 |
|---|---|
| conda 环境数 | **7 个**（simfoundry / hunyuan / any6d / da3 / void / nerfstudio_simfoundry / 3dgrut），铰接再加 2 个（articulate-anything-{hunyuan,partfield}） |
| 磁盘 | **~250 GB**（conda envs ≈100 GB，`deps/` ≈82 GB 其中 VOID 模型单个 41 GB，HF cache ≈12 GB） |
| VRAM | 最低 16 GiB；stage 7 网格生成默认需 ~29 GiB，**24 GiB 卡必须加 `s7_mesh.low_vram=true`**（CPU offload 降到 ~6 GiB，代价是慢）；铰接 stage 8b 最低 18 GiB 且**不受流水线 VRAM 调度器管辖** |
| CUDA | 铰接需要 **CUDA 12.8 在 `/usr/local/cuda-12.8`**（flash-attn / spconv 编译）；自动背景需 CUDA 12.x 工具链编译 gsplat |
| Git LFS | 必须（docs 资产 + 铰接依赖的 embedding/mesh） |
| 外部服务 | **Google Cloud Vertex AI（Gemini）项目 + 计费**，或 Gemini API key；Hugging Face 账号 + **3 个 gated 模型的审批**（facebook/sam3、facebook/dinov3-vitl16、briaai/RMBG-2.0，README 明说审批要等） |

**关键结论：这不是一个能离线部署的系统。** 重建 stage 3/5/6/10 和整个 Pipeline B 的 VLM 调用全部走 Vertex AI / Gemini。没有网络或没有 GCP 账号，流水线在 stage 3 就停。仓库提供 `--cache-mode` / `--test-mode` 缓存远程响应用于可复现调试，但那是回放缓存，不能替代首次运行。

### 9.3 许可证雷区（`docs/INSTALL.md` 最后一节，写得异常坦诚）

- 安装脚本**代你接受** NVIDIA Omniverse EULA（Isaac Sim / Kit / pxr / NuRec）、BEHAVIOR-1K/OmniGibson 数据集条款、Anaconda 频道 ToS。
- 多个组件是 **non-commercial / research-only / source-available / 无许可证**：SAM 3、Any6D、Hunyuan3D-2.1、Hunyuan3D-Part、PartField、FoundationPose、FoundationStereo、nvdiffrast、cuRobo、Depth Pro、VOID/CogVideoX 权重、OpenPI/Gemma 权重、**CoTracker（CC-BY-NC-4.0）**。
- **TeleMoMa 无 license 文件 = all-rights-reserved**，仓库拒绝安装/分发/镜像，遥操作 stage（C 的 2/2b）会 lazy import 并报错。**要用遥操作采数据，你得自己解决 TeleMoMa 的授权问题**——而遥操作正是整条 sim-to-real 数据生成链的第一步。这是一个**在开源链路上的实质性断点**。
- Hunyuan3D-2.1 社区许可的 Territory **排除欧盟、英国、韩国**（不含中国大陆，对本仓库不构成障碍，但记一笔）。

### 9.4 真实的「跑通到出结果」难度

| 目标 | 难度 | 主要障碍 |
|---|---|---|
| 跑通 Pipeline A，得到一个 OmniGibson 场景 JSON | **中** | 装 7 个 env + 250 GB + GCP 账号 + HF 审批；文档质量不错（有 dry-run、smoke test、`AGENT_INSTALL.md` 可让 agent 代装） |
| 跑通 Pipeline B 出 cousins | **中** | 依赖 A 的产物；VLM 调用量大 = 真金白银 |
| 在 OmniGibson 里评测一个 π0/GR00T checkpoint | **中高** | 需起 policy server（openpi:8000 / gr00t:5555）；配置里机器人硬编码为 Franka Panda；需要 BEHAVIOR-1K 资产 |
| **换成自己的机器人** | **高** | 需往 OmniGibson 注册 robot class + USD + 控制器 + CuRobo 配置；配置文件里 prim path 是硬编码的 |
| **复现论文的 0.911 相关性** | **不可能** | 场景/资产未放；且需要你自己的真机数据 |
| **复现 sim-to-real 训练数字** | **不可能** | 训练代码未放；遥操作依赖 TeleMoMa（授权断点）；CuRobo 首次 JIT 编译 5–15 分钟 |

### 9.5 最有价值的一处「代码比论文诚实」

`scripts/pipeline/C_application/stages/1_eval_policy_og_scene.py` 顶部有一段开发者注释，标题是 **"NOTES FROM JOSIAH on tuning sim"**，列的是让仿真评测与真机对上所需的手工调参项：

- 调渲染后处理的色彩校正参数
- 回放真机数据集里的机器人动作，确认整体轨迹与仿真设置对得上（尤其 z 轴）
- 把 3DGS 背景换成 mesh 背景
- 把 assisted grasping 换成 physical grasping
- 给场景里所有物体加摩擦
- 调所有场景物体和机器人的位姿
- 调所有场景物体的尺度以匹配真实世界
- 降低机器人手指最大速度（注释里还抱怨「这个值不知道被哪里外部设置了，每次都得手动覆盖」）

**这份清单应当被视为论文 §5.1 的隐含方法部分。** 它说明：**资产生成是自动的，但「让仿真评测与真机对齐」这一步是手工的、逐场景的、且需要真机数据回放来校准。** 论文 FAQ #3 的说法（「几分钟的交互式微调」）与这份清单之间存在明显落差。对任何打算复用这套评测相关性的人，这是最重要的一条信息——**0.911 的背后有一个未被论文量化的人力项。**

### 9.6 工程质量的正面评价

不能只挑刺。这个仓库的工程质量在学术开源里属于上乘：

- Hydra 配置化（`real2sim_cfg.yaml` 722 行，每个 stage 一个 block，注释详尽到写明每个 stage 跑在哪个 conda env）。
- 有 pipeline orchestrator + resource scheduler（按 GPU 显存比例调度流式 stage，`max_vram_frac` 默认 0.9，同一套配置在 24 GiB 和 96 GiB 卡上都能跑）。
- 有 `--dry-run`、`--include/--exclude`、`--cache-mode/--test-mode`、smoke test、pytest 测试套件（其中 4 个文件无需任何运行时依赖即可跑）。
- 拍摄须知写得很实在（一个平面、俯角拍摄、平移不要原地旋转、每帧保持所有物体可见、铰接物体全部关闭）——这些是论文里没有但决定成败的操作知识。
- 第三方许可披露矩阵（`THIRD_PARTY_LICENSES.md §6`）逐组件记录必需/可选、获取方式、版本 pin、源码与权重各自的条款——罕见的合规诚实度。

---

## 评分

- **新颖性 6/10**：没有新算法。价值在系统整合与 cousin taxonomy 的三轴化。诚实地说，这是一篇优秀的工程论文而非科学论文——作者自己在 FAQ #1 也是按「feature-complete automation」而非「新方法」来辩护的。
- **实验充分度 6/10**：广度好（7 任务 × 5 策略 × 2 本体 × 12 重建场景），深度不足（N=3~5 算 Pearson、25 rollout、无 CI、无跨场景外推、摘要口径与正文不一致）。核心结论方向大概率正确，但精确数字不该被当真。
- **写作质量 8/10**：正文清晰，附录极其详尽（A–L），FAQ 与 Limitations 章节坦诚度高于平均水平（例如主动说明机器人实验用的是 mesh 背景而非自动 3DGS）。
- **复现/落地价值 5/10**：重建与增广部分可复现且有实用价值；评测相关性与 sim-to-real 训练部分不可复现（资产未放、训练代码未放、TeleMoMa 授权断点）。安装成本（7 env / 250 GB / GCP 账号）会劝退相当一部分团队。
- **对当前研究的启发 8/10**：对本仓库而言，**协议层面的启发（子任务评测、5×5 网格采样、MMRV）价值远高于系统本身**，且这三样零成本可用。

---

## 实践 takeaway

**可以借的**：
1. 子任务评测协议 + 5×5 网格无放回初始条件采样 + MMRV 指标——三样都不依赖 SimFoundry，本周就能落到现有仿真与真机评测里，直接对症 `real-world-policy-evaluation-2026-08.md` 的统计缺口。
2. object / scene / task 三轴 cousin taxonomy，作为数据多样性设计的检查表。
3. 「VLM 生成 → 仿真执行 → critic VLM 判定 → 迭代」的资产生成闭环。
4. 若要接入系统，只取 Pipeline A/B 当**资产工厂**（输出标准 USD/URDF），导入自己的仿真栈，绕开 OmniGibson 依赖。

**要当心的**：
1. **仿真绝对成功率不可用**（真机 100% ↔ 仿真 34%），只有排序可用。
2. **0.911 需要你自己的真机数据来重新标定**，不能直接继承；建立它本身要约 700+ 次真机 rollout。
3. **摘要的 17%/21%/40% 口径与正文不一致**，task cousins 的 40% 是仿真内数字。
4. **论文的 0.911 不是全自动流程跑出来的**：背景用的是手机扫描的 mesh，且有一份未被量化的手工对齐清单（§9.5）。
5. **它发现不了部署链路失配**（Hermite S 曲线、gripper 尺度这类问题），与本仓库方案 2 是互补而非替代。
6. 安装是 7 个 conda env / 250 GB / 强制 GCP Vertex AI / 3 个 gated HF 模型审批；遥操作链路卡在 TeleMoMa 的授权断点上。

**它打开的问题**：相关性能否跨场景外推？如果每个新场景都要重新标定，那么「real-to-sim 评测」相对「直接真机评测」的成本优势，究竟在多少次策略迭代之后才回本？论文没答，而这是决定这条路线是否值得投入的唯一问题。
