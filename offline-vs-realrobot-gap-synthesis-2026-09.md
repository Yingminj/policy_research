# 离线指标与真机结果为什么对不上：成因、文献与本项目证据的重新定级

**编撰日期** 2026-09-11 · **范围** `tidy_up_stationery_le` 上 `act_eef` 线的离线评测与真机成功率之间的分歧
（主对象：`experiment_report/act/act_eef-realrobot-vs-offline-2026-09.md`，下称 **R9**；
`ACT-ACTEEF-results-2026-09.md`，下称 **汇总报告**；`MASTER-LEDGER-2026-09.md`，下称 **总账**）。

**本文做了三件事：**
(1) 把"离线-真机不一致"拆成六类可区分的成因，每类配上已核实的文献与本项目的对应证据；
(2) 用使用者 09-11 补充的真机协议信息（**按权重分块连续测试**、**初始摆放基本固定**、成功 = 整任务二值）
对 R9 的真机统计做了一次独立复核，**结论有实质变化**（§1.2）；
(3) 给出一套评测协议 v2，目标是让下一轮离线数字**先被真机标定、再被用来做决策**。

**核实方法。** 全部 27 篇文献均经 WebFetch 抓取 `arxiv.org/abs/<id>` 逐篇核对标题、作者、v1 日期与摘要原文；
正文中的具体数字另抓 `arxiv.org/html/<id>` 或 PDF 核对，核实等级见 §7。未使用任何来自模型记忆的引用。
本地数字全部引自上述三份报告，**本文没有重跑任何离线评测**；§1.2 的统计复核脚本见附录 A（仅标准库，可直接运行）。

**AI 使用声明。** 本文的文献检索、核实、统计复核与撰写由 AI 研究助手（Claude）在使用者指导下完成；
所有结论需经使用者对照原始数据复核后方可引用。

---

## 0. 结论先行

| # | 结论 | 依据 |
|---|---|---|
| **K1** | **"离线指标与真机相反"这件事，大部分不是"模型行为反常"，而是离线指标在量一个和真机成功不同的量。** 在 TE 归约下，逐帧 MAE 主要量的是**长视野预测误差被加权平均后的滞后与欠冲**，而静态桌面任务在闭环里几乎不为滞后付费、却要为计划不一致付费。文献里这类构念错位（construct mismatch）是被反复测到的普遍现象，不是本项目的特例。 | §2.1–2.3；Codevilla 2018、robomimic、SIMPLER、CI-MSE、Tune to Learn |
| **K2** | **R9 最强的一条结论（coeff 0.01 > −0.05）经复核后仍然成立，其余三条"离线押反"的判决降级为"未测出"。** 复核发现：真机表只有 **10 个唯一条件 = 300 次试验**（不是 480）；**10 格成功数全部是 3 的倍数**（独立二项下概率 ≈ 1.7 × 10⁻⁵），**孪生两份权重三格完全相同**（≈ 0.0015），**三份权重九格完全相同**（≈ 3.3 × 10⁻⁶）。这说明试验之间**不独立**或**百分比是取整后的**。按有效样本 n = 10/格重算：coeff 对比 p = 0.029（仍成立）；533 vs 其余 p = 0.23、200k vs 100k p = 0.55（**都不成立**）。再叠加"按权重分块测试"，533 = 60% 与测试时段完全共线。 | §1.2、附录 A |
| **K3** | **跨权重那一段，离线侧本来就分辨不出来。** 三份权重在 `ens_0.01` 下的离线跨度：MAE 5.6%、二阶差分 6.7%、`replan_1` 位置 2.7%，全部落在总账实测的**评测集判读线（中位 5.86%）**附近或以下。离线侧"533 最好"与"616as 最平滑"这两个读数**都不该拿来和真机对符号**。因此汇总报告表 19 的 ρ = −0.949 / +0.973 实质上只由**一个对比（coeff 轴）**撑起，精确置换检验 p = 0.067。 | §1.3、§3 |
| **K4** | **离线评测集和真机测试不是同一个分布。** 真机是"基本固定"的摆放、反复试；离线是 53/100 条独立录制的示教，且不含左腕触限工况。文献的两条对应：Codevilla 2018 把 MSE 与驾驶成功率的相关从 \|r\| = 0.39 提到 0.77，**唯一改动是换验证数据**（加入偏离专家轨迹的恢复样本）；ALT（Demystifying Diffusion Policies）指出小数据下策略的有效机制是**检索记忆**。后者给出了一个可检验的解释：**在固定摆放上，记得越牢（200k、训练集覆盖该摆放越密）越好，而独立留出集恰好惩罚记忆。** | §2.4 |
| **K5** | **本项目离线侧已经测到的"唯一未被证伪"的平滑度 / 计划一致性指标，在文献里有独立支持**：ACT 原文 TE 给参数化策略 +3.3%、给检索式策略反而掉点；2026-08 的 *Why Does Action Chunking Improve BC* 把 chunking 的收益归于**隐式集成**，并明确指出 ACT 式指数加权"显著削弱集成效果"、线性（等权）集成更好——这正是 R9 建议 1（试 `coeff = 0`）的先验。VLA-FAIL 的 action chunk consistency 与 *How VLAs Fail Differently* 的方向反转率，是同一类"不用 GT 的一致性指标"作为失败预测子的证据。 | §2.3 |
| **K6** | **下一步最值钱的不是再换一个离线指标，而是把真机协议改成能被统计解读的样子**：交错 / 随机顺序（ABAB）、摆放从一个**写明的分布**里抽并记录编号、记录分阶段进度、用序贯检验决定何时停。在此之前，任何离线指标都没有"真值"可以对齐。 | §4.1；Kress-Gazit 2024、TRI LBM 2025、STEP 2025、*Beyond Binary Success* 2026 |

**一句话：** 你们的离线尺子没有"坏掉"，它在认真地量**开环、逐帧、对单条示教的偏差**。真机成功取决于**闭环一致性、关键时刻的对错、以及测试分布是否被训练覆盖**，这三件事离线尺子要么没量（一致性）、要么平均掉了（关键时刻）、要么用错了样本（分布）。同时，**现在的真机数据分辨力比 R9 以为的低**，所以"离线押反"里有一半其实是"两边都没测出来"。

---

## 1. 问题界定：到底是哪几处对不上

### 1.1 R9 / 汇总报告列出的分歧（原文数字，未改动）

| # | 离线说 | 真机说 | 出处 |
|---|---|---|---|
| D1 | `ens_-0.05` 的 MAE 比 `ens_0.01` 好 1.6× | 0.01 好 30 点（72/90 vs 45/90） | R9 §1-1 |
| D2 | `acteef_533` 离线各栏第一 | 533 = 60%，其余三份 80% | R9 §1-3、§6 |
| D3 | `acteef_505` 离线四项垫底 | 505 = 80%（并列第一） | R9 §1-4 |
| D4 | 训练曲线在 50k–100k 触底 | 200k 比 100k 好 10 点（3/3 同向） | R9 §7 |
| D5 | 四个准确度指标对真机 ρ = −0.949；二阶差分 +0.973 | —— | 汇总报告表 19 |
| D6 | chunk + K=40 桥链路上的全部数字 | 真机在 TE 模式下根本不跑这条链 | 汇总报告 §1.3 |

D6 已经被 09-07 那一轮识别并修正（改用 `ens_*` 行），本文不再展开。下面重点是 D1–D5。

### 1.2 复核：真机数据能承载多少结论（**本文新增**）

使用者 09-11 确认了三件 R9 §8-1 列为"最大单一威胁"的协议信息：

- **测试顺序：按权重分块**——同一份权重的 30 次连续跑完，再换下一份；
- **初始摆放：基本固定**，不是每次随机；
- **成功判据：整任务二值**。

据此复核真机表（脚本见附录 A）：

**(a) 试验数。** 表 A 与表 B 的 `200k @ 0.01` 那一列是**同一个条件**。去重后唯一条件为 **10 个，共 300 次试验**。
R9 写的"16 个非空格、480 次"只有在该列被真的重复跑了一遍时才成立——**请确认**；若只跑了一次，R9 所有合并检验的分母不变（它们本来就是按唯一格合并的），但"480 次"的表述需要改。

**(b) 计数的离散结构。**

| 检查 | 观测 | 若每次试验独立、Binom(30, p) | 
|---|---|---:|
| 10 格成功数都能被 3 整除 | 24 / 21 / 18 / 15，全是 3 的倍数 | **1.7 × 10⁻⁵** |
| 孪生 `616` 与 `616as` 三格完全相同 | 24=24、21=21、15=15 | **0.0015** |
| `505` / `616` / `616as` 九格完全相同 | 三份权重三格全等 | **3.3 × 10⁻⁶** |

R9 §2.3 只算了单格相等的概率（约 13%）并判为"有运气成分"；**三格联合、以及三份权重联合的概率低三到五个数量级**，不能再用运气解释。两种可能（需使用者确认哪一种）：

1. **百分比是取整到 10% 后报的**——那么真实计数未知，所有 Fisher p 值都是按一个不存在的计数算的；
2. **结果在试验间不独立**——例如"约 10 个固定摆放 × 各跑 3 次"，而策略在同一摆放上近乎确定性地成功或失败
   （ACT 推理取 z = 0、TE 是确定性归约，这在机制上完全说得通）。此时每格的**有效样本量接近摆放数而不是 30**。

**(c) 按有效 n = 10/格重算（情形 2 的保守读法）：**

| 比较 | R9（n = 30/格） | 有效 n = 10/格 | 判读 |
|---|---:|---:|---|
| coeff 0.01 vs −0.05（合并三份） | 4.0 × 10⁻⁵ | **0.029** | **仍成立** |
| 533 vs 其余三份 | 0.049 | 0.23 | **不成立** |
| 533 vs 两份孪生 | 0.075 | 0.38 | 不成立 |
| 200k vs 100k（合并三份） | 0.17 | 0.55 | 不成立 |

**(d) 分块顺序。** 按权重分块意味着"权重"与"测试时段"（光照、物体磨损、操作者熟练度、夹爪/相机微漂）**完全共线**。
533 只有一格、且与另外三份权重不在同一时段——**D2 在当前设计下没有可辨识性**。
coeff 对比是在**每份权重的块内**做的，受分块影响小得多；但如果每个块内 −0.05 总是排在最后跑，那它仍与"块内时间"共线（**请确认块内顺序**）。

**复核结论：D1 站得住，D2 / D3 / D4 目前只能读作"真机没测出差别"，而不是"真机和离线方向相反"。**

### 1.3 离线侧的分辨力：跨权重那一段本来就在噪声里

总账 §3 已实测两条判读线：采样判读线 1.70%（flow）/ 2.00%（diffusion），**评测集判读线中位 5.86%、最坏 15.21%**
（两个互不相交留出集上 10 个权重）。与之对照，三份真机权重在真机实际执行的归约下的离线跨度：

| 指标（`ens_0.01` / `replan_1`） | 533 → 616 跨度 | 相对评测集判读线 5.86% |
|---|---:|---|
| `ens_0.01` 平坦 MAE | 5.6% | 贴线，读不出 |
| `ens_0.01` 二阶差分 | 6.7% | 贴线，读不出 |
| `replan_1` 末端位置 mm | 2.7% | 线下，读不出 |

所以 D2 那一段"离线说 533 最好 / 616as 最平滑"，**两边的方向都不可信**。汇总报告表 19 的 5 个点里，真正有分辨力的只有 coeff 轴那一个对比；
在含并列的 5 点上做精确置换检验，|ρ| = 0.949 与 0.973 的双侧 p 都是 **0.067**。
**"平滑度押对、准确度押反"应当读作：在 coeff 这一个轴上，逐帧 MAE 的方向与真机相反，平滑度的方向与真机一致。** 这一条本身是真的，也足以说明问题，但它是一条关于**归约方式**的结论，不是关于**权重排序**的。

---

## 2. 成因分类：六类机制，各自的文献与本项目证据

每类按"机制 → 文献 → 本项目对应 → 可证伪检验"组织。六类互不重叠，但在同一个数字上可以叠加。

### 2.1 G1 构念错位：开环逐帧误差 ≠ 闭环任务成功

**机制。** 离线评测是 teacher-forced：每一帧的观测都来自示教，模型永远处在专家的状态分布上；真机是闭环：模型自己的误差会改变下一帧看到什么。两者之间的缺口不是噪声，是结构性的。

**文献。**
- Ross 等（DAgger，arXiv:1011.0686）给出这一问题的奠基表述：序列预测"violate the common i.i.d. assumptions made in statistical learning"，训练分布上的误差不能约束执行分布上的表现。
- Simchowitz 等（arXiv:2503.09722，COLT 2025）在连续动作下给出更强的负面结果：即使动力学指数稳定、专家光滑确定，任何光滑确定的模仿策略在执行时的误差都会比专家数据分布下的误差"exponentially larger, as a function of problem horizon"；只有非光滑、**非马尔可夫**或状态相关随机性很强的"improper"策略（原文点名 action-chunking 与 diffusion policy）才能绕开。
  Zhang 等（arXiv:2507.09061）进一步说明 action chunking 与探索式数据采集在不同条件下规避指数级复合误差，关键机制是控制论意义上的稳定性。
  **推论：**离线逐帧误差和闭环误差之间的换算系数取决于执行方式（chunk、TE），**同一个离线误差在不同执行方式下对应不同的闭环结果**——这正是 D1 的形式。
- Codevilla 等（arXiv:1809.04843，ECCV 2018）在 CARLA 上训练 45 个模型、每个模型多个训练步各测一次在线与离线：
  "offline prediction accuracy and actual driving quality are surprisingly weakly correlated"，
  单前视相机、无动作噪声的标准验证集上 MSE 与成功率 |r| 仅 **0.39**；并给出案例——两个离线误差相近的模型，一个每次都撞，一个大多成功（§4.3 Fig. 5）。
- robomimic（Mandlekar 等，arXiv:2108.03298）项目页原文："the best validation loss does not correspond to the best performing policy"，
  "the best validation policy is 50 to 100% worse than the best performing policy"。
- SIMPLER（Li 等，arXiv:2405.05941）正文：validation MSE "is *not* a good proxy for a policy's real-world performance"；Google Robot 设置下 validation MSE 对真机的 MMRV 0.412 / Pearson r 0.464，而仿真评测为 0.031 / 0.976。
- CI-MSE（Huang 等，arXiv:2606.29898）："validation loss on expert demonstrations ... is often poorly correlated with real-world performance"；27 个 checkpoint 上原始 MSE 与 rollout 的 Spearman ρ = −0.61。
- *Tune to Learn*（Bronars 等，arXiv:2604.02523）§V-A："Lower imitation loss does not translate to better policy performance in our experiments"——柔顺 / 过阻尼增益下训练与验证 MSE 更高，闭环成功率反而更好。
  **这一条值得单独记住：执行层（控制器增益）可以反转"loss 低 = 好"的关系，而本项目的 TE 系数正是一个执行层参数。**

**本项目对应。** 全部离线轮次均为 teacher-forced（汇总报告 §10.2 第一条）。失败数据消融里"离线最多 +10%、闭环主观明显变差"（汇总报告 §8.2 读数 5）是同一机制的另一个实例。

**可证伪检验。** 如果 G1 是主因，那么把离线评测换成"有闭环的代理"（见 §4.2 的扰动恢复集）之后，与真机的秩相关应当显著上升；如果不升，G1 不是主因。

### 2.2 G2 归约错位：TE 下的逐帧 MAE 主要量的是"时序"而不是"对错"

**机制（本文的解释，可检验）。** `ens_<coeff>` 在第 t 帧执行的动作，是过去 100 次推理对"第 t 帧"的预测的加权平均；
第 t−k 次推理对第 t 帧的预测，就是那条 chunk 的第 k 步。因此 **`ens_*` 的逐帧 MAE ≈ 各视野 k 上预测误差按 TE 权重的加权平均**。
而本项目各视野误差随 k 单调上升（汇总报告表 4：@1 0.017 → @60 0.037），coeff = 0.01 把权重压向老预测（加权平均"年龄" 1.92 s，附录 A 独立复算一致），
所以 `ens_0.01` 的 MAE 基本就是**长视野误差**。长视野误差里最大的一块，是示教者**什么时候**动——人类操作者的停顿与节奏不可预测，
ACT 在 z = 0 下给出的是条件均值，表现为"更慢、更平、欠冲"。离线尺子把这个时序偏差按全额计入；
**真机的桌面静态任务不设时限，物体会等机器人，时序偏差几乎不花钱。**

这解释了 D1 的方向：−0.05 把权重压向短视野（最近 10 次推理占 39.6%），逐帧 MAE 当然好 1.6×，但它量的"好"是时序贴合，不是任务对错。

**文献。**
- CI-MSE 的对齐流程正是为此设计：先按 rollout 时的 TE / RTC 方式把离线预测合成执行动作（其式 3 是等权 1/H 平均），再用**窗口约束的 DTW** 对专家动作计误差。
  消融（§4.5）：不做 TE 对齐 ρ = −0.80 → 做 −0.87；不做 DTW −0.80 → W = 1 时 −0.87。**两个对齐各自贡献一截，都是"让离线误差不再为时序错位付费"。**
- Codevilla 2018 §3.3 的原文直觉可以直接搬来："the average prediction error may not be characteristic of the driving quality, since it does not take into account the temporal correlations in the errors"——
  时间上不相关的噪声只造成轻微摆动仍可成功，而**持续同向的偏置才致命**。
- Dauner 等（arXiv:2306.07962）在 nuPlan 上发现短期规划（闭环）与长时自车预测（开环）"are fundamentally misaligned"，开环子任务的最优解甚至只用中心线、忽略地图与其他车辆。
  Wang 等（arXiv:2605.00066）在 NAVSIM 与 Bench2Drive 上：传统开环 ADE / FDE 与闭环驾驶分"no dependable connection"；安全感知的开环 PDMS 相关虽强但仍有明确的排序倒挂。
  **启示：开环指标里对时序 / 进度的处理方式，决定它能不能预测闭环。**
- *Do Better Imagined Rollouts Mean Better Robot Control?*（Raghavan & Singh，arXiv:2609.02811）在受控路径跟踪上测得：
  带测量更新的回放误差与闭环 ρ = 0.923，无测量的 20 步开环外推只有 0.774，且后者在 24 个条件里 18 个选错了最优估计器。
  **离线评测的"校正节奏"要与闭环一致**——对本项目即：离线归约要按真机的重规划与集成方式来做，而 09-07 那一轮已经做到了一半（归约对了，误差计法没对）。

**本项目对应。** 汇总报告 §5.3 已观察到"末端位置 mm 在 `replan_1` 归约下押对、在 `ens_0.01` 归约下押反"（注：按 §1.3，这两个方向在跨权重段都在判读线内）。
R9 §10.2 第二条"TE 滞后的直接测量未做"——**这正是 DTW / 最优时移对齐要补的那一步。**

**可证伪检验（零 GPU）。** 在 09-08 的 `results/` 上，对 `ens_*` 执行轨迹做"逐 episode 最优时移"或窗口 DTW 后再算误差。
若 G2 成立，**时移后 `ens_0.01` 的误差应降到接近 `replan_1`，且 0.01 与 −0.05 的离线排序应翻转或持平**。

### 2.3 G3 离线看不见的收益：一致性 / 集成在闭环里值钱

**机制。** TE 买的是**相邻推理之间的一致性**（二阶差分降一个数量级，汇总报告表 13）。在闭环里，一个每拍都在改计划的策略会在两个模式之间摇摆（到了该抓的位置反而悬停），而逐帧 MAE 对"摇摆"与"稳定偏移"一视同仁。

**文献。**
- ACT 原文（Zhao 等，arXiv:2304.13705）TE 消融："a 3.3% gain for our method"，BC-ConvMLP 得益最多（4%），而检索式的 VINN 反而掉点；
  原文解释 "temporal ensemble mostly benefits parametric methods by smoothing out modeling errors"。
- *Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?*（Lazzati 等，arXiv:2608.02547，v1 2026-08-03）：
  现有三个解释（时间一致性、视野缩短、表征学习）都不足以解释 chunking 的收益；除非马尔可夫表达力与复合误差外，还有一项"**implicit ensembling**"——
  学到多种时间关系的 chunk 策略"exhibit behavior matching that of a model ensemble, increasing their robustness"。
  其脚注 4 专门比较了 ACT："previous works apply an exponential weighting, significantly downweighting the contributions of more recent timesteps, and mitigating the ensembling effects"，而他们用**线性（等权）**组合。
  **对本项目的直接含义：`coeff = 0`（等权，N_eff = 100）在文献先验上应不差于 0.01**，与 R9 建议 1 的离线证据同向；也给"−0.05 输在 N_eff 从 92 掉到 40"这一假设提供了机制名称。
- BID（Liu 等，arXiv:2408.17355，ICLR 2025）："action chunking allows the learner to better capture the temporal dependencies in demonstrations but at the cost of reduced reactivity"——一致性与反应性的权衡。
  本项目是静态、摆放固定的场景，**反应性的价值低、一致性的价值高**，TE 高 N_eff 占优在这个权衡里是预期内的；换到动态场景这一结论可能反转。
- *Revisiting Open-Loop Execution in Robotics*（Zeng 等，arXiv:2608.15938，Tedrake 组）：长开环执行的主要作用是帮助**短上下文**策略模仿"non-Markovian demonstrations"；给足上下文后，最反应式的闭环策略最好。
  本项目 `act_eef` 为 `n_obs_steps = 1`（短上下文），而人类示教的停顿与节奏是典型的非马尔可夫性——TE 在这里补的正是上下文。
- 不用 GT 的一致性指标作为失败预测子：VLA-FAIL（arXiv:2606.21386）的 action chunk consistency 利用 receding-horizon 的 chunk 重叠检测失败；
  *How VLAs Fail Differently*（arXiv:2605.28726）在 ACT / Diffusion / VQ-BeT 上测得方向反转率是通用失败预测子（AUROC 0.93 / 0.79 / 0.91），而速度越限在 ACT 上几乎无信号（0.52）。

**本项目对应。** 二阶差分（ρ = +0.973，仅在 coeff 轴上有分辨力）、`plan_consistency`（act 0.0027 vs patch 0.0071–0.0082）。
**文献把这一类量从"唯一没被证伪的候选"升级为"有独立失败预测证据的候选"，但仍不是"在本任务上被标定过的预测器"。**

**可证伪检验。** R9 建议 2 的 `coeff = +0.05`（N_eff 与 −0.05 相同、滞后反向）仍是分离"集成次数"与"时序方向"的最干净实验。
按 2608.02547 的先验，预测 +0.05 也会掉点（集成次数少），`coeff = 0` 不差于 0.01。

### 2.4 G4 分布错位：离线评测集与真机测试不是同一个分布

**机制。** 离线评测集是独立录制的 53 + 100 条示教；真机是"基本固定"摆放上的重复试验。
前者问的是"对新摆放的平均泛化"，后者问的是"在这一个（几个）摆放上稳不稳"。两者对"记忆"的奖惩方向相反。

**文献。**
- Codevilla 2018 §4.3："This shows that a successful policy must not only predict the actions of an expert on the expert's trajectories, but also for observations away from the expert's trajectories. Proper validation data should therefore include examples of recovery from perturbations."
  仅把验证数据从单相机换成三相机（侧向相机相当于偏离专家轨迹的状态），MSE 与成功率的 |r| 从 **0.39 → 0.77**；加入动作噪声 → 0.54。**换数据比换指标的收益更大**（换指标最好只到 0.65）。
- ALT（He 等，arXiv:2505.05787）："diffusion policies essentially memorize an action lookup table -- and this is beneficial"，在稀疏数据下"find the closest training image to the test image in a latent space, and recall the associated training action sequence"。
  **若本项目 ACT 在 616 集规模下也近似"检索 + 回放"，那么在固定摆放上：训得越久（记得越牢）越好，训练集中该摆放附近的样本越密越好；而独立留出集惩罚的正是这种记忆。**
  这同时给 D3（505 ⊂ 616 同为 80%、533 有 67 集不在其中）与 D4（200k > 100k）一个统一的、可检验的解释。
- Li 等（*Is Ego Status All You Need*，arXiv:2312.03031）与 Zhai 等（AD-MLP，arXiv:2305.10430）：nuScenes 开环规划里，只用自车状态、不看感知的 MLP 就能把 L2 做到与感知方法相当甚至低约 20%——
  **场景太简单时，开环指标被"外推当前状态"主导。** 本项目的对应物是 `hold_state`：h1 上 0.02091，只比 `replan_1` 差 24%；姿态维只好 1.05×（汇总报告附录 B）。
  离线分数的大部分被"照抄当前位姿"就能拿到的部分占据，**真正决定成败的那一小段被平均掉了**。
- *Active Real-World Factor-Based Evaluation*（Liao 等，arXiv:2607.14439）：真机表现依赖"a large combinatorial space of task factors including object poses and camera viewpoints"，
  "narrow test suites ... can miss critical failure modes and misrepresent true deployment readiness"。**基本固定的摆放就是最窄的测试集。**

**本项目对应。** (i) 两个评测集都不含 `J7_L` 触限 episode（0/53、0/100，训练集 3%；汇总报告 §2.3.7）；
(ii) 真机摆放基本固定（09-11 确认）；(iii) `acteef_533` 在其独有 67 集上是 7.23× null（记忆读数），616 未见时 2.35×（汇总报告表 9）——**记忆效应在本数据上的量级是 3 倍**，足以主导一个固定摆放上的真机结果。

**可证伪检验（零训练）。** 拍下真机测试的初始场景（或取真机日志首帧），用任一冻结图像编码器在 505 / 533 / 616 三个训练集的首帧上做最近邻：
若 G4 成立，**真机成功率应与"训练集中距测试摆放 ε 内的 episode 数"同序**（预测：533 的近邻密度低于 505 / 616）。

### 2.5 G5 指标聚合与标签伪影

**机制。** 一个标量把"哪里错、错多少、错的是不是关键时刻"全部平均掉。

**文献。**
- Codevilla 2018 表 1 / 图 3：同一验证集上，MSE |r| = 0.39、绝对误差 0.61、量化分类误差 0.65、阈值化相对误差 0.64；
  §4.4 在真实 BDDV 数据上发现"The prediction error in the turns is most informative"——**关键片段上的误差比全程平均更有信息**，这是 CI-MSE 的前身。
- CI-MSE 把误差限制在 VLM 标注的任务关键区间，ρ 从 −0.61 → −0.87；其作者也写明了在评测分布漂移下的"design boundaries"。
- *Beyond Binary Success*（Snyder 等，arXiv:2603.13616）："competing policies can be separated more quickly when using fine-grained task progress than binary success metrics"。

**本项目对应。** 汇总报告 §4.4：平坦 14 维 MAE 近似是一个"左腕姿态指标"，其中一半量的是关节限位而非策略误差；
阈值指标文档（`offline-metric-threshold-accuracy-plan-2026-09.md` §6.2）：在 0.05σ 容差上策略与 null 只差 2 个百分点。
**这一类问题本项目已经诊断得比文献更细，缺的是"关键片段"这一刀。**

### 2.6 G6 分辨力与实验设计

**机制。** 离线与真机两把尺子各有分辨力下限，两者都在线下时，"方向相反"不是信息。

**文献。**
- Kress-Gazit 等（arXiv:2409.09491）§2.3 建议"A/B testing, i.e., interleaving policy rollouts when comparing different policies in a way that is blind to the evaluator"，并在同一时段内评所有策略；§2.2 要求尽量消除初始条件的非预期变化；§4.2 要求报区间估计。
- TRI LBM（arXiv:2507.05331）以"blind, randomized trials in a controlled setting"作为大规模真机比较的前提。
- STEP（Snyder 等，arXiv:2503.10966，RSS 2025）：小样本下的序贯比较检验，可按中间结果决定是否加试验而不引入 p-hacking，最多省 32% 试验。
- *Beyond Binary Success*（arXiv:2603.13616）：把序贯检验推广到分阶段进度与平滑度等连续指标，相对批量检验最多省 70%。
- SureSim（Badithela 等，arXiv:2510.04354）：用少量"真机 + 代理"配对数据校正大规模代理评测的偏差（prediction-powered inference），省 20–25% 真机试验。
  **本项目的离线评测可以扮演其中"代理"的角色**——前提是先攒够配对数据。

**本项目对应。** §1.2、§1.3 的全部内容。

---

## 3. 重新定级：现有七条"离线-真机判决"哪些站得住

对照 R9 §10 的一页速览，按 §1.2 与 §1.3 的复核重新判读：

| R9 的判决 | R9 判为 | 本文复核 | 理由 |
|---|---|---|---|
| coeff −0.05 vs 0.01：离线反了 | 离线反了（强） | **成立** | 有效 n = 10 下 p = 0.029；块内比较不受跨块漂移影响；机制（G2 时序 + G3 集成）有文献支持 |
| 权重排序：533 离线第一、真机最后 | 离线反了 | **两边都未测出** | 真机 p = 0.23（有效 n）且与测试时段共线；离线跨度 5.6% ≈ 评测集判读线 |
| 505：离线垫底、真机并列第一 | 离线反了 | **两边都未测出** | 505 无 TE 口径离线数字；chunk 口径离线跨度 7.4% 含 2% 训练噪声；真机与 616 系在同一时段外 |
| 200k vs 100k | 离线反了（弱） | **真机未测出** | p = 0.55（有效 n）；但 G4 给出了"固定摆放奖励记忆"的可检验解释，**不应把"200k 更好"当默认** |
| `replan_1` 位置 mm 押对 | 离线对了 | **未测出** | 2.7% 跨度，低于判读线 |
| 右夹爪 MAE 押对 | 离线对了 | **未测出** | 同上，且只对应一个真机对比 |
| 二阶差分押对 | 离线对了 | **在 coeff 轴上成立，跨权重未测出** | 6.7% 跨度贴线 |

**净结论：**本轮真机实验**唯一可以带走的因果结论是 TE 系数**；关于"哪份权重更好"，现有离线与真机数据**都没有给出可辨识的答案**。
R9 §9 建议 3（把 `deploy_config_eef.yaml` 从 533 换到 616as）**没有害处但也没有证据支持**，若要换，理由应写成"便于用孪生做复核"，而不是"真机更好"。

---

## 4. 建议：评测协议 v2

按"每单位成本能换来的可辨识性"排序。前五条不需要 GPU。

### 4.1 真机侧（先做，否则离线无从标定）

1. **确认 §1.2 的三个问题**（零成本）：(a) `200k @ 0.01` 那列是跑了一次还是两次；(b) 百分比是否取整、原始成功次数是多少；(c) 30 次里是几个摆放、每个摆放几次、块内条件的先后顺序。
   **(b) 与 (c) 决定了本轮全部 p 值是否有效。**
2. **交错顺序 + 摆放清单。** 每轮比较只放 2–3 个条件，按 ABBA / 随机顺序交错；摆放从一个**写成清单的分布**里抽（例如 10 个编号摆放覆盖工作区，每个条件每个摆放各 1–3 次），
   每次试验记录 `(条件, 摆放编号, 时间, 成败, 失败阶段)`。这样可以按摆放做配对分析（McNemar / 分层），**摆放间的难度差不会再污染条件间的比较**。依据：Kress-Gazit 2024 §2.2–2.3、TRI LBM 2025。
3. **记录分阶段进度**（如：到达 / 抓取 / 搬运 / 放置 / 复位），不只记整任务二值。依据：*Beyond Binary Success* 2026——同样的试验数下分得更开。
4. **用序贯检验决定何时停**，不预先定 30 次。依据：STEP 2025。coeff 那类大效应往往 10–15 次就能定案，把省下的预算给小效应。
5. **测一个"泛化摆放"子集。** 在固定摆放之外，每个条件额外跑 10 次在**未见摆放**上。两组数分别报告。
   这是 G4 的直接检验：若"200k > 100k"只在固定摆放上成立而在未见摆放上消失或反转，记忆解释即被证实。

### 4.2 离线侧（改尺子，不改模型）

6. **给 `ens_*` 加时序对齐误差**（零 GPU，§2.2 的检验）：逐 episode 最优时移或窗口 DTW（W 取 1–3 帧 × chunk 步长），报对齐后的末端位置 / 姿态误差。
   依据：CI-MSE 的 TE 对齐 + DTW 消融（−0.80 → −0.87 各一截）。
7. **关键片段误差**：本任务的关键片段（夹爪闭合前后 ±N 帧、放置前 ±N 帧）不需要 VLM，**直接用夹爪通道的开合切换定位**。报关键片段内的末端位置误差与夹爪时序误差。
   依据：CI-MSE、Codevilla §4.4（转弯片段最有信息）。
8. **扰动恢复集**：`J7_L` 触限的 19 条（汇总报告建议 10）是现成的一半；另一半是在现有评测 episode 上对 `observation.state` 加小扰动（例如末端 ±5 mm / ±2°），看策略是否拉回示教轨迹。
   依据：Codevilla 2018——换成含恢复样本的验证数据是它唯一把 |r| 从 0.39 提到 0.77 的做法。
9. **一致性指标进主 harness**：`plan_consistency`、执行轨迹二阶差分、方向反转率。都不需要 GT，历史权重可免费重读。依据：VLA-FAIL、*How VLAs Fail Differently*、本项目 R9。
10. **每个离线数字都带两条判读线**（采样线、评测集线），并在跨权重比较时写明"是否超过 5.86%"。这一条总账已经在做，本文只是把它推广到与真机对照的场合。

### 4.3 标定流程（让离线数字有资格替真机做决定）

11. **先攒配对数据再选指标。** 目标 ≥ 8–10 个 checkpoint（跨 coeff、跨训练步、跨权重），每个都有按 §4.1 协议测的真机成功率与全套离线指标；
    在这批配对上算每个离线指标的 Spearman ρ 与 MMRV（SIMPLER 用的排序违反度量），**选出来的指标才写进主表**。
    在此之前，所有"主排序键"的更换（包括汇总报告建议 9）都是基于 1 个对比的推断。
12. **攒够之后用 SureSim 式的校正**：离线指标当代理、少量真机当校正，给真机成功率出置信区间。依据：arXiv:2510.04354。

### 4.4 机制实验（与 R9 一致，文献先验更新）

13. `coeff = 0`：2608.02547 的等权集成先验 + R9 §5.4 的离线证据同向。**首选。**
14. `coeff = +0.05`：分离"集成次数"与"时序方向"。先验预测掉点。
15. 以上两条都按 §4.1 协议做（交错、摆放清单），否则再得到的一组"80 / 50"仍然无法与本轮合并。

**不建议做的：** 在 §4.1 落地之前继续加真机格子比较权重（533 vs 616 系那类 ≤ 20 点的差别，在分块 + 非独立试验下需要的试验量远超预算）；
在配对数据 < 8 个 checkpoint 时宣布任何离线指标"被证实"。

---

## 5. 反方意见

1. **"固定摆放的真机结果才是部署真正关心的，泛化到新摆放不重要。"** 若部署场景确实固定，那么 G4 不是缺陷而是选择——此时离线评测集应**换成固定摆放下录的示教**，而不是反过来去解释真机。
   本文建议 5 的双子集设计正是为了让这个选择显式化。
2. **"计数是 3 的倍数也可能就是巧合 / 或者记录习惯。"** 独立二项下 10 格全中的概率 1.7 × 10⁻⁵；更可能的是取整或摆放重复。只要原始计数能拿出来，这一条立刻可以定案——**这也是本文最想被证伪的一条**。
3. **"G2 的时序解释只是假设。"** 是。它给出了一个零 GPU 可证伪的预测（§2.2：时移对齐后排序翻转）。在跑完之前，它的地位与 R9 的 N_eff 解释相同。
4. **"文献多数来自仿真或驾驶，外推到双臂桌面任务要打折。"** 成立。Codevilla、Dauner、Wang 2026 是驾驶；CI-MSE、SIMPLER 有真机但任务不同。本文只用它们支撑**机制类别**，定量结论全部来自本地数据。
5. **"有效 n = 10 太保守。"** 若实际是 30 个互不相同的摆放、只是取整，那么 R9 原 p 值近似有效，§3 表中 533 vs 其余会回到 p ≈ 0.05 压线——**但分块顺序的共线性依然存在**，D2 仍不可辨识。

---

## 6. 本文未覆盖 / 未测

- 没有重跑任何离线评测；§2.2、§2.4 的可证伪检验都未执行。
- 世界模型评测（WorldEval arXiv:2505.19017 等）与仿真评测（SIMPLER）是另一条路线：它们在文献上与真机相关性更高，但需要为本工位建模，成本不在本文建议范围内，仅列作远期选项。
- patch_policy 线从未上真机，本文结论不外推到它；汇总报告表 12 中 patch 权重"MAE 全表最好"的读数，按 G2 / G3 应视为**待真机检验**，而非优势。

---

## 7. 参考文献与核实状态

**A** = 抓 `arxiv.org/abs/<id>` 逐字核对标题 / 作者 / v1 日期 / 摘要；**B** = 另抓正文 HTML / PDF / 作者项目页核对引用的具体数字或句子。

| # | 文献 | 本文用途 | 核实 |
|---|---|---|---|
| 1 | Ross, S., Gordon, G. J., & Bagnell, J. A. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. arXiv:1011.0686, v1 2010-11-02 | G1 | A |
| 2 | Codevilla, F., López, A. M., Koltun, V., & Dosovitskiy, A. *On Offline Evaluation of Vision-based Driving Models*. arXiv:1809.04843, v1 2018-09-13（ECCV 2018） | G1 / G2 / G4 / G5；\|r\| 0.39 / 0.54 / 0.77、0.61 / 0.65 / 0.64 | A + B（PDF 第 2、7、9–11 页） |
| 3 | Mandlekar, A., Xu, D., Wong, J., Nasiriany, S., Wang, C., Kulkarni, R., Fei-Fei, L., Savarese, S., Zhu, Y., & Martín-Martín, R. *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*. arXiv:2108.03298, v1 2021-08-06 | G1；"50 to 100% worse" | A + B（项目页 robomimic.github.io/study） |
| 4 | Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（ACT）. arXiv:2304.13705, v1 2023-04-23 | G3；TE +3.3% / +4% / VINN 掉点 | A + B |
| 5 | Zhai, J.-T., Feng, Z., Du, J., Mao, Y., Liu, J.-J., Tan, Z., Zhang, Y., Ye, X., & Wang, J. *Rethinking the Open-Loop Evaluation of End-to-End Autonomous Driving in nuScenes*. arXiv:2305.10430, v1 2023-05-17 | G4 | A |
| 6 | Dauner, D., Hallgarten, M., Geiger, A., & Chitta, K. *Parting with Misconceptions about Learning-based Vehicle Motion Planning*. arXiv:2306.07962, v1 2023-06-13 | G2 | A |
| 7 | Li, Z., Yu, Z., Lan, S., Li, J., Kautz, J., Lu, T., & Alvarez, J. M. *Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?* arXiv:2312.03031, v1 2023-12-05 | G4 | A |
| 8 | Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., Levine, S., Wu, J., Finn, C., Su, H., Vuong, Q., & Xiao, T. *Evaluating Real-World Robot Manipulation Policies in Simulation*（SIMPLER）. arXiv:2405.05941, v1 2024-05-09 | G1；MMRV 0.412 / r 0.464 vs 0.031 / 0.976 | A + B |
| 9 | Liu, Y., Hamid, J. I., Xie, A., Lee, Y., Du, M., & Finn, C. *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling*. arXiv:2408.17355, v1 2024-08-30 | G3 | A |
| 10 | Kress-Gazit, H., Hashimoto, K., Kuppuswamy, N., Shah, P., Horgan, P., Richardson, G., Feng, S., & Burchfiel, B. *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation*. arXiv:2409.09491, v1 2024-09-14 | G6、§4.1 | A + B（§2.2、2.3、4.2） |
| 11 | Simchowitz, M., Pfrommer, D., & Jadbabaie, A. *The Pitfalls of Imitation Learning when Actions are Continuous*. arXiv:2503.09722, v1 2025-03-12 | G1 | A |
| 12 | Snyder, D., Hancock, A. J., Badithela, A., Dixon, E., Miller, P., Ambrus, R. A., Majumdar, A., Itkina, M., & Nishimura, H. *Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping*. arXiv:2503.10966, v1 2025-03-14 | G6、§4.1 | A |
| 13 | He, C., Liu, X., Sznaier Camps, G., Sartoretti, G., & Schwager, M. *Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives*. arXiv:2505.05787, v1 2025-05-09 | G4 | A |
| 14 | Li, Y., Zhu, Y., Wen, J., Shen, C., & Xu, Y. *WorldEval: World Model as Real-World Robot Policies Evaluator*. arXiv:2505.19017, v1 2025-05-25 | §6 远期选项 | A |
| 15 | Black, K., Galliker, M. Y., & Levine, S. *Real-Time Execution of Action Chunking Flow Policies*. arXiv:2506.07339, v1 2025-06-09 | 背景（CI-MSE 的 RTC 对齐） | A |
| 16 | TRI LBM Team, Barreiros, J., Beaulieu, A., Bhat, A., Cory, R., et al.（共 82 位作者）. *A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*. arXiv:2507.05331, v1 2025-07-07 | G6 | A |
| 17 | Zhang, T. T., Pfrommer, D., Pan, C., Matni, N., & Simchowitz, M. *Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control*. arXiv:2507.09061, v1 2025-07-11 | G1 | A |
| 18 | Badithela, A., Snyder, D., Zha, L., Mikhail, J., O'Kelly, M., Dixit, A., & Majumdar, A. *Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators*（SureSim）. arXiv:2510.04354, v1 2025-10-05 | G6、§4.3 | A |
| 19 | Snyder, D., Badithela, A., Matni, N., Pappas, G., Majumdar, A., Itkina, M., & Nishimura, H. *Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison*. arXiv:2603.13616, v1 2026-03-13 | G5 / G6、§4.1 | A |
| 20 | Bronars, A., Park, Y., & Agrawal, P. *Tune to Learn: How Controller Gains Shape Robot Policy Learning*. arXiv:2604.02523, v1 2026-04-02 | G1；§V-A 原句 | A + B |
| 21 | Wang, Y., Jiang, A., Wang, S., Heng, Y., Yang, H., Chen, Y., & Sun, H. *Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and Bench2Drive*. arXiv:2605.00066, v1 2026-04-30 | G2 | A |
| 22 | Gupta, K. *How VLAs Fail Differently: Black-Box Action Monitoring Reveals Architecture-Specific Failure Signatures*. arXiv:2605.28726, v1 2026-05-27 | G3 | A |
| 23 | Seligmann, F., Gospodinov, E., Dincer, E. U., & Neumann, G. *VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models*. arXiv:2606.21386, v1 2026-06-19 | G3 | A |
| 24 | Huang, H., Zheng, T., Chen, Y., You, J., & Gao, Y. *Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies*. arXiv:2606.29898, v1 2026-06-29 | G1 / G2 / G5；ρ −0.61 → −0.87，TE / DTW 消融 | A + B（§3.2、§4.2、§4.5） |
| 25 | Liao, A., Cui, H., Desingh, K., & Deshwal, A. *Active Real-World Factor-Based Evaluation for Generalist Robot Policies*. arXiv:2607.14439, v1 2026-07-16 | G4 | A |
| 26 | Lazzati, F., Stachowicz, K., Chen, W., Metelli, A. M., Wagenmaker, A., & Levine, S. *Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?* arXiv:2608.02547, v1 2026-08-03 | G3；脚注 4（指数加权削弱集成） | A + B |
| 27 | Zeng, M., Agarwal, A., Bati, A., Lee, B., Ancha, S., & Tedrake, R. *Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies*. arXiv:2608.15938, v1 2026-08-16 | G3 | A |
| 28 | Raghavan, D., & Singh, A. *Do Better Imagined Rollouts Mean Better Robot Control? A Controlled Study of World-Model Evaluation Under Feedback*. arXiv:2609.02811, v1 2026-09-02 | G2 | A |

**检索时命中但未采用：** dWorldEval（arXiv:2604.22152）、DreamDojo（arXiv:2602.06949）、Ctrl-World（arXiv:2510.10125）、PiL-World（arXiv:2606.05773）——世界模型评测路线，与本文问题相关但未逐篇核实，未进入正文。
检索摘要里出现过"robomimic 10–100% 退化"的说法，与作者项目页原文"50 to 100%"不一致，**本文以项目页为准**。

**本地产物引用**

| 出处 | 用于 |
|---|---|
| `experiment_report/act/act_eef-realrobot-vs-offline-2026-09.md`（R9） | §1、§3 全部真机数字 |
| `experiment_report/act/ACT-ACTEEF-results-2026-09.md` 表 4、9、12、13、14、19，§2.3、§10 | 离线数字、限位工况、记忆读数 |
| `experiment_report/MASTER-LEDGER-2026-09.md` §3 | 两条判读线（1.70% / 5.86%） |
| `offline-metric-threshold-accuracy-plan-2026-09.md` §6 | acc@τ 判别力 |
| `accuracy-root-cause-literature-survey-2026-09.md` §6 | 轴 E 的前一版综述（本文是它的展开） |

---

## 附录 A　§1.2 / §1.3 的复核脚本（仅标准库）

```python
# 用法: python3 stats.py
from math import comb, prod, exp
import itertools

def bpmf(k, n, p): return comb(n, k) * p**k * (1 - p)**(n - k)
def fisher(a, n1, b, n2):  # 双侧 Fisher 精确检验
    K, N = a + b, n1 + n2
    P = lambda x: comb(n1, x) * comb(n2, K - x) / comb(N, K)
    po = P(a)
    return sum(P(x) for x in range(max(0, K - n2), min(n1, K) + 1) if P(x) <= po * (1 + 1e-9))
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]: j += 1
        for t in range(i, j + 1): r[s[t]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(x, y):
    rx, ry = rank(x), rank(y); mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    c = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return c / (sum((a - mx)**2 for a in rx) * sum((b - my)**2 for b in ry))**.5

# 唯一条件 -> 成功次数/30（R9 表 A、B 去重）
cells = {('505',200,.01):24, ('505',100,.01):21, ('505',200,-.05):15, ('533',200,.01):18,
         ('616',200,.01):24, ('616',100,.01):21, ('616',200,-.05):15,
         ('616as',200,.01):24, ('616as',100,.01):21, ('616as',200,-.05):15}
print("唯一条件", len(cells), "-> 试验", 30 * len(cells))
pdiv = [sum(bpmf(k, 30, c / 30) for k in range(0, 31, 3)) for c in cells.values()]
print("10 格都能被 3 整除: %.1e" % prod(pdiv))
peq = lambda p, m: sum(bpmf(k, 30, p)**m for k in range(31))
print("孪生三格全等: %.4f ; 三份权重九格全等: %.1e"
      % (prod(peq(p, 2) for p in (.8, .7, .5)), prod(peq(p, 3) for p in (.8, .7, .5))))
for lab, (a, n1, b, n2) in {"coeff": (72,90,45,90), "533 vs 其余": (18,30,72,90),
                            "533 vs 孪生": (18,30,48,60), "200k vs 100k": (72,90,63,90)}.items():
    print("%-12s n=30: p=%.2g | 有效 n=10: p=%.3f"
          % (lab, fisher(a, n1, b, n2), fisher(a // 3, n1 // 3, b // 3, n2 // 3)))
real = [60, 80, 80, 50, 50]
for name, m in (("MAE", [.04391, .04638, .04626, .02803, .02817]),
                ("二阶差分", [.000224, .000212, .000210, .000256, .000256])):
    obs = abs(spearman(real, m)); ps = list(itertools.permutations(m))
    print("%s |rho|=%.3f 精确置换 p=%.3f"
          % (name, obs, sum(abs(spearman(real, q)) >= obs - 1e-9 for q in ps) / len(ps)))
for c in (0.01, 0, -0.05, 0.05):  # TE 几何复算: w_i=exp(-c*i), i=0 为最老
    w = [exp(-c * i) for i in range(100)]; S = sum(w)
    lag = sum(wi * (99 - i) for i, wi in enumerate(w)) / S
    print("coeff %+.2f 滞后 %.2f s N_eff %.1f" % (c, lag / 30, S * S / sum(x * x for x in w)))
```

期望输出（2026-09-11 实跑）：

```
唯一条件 10 -> 试验 300
10 格都能被 3 整除: 1.7e-05
孪生三格全等: 0.0015 ; 三份权重九格全等: 3.3e-06
coeff        n=30: p=4e-05 | 有效 n=10: p=0.029
533 vs 其余  n=30: p=0.049 | 有效 n=10: p=0.232
533 vs 孪生  n=30: p=0.075 | 有效 n=10: p=0.384
200k vs 100k n=30: p=0.17 | 有效 n=10: p=0.552
MAE |rho|=0.949 精确置换 p=0.067
二阶差分 |rho|=0.973 精确置换 p=0.067
coeff +0.01 滞后 1.92 s N_eff 92.4
coeff +0.00 滞后 1.65 s N_eff 100.0
coeff -0.05 滞后 0.63 s N_eff 39.5
coeff +0.05 滞后 2.67 s N_eff 39.5
```

TE 几何四行与 R9 表 17 逐位一致，作为脚本自检。
