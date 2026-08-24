# DAgger 这条线到底在解什么问题——两篇原始论文精读 + 研究方向综述 + 对我们数据配比的结论

> 撰写日期：2026-08-24
> 触发问题：`https://www.alphaxiv.org/pdf/2506.16685v5`（CR-DAgger）与 `https://www.alphaxiv.org/pdf/1011.0686v3`（DAgger 原文）讲了什么；
> DAgger 这条研究线的全貌；有没有同构的替代策略；对我们的数据配比意味着什么。
> 配套阅读：[`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md)（本文是它的文献侧补全，会**修正**其中 §6.2 的两处档位设定）、
> [`team-division-strategy-2026-08.md`](./team-division-strategy-2026-08.md) §「cfw + qcy · DAgger 数据」
>
> **文献验证方法**：本文引用的每一篇论文都逐篇 WebFetch 了 `arxiv.org/abs/<id>` 活页，
> 核对标题、作者、v1 日期与摘要原文；两篇主论文额外下载 PDF 用 `pdftotext -layout` 取全文，
> 所有数字均从正文/表格原文抄录，不使用摘要转述。
> **一处必须说明的坑**：对 1011.0686 的 PDF 做自动摘要时，摘要器编造了三组实验数字
> （"手写识别 53.3% → 97.8%"、"Mario 20 条示教"、"O(ε+√(NTε)) 界"），
> 与原文（82% / 83.6% / 85.5%，5000 点每轮，J(π̂) ≤ J(π\*)+uTε_N）**全部不符**。
> 本文用的是 PDF 原文。引用二手摘要的风险在这里有实例。

---

## 0. 一页结论

**三句话：**

1. **DAgger 从来不是"往训练集里掺失败数据"的方法。** 它的全部收益来自一件事：
   标签是在**当前策略自己走出来的状态**上取的。Ross 等人证明的是 BC 的 `T²ε` 退化在
   on-policy 标注下降到 `uTε`——这是**分布**的性质，不是**数据量**或**配比**的性质。
   我们那 36/72 条人演的失败 episode 不满足这个前提，所以 DAgger 的任何定理和任何配比经验对它都不成立。
   这不是措辞问题：它决定了"调比例"这件事在我们当前数据上**先验期望就是 0**。

2. **"纠错数据该占多少比例"这个问题，文献的答案是：这个问题本身提错了。**
   十五年里所有 work 的方案只有三种，没有第四种——
   (a) **加权到采样后各占 50%**（IWR、Sirius，两篇独立收敛到 0.5）、
   (b) **完全隔离到一个单独的小网络**（CR-DAgger 残差，主策略见到 0% 纠错数据）、
   (c) **直接扔掉一边**（HG-DAgger 只留接管段）。
   **"把少量纠错数据按原比例混进大池子里均匀采样"这一档，是被实测点名过的失败档**：
   IWR-NB（同样的数据，只是不做平衡采样）87.3% → 74.7%，掉回 Full-Demos 基线水平；
   CR-DAgger 的 retrain 基线在书本任务上是 **−1.67%**。我们现在正好在这一档。

3. **对我们最直接的三条：**
   - 我们 §6.2 方案 C 写的"权重 3–5×"**偏低**。要打到文献口径的 50/50，权重应满足 `w = (1−p)/p`；
     若只升权真正的纠错片段（≈总帧的 2–3%），`w` 在 **30–40×** 量级，不是 3–5×。（§3.3）
   - 我们 §6.2 第 3 条"只保留纠错片段 **+ 短前导**"里的**前导要删掉**，而且不是"少放一点"，是**权重设 0**。
     两篇论文各自独立测到这一点：CR-DAgger 的 dense-around 采样比 dense-after 差，
     原文归因为"这些大多是负样本"；Sirius 直接设 `P*(preintv) = 0`。（§3.4）
   - **批次要大、轮次要少。** CR-DAgger：batch=10 的多轮 DAgger 在书本任务上最终 **0%**，
     batch=50 单批 **100%**，同样的数据量同样的 epoch；belt 任务三批（90.6 → 96.8 → 87.5）在第三批**掉头向下**。
     我们不要规划"每周回流一点失败数据"的长流水线。（§3.5）

**一句话行动版**：在补上闭环评测（`failure-data` §6.4 #1）和 on-policy 采集之前，
不要再花时间调 36/72/108 这个比例；那条曲线在文献里是平的。

---

## 1. 两篇论文

### 1.1 DAgger 原文：A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

| 项 | 内容 |
|---|---|
| 作者 | Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell |
| arXiv | [1011.0686](https://arxiv.org/abs/1011.0686)，v1 2010-11-02，v3 2011-03-16 |
| 发表 | AISTATS 2011 |

#### 核心问题

序贯决策里，`s_{t+1}` 依赖于 `a_t = π(s_t)`，所以**学出来的策略自己改变了测试分布**，
i.i.d. 假设破了。论文用 `d_π = (1/T)Σ_t d^t_π` 记策略 π 诱导的平均状态分布，
`J(π) = T·E_{s~d_π}[C_π(s)]` 记 T 步总代价。

#### 核心结论（这是全文的主轴，也是我们最该记住的一句）

**Theorem 2.1**（引自 Ross & Bagnell 2010）：若 `E_{s~d_{π*}}[ℓ(s,π)] = ε`（即在**专家分布**上的误差是 ε），
则

```
J(π) ≤ J(π*) + T²ε
```

论文明确写了这个界**是紧的**（tight）——存在问题实例使额外代价真的按 `T²` 增长
（脚注引 Kääriäinen 2006 的构造，误差期望为 `T/2 − (1−(1−2ε)^{T+1})/(4ε) + 1/2`，小 ε 下行为是 `Θ(T²ε)`）。
**这就是纯 BC 的天花板，而且它是分布的问题，不是模型容量或数据量的问题。**

**Theorem 2.2**（本文推广版）：若 π 在**自己诱导的分布**上误差是 ε，且 `Q*_{T−t+1}(s,a) − Q*_{T−t+1}(s,π*) ≤ u`，则

```
J(π) ≤ J(π*) + uTε
```

`T²` → `uT`。整篇论文剩下的部分就是造一个能达到第二个式子的算法。

#### 算法（Algorithm 3.1 原文）

```
D ← ∅ ;  π̂₁ ← Π 中任意策略
for i = 1..N:
    π_i = β_i·π* + (1−β_i)·π̂_i          # 混合策略，专家以 β_i 概率接管
    用 π_i 采 T 步轨迹
    D_i = {(s, π*(s))}                    # 状态由 π_i 访问，动作由专家标注
    D ← D ∪ D_i                           # 关键：聚合，不是替换
    在 D 上训练 π̂_{i+1}
返回验证集上最好的 π̂_i
```

三个必须记住的设计点：

1. **状态来自策略，动作来自专家。** 这就是 on-policy correction 的原始定义。
2. **`D ← D ∪ D_i` 是聚合。** 每轮的新数据在总池里只占 `1/N`，
   **DAgger 从不调"新数据占多少"**——因为它池子里每一批都是 on-policy 的，不存在"两类数据"。
   我们的场景里存在两类数据，这本身就说明我们做的不是 DAgger。
3. **`β_i` 的唯一要求是 `β̄_N = (1/N)Σβ_i → 0`。** 论文说 `β_i = I(i=1)`
   （第一轮纯专家，之后纯策略）这个 parameter-free 版本"在实践中常常最好"。
   Follow-The-Leader 的 no-regret 性质是全部理论的来源；把 no-regret learner 换成别的也成立。

**Theorem 3.1 / 3.2**：`N` 取 `Õ(T)` / `Õ(uT)` 时，序列中存在 π̂ 使
`E_{s~d_π̂}[ℓ] ≤ ε_N + O(1/T)`，`J(π̂) ≤ J(π*) + uTε_N + O(1)`。
有限样本版（Thm 3.3/3.4）：每轮采 `m = O(1)` 条轨迹、`N = O(T²log(1/δ))`（利用强凸性可降到 `Õ(T log(1/δ))`）。

#### 实验（原文数字，非转述）

| 任务 | 设置 | 结果 |
|---|---|---|
| Super Tux Kart | 每轮 1 圈（≈1000 个点），共 20 轮，线性 ridge 回归控制器，5 Hz | DAgger 15 轮后**再也不掉下赛道**；SMILe 20 轮后仍平均每圈掉 2 次；**supervised 随数据增加不改善** |
| Super Mario Bros | 每轮 5000 点（每关约 150 点），20 轮，4 个线性 SVM | `β_i=0.5^{i−1}` 得 3030，`β_i=I(i=1)` 得 2980，`β_i=0.9^{i−1}` 收敛明显更慢；supervised 停滞 |
| 手写识别（Taskar OCR，≈6600 词/52000 字符） | 20 轮，线性 SVM，贪心一遍解码 | 无结构 82% → supervised 83.6% → **DAgger 85.5%** |

**Super Tux Kart 那句话是我们最该抄在墙上的**：原文解释 supervised 不涨的原因是
*"most of the training laps are all very similar and do not help the learner to learn how to recover from mistakes it makes"*。
我们有 361 条成功 episode、全部来自同一套摆位分布——第 362 条的闭环边际价值，先验就接近 0。

**Mario 那条 β 的观察也值得记**：纯 on-policy（`β=I(i=1)`）反而略差于 `β=0.5^{i−1}`，
原文归因为纯 on-policy 早期会产出大量"Mario 卡在同一个地方"的**冗余重复状态**；
让专家偶尔介入既能观察到那些状态、又能把它解卡，从而采到更宽的有用数据。
**"纠错数据里混一点点专家接管"在原始论文里是有正收益的**，但方向和我们做的相反：
它是在 on-policy 数据里掺专家，不是在专家数据里掺失败。

---

### 1.2 CR-DAgger：Compliant Residual DAgger

| 项 | 内容 |
|---|---|
| 作者 | Xiaomeng Xu\*, Yifan Hou\*, Chendong Xin, Zeyi Liu, Shuran Song（Stanford） |
| arXiv | [2506.16685](https://arxiv.org/abs/2506.16685)，v1 2025-06-20，v5 2025-12-25 |
| 发表 | NeurIPS 2025（会议版）；v5 是期刊扩展版 |
| 版本沿革 | **v1 标题是 _What Matters in DAgger? An Empirical Study on Improving Real World Robot Learning with Human Corrections_**——这个旧标题更贴近它对我们的用途 |

v5 相对会议版增加：gear insertion + cable routing 两个任务；§3.1 的 intention-misinterpretation 分析与修复；§4.5 追踪误差消融。

#### 它提的两个问题，就是我们的两个问题

原文 Introduction 逐字：

- *How to collect informative human correction data?*
- *How to effectively update the policy with new data?*

#### 组件 A：Compliant Intervention Interface（怎么采）

不是接管（take-over），是**在策略继续跑的同时叠加 delta 力**：

- 末端装把手，人直接推机械臂；背景跑**导纳控制**（admittance control）
  `M q̈ = K(q_ref − q) − D q̇ + F`，虚拟刚度 ≈ **1000 N/m**；
  `q_ref` 是基础策略的输出，人只能"影响"不能"覆盖"。
- ATI 六维力传感器记录接触力；把手上一个按键记录介入起止时刻。
- 修正的关键 bug（v5 新增）：**intention misinterpretation**。
  纠错量 `q_delta[t] = q[t] − q_ref[t]`，当**跟踪误差大于人的纠错幅度**时，
  记录下来的纠错方向可能**与人的真实意图相反**。修复两条：
  控制律加速度跟踪项 `M q̈ = K(q_ref−q) + D(q̇_ref−q̇) + F`（空间上降误差）；
  取反馈时加 look-ahead `q_delta[t] = q[t] − q_ref[t−Δt]`（时间上降误差）。
  最大跟踪误差从 **30 mm 降到 <5 mm**。在 gear insertion 上，修复前残差策略学会的是**把齿轮往外拔**。

> 这条对我们有直接映射：我们的 `vita-deploy` 链路里 100% 的 chunk 被替换成 Hermite S 曲线、
> 每周期还有 60–100 ms 硬停（见 `team-division-strategy` §0）。
> **在那种链路上采 delta 纠错数据，采到的就是 S 曲线的误差，不是人的意图。**
> CR-DAgger 这一节等于提前告诉我们：链路不修，DAgger 数据不能采。

#### 组件 B：Compliant Residual Policy（怎么用）

| | 基础策略 | 残差策略 |
|---|---|---|
| 模型 | Diffusion Policy | 冻结基础策略的图像编码器 + TCN 编力 + MLP 头（**≈2 MB 可训练权重**） |
| 频率 | 1 Hz | **50 Hz** |
| 输入 | 图像 `I_t`、本体 `P_t` | 同上 + **力 `F_t`** |
| 输出 | 32 帧末端位姿，间隔 0.1 s | 5 帧 delta 位姿 + **目标力**，间隔 0.02 s（15 维：9 维 SE3 delta + 6 维期望 wrench） |
| 训练数据 | 150–405 条示教 | **只有 50–100 条纠错 episode** |

三个 advantage 原文列的：sample-efficient（小网络小数据）、
**可以引入基础策略没有的新模态**（力）、高频推理（接触事件里的反应式纠错）。

**训练策略里两条与配比直接相关：**

1. *Ensure sufficient coverage of in-distribution data* ——
   只用纠错时刻训练，残差会在"不该纠错"的地方外推爆掉。两种解法：
   (1) 把 no-correction 段也放进来、**标成零残差动作**；
   (2) 采少量"全程标为纠错"的轨迹（§A.3：**50 条里的 15 条**）。
   原文经验：基础策略稳、成功率高时用 (1)；否则用 (2)。书本任务用 (1)，其余三个任务用 (2)。
2. *Prioritize correction data over no-correction data* ——
   **改采样频率**，在介入开始后的一小段窗口内把采样频率**提高 4 倍**。

#### 实验与消融（原文数字）

基础策略与失败模式：

| 任务 | 基础策略示教数 | 基础成功率 |
|---|---:|---:|
| Book Flipping | 150 | 40% |
| Belt Assembly | 405 | 15.6%（加力输入 → 43.8%） |
| Cable Routing | 200 | 25% |
| Gear Insertion | 150 | 5% |

**主结果（Fig. 9）**：CR + On-Policy-Delta 在书本 40 → 100%、belt 15.6 → 96.8%、
gear 5 → 90%、cable 25 → 100%；摘要口径"四任务平均 +64%"。

**五条 Key Finding，第 4、5 条是给我们的：**

| # | 结论 | 数字 |
|---|---|---|
| 1 | 残差能大幅超基础策略 | 书本 +60%、belt +50% |
| 2 | 残差可以引入新模态（力） | 比最好的 position-only 高 45%（书本）/ 53%（belt） |
| 3 | On-policy delta 数据比 take-over 数据更稳 | 同为残差策略，书本任务 **+45%**；Fig.10 显示 take-over 在控制权切换瞬间速度尖峰明显更大；Fig.11 显示 delta 数据的轨迹分布与基础策略训练分布更贴合 |
| 4 | **retrain（把纠错数据和原数据混在一起从头训）稳，但学不到纠错** | 书本 **−1.67%**、belt **+18.8%**，都远低于残差。原文归因：*"its behavior is less affected by the small amount of correction data compared to the dominant portion of initial data"* |
| 5 | **finetune（只用纠错数据微调）最差，还不如基础策略** | 书本 **−30%**、belt **−15.6%**；加 KL 正则能压住噪声动作但总成功率仍低于其他基线 |

**§4.5 三组消融（这三组是本文对我们最值钱的部分）：**

| 消融 | 设置 | 结果 |
|---|---|---|
| **批次大小** | batch=50 单批 vs. 20 条预热 + 3×10 条增量（总量与 epoch 相同） | **batch=50 → 100%；batch=10 多轮 → 最终 0%**（评测时总是插得太高而失败）。原文归因：小批多轮 = 非平稳分布上反复微调 = 灾难性遗忘，且随轮次增加人不得不给越来越大的纠错 |
| **批次数量** | belt 任务 1/2/3 批 | **90.6% → 96.8% → 87.5%**，第三批掉头。原文归因：每一批采集时失败模式都变了，50 条新数据不足以覆盖新分布 |
| **采样策略** | uniform / dense-around-intervention-start / dense-after-intervention-start | **dense-after 最好**。dense-around 更差，原文逐字：介入开始**之前**那段是人刚看出要失败的时刻，*"These are mostly negative data, and using them for training decreases the policy success rate."* |

**Limitation 里一条我们用得上的经验法则**：
*"we recommend starting to collect correction data for the residual policy when the base policy has at least 10% ∼ 20% success rate."*
——基础策略太差时，纠错数据没有意义。我们白板上的 60–70%（虽然无出处）远在这条线之上，
所以**"该不该做 DAgger"这个问题对我们是 yes**；问题只在于怎么做。

---

### 1.3 十五年之间，什么变了什么没变

| | Ross 2011 | CR-DAgger 2025 |
|---|---|---|
| **没变**：收益来源 | 标签取在 `d_π̂` 上 | 同左（原文明确引 Ross et al. 2011 论证 delta 修正的合法性） |
| **没变**：BC 的敌人 | `T²ε` 复合误差 | 同左 |
| 专家怎么给 | 全程标注所有访问状态（`π*(s)`，人不控制机器人） | 人加力，只给 delta，机器人一直在跑 |
| 谁决定何时介入 | 算法（`β_i`） | 人（human-gated） |
| 轮次结构 | `N = Õ(T)` 轮，每轮 `m = O(1)` 条 | **1–2 批，每批 50 条**（多轮实测更差） |
| 新数据怎么进模型 | 聚合进同一个池子重训（每轮都是 on-policy，没有"两类数据"） | **单独的 2 MB 残差网络**，主策略冻结 |
| 理论 | 有（no-regret 归约） | 无（纯实证；Limitation 里把 base/residual 预算权衡列为 open problem） |

**要点**：CR-DAgger 把 DAgger 的**理论内核（on-policy 标注）保留**，
把 DAgger 的**工程外壳（多轮小批 + 聚合重训）几乎全部推翻**。
这两条恰好对应我们要做的两个决定：怎么采（保留内核）、怎么用（不要照抄外壳）。

---

## 2. DAgger 这条研究线的全貌

### 2.1 三条正交主轴

十五年的 variants 几乎都可以放进这三个坐标：

```
轴 A：谁决定何时介入 / 何时查询专家
  算法门控 (algorithm-gated) ────── 机器人门控 (robot-gated) ────── 人门控 (human-gated)
  DAgger β_i                        SafeDAgger / EnsembleDAgger /    HG-DAgger / IWR / Sirius /
                                    DropoutDAgger / ThriftyDAgger /  RoboCopilot / CR-DAgger
                                    LazyDAgger / Diff-DAgger / ARMADA

轴 B：专家给的是什么
  全动作标注 ── 接管轨迹段 ── delta 力/位姿修正 ── 偏好/评价 ── 语言批评
  DAgger        HG-DAgger      CR-DAgger / TER-DAgger    RLIF        Language-Critique

轴 C：新数据怎么进模型
  聚合重训 ── 加权 BC ── 单独残差网络 ── 隐空间适配 ── 当作 RL 奖励 ── 数据筛选
  DAgger      IWR/Sirius   CR-DAgger/PolicyDecorator  FlowDAgger  RLIF/HIL-SERL  CUPID/Demo-SCORE
```

**我们的问题全部落在轴 C，而轴 A/B 是轴 C 的前提。**
这是本报告最重要的结构性判断：我们现在在争论轴 C（比例/权重），
但我们的数据在轴 A 上是"没有门控"（人预先演出来的失败），在轴 B 上是"完整替代轨迹"。
**轴 A/B 不成立时，轴 C 的所有已知结论都不适用。**

### 2.2 谱系（全部经 arXiv 活页验证）

**理论主干**

| 论文 | ID / 时间 | 与我们的关系 |
|---|---|---|
| A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | [1011.0686](https://arxiv.org/abs/1011.0686)，2010-11 | `T²ε → uTε`；本文 §1.1 |
| Reinforcement and Imitation Learning via Interactive No-Regret Learning（AggreVaTe） | [1406.5979](https://arxiv.org/abs/1406.5979)，2014-06 | Ross & Bagnell 把 DAgger 从"模仿动作"扩展到"利用 cost-to-go"；同一作者的后续统一视角 |
| Toward the Fundamental Limits of Imitation Learning | [2009.05990](https://arxiv.org/abs/2009.05990)，2020-09 | Rajaraman 等给出 minimax 下界 `Ω(|S|H²/N)`——**`H²` 是纯离线模仿的信息论极限，与算法无关**。这一条比 DAgger 更硬：不做交互，二次项就消不掉 |

**机器人门控（robot-gated）：让机器人自己判断什么时候求助**

| 论文 | ID / 时间 | 一句话 |
|---|---|---|
| SafeDAgger（Query-Efficient Imitation Learning for End-to-End Autonomous Driving） | [1605.06450](https://arxiv.org/abs/1605.06450)，2016-05 | Zhang & Cho；学一个 safety 分类器决定何时查专家 |
| EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning | [1807.08364](https://arxiv.org/abs/1807.08364)，2018-07 | 集成方差当置信度，同时约束"与专家的偏差"和"认知不确定性" |
| LazyDAgger: Reducing Context Switching in Interactive Imitation Learning | [2104.00053](https://arxiv.org/abs/2104.00053)，2021-03 | 上下文切换比 SafeDAgger 少 60%，同时成功率高 60% |
| ThriftyDAgger: Budget-Aware Novelty and Risk Gating | [2109.08273](https://arxiv.org/abs/2109.08273)，2021-09 | **给定人力预算**下的主动查询；novelty + risk 双门 |
| Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision | [2206.14349](https://arxiv.org/abs/2206.14349)，2022-06 | 100 机器人 / 少数人；提出 ROHE（Return on Human Effort）指标 |
| Diff-DAgger: Uncertainty Estimation with Diffusion Policy | [2410.14868](https://arxiv.org/abs/2410.14868)，2024-10 | 扩散策略的不确定度怎么算（把"多模态分歧"与"真实不确定"分开）；失败预测 +39.0%，ICRA 2025 |
| ARMADA: Autonomous Online Failure Detection and Human Shared Control | [2510.02298](https://arxiv.org/abs/2510.02298)，2025-10 | FLOAT 失败检测 ≈95% 准确；多轮部署成功率 >4×，人介入率 <1/2 |

**人门控（human-gated）：人看着，觉得要坏了就动手**

| 论文 | ID / 时间 | 一句话 |
|---|---|---|
| HG-DAgger: Interactive Imitation Learning with Human Experts | [1810.02890](https://arxiv.org/abs/1810.02890)，2018-10 | 人接管；**只保留接管段**，丢弃策略段 |
| Human-in-the-Loop Imitation Learning using Remote Teleoperation（**IWR**） | [2012.06733](https://arxiv.org/abs/2012.06733)，2020-12 | Mandlekar 等；**本文 §3.2 的核心证据** |
| Robot Learning on the Job（**Sirius**） | [2211.08416](https://arxiv.org/abs/2211.08416)，2022-11 | 四类样本加权 BC；RSS 2023，IJRR 2025 |
| RoboCopilot: Human-in-the-loop Interactive Imitation Learning | [2503.07771](https://arxiv.org/abs/2503.07771)，2025-03 | 双臂、双边遥操，人机控制权无缝切换 |
| **Compliant Residual DAgger** | [2506.16685](https://arxiv.org/abs/2506.16685)，2025-06 | 本文 §1.2 |
| Force-Aware Residual DAgger via Trajectory Editing（TER-DAgger） | [2603.04038](https://arxiv.org/abs/2603.04038)，2026-03 | 与 CR-DAgger 高度同构的独立工作：优化式轨迹编辑学残差 + 力感知失败检测 + Cartesian 阻抗控制，报 +37% |
| DexHiL: Human-in-the-Loop for VLA Post-Training in Dexterous Manipulation | [2603.09121](https://arxiv.org/abs/2603.09121)，2026-03 | **intervention-aware 采样策略：优先采纠错段**做 VLA 后训练；比离线微调高约 25% |
| FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space | [2607.08877](https://arxiv.org/abs/2607.08877)，2026-07 | action inversion：把人的动作反演成冻结基础模型的噪声，在隐空间学轻量策略；**保住 held-out 任务上的原技能** |

**把介入当奖励（RL 化）**

| 论文 | ID / 时间 | 一句话 |
|---|---|---|
| RLIF: Interactive Imitation Learning as Reinforcement Learning | [2311.12996](https://arxiv.org/abs/2311.12996)，2023-11 | **把"人介入了"本身当负奖励**，不需要人给出正确动作；专家次优时优于 DAgger；ICLR 2024 |
| Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL（HIL-SERL） | [2410.21845](https://arxiv.org/abs/2410.21845)，2024-10 | 示教 + 人纠错 + 高效 RL，1–2.5 小时真机训练，成功率平均 2× |
| Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model | [2412.13630](https://arxiv.org/abs/2412.13630)，2024-12 | 与 CR-DAgger 同构但**残差用 RL 在线学**；保住 IL 的平滑运动，避免纯 RL 的抽动 |

**综述**

| 论文 | ID / 时间 |
|---|---|
| Interactive Imitation Learning in Robotics: A Survey | [2211.00600](https://arxiv.org/abs/2211.00600)，2022-10（Celemin, Kober 等，10 作者） |
| Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives | [2506.00098](https://arxiv.org/abs/2506.00098)，2025-05 |

### 2.3 同构但不叫 DAgger 的策略（"有没有类似的做法"）

这四族解的是同一个问题——**如何让训练分布覆盖策略实际会到的状态**——但手段完全不同。
对我们的价值在于：**它们中有两族不需要 on-policy 采集，因此在我们链路修好之前就能做。**

**(1) 噪声注入：把覆盖做在采集时，而不是采集后**

- **DART: Noise Injection for Robust Imitation Learning**（[1703.09327](https://arxiv.org/abs/1703.09327)，2017-03，Laskey/Goldberg）。
  在专家演示时注入噪声，逼专家演出"从偏离状态恢复"的行为，噪声幅度自动匹配学出来策略的误差分布。
  完全 off-policy，人不需要盯着机器人。原文报：高维任务上比 DAgger **快 3×**，
  且专家自身回报只掉 5%（DAgger 掉 80%），抓取任务比 BC **+62%**。
  **对我们：这是唯一一个"不改部署链路就能做"的覆盖类方法。** 代价是需要能在采集时给遥操作叠噪声。

**(2) 数据筛选/影响函数：不加数据，减数据**

- **CUPID: Curating Data your Robot Loves with Influence Functions**（[2506.19121](https://arxiv.org/abs/2506.19121)，2025-06，CoRL 2025）。
  用影响函数估计每条示教对**闭环期望回报**的因果影响，据此排序。
  两个用法：**滤掉有害的训练示教**、**从新采的数据里挑最有用的**。
  原文：**用不到 33% 的精选数据**就能在 robomimic 上得到 SOTA 扩散策略。
- **Demo-SCORE（Curating Demonstrations using Online Experience）**（[2503.03707](https://arxiv.org/abs/2503.03707)，2025-03，Chen/Finn）。
  用策略自己的 rollout 训一个"成功 vs 失败"分类器，反过来筛训练示教。绝对成功率 **+15–35%**。
- **Quality over Quantity: Demonstration Curation via Influence Functions**（[2603.09056](https://arxiv.org/abs/2603.09056)，2026-03，ICRA 2026）。
  把"质量"定义为**单个样本对验证示教损失的贡献**；两处改动：取跨验证样本的**最大**影响、在轨迹内聚合分数。
- **Data Quality in Imitation Learning**（[2306.02437](https://arxiv.org/abs/2306.02437)，2023-06，Belkhale/Sadigh）。
  提出 action divergence 与 transition diversity 两个属性，并明确指出 **state diversity 不总是有益**。

  > **这一族对我们特别重要**：`failure-data` §3.3 测到策略对背景像素的依赖强于对物体的依赖。
  > 那正是 CUPID / Demo-SCORE 会打出负影响分的那类数据。
  > **"该加什么数据"和"该删什么数据"是同一个问题的两面，而删这一面我们一次都没做过。**

**(3) 负样本引导：把失败当"往哪儿别走"的信号，而不是当监督**

- **AFIL / Failing Forward: Adaptive Failure-Informed Learning for VLA**（[2605.08434](https://arxiv.org/abs/2605.08434)，2026-05）。
  训**两个**动作生成器（成功 / 失败）共享视觉-语言骨干，推理时用失败生成器**把动作推离失败区**，
  引导强度由逐步的成功/失败分布距离决定。失败轨迹由预训练 VLA 自己 rollout 产生，不用人演。
- **How to Utilize Failure Demo Data?**（[2605.07560](https://arxiv.org/abs/2605.07560)，2026-05）。
  把失败数据的处理路线分成三类（评价式学习 / 人介入纠错 / 从不完美示教提取信息），
  提出用注意力分布差异做**失败样本选择**的后训练指标——即"哪些失败样本值得留"。
- **EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration**（[2607.19745](https://arxiv.org/abs/2607.19745)，2026-07）。
  人手直接演恢复动作，映射到"corrective-intent space"与机器人对齐，另配一个 recovery gate。
  原文：人演恢复数据的产出速率是遥操作的 **>10×/小时**。
- **Language-Critique Imitation Learning from Suboptimal Demonstrations**（[2607.01225](https://arxiv.org/abs/2607.01225)，2026-07）。
  用语言标签描述进度、指出次优行为、给出纠正指导，配 language-critique 损失。
- **Using Non-Expert Data to Robustify Imitation Learning via Offline RL**（[2510.19495](https://arxiv.org/abs/2510.19495)，2025-10）。
  用离线 RL（而不是 BC）吃下 play data / 次优示教 / 半成品轨迹——
  **这是"失败数据"最有原则的用法：BC 无法表达"这条轨迹是坏的"，价值函数可以。**

**(4) 数据规模侧：先确认边际收益还在不在**

- **Data Scaling Laws in Imitation Learning for Robotic Manipulation**（[2410.18647](https://arxiv.org/abs/2410.18647)，2024-10，v4 2026-06）。
  40000+ 条示教、15000+ 次真机 rollout。核心结论：**泛化性能对"环境数"和"物体数"近似幂律，
  而对"每个环境/物体的示教条数"在越过某个阈值后几乎无效。**
- **What Matters in Learning from Offline Human Demonstrations**（robomimic，[2108.03298](https://arxiv.org/abs/2108.03298)，2021-08）。
  六个算法 × 五个仿真 + 三个真机任务；算法对数据质量的敏感度、评测指标选择是主要变量。

---

## 3. 对我们数据配比的启示

以下每一条都锚定 `failure-data-in-imitation-2026-08.md` 里的具体事实。

### 3.1 第一性结论：我们那 36/72 条不是 DAgger 数据

DAgger 家族的收益来自 `d_π̂` 上的标注。我们的失败 episode（`failure-data` §6.3 第 9 条）：

- 是**人演的**失败，不是策略走出来的失败；
- `§2.1` 显示失败与成功 episode 的夹爪翻转次数**完全相同**，差别只是接近段慢 56%；
- 因此它既不是 `d_π̂` 的样本，也不含 `π*(s)` 在 `d_π̂` 上的标注。

**结论**：Theorem 2.2 的 `uTε` 不适用；IWR/Sirius 的 50/50 不适用；CR-DAgger 的 batch=50 不适用。
**它们全都是关于 on-policy 数据的结论。**
把这批数据按任何比例混进去，先验期望收益 ≈ 0——这与 `failure-data` §6.1 的实测判断一致，
本报告只是给出了它为什么必然如此的理由。

> 这也解释了 `failure-data` §4 的一个费解现象：离线指标上"变差"几乎看不出来。
> 因为这批数据**在离线分布上确实无害**——它只是无关。伤害发生在闭环，
> 而闭环恰好是 DAgger 理论唯一关心的那个分布。

### 3.2 "比例"这个问题，文献只有三个答案，且没有一个是"调一个小比例"

把所有实测放在一起：

| 方案 | 纠错数据在**采样后**的占比 | 实测 |
|---|---|---|
| 均匀混合大池子（我们现在） | 原始比例（我们是 **18.5% 帧**） | **IWR-NB 74.7%**（vs IWR 87.3%，同一份数据）；**CR-DAgger retrain 书本 −1.67%** |
| 加权到 50% | **50%** | IWR 87.3% / 87.5%；Sirius `P*(intv)=0.5`，消融在 0.5 处**取峰值**，两侧都更差 |
| 只用纠错数据微调 | 100% | **CR-DAgger finetune 书本 −30%、belt −15.6%**；HG-DAgger 在 IWR 表里 75.3%（≈ Full Demos） |
| 隔离到残差网络 | 主策略 0%，残差 100% | CR-DAgger 书本 +60%、belt +50%；Policy Decorator、TER-DAgger、FlowDAgger 同路线 |

IWR 完整对照表（原文 Table I，Threading 任务，单操作员，3 seed）：

| 方法 | Round 1 | Round 2 | Final |
|---|---|---|---|
| Base | — | — | 58.0 ± 9.2 |
| Full Demos（同等样本量的完整示教） | — | — | 76.7 ± 2.3 |
| HG-DAgger（只留接管段） | 57.3 ± 9.5 | 62.7 ± 5.0 | 75.3 ± 8.1 |
| **IWR-NB（同数据，不做平衡采样）** | 76.0 ± 6.9 | 72.0 ± 3.5 | **74.7 ± 1.2** |
| **IWR（平衡采样）** | 84.0 ± 5.3 | 90.7 ± 3.1 | **87.3 ± 5.0** |

**IWR-NB 这一行是整份报告里最硬的一条证据**：
数据完全相同、网络完全相同、轮次完全相同，**唯一的差别是采样器做不做类别平衡**，
成功率 87.3% → 74.7%，直接掉回"根本没做纠错"的水平。

**所以：`failure-data` §6.2 表里的方案 A（"降到 ≤5% 只作为确认无害的对照"）判断是对的，
但理由可以更强——不是"无害但也无大益"，而是"这一档有实测证明是无效档"。**

### 3.3 修正一：方案 C 的权重 3–5× 偏低

要把采样后占比压到目标 `q`，若纠错帧原始占比为 `p`，权重需满足

```
w·p / (w·p + (1−p)) = q     ⟹     w = q(1−p) / (p(1−q))
```

取 IWR/Sirius 的 `q = 0.5`：`w = (1−p)/p`。代入我们的数：

| 升权对象 | 原始帧占比 `p` | 打到 50% 需要的 `w` |
|---|---:|---:|
| 整条失败 episode（现状，fail_72） | 18.5% | **4.4×** |
| 整条失败 episode（fail_36） | 10.6% | **8.4×** |
| **只升权真正的纠错片段**（按 `failure-data` §2.2 的 ≈14%/episode 估） | ≈2.6% | **≈37×** |

`failure-data` §6.2 方案 C 写的是"帧比例 5–15%，权重 3–5×"。
若按整条 episode 升权，3–5× 大致落在 30–45% 的采样后占比，接近但偏低；
**若按正确做法（只升权纠错片段），3–5× 只能到 8–12%，远低于文献口径。**

**建议改成：不要指定权重倍数，指定采样后目标占比 `q=0.5`，让 sampler 自己按 `w=(1−p)/p` 算。**
这也正是 IWR 的原文做法——原文说他们不逐数据集调 `α`，而是直接取 `α = |D_R|/|D_I|`
使两个池子等比例采样；Sirius 也用同一个式子 `w(s,a,c) = P*(c)/P(c)`。
**两篇独立工作收敛到"设目标分布而非设倍数"，这个工程细节值得照抄。**

### 3.4 修正二：介入前的"短前导"要删掉，而且是权重设 0

`failure-data` §6.2 第 3 条写"只保留纠错片段 **+ 短前导**"。文献两处独立反对：

- **Sirius** 定义 `preintv` 类（介入前长度 `ℓ` 的一段），显式设 **`P*(preintv) = 0`**，
  原文措辞 *"essentially nullifying the impact of pre-intervention samples"*；
  另外三类：`P*(intv) = 1/2`、`P*(demo) = P(demo)`、`P*(robot)` 自动补齐。
  Fig. 6 右图消融显示**去掉任一类别的权重设计都会掉点**——包括这一条。
- **CR-DAgger** §4.5 采样消融：dense-around（同时加密介入前后）比 dense-after（只加密介入后）差，
  原文逐字：*"Sampling denser around the start of a human intervention also adds more samples right before
  the intervention starts, which is where humans observe signs of failures. These are mostly negative data,
  and using them for training decreases the policy success rate."*

**逻辑很直白**：介入前那一段，正是策略**正在犯错**的动作。
把它当监督信号，等于教策略去做那个错。它有用的只是"这里是个瓶颈"的**位置信息**，
而位置信息已经被"介入开始时刻"这个标注承载了，不需要再喂动作。

**注意 Sirius 的 `ℓ` 是超参**，原文说它取决于人的反应时间。
我们如果按键标注介入起止，人的反应延迟会自动把一段"策略犯错"的帧划进"纠错"区间——
所以标注时**介入起点要往后切一个反应时间**，宁可少标不要多标。

### 3.5 批次策略：大批、少轮，不要做长流水线

CR-DAgger 的两组消融合起来是一条明确的工程告诫：

- batch=10 的多轮增量 DAgger：书本任务最终 **0%**；batch=50 单批：**100%**。同样数据量同样 epoch。
- belt 任务批次数 1/2/3：**90.6 → 96.8 → 87.5**，第三批**掉头**。

原因链条（原文给的）：小批多轮 = 在非平稳分布上反复微调 = 灾难性遗忘；
且每批采集时策略的失败模式都变了，50 条覆盖不了新分布。

**对我们（`team-division-strategy` 里 cfw + qcy 的 DAgger 数据回路）：
不要设计"每周回流 20 条纠错数据"的常态流水线。设计成"一次采 ≥50 条，最多做两轮"。**
IWR 的对照实验给了同向但更细的结论：把三轮的数据量压成一轮，
threading 任务**无显著差异**，coffee machine 任务（更长、瓶颈更多）**低约 7%**。
所以多轮的价值随任务瓶颈数增长——我们的 tidy-up-stationery 是多阶段任务，
可能落在"两轮有价值"这一档，但**三轮以上没有任何文献支持**。

### 3.6 数量下界：比例讨论的前提

`failure-data` §4.2(3) 测到：36 条纠错数据在纠错模式上的泛化误差是拟合误差的 **4.3×**。
文献侧的对应数字：

- **CR-DAgger 全部实验用 50–100 条纠错 episode**，这是一个 ≈2 MB 的残差 MLP；
  他们的 Limitation 明确说这个量级的代价是"缺乏大扰动下的极端鲁棒性"。
- **IWR** 每轮采到"纠错样本数 ≈ 初始数据集样本数的 33%"，三轮。
  按我们 300689 帧的初始集，这个口径下**每轮 ≈ 10 万纠错帧**——比我们现在整个 fail_72（68268 帧）还多。
  注意 IWR 是仿真任务，这个绝对数不能直接搬，但**"纠错帧应与初始集同量级"这个相对口径值得记**。

**结论**：72 条大概率不够，这与 `failure-data` §6.2 末尾的判断一致。
但更重要的是：**在数据是 off-policy 的前提下，加到 200 条也不会有收益**。
先解决 §3.1，再谈量。

### 3.7 采集侧：CR-DAgger 补充给我们清单的三条

`failure-data` §6.3 已有 10 条。文献补三条，都不在原清单里：

1. **先确认基础策略成功率 ≥10–20%**（CR-DAgger Limitation）。我们的 60–70% 满足——
   但**那个 60–70% 目前没有可复现出处**（`team-division-strategy` §0 P0#1）。这条先要落实。
2. **采纠错数据前，先量跟踪误差，并确认它 << 人的纠错幅度**（CR-DAgger §3.1 + §4.5）。
   他们在 gear insertion 上因为这个 bug 学出了**方向相反**的残差。
   我们的部署链路 100% chunk 被替换成 Hermite S 曲线 + 每周期 60–100 ms 硬停
   （`team-division-strategy` §0 P0#2）——**这个量级的跟踪误差足以让 delta 纠错数据符号翻转。**
   **在 jj 的链路修好之前，不要开始采 delta 式纠错数据。** 这是本报告给出的一个硬性先后次序。
3. **要采"零纠错"的正样本，并显式标零残差**（CR-DAgger §3.2 训练策略 + §A.3）。
   否则纠错模型在"不该纠"的地方外推爆掉。两种策略二选一：
   基础策略稳 → 把无纠错段标零残差；基础策略不稳 → 采少量"全程标纠错"的轨迹（他们 50 条里放了 15 条）。

### 3.8 一个我们完全没做过的方向：删数据

`failure-data` §3.3 的遮挡消融（涂掉纯背景条带，开环误差涨 2.52×；涂掉桌面任务区，只涨 2.00×）
说明训练集里存在**被策略当特征用的有害相关性**。
CUPID / Demo-SCORE / QoQ 这一族恰好是量化并删除这类数据的工具：

- **Demo-SCORE** 的做法在我们这里几乎是现成的：用策略 rollout 训"成功 vs 失败"分类器再回头筛示教。
  **但它需要闭环 rollout 数据**——又回到 `failure-data` §6.4 #1。
- **CUPID** 需要影响函数计算，工程量更大，但它给的是"这条示教对闭环回报的因果影响"，
  正好是我们现在完全没有的那个量。
- **QoQ**（ICRA 2026）只需要一个验证示教集 + 影响函数，不需要 rollout，**在我们现在的条件下就能跑**。

**建议把"删数据"作为 yw 那条线（数据质量验证）与 cfw/qcy 这条线的交汇点**：
在加任何纠错数据之前，先量一遍现有 361 条成功示教里有没有负影响样本。
按 CUPID 的数字（<33% 精选数据达到 SOTA），这条路径的潜在收益不比加数据小，而且**不需要采集**。

---

## 4. 修正后的方案表（替换 `failure-data` §6.2）

| 方案 | 前提 | 配比设定 | 期望 | 优先级 |
|---|---|---|---|---|
| **A. 调比例（≤5%）** | 无 | — | **文献实测无效档**（IWR-NB / CR-DAgger retrain）。仅可作"确认无害"对照 | 跳过 |
| **A′. 删数据**（QoQ / Demo-SCORE） | QoQ 只需验证集；Demo-SCORE 需闭环 rollout | 不加数据，删 | CUPID 口径：<33% 数据达 SOTA | **立刻可做（QoQ）** |
| **B. 条件化 + 高比例** | 数据要有标签 | 纠错片段帧 20–50% | 有正收益但依赖条件信号真的可分 | 中 |
| **C. 加权 BC（IWR/Sirius）** | 只改 sampler | **设目标占比 `q=0.5`，`w=(1−p)/p` 自动算；`preintv` 权重设 0** | IWR：87.3% vs 74.7%（同数据） | **改动最小，先做** |
| **D. 残差策略（CR-DAgger）** | 需 on-policy delta 采集 + 低跟踪误差链路 | 主策略 0%，残差 100%，50–100 条，1–2 批 | 四任务 +64% | 目标态，但**被链路阻塞** |
| **E. 噪声注入（DART）** | 采集端能叠噪声 | 不涉及配比 | 唯一不需要改部署链路的覆盖类方法 | **值得评估** |

**路径建议：A′ → C → （链路修好后）D。**
B 的位置从"唯一稳定正收益档"下调——它的证据基础（2605.07560 等）弱于 C 的 IWR/Sirius 对照实验，
且 C 不需要动模型结构。

---

## 5. 反方与局限（devil's advocate）

请连同结论一起读。

1. **IWR/Sirius 的 50/50 里，"非纠错"那一半是机器人自己的 on-policy 轨迹，不是人的示教。**
   IWR 原文说非纠错样本的作用是 *"regularization that keeps the policy close to previous policy iterates"*。
   我们的非纠错数据是 361 条人类示教——它承担不了"贴近上一轮策略"这个角色。
   **所以 `q=0.5` 这个数搬到我们这里，理论依据是打折的。**
   本文把它作为起点值而不是最优值，真正的最优值必须我们自己测。
2. **CR-DAgger 的 batch=50 是在 150–405 条基础示教、任务较短、残差网络仅 2 MB 的设定下测的。**
   我们的 ACT 是端到端重训，不是小残差；"批次大小"在两种设定下不是同一个量。
   可迁移的是**方向**（大批优于小批多轮），不是**数值**。
3. **CR-DAgger 与 TER-DAgger 都是接触密集的精密装配任务（belt/gear/insertion），力反馈是它们收益的主要来源之一**
   （去掉力后书本掉 45%、belt 掉 53%）。我们的 tidy-up-stationery 不是力主导任务，
   **所以 CR-DAgger 的绝对增益不能期待**，能迁移的是它的数据侧结论（批次、采样、前导）。
4. **本报告所有对我们数据的判断，仍然建立在没有闭环成功率的基础上。**
   `failure-data` §7.2 的六条局限全部继承。文献能告诉我们"哪些方案在别人那里无效"，
   不能替我们测"我们的方案有没有效"。
5. **我没有找到任何一篇论文，测过"人预先演的失败数据按小比例混入成功数据"这个确切设定。**
   最接近的是 2605.07560（失败数据选择）和 AFIL（失败当负引导），但两者都不是 naive 混合。
   这既意味着我们的负面观察没有直接文献反证，也意味着**这个设定之所以没人测，
   很可能是因为它在理论上就没有理由起作用**——本报告 §3.1 就是这个理由。

---

## 6. 参考文献

全部经 `arxiv.org/abs/<id>` 活页逐篇验证（标题 / 作者 / v1 日期 / 摘要原文），2026-08-24。
两篇主论文及 IWR、Sirius 另经 `pdftotext -layout` 取全文核对数字。

**主论文**

1. Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) (v1 2010-11-02, v3 2011-03-16)
2. Xu, X., Hou, Y., Xin, C., Liu, Z., & Song, S. (2025). *Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections*. NeurIPS 2025（扩展版）. [arXiv:2506.16685](https://arxiv.org/abs/2506.16685) (v1 2025-06-20, v5 2025-12-25)

**理论主干**

3. Ross, S., & Bagnell, J. A. (2014). *Reinforcement and Imitation Learning via Interactive No-Regret Learning*. [arXiv:1406.5979](https://arxiv.org/abs/1406.5979)
4. Rajaraman, N., Yang, L. F., Jiao, J., & Ramachandran, K. (2020). *Toward the Fundamental Limits of Imitation Learning*. [arXiv:2009.05990](https://arxiv.org/abs/2009.05990)

**机器人门控**

5. Zhang, J., & Cho, K. (2016). *Query-Efficient Imitation Learning for End-to-End Autonomous Driving*（SafeDAgger）. [arXiv:1605.06450](https://arxiv.org/abs/1605.06450)
6. Menda, K., Driggs-Campbell, K., & Kochenderfer, M. J. (2018). *EnsembleDAgger: A Bayesian Approach to Safe Imitation Learning*. IROS 2019. [arXiv:1807.08364](https://arxiv.org/abs/1807.08364)
7. Hoque, R., Balakrishna, A., Putterman, C., Luo, M., Brown, D. S., Seita, D., Thananjeyan, B., Novoseller, E., & Goldberg, K. (2021). *LazyDAgger: Reducing Context Switching in Interactive Imitation Learning*. [arXiv:2104.00053](https://arxiv.org/abs/2104.00053)
8. Hoque, R., Balakrishna, A., Novoseller, E., Wilcox, A., Brown, D. S., & Goldberg, K. (2021). *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning*. CoRL 2021 (Oral). [arXiv:2109.08273](https://arxiv.org/abs/2109.08273)
9. Hoque, R., Chen, L. Y., Sharma, S., Dharmarajan, K., Thananjeyan, B., Abbeel, P., & Goldberg, K. (2022). *Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision*. CoRL 2022 (Oral). [arXiv:2206.14349](https://arxiv.org/abs/2206.14349)
10. Lee, S.-W., Kang, X., & Kuo, Y.-L. (2024). *Diff-DAgger: Uncertainty Estimation with Diffusion Policy for Robotic Manipulation*. ICRA 2025. [arXiv:2410.14868](https://arxiv.org/abs/2410.14868)
11. Yu, W., Lv, J., Ying, Z., Jin, Y., Wen, C., & Lu, C. (2025). *ARMADA: Autonomous Online Failure Detection and Human Shared Control Empower Scalable Real-world Deployment and Adaptation*. [arXiv:2510.02298](https://arxiv.org/abs/2510.02298)

**人门控**

12. Kelly, M., Sidrane, C., Driggs-Campbell, K., & Kochenderfer, M. J. (2018). *HG-DAgger: Interactive Imitation Learning with Human Experts*. [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
13. Mandlekar, A., Xu, D., Martín-Martín, R., Zhu, Y., Fei-Fei, L., & Savarese, S. (2020). *Human-in-the-Loop Imitation Learning using Remote Teleoperation*（IWR）. [arXiv:2012.06733](https://arxiv.org/abs/2012.06733)
14. Liu, H., Nasiriany, S., Zhang, L., Bao, Z., & Zhu, Y. (2022). *Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment*（Sirius）. RSS 2023 / IJRR 2025. [arXiv:2211.08416](https://arxiv.org/abs/2211.08416)
15. Wu, P., Shentu, Y., Liao, Q., Jin, D., Guo, M., Sreenath, K., Lin, X., & Abbeel, P. (2025). *RoboCopilot: Human-in-the-loop Interactive Imitation Learning for Robot Manipulation*. [arXiv:2503.07771](https://arxiv.org/abs/2503.07771)
16. Huang, Y., Ma, N., Zhao, W., Liu, Z., Sun, J., Wang, Q., & Chen, Y. (2026). *Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control*（TER-DAgger）. [arXiv:2603.04038](https://arxiv.org/abs/2603.04038)
17. Han, Y., Chen, Z., Zhao, Y., Xu, C., Shao, Y., Peng, Y., Mu, Y., & Lian, W. (2026). *DexHiL: A Human-in-the-Loop Framework for Vision-Language-Action Model Post-Training in Dexterous Manipulation*. [arXiv:2603.09121](https://arxiv.org/abs/2603.09121)
18. Murray, M., Chen, D., Bagaria, S., Fortier, D., Hellebrekers, T., Mullins, G., Gajarla, H., Mees, O., Cakmak, M., & Kolobov, A. (2026). *FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space*. [arXiv:2607.08877](https://arxiv.org/abs/2607.08877)

**介入即奖励**

19. Luo, J., Dong, P., Zhai, Y., Ma, Y., & Levine, S. (2023). *RLIF: Interactive Imitation Learning as Reinforcement Learning*. ICLR 2024. [arXiv:2311.12996](https://arxiv.org/abs/2311.12996)
20. Luo, J., Xu, C., Wu, J., & Levine, S. (2024). *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*（HIL-SERL）. [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)
21. Yuan, X., Mu, T., Tao, S., Fang, Y., Zhang, M., & Su, H. (2024). *Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model*. [arXiv:2412.13630](https://arxiv.org/abs/2412.13630)

**同构策略：噪声注入 / 数据筛选 / 负样本 / 规模**

22. Laskey, M., Lee, J., Fox, R., Dragan, A., & Goldberg, K. (2017). *DART: Noise Injection for Robust Imitation Learning*. CoRL 2017. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)
23. Agia, C., Sinha, R., Yang, J., Antonova, R., Pavone, M., Nishimura, H., Itkina, M., & Bohg, J. (2025). *CUPID: Curating Data your Robot Loves with Influence Functions*. CoRL 2025. [arXiv:2506.19121](https://arxiv.org/abs/2506.19121)
24. Chen, A. S., Lessing, A. M., Liu, Y., & Finn, C. (2025). *Curating Demonstrations using Online Experience*（Demo-SCORE）. [arXiv:2503.03707](https://arxiv.org/abs/2503.03707)
25. Lee, H., Min, T., Kim, J., Kang, S., Liu, F., Pinto, L., & Lee, K. (2026). *Quality over Quantity: Demonstration Curation via Influence Functions for Data-Centric Robot Learning*. ICRA 2026. [arXiv:2603.09056](https://arxiv.org/abs/2603.09056)
26. Belkhale, S., Cui, Y., & Sadigh, D. (2023). *Data Quality in Imitation Learning*. [arXiv:2306.02437](https://arxiv.org/abs/2306.02437)
27. Zheng, M., Marri, S., Choudhuri, A., Planche, B., Gao, Z., Nguyen, V. N., Chen, T., Chowdhary, G., & Wu, Z. (2026). *Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models*（AFIL）. [arXiv:2605.08434](https://arxiv.org/abs/2605.08434)
28. Miyamoto, K., Suzuki, K., & Ogata, T. (2026). *How to Utilize Failure Demo Data?: Effective Data Selection for Imitation Learning Using Distribution Differences in Attention Mechanism*. [arXiv:2605.07560](https://arxiv.org/abs/2605.07560)
29. Ge, Z., Zhou, Y., Zhou, W., Li, M., Li, X., Wu, C., Zhao, H., Wang, H., Wu, Z., Jia, X., & Jiang, Y.-G. (2026). *EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration*. [arXiv:2607.19745](https://arxiv.org/abs/2607.19745)
30. Huang, K., Scalise, R., Winston, C., Agrawal, A., Zhang, Y., Baijal, R., Grotz, M., Boots, B., Burchfiel, B., Itkina, M., Shah, P., & Gupta, A. (2025). *Using Non-Expert Data to Robustify Imitation Learning via Offline Reinforcement Learning*. [arXiv:2510.19495](https://arxiv.org/abs/2510.19495)
31. Lin, F., Hu, Y., Sheng, P., Wen, C., You, J., & Gao, Y. (2024). *Data Scaling Laws in Imitation Learning for Robotic Manipulation*. [arXiv:2410.18647](https://arxiv.org/abs/2410.18647)
32. Mandlekar, A., Xu, D., Wong, J., Nasiriany, S., Wang, C., Kulkarni, R., Fei-Fei, L., Savarese, S., Zhu, Y., & Martín-Martín, R. (2021). *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*（robomimic）. CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)

**其他**

33. D'urso, G., Roy, K., Lawrance, N., & Tidd, B. (2026). *It's Not Just More Demos: Counterfactual Action Sensitivity Coverage for Data-Efficient Robust Robot Imitation*. [arXiv:2607.27261](https://arxiv.org/abs/2607.27261)
34. *Language-Critique Imitation Learning from Suboptimal Demonstrations* (2026). [arXiv:2607.01225](https://arxiv.org/abs/2607.01225)

**综述**

35. Celemin, C., Pérez-Dattari, R., Chisari, E., Franzese, G., de Souza Rosa, L., Prakash, R., Ajanović, Z., Ferraz, M., Valada, A., & Kober, J. (2022). *Interactive Imitation Learning in Robotics: A Survey*. [arXiv:2211.00600](https://arxiv.org/abs/2211.00600)
36. *Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives — A Survey* (2025). [arXiv:2506.00098](https://arxiv.org/abs/2506.00098)

---

## 7. AI 使用声明

本报告使用 AI 辅助完成文献检索、PDF 全文抽取与综合。
所有引用文献均经 arXiv 活页独立验证（见 §6 抬头）；两篇主论文及 IWR、Sirius 的全部数字来自 PDF 原文抽取，
不来自模型记忆或二手摘要（首页的摘要器编造实例即为此规程存在的理由）。
对我们内部数据的所有引用均指向 `failure-data-in-imitation-2026-08.md` 中的具体节号，未新增未经测量的数字。
