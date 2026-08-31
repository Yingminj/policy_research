# PARIL 的相关工作、新颖性判定与 Related Work 草稿

> 撰写日期：2026-08-30
> 触发问题：PARIL introduction 草稿的（a）相关研究进展梳理、（b）可行性评估、（c）与已有工作的重叠判定。
> 配套阅读：[`dagger-strategy-and-data-allocation-2026-08.md`](./dagger-strategy-and-data-allocation-2026-08.md)（本文是它在
> **"进度/价值"这一轴**上的补全——那份报告的谱系 §2.2 覆盖了门控轴与残差轴，但没有覆盖进度估计这一族）、
> [`experiment_report/act/failure-data-in-imitation-2026-08.md`](./experiment_report/act/failure-data-in-imitation-2026-08.md)（§4 现状落差的事实来源）。
>
> **文献验证方法**：本文新增的每一篇论文都逐篇 WebFetch 了 `arxiv.org/abs/<id>` 活页，
> 核对标题、作者、v1 日期与摘要原文（验证日期 2026-08-30）。
> 复用 `dagger-strategy` §6 的 36 篇不重复验证，其验证记录见该文件抬头。
> 本文**不引用任何未经活页核对的条目**；搜索结果里出现但未核对的，一律列在 §7「待核对」而不进正文。

---

## 0. 一页结论

**三句话：**

1. **Introduction 里列的三条贡献，前两条各自已被一篇论文作为主标题claim 发表过，且都不是边缘工作。**
   贡献 2（从人类纠错学残差策略、面向接触密集精密操作）= **CR-DAgger**，NeurIPS 2025，
   四任务 +64%，且它的对照组正是我们写的两个 baseline（retrain / finetune）。
   贡献 1（用任务进度区分"致败行为"与"有益纠正行为"并据此重加权 BC）= **ReTVL**，2026-06-23 上 arXiv，
   摘要里 "distinguish harmful execution errors from beneficial corrective behaviors" 与我们
   "distinguish these behaviors according to task progress" 是同一句话。
   **以目前写法，PARIL = CR-DAgger + ReTVL，两条都不是我们的。**

2. **真正没人做的是那个"路由规则"，而它现在被我们写成了自相矛盾。**
   贡献 1 说"把致败行为滤掉，别污染策略"，贡献 2 说"正是要从误差导致的边界状态学习"——
   同一批数据，两种相反处理，Introduction 通篇没有给出切分规则。
   把它写成规则就是方法本身：**progress 下降段的"动作"不进主策略 BC，但它的"状态"保留下来当残差策略的训练分布；
   progress 上升的恢复段供给残差策略的动作监督。** 这个组合在 §2 列出的全部 12 篇里都没有出现。
   `dagger-strategy` §2.1 的坐标系里，这相当于**在轴 C 上同时用 (b) 残差 与 (f) 数据筛选，并让筛选结果决定路由方向**。

3. **最大的风险不在文献，在我们自己的实验状态。** Introduction 承诺
   "six manipulation tasks across three robotic platforms" + "real-world task success rates"，
   而 `failure-data` §0 抬头写的是：磁盘上**没有任何 rollout 成功率记录**，`env_eval_freq` 全为 0；
   数据集只有 `tidy_up_stationery` 一个任务；`dagger-strategy` §4 把残差路线（方案 D）标为
   **"被链路阻塞"**（跟踪误差足以让 delta 纠错数据符号翻转）。
   **贡献 3 目前没有任何可支撑的证据，而它恰好是审稿人唯一会真正检验的那一条。**

**一句话行动版**：文献侧的补救是重写贡献列表（§3）；实验侧的补救是先做闭环评测和 on-policy 采集，
在这两件事完成之前，Related Work 可以写，Contribution 3 不能写。

---

## 1. 本文相对 `dagger-strategy` §2.2 新增了什么

那份报告把 DAgger 谱系按三条轴切得很干净，但它的轴 C（"新数据怎么进模型"）里
"数据筛选"一格只放了 CUPID / Demo-SCORE / QoQ——**全部是"影响函数 / 成功分类器"路线**。
**用"任务进度 / 价值函数"来切分同一条轨迹内部的好坏片段，是另外一族，那份报告完全没有覆盖。**
而 PARIL 贡献 1 恰好落在这一族的正中间。

补上这一格：

### 1.1 进度 / 价值驱动的片段级筛选（新增，与贡献 1 直接竞争）

| 论文 | ID / 时间 | 一句话 | 与 PARIL 贡献 1 的关系 |
|---|---|---|---|
| **ReTVL: Beyond Monotonic Progress — Retry-Supervised Value Learning for Robot Imitation** | [2606.24633](https://arxiv.org/abs/2606.24633)，v1 2026-06-23（v2 07-06） | 全局进度校准 + 基于**标注 retry 关键点**的局部成对偏好；学到的价值函数用于**选择性重加权** BC | **近乎同题**。它明确处理"非单调进度"，即我们说的"纠错段进度先降后升"。我们必须逐点区分 |
| **SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning** | [2505.22626](https://arxiv.org/abs/2505.22626)，v1 2025-05-28 | **自监督**任务进度预测器 + 去重，在 **state-action pair 粒度**滤掉低质量样本；平均 +15.4% | 提供了"进度预测器不需要人标注"的路线——是我们 R1 的一个出路，也是我们"新颖性"的一个威胁 |
| **ValueFormer: A Causal Transformer Value Function with Stage-Aware Labels** | [2608.02958](https://arxiv.org/abs/2608.02958)，v1 2026-08-03 | 一次前向出两个信号：平滑 MC 价值（做 advantage）+ 尖锐二值（在线错误检测）；失败 episode 用 **success-then-decay** 的 stage-aware 回报，**保留失败阶段之前的成功曲线**；critic 导出的逐帧权重把完成率 70%→85% | 它的 stage-aware 标签正是"如何给含错误的轨迹打进度标签"的一个已发表答案 |
| **ROVE: Unlocking Human Interventions for Humanoid Manipulation via RL** | [2606.17011](https://arxiv.org/abs/2606.17011)，v1 2026-06-15 | 摘要直陈"介入轨迹常常次优，把介入当专家监督会吸收犹豫、低效甚至错误的行为"；用 Optimistic Value Estimation 在混合质量轨迹里挑高价值行为 | **把我们 Introduction 第一段的 motivation 原样写在了摘要里** |
| **RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction** | [2509.07953](https://arxiv.org/abs/2509.07953)，v1 2025-09-09 | IL 预训练后加一段 human-in-the-loop rollout 微调：人先把机器人**倒回 in-distribution 状态**再给纠正段；比 SOTA 少 **10×** 数据；**恢复段数量与成功率近似线性** | **把我们第二段 motivation 变成了已测结果**。"纠错数据扩展状态分布、提升鲁棒性"不再是假设 |
| **FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement** | [2607.01111](https://arxiv.org/abs/2607.01111)，v1 2026-07-01 | 失败对比偏好适应 + 重试时轻量动作扰动；成功恢复轨迹回流做持续改进；仿真 +17.6%、真机 +11.7% | 测试期恢复这一侧的最新对照 |

### 1.2 残差策略这一格的补全（新增，与贡献 2 直接竞争）

`dagger-strategy` §2.2 已验证 CR-DAgger、TER-DAgger、Policy Decorator、FlowDAgger。补三篇：

| 论文 | ID / 时间 | 一句话 |
|---|---|---|
| **TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction** | [2405.10315](https://arxiv.org/abs/2405.10315)，v1 2024-05-16，**CoRL 2024** | **从人类在线纠错学残差策略**并与基础策略合并自主执行。比 CR-DAgger 早一年，是"残差 ← 人类纠错"这个组合的更早出处 |
| **ResiP: From Imitation to Refinement — Residual RL for Precise Assembly** | [2407.16677](https://arxiv.org/abs/2407.16677)，v1 2024-07-23 | 给**冻结的 chunked BC 模型**加一个全闭环 RL 残差策略。"冻结基础 + 残差"在精密装配上的标准形 |
| **Residual Off-Policy RL for Finetuning Behavior Cloning Policies** | [2509.19301](https://arxiv.org/abs/2509.19301)，v1 2025-09-23 | 把 BC 当黑箱基座，用稀疏二值奖励的 off-policy RL 学逐步残差；首次在带灵巧手的人形上完成真机 RL |
| **HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual RL** | [2606.22860](https://arxiv.org/abs/2606.22860)，v1 2026-06-22 | 人在环残差 RL 适配器，纠正 VLA 的次优动作与分布偏移；真机 1.5 小时在线 RL 后 >95% |

### 1.3 人门控一格的补全

| 论文 | ID / 时间 | 一句话 |
|---|---|---|
| **Sirius-Fleet: Multi-Task Interactive Robot Fleet Learning with Visual World Models** | [2410.22689](https://arxiv.org/abs/2410.22689)，v1 2024-10-30，**CoRL 2024** | 视觉世界模型预测未来隐embedding，异常预测器**随自主性提升自动调整阈值**，人力需求随时间下降 |

---

## 2. 三条贡献的逐条重叠判定

### 2.1 贡献 2（残差策略）——**已被直接发表**

**CR-DAgger**（[2506.16685](https://arxiv.org/abs/2506.16685)，v1 2025-06-20，NeurIPS 2025）
是贡献 2 的逐字版本：面向**真实世界接触密集精密操作**，从**人类 DAgger 纠错**学一个**残差策略**叠在基础策略上，
四任务（book flipping / belt assembly / cable routing / gear insertion）比基础策略 **+64%**，
且**同时优于 retrain-from-scratch 与 finetuning**——正是我们 Introduction 写的
"naive data aggregation and base-policy-only learning"两个 baseline。
它的 v1 标题还是 *"What Matters in DAgger? An Empirical Study on Improving Real World Robot Learning with Human Corrections"*，
连"实证研究"这个 framing 也占了。

上游：TRANSIC（CoRL 2024）已经是"残差 ← 人类在线纠错"；Policy Decorator（ICLR 2025）是
"残差 ← 冻结 BC 基座"的 model-agnostic 形；ResiP / Residual Off-Policy RL 是 RL 残差侧；
HiL-ResRL 是 VLA 侧。**2026 年"从人类纠错学残差策略"本身不构成贡献。**

### 2.2 贡献 1（进度感知筛选）——**已被直接发表，两个月前**

**ReTVL**（[2606.24633](https://arxiv.org/abs/2606.24633)，2026-06-23）。逐条对照：

| PARIL Introduction | ReTVL 摘要 |
|---|---|
| "trajectory may contain erroneous actions ... whereas the subsequent correction contains useful recovery behaviors" | "human demonstrations often contain mistakes and corrective behaviors, such as imprecise grasps, object misalignment, unstable contact, and repeated attempts" |
| "distinguish these behaviors according to task progress" | "distinguish harmful execution errors from beneficial corrective behaviors" |
| "retaining informative corrective and recovery actions while preventing failure-inducing behaviors from contaminating" | "selective reweighting of demonstration data for improved behavior cloning" |
| （未写：进度非单调怎么办） | **"Beyond Monotonic Progress"——这是它的标题** |

同向收敛的还有 SCIZOR（自监督进度预测器、pair 粒度筛选）、ValueFormer（stage-aware 标签 + critic 逐帧权重）、
ROVE（混合质量介入里挑高价值行为）。加权侧的祖先 IWR / Sirius 见 `dagger-strategy` §2.2。

### 2.3 motivation 第二段——**已被 RaC 变成带 scaling law 的实测结果**

RaC（[2509.07953](https://arxiv.org/abs/2509.07953)）已经证明：rewind-then-correct 的 human-in-the-loop rollout
把 retry/adaptation 行为纳入技能库，**比 SOTA 少 10× 数据**，且**恢复段数量与成功率近似线性**。
我们不能再把"纠错数据扩展状态分布、带来鲁棒性"当作待验证的假设来写——它是一条必须引用并超过的已知结果。

### 2.4 名字

"PARIL" 在 arXiv 上没有检索到冲突（2026-08-30）。相近的 PARL = Peer-Assisted Robotic Learning，不构成冲突。

---

## 3. 还剩什么可做（重写后的贡献列表建议）

**保留项目，重写 claim。把两条借来的贡献压成一条机制 + 一条证据：**

> **C1（机制）——进度信号作为"路由器"而非"过滤器"。**
> 现有工作把进度/价值信号用于**给主策略重加权或筛数据**（ReTVL、SCIZOR、ValueFormer、ROVE），
> 或把纠错数据整体喂给**一个残差策略**（CR-DAgger、TRANSIC）。
> **没有工作用进度信号决定"哪部分纠错数据进主策略、哪部分进残差策略"。**
> 关键的不对称是：致败片段的**动作**有害，但它的**状态**正是我们要覆盖的边界分布——
> 前者从主策略 BC 中剔除，后者保留为残差策略的训练分布。这一条同时解决了
> Introduction 里贡献 1 与贡献 2 的表面矛盾。
>
> **C2（证据）——跨平台。** 三平台 × 六任务的规模宽于 CR-DAgger（单一 setup、四任务）、
> ReTVL、RaC（三个双臂任务）。不是机制贡献，但支撑一条它们都做不出的 claim。
> **前提是这些数字真的跑得出来，见 §4。**

**新颖性 claim 的写法**（#548 口径）：不要写 "we are the first"，写成检索有界的形式——
"To the best of our knowledge, and based on a systematic search of arXiv up to 2026-08 covering
interactive imitation learning, residual policy learning, and progress/value-based data curation,
no prior method uses a progress signal to route correction data between the base policy and a residual policy."
并在同句点名最近邻（CR-DAgger、ReTVL）。

---

## 4. 与我们实际实验状态的落差（**优先级高于文献问题**）

Introduction 承诺 vs 磁盘事实：

| Introduction 写的 | 磁盘上的事实 | 出处 |
|---|---|---|
| "six manipulation tasks" | 只有 `tidy_up_stationery` 一个任务的数据集 | `failure-data` 抬头数据路径 |
| "three robotic platforms" | 未见跨平台数据集；ACT / ACT-DiT / pi0.5 是**模型**不是平台 | `experiment_report/` 目录结构 |
| "real-world task success rates" | **没有任何 rollout 成功率记录**，三个 job 的 `env_eval_freq` 全为 0，无评测目录 | `failure-data` §0 抬头 |
| "improvements over naive data aggregation" | 现有"变差"结论**只有主观观察支撑**，离线指标上几乎看不出来 | `failure-data` §0 抬头 |
| 贡献 2 残差策略 | 方案 D 被标记 **"被链路阻塞"**——跟踪误差足以让 delta 纠错数据**符号翻转** | `dagger-strategy` §3.7 / §4 |
| "corrections from policy rollouts" | 现有 36/72 条是**人演的**失败，不是策略走出来的，**不满足 DAgger 前提** | `dagger-strategy` §3.1 |

**最后一行是致命的**：Introduction 第一段写 "policy rollouts with human interventions"，
而我们手上的数据不是 rollout 产生的。`dagger-strategy` §3.1 的判定是
"DAgger 的任何定理和任何配比经验对它都不成立"。**在 on-policy 采集打通之前，
这篇论文的整个 setup 在事实上还不存在。**

另有一个方法论上的坑：`failure-data` §3.3 测到策略对**纯背景像素**的依赖强于对物体的依赖
（遮挡消融 2.52× vs 2.00×），而失败数据是**独立会话块**采的（背景亮度 std 5.9 vs 25.4，
单标量 66.9% 可分）。**任何"进度预测器"在这样的数据上会先学会用背景判断会话块，而不是判断任务进度。**
这不是理论担忧，是已经测到的捷径。

---

## 5. Related Work 草稿

见同目录 [`related_work.tex`](./related_work.tex) 与 [`paril.bib`](./paril.bib)。
草稿按五段组织：交互式模仿学习与 DAgger 家族 → 不完美纠错数据的使用（加权/筛选/价值）→
残差策略 → 进度与价值估计 → 定位段。**定位段是唯一需要你亲自改的部分**，
因为它必须与最终的方法一致；目前按 §3 的 C1 写。

---

## 6. 十条优化建议（按优先级排序）

1. **先补闭环评测，再谈其他任何事。** `failure-data` §6.4 #1 已经把它列为第一优先级，
   `dagger-strategy` §0 又重复了一遍。现在第三次：Contribution 3 的每个字都依赖它。
   在 `env_eval_freq > 0` 跑出第一条成功率曲线之前，不要写 experiment section。

2. **把"路由规则"从矛盾改写成方法（§3 的 C1）。** 这是唯一能把论文从"两篇工作的并集"救回来的改动，
   而且它是免费的——只需要把 Introduction 现有的两条贡献重新组织，不需要新实验。
   同时它给出了一个可证伪的预测：**只筛选**和**只残差**都应该弱于路由。这个预测就是第 3 条的消融。

3. **补三个消融，否则组合性 claim 不成立。** filter-only（进度筛选 + 主策略，≈ReTVL）、
   residual-only（全部纠错进残差，≈CR-DAgger）、routing（我们）。
   没有这张表，审稿人无法判断增益来自哪一半——而他会默认来自已发表的那一半。

4. **升级 baseline。** "naive aggregation + base-policy-only" 是 2023 年的对照组，
   且恰好是 CR-DAgger 已经打过的那两个。至少补：IWR/Sirius 式加权（`dagger-strategy` §4 方案 C，
   改动最小）、RaC 式全量恢复段训练、以及一个 oracle-filter 上界。

5. **明确进度标签的监督来源，并在 Method 第一段就交代。** 四选一，各有代价：
   时间倒数回归（在非单调段上崩，正是 ReTVL 标题在说的）、
   人标 retry 关键点（ReTVL 的做法，但与"simple"冲突）、
   自监督成对进度（SCIZOR，最省人力，但要防 §4 的背景捷径）、
   介入起点当弱标签（免费，但受第 6 条的延迟污染）。**这是决定论文成立与否的单一问题。**

6. **把人类反应延迟建进片段边界。** Sirius 实测介入延迟约 2 秒 ≈ **15 个动作步**。
   若以介入起点切分，前 15 步真正的坏动作会被标成"专家邻近"，而纠正段开头混着接管瞬态。
   `dagger-strategy` §3.4 已经给出对应结论（前导段权重设 0，两篇论文独立测到）——
   把它写成方法里的一个显式偏移量，而不是隐含假设。

7. **筛选时保状态、弃动作。** 这是第 2 条的技术内核，也值得单列：
   整段丢弃会连同"边界状态覆盖"一起丢掉，而那正是贡献 2 的全部资产。
   实现上就是两个 sampler：主策略 BC 采 progress 非降段，残差策略在全部段的**状态**上训练。

8. **验证残差监督在我们的硬件上取得到。** CR-DAgger 之所以要做 compliant intervention interface，
   就是为了让基础策略在人介入时**继续运行**，从而算得出 delta。
   若我们是 HG-DAgger 式硬接管，残差目标根本不存在，必须改走 RL（要奖励）或改采集接口。
   `dagger-strategy` §3.7 已经把它列为硬性先后次序：**jj 的链路修好之前不要开始采 delta 式纠错数据。**

9. **降级或去掉 "offline action prediction accuracy" 这个主指标。** 动作 MSE 与成功率相关性差，
   在多模态纠错数据上尤其误导（多个动作都对）。`failure-data` §4 自己就测到
   "离线指标上变差几乎看不出来，伤害必然发生在闭环"。把它降为诊断量，主结果用真机成功率。

10. **先做"删数据"再做"加数据"。** `dagger-strategy` §3.8 的建议在 PARIL 语境下更强了：
    QoQ（ICRA 2026）只需验证集 + 影响函数，**当前条件就能跑**，
    而且它给的是与进度筛选正交的第二个证据源——如果两种筛选选出同一批坏样本，
    这本身就是 Method 里一个有力的 sanity check。

---

## 7. 参考文献

**新增（本文逐篇经 `arxiv.org/abs/<id>` 活页验证，2026-08-30）**

1. Qin, X., Lu, J., Wang, K., Zhang, C., Kang, S., Lee, K., Xu, M., Liang, B., Yang, J., & Zhao, L. (2026). *Beyond Monotonic Progress: Retry-Supervised Value Learning for Robot Imitation*（ReTVL）. [arXiv:2606.24633](https://arxiv.org/abs/2606.24633) (v1 2026-06-23, v2 2026-07-06)
2. Zhang, Y., Xie, Y., Liu, H., Shah, R., Wan, M., Fan, L., & Zhu, Y. (2025). *SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning*. [arXiv:2505.22626](https://arxiv.org/abs/2505.22626) (v1 2025-05-28)
3. Sa, I., Stulov, K., & Bhageria, R. (2026). *ValueFormer: A Causal Transformer Value Function with Stage-Aware Labels for Semi-Autonomous Vision-Language-Action Policies*. [arXiv:2608.02958](https://arxiv.org/abs/2608.02958) (v1 2026-08-03)
4. Xiao, W., Tang, W., Ge, Y., Zhou, H., Mu, Y., Zhang, L., & Ge, Y. (2026). *ROVE: Unlocking Human Interventions for Humanoid Manipulation via Reinforcement Learning*. [arXiv:2606.17011](https://arxiv.org/abs/2606.17011) (v1 2026-06-15)
5. Hu, Z., Wu, R., Enock, N., Li, J., Kadakia, R., Erickson, Z., & Kumar, A. (2025). *RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction*. [arXiv:2509.07953](https://arxiv.org/abs/2509.07953) (v1 2025-09-09)
6. Hao, H., Syed, S. N., Ichnowski, J., & Schneider, J. (2026). *FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement*. [arXiv:2607.01111](https://arxiv.org/abs/2607.01111) (v1 2026-07-01)
7. Jiang, Y., Wang, C., Zhang, R., Wu, J., & Fei-Fei, L. (2024). *TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction*. CoRL 2024. [arXiv:2405.10315](https://arxiv.org/abs/2405.10315) (v1 2024-05-16)
8. Ankile, L., Simeonov, A., Shenfeld, I., Torne, M., & Agrawal, P. (2024). *From Imitation to Refinement — Residual RL for Precise Assembly*（ResiP）. [arXiv:2407.16677](https://arxiv.org/abs/2407.16677) (v1 2024-07-23)
9. Ankile, L., Jiang, Z., Duan, R., Shi, G., Abbeel, P., & Nagabandi, A. (2025). *Residual Off-Policy RL for Finetuning Behavior Cloning Policies*. [arXiv:2509.19301](https://arxiv.org/abs/2509.19301) (v1 2025-09-23)
10. Liu, J., Mai, Z., He, S., Ren, H., Wang, C., Zhou, S., Wu, X., & Zhang, H. (2026). *HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual Reinforcement Learning*. [arXiv:2606.22860](https://arxiv.org/abs/2606.22860) (v1 2026-06-22)
11. Liu, H., Zhang, Y., Betala, V., Zhang, E., Liu, J., Ding, C., & Zhu, Y. (2024). *Multi-Task Interactive Robot Fleet Learning with Visual World Models*（Sirius-Fleet）. CoRL 2024. [arXiv:2410.22689](https://arxiv.org/abs/2410.22689) (v1 2024-10-30)

**复用**：`dagger-strategy-and-data-allocation-2026-08.md` §6 的 36 篇（该文件已逐篇活页验证，2026-08-24）。
其中本文正文直接引用的有：DAgger [1011.0686]、CR-DAgger [2506.16685]、HG-DAgger [1810.02890]、
IWR [2012.06733]、Sirius [2211.08416]、RoboCopilot [2503.07771]、TER-DAgger [2603.04038]、
DexHiL [2603.09121]、FlowDAgger [2607.08877]、Policy Decorator [2412.13630]、RLIF [2311.12996]、
HIL-SERL [2410.21845]、DART [1703.09327]、CUPID [2506.19121]、Demo-SCORE [2503.03707]、
QoQ [2603.09056]、Celemin et al. survey [2211.00600]。

**待核对（搜索结果中出现，尚未活页验证，不得进正文）**

- Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning — [2604.16683](https://arxiv.org/abs/2604.16683)
- Phase-Conditioned Imitation Learning with Autonomous Failure Recovery — [2605.29407](https://arxiv.org/abs/2605.29407)
- RecoveryChaining: Learning Local Recovery Policies for Robust Manipulation — [2410.13979](https://arxiv.org/abs/2410.13979)
- Improving Robustness to OOD States via Deep Koopman-Boosted Diffusion Policy — [2511.00555](https://arxiv.org/abs/2511.00555)
- Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism — [2506.09176](https://arxiv.org/abs/2506.09176)
- Human-in-the-loop Online Rejection Sampling for Robotic Manipulation — [2510.26406](https://arxiv.org/abs/2510.26406)
- MILE: Model-based Intervention Learning — [2502.13519](https://arxiv.org/abs/2502.13519)
- Predictive Preference Learning from Human Interventions — [2510.01545](https://arxiv.org/abs/2510.01545)

---

## 8. AI 使用声明

本报告使用 AI 辅助完成文献检索与综合。§7「新增」的 11 篇均经 `arxiv.org/abs/<id>` 活页独立核对
（标题 / 作者 / v1 日期 / 摘要原文），未使用模型记忆或二手摘要；
§7「待核对」的 8 篇仅来自搜索结果标题，**未经核对，因此未进入正文与 `paril.bib`**。
对内部实验状态（§4）的全部引用均指向 `failure-data-in-imitation-2026-08.md` 与
`dagger-strategy-and-data-allocation-2026-08.md` 的具体节号，未新增任何未经测量的数字。
