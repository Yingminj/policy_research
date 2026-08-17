# 真实场景策略评估：文献调研与十个可落地方案

**范围。** 当前训练/评估链路只有训练集数值指标（loss），真机只有二值成功率，仿真只评估动作本身而缺视觉监督——本文调查该问题在文献中的已有解法，并给出 10 个可在 `lerobot_vlahost` / `robot_data_platform` / `Apex_Deploy_new` 三仓库落地的方案。

编撰于 2026-08-15。引用标注说明：**[✓]** = 已直接抓取原文核实标题/作者/摘要/数字；**[~]** = 仅来自检索结果元数据与摘要，未逐字核实原文，引用其数字前请自行复核。

---

## 结论先行

1. **你诊断对了一半。** "训练 loss 不反映真实表现"这个判断，文献不但同意，而且量化过：robomimic 的经典结论是**按验证集选出的最佳 checkpoint 可能比真正最好的策略差 50–100%** [~]。但结论不是"离线指标没用"，而是"**朴素的 MSE 没用**"——2026 年的 CI-MSE 把误差限制在任务关键区间并做动作对齐后，与 rollout 成功率的 Spearman 秩相关从 −0.61 提升到 **−0.87** [✓]。离线信号的天花板远比你现在测的高。

2. **"仿真只评估动作质量，缺视觉监督"这个判断需要修正。** 这是**通用 benchmark**（LIBERO/CALVIN/MetaWorld）的性质，不是仿真本身的性质。SIMPLER 的核心贡献恰恰是**视觉匹配（visual matching）**——不追求全保真数字孪生，而是把仿真渲染对齐到真机相机的观测分布上，在 1500+ 组配对 sim-real 评估中取得了高 Pearson 相关和低 MMRV [~]。你要的"视觉监督"在文献里是有解的，代价是**为你自己的场景建环境**，而不是用别人的 benchmark。

3. **最被低估的问题不是"缺指标"，而是"缺统计"。** 2026 年的一项系统审计发现：LIBERO 上报告的"进展"中只有 **19.8%** 在统计上可证显著，SimplerEnv 为 **19.7%** [✓]。真机侧更糟：近年 VLA 论文每条件试验数的众数是 **10–20 次**，且几乎不报置信区间 [~]。以 90% 成功率、70 次 rollout 计，95% Clopper-Pearson 区间宽达 **15.4 个百分点**（80.5%–95.9%）[~]。**你现在真机上观察到的"成功率没提升"，很可能连"有没有变化"都还没测出来。**

4. **你的仓库里有一处文献没覆盖、但对你影响最大的缺口：部署链路与训练分布不一致。** 此前审计已确认：下发给机器人的是合成 Hermite 零速 S 曲线而非策略原始输出；gripper 状态尺度约为训练范围的 1.25 倍。这意味着**即使你把上述所有评估指标都建起来，如果离线评估不走部署的同一条后处理链路，你测的仍然不是被部署的那个策略**。方案 2 专门处理这一点。

5. **建议的投入顺序**：方案 4 → 2 → 1 → 9 → 3 → 5，这六项在 6–8 周内可完成，不需要新硬件，且能立刻把"烧真机时间才能发现问题"的周期压缩到小时级。方案 6/7/8 是季度级投入。方案 10 是低成本高杠杆的补充。

---

## 1. 现状审计：三个仓库里已经有什么、缺什么

### 1.1 `robot_data_platform`（训练与调度侧）

| 项 | 现状 | 证据 |
|---|---|---|
| 训练指标 | 仅训练集 loss，写入 `log.jsonl` | `apps/lelab/lelab/jobs.py` 的 `TrainingMetrics` / `parse_metrics_into` |
| 验证集 | **无**。`splits = {"train": "0:30"}` | `act_delta` 报告 §3.1 |
| `env_eval_freq` | **0**，eval 从未跑过 | 同上 |
| 评估作业类型 | **不存在**。`apps/lelab/lelab/` 下无任何 eval/metric 模块 | 对该目录的 eval/success_rate/metric 全文检索无命中 |
| 数据侧审计 | **较强**：`conversion_manifest.json`、`audit.hold`、`unique_ratio`、`end_effector_state_source` 等 | `tool/CONVERSION.md` |

**判断**：数据侧的审计文化已经建立得相当好（`rdp check-rosbag` 的拓扑清单是一个很好的先例），但这套严谨性完全没有延伸到模型侧。评估基础设施在这个仓库里是零。

### 1.2 `lerobot_vlahost`（策略与部署引擎侧）

| 项 | 现状 |
|---|---|
| 仿真环境封装 | 已有 `envs/{libero,robocasa,robotwin,vlabench,metaworld,robomme}.py` |
| 评估脚本 | 已有 `scripts/lerobot_eval.py`、`rl/eval_policy.py`、`configs/eval.py` |
| Rollout 引擎 | `scripts/lerobot_rollout.py`，策略：`base/sentry/highlight/dagger/episodic`；推理后端：`sync/rtc` |
| 奖励模型 | **已有 `rewards/topreward/`**——TOPReward，零样本 VLM 奖励（Qwen3-VL 骨干），返回 `log P("True" | video + instruction)`，inference-only |
| 策略族 | act / act_delta / diffusion / pi0 / pi05 / smolvla / vita / rtc / groot / xvla 等 |

**判断**：这是三个仓库里评估资产最厚的一个，但资产是**孤立的**——`envs/` 里的仿真封装接的是公开 benchmark（正是 §2.3 指出的有 shortcut 问题的那批），`topreward` 定位是 RL 奖励而非评估裁判，`dagger` 策略产出的干预数据没有被当作评估指标使用。**大部分方案的实现工作是"接线"而不是"从零造"。**

### 1.3 `Apex_Deploy_new`（真机执行侧）

| 项 | 现状 |
|---|---|
| 全量记录 | `components/log_recorder_nodes_py/all_topic_log_recorder.py`——**全 topic 录制** |
| 推理节点 | `robot_node/vlahost/vlahost/vla_node.py`（1234 行） |
| chunk 播放器 | 500 Hz 开环插值，`chunk_waypoint_hz=30`，`chunk_blend_from_current=True` |
| chunk 触发 | `_check_need_new_chunk()`（`vla_node.py:434`）：剩余位点 ≤0 **且** `_has_reached_target()`（2° 阈值，`vla_node.py:464`），或 5 秒无新 chunk |
| 仿真 | `robot_node/marvin_hardware_sim/`、`marvin_description/{mjcf,urdf}` |
| 控制器 | `marvin_qp_controller`（可脚本化 → 可用于自动复位） |

**判断**：**这个仓库的评估潜力被严重低估了。** 全 topic 录制意味着每次真机 rollout 都已经落盘了算行为指标所需的全部原始数据——轨迹平滑度、双臂协调、抓取滑移、阶段耗时，都不需要新增任何硬件或传感器，只需要一个离线分析器。同时 `marvin_description` 的 mjcf + `marvin_hardware_sim` 是方案 6（real-to-sim）的现成起点。

### 1.4 缺口总结：评估阶梯的四个层级

```
L0  训练集 loss              ← 你现在唯一自动化的东西（无泛化信息）
L1  离线验证指标              ← 缺失（方案 1、2）
L2  仿真 rollout             ← 有封装但只接公开 benchmark，无自建场景（方案 6、7）
L3  真机 rollout             ← 只有二值成功率，无统计、无行为指标、人工判定（方案 3、4、5、8）
L∞  在线监控                  ← 缺失（方案 9）
```

文献的共识是：**没有任何单层能替代其他层**，但每一层都能过滤掉一部分候选，从而让下一层的昂贵试验花在值得的地方。你现在的问题是 L1、L2 完全空缺，导致每个假设都要用 L3 的真机时间去验证——这正是 `act_delta` 那轮实验"花了一整轮真机才发现相对动作根本没开启"的结构性原因。

---

## 2. 文献综述

### 2.1 离线指标为什么失效，以及怎么修

**失效证据。** robomimic 的大规模研究给出了该领域最常被引用的结论：训练目标与评估目标不匹配导致策略选择失效，**按验证集选出的最佳策略可能比真正最好的策略差 50–100%**，因此每个 checkpoint 都必须上真机试 [~]。这是 2021 年的结论，也是"离线指标没用"这一流行说法的来源。

**修复路线一：只在关键区间上算误差。** *Critical Interval MSE*（Huang, Zheng, Chen, You, Gao；arXiv 2606.29898，2026-06-29）[✓] 指出朴素 MSE 的问题是**在整条轨迹上平均**——而机器人任务的成败通常只由少数几个关键时刻（接近、闭合、插入）决定，这些时刻在时间轴上占比很小，其误差被大量"空闲移动"帧稀释。CI-MSE 把误差计算限制在任务关键片段，并配合动作对齐流程使之匹配 rollout 时的实际行为。结果：与 rollout 表现的 Spearman 秩相关 **−0.87**，朴素 MSE 为 **−0.61**。作者同时做了敏感性分析与分布偏移下的表现分析。

**修复路线二：换一个理论上更合适的离线目标。** *Offline Policy Evaluation for Manipulation Policies via Discounted Liveness Formulation*（arXiv 2605.11479）[~] 从"存活性/可达性"角度重构 OPE 目标，而非回归动作。

**修复路线三：预测数据影响力而非策略表现。** *DataMIL*（arXiv 2505.09603，ICLR 2026）[~] 把 datamodels 扩展到机器人，训练一个数据质量估计器，用可解的代理目标预测数据影响，从而无需真机 rollout 就能做数据选择。

**对你的意义**：CI-MSE 是本节里性价比最高的一项——它不改变训练流程，只改变**你怎么算验证误差**，且直接解决 `act_delta` 报告 §3.3 提出但从未实现的"离线 MAE"需求。

### 2.2 二值成功率不够，需要什么指标

**方法论层面的呼吁。** *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation*（Kress-Gazit, Hashimoto, Kuppuswamy, Shah, Horgan, Richardson, Feng, Burchfiel；arXiv 2409.09491）[~] 直指论文普遍只报成功率，且**不报试验次数、不报初始条件、不报成功判据、不做统计分析、不描述观察到的行为与失败模式**。他们的建议是三条：显式报告实验条件、使用若干与成功率互补的指标、做统计分析。

**指标层面的实现。** *RoboEval*（arXiv 2507.00435）[~] 提供 8 个双臂任务、3000+ 专家演示与一套标准化指标，分三类：**效率**（轨迹流畅度、路径长度）、**协调**（双臂对称性/同步性）、**安全与稳定**（滑移、抖动）；以及**分阶段进展**的结果类指标用于定位失败模式。关键发现有两条：行为指标在**超过一半**的 task-metric 对上与成功率相关；且**在二值成功率饱和时仍然有区分度**——两个成功率相近的策略，行为指标能区分出谁的动作更平滑、滑移更少。*Beyond Binary Success: A Diagnostic Meta-Evaluation Framework*（arXiv 2605.19986）[~] 沿同一方向做细粒度诊断。

**连续型人力指标。** 交互式模仿学习传统里的 **ROHE（Return On Human Effort）** 与其变体**干预率（intervention rate）**是被长期使用的连续指标 [~]。ThriftyDAgger [~] 用集成动作方差（新颖度）+ 风险自动触发干预；Sirius / *Robot Learning on the Job* [~] 在部署中用人类干预同时获得指标与训练数据；*Easy-IIL*（arXiv 2603.12769）[~] 用干预率量化人的操作负担。**干预率的信息量远高于成功率**：它是连续的、每条 episode 内多次采样的，且顺带产出改进数据。

### 2.3 benchmark 本身是否可信

*What Are We Actually Benchmarking in Robot Manipulation?*（Jiang, Tan, Wheeler, Sun, Ayalew, Walter；arXiv 2606.04233，2026-06-02）[✓] 是本次调研中对你"仿真评估靠不住"直觉最有力的支持，但它给出的理由和你想的不同。四种失效模式：

1. **捷径可解性（Shortcut Solvability）**：一个 90M 参数、**不理解语言**的小模型在 LIBERO 上就能追平 SOTA——说明 benchmark 允许用非预期能力通关。
2. **统计不显著**：LIBERO 上报告的进展只有 **19.8%** 可证显著，SimplerEnv **19.7%**，而 RoboTwin 2.0 达 **73.7%**。
3. **渐进过拟合**：在 CALVIN 上仅仅**在训练范围内重采样物体位姿**，性能就从 4.17 掉到 3.14 个任务——模型过拟合的是狭窄的测试分布，而非学到稳健技能。
4. **数据源依赖**：用 120 条贴近测试场景的脚本化演示训的小模型达到 94.8% 成功率，几乎追平 900M 基础模型。

排名结论：RoboCasa 与 RoboTwin 2.0 通过的诊断更多；**LIBERO、CALVIN、SimplerEnv 在多项诊断上有系统性弱点**。

**对你的意义**：`lerobot_vlahost/src/lerobot/envs/` 里已封装的 `libero.py` 恰好在弱点名单上。**不要用这些 benchmark 的分数指导你自己场景的模型选型**——它们的用途是与文献对齐，不是预测你的真机表现。同方向的相关工作还有 *Trustworthy Evaluation of Robotic Manipulation: A New Benchmark*（arXiv 2601.18723）[~]。

### 2.4 仿真评估怎样才能预测真机

**SIMPLER**（Li et al., *Evaluating Real-World Robot Manipulation Policies in Simulation*；arXiv 2405.05941，CoRL 2024）[~] 是这条线的奠基工作。核心论点：真机与仿真的差距可分解为**控制差距**与**视觉差距**，两者都可以在**不构建全保真数字孪生**的前提下缓解。其 **Visual Matching** 设置——把仿真渲染主动对齐到真机相机的观测统计——在 **1500+ 组配对 sim-real 评估**（2 种本体、8 个任务族）中取得高 Pearson 相关与低 **MMRV（Mean Minimum Rank Violations）**。且 SIMPLER 能复现单个策略对各类分布偏移的敏感性。

**注意这里的张力**：SIMPLER 自证有效，但 §2.3 的诊断把 SimplerEnv 列为弱点较多的 benchmark。这两件事不矛盾——**SIMPLER 作为"为特定真机场景定制的评估方法论"是成立的，作为"通用公开 benchmark"被反复刷榜后就失去了区分度**。你要学的是它的方法，不是用它的排行榜。

后续工作把保真度继续往上推：*Real-is-Sim*（arXiv 2504.03597）[~] 用 Embodied Gaussians 构建**贯穿数据采集、训练、评估、部署全流程**的动态数字孪生；*Real-to-Sim Robot Policy Evaluation with Gaussian Splatting of Soft-Body Interactions*（arXiv 2511.04665）[~] 处理毛绒玩具装袋、绳索走线等**软体**任务；*PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies*（Jain, Zhang, Arora, Chen, Torne, Irshad, Zakharov, Wang, Levine, Finn, Ma, Shah, Gupta, Pertsch；arXiv 2512.16881，2025-12-16）[✓ 标题/作者/日期，相关性数字未核实] 把流程自动化：多视角拍摄 → COLMAP + Gaussian Splatting 重建 → 导入 Isaac Sim → 策略评估。

**把仿真从定性变定量的关键一步**：*Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators*（SureSim；arXiv 2510.04354）[~] 把"合并真机与仿真评估"形式化为 **prediction-powered inference** 问题——用少量**配对的**真机/仿真评估去校正大规模仿真的偏差，再用非渐近均值估计给出真机性能的**置信区间**。报告在扩散策略与多任务微调策略上节省 **20–25%** 的硬件评估工作量。这直接回答了"我的仿真到底能不能信"：不是信或不信，而是**先测出偏差，再用统计手段扣除它**。

### 2.5 真机评估的自动化

**AutoEval**（Zhou et al.；arXiv 2503.24278，CoRL 2025）[~]：像提交 Slurm 作业一样提交评估作业，系统提供**自动成功判定**（基础模型微调的成功分类器，输入机器人状态图像，输出二值标签）与**自动场景复位**（微调的 reset policy），实现全天候无人值守评估。报告**减少人工监督时间 99% 以上**，且评估结果与人工评估高度吻合，同时**比照片级仿真环境和验证误差等离线指标都更可靠**。公开了 BridgeData/WidowX 平台上的多个 AutoEval 场景。

**RoboArena**（Atreya, Pertsch, Lee, Kim, Jain, Kuramshin, Eppner, Neary et al.；arXiv 2506.18123，CoRL 2025）[~]：换一个思路——**不标准化任务/环境/地点，而是众包评估**。评估者自由选择任务与环境（从而天然扩展多样性），但必须做**双盲的成对策略比较**。通过在 7 所机构、DROID 平台上的 **600+ 次成对真机评估**、7 个通用策略，证明众包成对比较**比传统集中式评估更准确地排出策略优劣**，且更可扩展、更抗操纵。

**LBM 的评估范式**（*A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*；arXiv 2507.05331 / Science Robotics 2025）[~]：这是目前工业界最严格的真机评估协议样板——**盲测、随机化 A/B 对照**，**1800 次真机 rollout + 47000+ 次仿真 rollout**，每个真机任务 50 次 rollout、每个仿真任务 200 次。他们的结论"多任务预训练确实更成功更稳健"之所以可信，正是因为这套协议。

### 2.6 统计方法

**为什么必须做。** TRI 的统计方法论文章 [~] 给出了最直观的数字：观察到 90% 成功率、70 次 rollout，95% Clopper-Pearson 区间是 **80.5%–95.9%**（宽 15.4 pp）；要收紧到 ±2 pp 需要 **1030 次 rollout**。而近年真机 VLA 论文每条件试验数众数是 **10–20 次**，13 篇标准实践论文中**没有一篇报告置信区间或配对检验** [~]。

**怎么做得更省。** *Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison*（**N-SCORE**；Snyder, Badithela, Matni, Pappas, Majumdar, Itkina, Nishimura；arXiv 2603.13616，2026-03-13）[✓]：基于 **safe, anytime-valid inference (SAVI)** 的**序贯假设检验**——构造一个乘性证据累加器，数据支持备择假设时增长、原假设成立时保持稳定，证据足够即可**提前停止**，同时在预设显著性水平上控制第一类错误。关键是它**不限于二值指标**，支持离散部分给分 rubric、连续 episodic reward、轨迹平滑度，参数与非参数分布皆可。报告：相比批量方法减少**最多 70%** 评估负担，相比仅支持二值结果的序贯方法减少**最多 50%**；在 LBM 1.0 硬件任务上比任意批量分配减少 **45%**，比二值序贯 SOTA 改进 **24–30%**。

**这一条对你尤其重要**：N-SCORE 同时解决了"试验太贵"和"必须有统计"这对矛盾，而且**它偏爱连续指标**——这与方案 3 的行为指标体系是互补设计，不是两件独立的事。

### 2.7 自动判定：VLM 作为裁判

- *Vision-Language Models for Robot Success Detection*（Luo, AAAI）[~]：MiniGPT-4 用测试分布轨迹微调后，成功检测准确率 **≥95%**，形式为 VQA 任务。
- *Score the Steps, Not Just the Goal*（**StepEval**；arXiv 2509.19524）[~]：成本感知的**插件式**评估框架，VLM 从录制的图像/视频中判定**子目标**结果；主张"每条轨迹输出一个逐子目标成功率向量"应成为常规做法，从而让部分能力可见（如"能抓起但倒不准"）。
- *TOPReward*（arXiv 2602.19313）[~]：**你仓库里已经有这个**（`lerobot_vlahost/src/lerobot/rewards/topreward/`）。零样本、无需微调权重。
- *Robometer / Large Reward Models*（arXiv 2603.16065）[~]：通用机器人奖励模型，输入指令+视频，零样本输出**稠密进度**与**二值成功**标签，跨任务跨本体。
- *Progress Reward Modeling for Robotic Learning: A Comprehensive Survey*（arXiv 2607.21655）[~]：该方向的综述，包含排序准确率、秩相关、单调性等评估裁判本身的指标。
- *VLAConf: Calibrated Task-Success Confidence for VLA*（arXiv 2605.29605）[~]：校准过的任务成功置信度。

**共同的方法论要求**：VLM 裁判本身必须先被评估——用人工标注集测其准确率与人机一致性（Cohen's κ），并报告这一数字。文献里评估裁判的标准做法是**排序准确率、秩相关、单调性**（后期状态应得高于前期状态的分数）[~]。

### 2.8 在线监控与失败检测

- **Sentinel**（Agia et al., *Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress*；arXiv 2410.04640，CoRL 2024）[~]：把失败分两类——**错乱失败（erratic）**用**动作 chunk 分布的时间一致性**统计量检测；**任务推进失败**用 VLM 做视频 QA 检测"策略自信且一致地在做无法解决任务的事"。两者结合比单用任一个**多检出 18%** 的失败。构建只需要一组成功 rollout + 任务描述。
- **SAFE**（*Multitask Failure Detection for VLA*；arXiv 2506.09937，NeurIPS 2025）[~]：分析 VLA 特征空间，发现 **VLA 内部特征已包含关于成败的高层知识，且跨任务通用**。SAFE-MLP / SAFE-LSTM 在已见和未见任务上均优于或持平基线。重要的负面结论：**token 级不确定性方法表现差**；采样一致性方法有效但**实时控制下算力不可承受**。
- **Foresight**（arXiv 2606.23085）[~]：用动作条件世界模型的隐状态做长时程失败检测。
- **Rewind-IL**（arXiv 2604.16683）[~]：在线失败检测 + 状态回溯重生，用 VLM 识别恢复时刻并抽取隐模板，执行时跟踪余弦相似度。

**Sentinel 的一致性检测对你近乎免费**：`vla_node.py` 的 chunk 架构下，连续两次推理产生的 chunk 在时间轴上有重叠区，**重叠区的分歧度直接就是时间一致性统计量**，不需要额外前向。

### 2.9 离线鲁棒性画像（红队）

**Predictive Red Teaming / RoboART**（Majumdar, Sharma, Kalashnikov, Singh, Sermanet, Sindhwani；arXiv 2502.06575，CoRL 2025）[~]：提出"预测式红队"问题——**在不做硬件实验的前提下，发现策略对环境因素的脆弱性并预测相应的性能下降**。RoboART 用生成式图像编辑修改标称观测以变化各环境因子（光照、视觉干扰物、物体位置），再用**策略专属的异常检测器**在编辑后的观测上运行来预测各条件下的表现。在 **12 种非标称条件、500+ 次硬件试验**上验证，预测成功率与真实成功率的**平均差异 < 0.19**。更实用的下游结论：**用预测为不利的条件去针对性采数并微调，把基线性能提升 2–7 倍**。

**这条线与你已有的工作直接咬合**：`literature-review.zh.md` + `test_raw_jpeg` 已经量化了图像压缩对像素与 DINO 特征的影响（PSNR ↔ CLS 余弦 ρ=0.927），但明确留下"像素保真度 ≠ 策略成功率"这个未解问题。RoboART 的异常检测器方法**正是把特征级扰动映射到成功率预测的那座桥**。

### 2.10 世界模型作为评估器（前沿）

**WorldEval**（arXiv 2505.19017）[~]：完全在线地评估真机策略——用视频世界模型生成"如果执行这个策略会发生什么"的视频，再据此判定。核心难点是让世界模型的输出**忠实跟随机器人动作**，直接输入动作或高维编码都会失败；其 **Policy2Vec** 把视频生成模型转成跟随**隐动作**的世界模拟器。报告能有效排出不同策略、以及同一策略不同 checkpoint 的优劣，并可作为**安全检测器**拦截危险动作。

定位：这是本文列出的所有路线中成熟度最低、但天花板最高的一条——它不需要建仿真器、不需要建数字孪生、不占用机器人。**建议观察而非现在投入。**

---

## 3. 十个方案

每个方案格式：**要解决的具体缺口 → 文献依据 → 在你三个仓库里怎么落地 → 成本与前置条件 → 怎么判定它成功了**。

---

### 方案 1：建立验证集 + 关键区间动作误差（CI-MSE）作为 L1 门控

**缺口。** 当前 `splits = {"train": "0:30"}`，无验证集；`env_eval_freq=0`；部署默认取 `last` checkpoint，即最过拟合的那个。106 个 epoch、30 条演示、无图像增广的条件下，训练 loss 从 0.026 降到 0.022 的那 74k 步**无法用现有数据区分"继续学"和"背轨迹"**。

**文献依据。** CI-MSE（2606.29898）[✓] 的 Spearman −0.87 vs 朴素 MSE −0.61；robomimic 的"最佳验证策略差 50–100%" [~]。核心机制：**误差要按任务关键区间加权，并做动作对齐**。

**落地。**

- `robot_data_platform`：
  - `tool/rdp convert` 增加 `--holdout-episodes N` 或 `--split-ratio`，在 `meta/info.json` 里写出真实的 `{"train": ..., "val": ...}`。**注意**：holdout 必须按 **episode** 切，不能按 frame 切——同一条 episode 的相邻帧高度相关，按帧切出的验证集是泄漏的。
  - `apps/lelab/lelab/train.py` + `runners/slurm.py`：新增 `val_loss` / `val_cimse` 指标解析进 `TrainingMetrics`，与现有 `parse_metrics_into` 同一路径落 `log.jsonl`。
  - 新增 `lelab/evaluate.py`：对 `{50k, 100k, 150k, 200k, last}` 全部 checkpoint 批量算离线指标，产出 checkpoint 排名表。
- `lerobot_vlahost`：实现 CI-MSE 计算本体。关键区间的识别，按成本从低到高有三种做法：
  1. **速度/夹爪事件**：末端速度过零点、夹爪开合跳变前后 ±k 帧——零额外标注，建议先做这个。
  2. **动作方差**：在演示集上按时间对齐后计算跨 episode 的动作方差，方差大的时刻即"关键"。
  3. **VLM 子目标切分**：用方案 5 的裁判标出子目标边界。

  必须按 **chunk 内位置 k 分桶**统计（k=0/20/50/79），这同时暴露 `act_delta` 报告 §2.3 指出的权重搬移效应。

**成本。** 1–2 人周。无硬件需求。**前置**：需要重新转换或重新切分现有数据集的 split（不需要重新采集）。

**判定成功。** 在至少 6 个已有 checkpoint（跨 2 个任务）上，CI-MSE 排名与真机成功率排名的 Spearman |ρ| > 0.7，且显著优于朴素 MSE 在同一组上的表现。**如果达不到，说明关键区间识别方式选错了，换第 2 或第 3 种，不要直接放弃离线指标。**

---

### 方案 2：部署保真离线回放（deploy-faithful replay）

**缺口。** 这是**文献没有专门覆盖、但对你影响最大**的一项。此前审计已确认三处训练/部署不一致：

1. 下发给机器人的是**合成 Hermite 零速 S 曲线**，不是策略输出的原始动作序列；
2. gripper 状态尺度约为训练范围的 **1.25 倍**，构成 OOD 输入；
3. `chunk_interval_s` 是死代码，真实节奏由 server 的 `need_new_chunk` 驱动（`vla_node.py:434`，含 `_has_reached_target` 的 2° 阈值门控）。

**后果**：无论方案 1 的离线指标做得多好，**如果它在 `policy.forward()` 的输出上算 MAE，它评的就不是被部署的那个东西**。真正被执行的是 `Hermite(chunk_player(postprocess(policy_output)))`，且下一次观测的时机由 `_has_reached_target` 决定而非固定周期。

**文献依据。** 没有直接对口的论文，但三条线支撑这个做法：Real-is-Sim（2504.03597）[~] 的核心主张正是**用同一个数字孪生贯穿采集/训练/评估/部署**以消除环节间的分布错位；CI-MSE 的"动作对齐流程使之匹配 rollout 时行为" [✓] 是同一思想的离线版本；Sentinel [~] 也建立在"监控的是策略实际输出的 chunk 分布"这一前提上。

**落地。**

- `Apex_Deploy_new`：把 `vla_node.py` 里的 chunk 播放器（Hermite 插值、`chunk_blend_from_current`、500 Hz 重采样）与 `_check_need_new_chunk` 触发逻辑**抽成一个不依赖 ROS 的纯函数库**（如 `vlahost/chunk_player_core.py`），ROS 节点变成它的薄封装。这一步是纯重构，行为不变，可用现有录制 bag 做回归验证。
- `lerobot_vlahost`：新增 `--replay` 评估模式——读一条真机 bag，逐帧喂观测给策略，走**完整后处理链路 + 上述 chunk_player_core**，输出"如果当时用这个策略会下发什么"，与 bag 中记录的真实下发指令、以及演示的真值动作做三方对比。
- `robot_data_platform`：转换管线在 `conversion_manifest.json` 里已经记录 profile 与对齐参数，追加记录**部署侧的 gripper 尺度与关节单位约定**，让"1.25 倍尺度失配"这类问题在转换时就能被 `rdp check-*` 检出而不是靠事后审计。

**成本。** 2–3 人周（重构占大头）。无硬件。

**判定成功。** 三项：(a) 用同一条 bag，`chunk_player_core` 离线复现的下发指令与 bag 中真实下发指令逐点误差 < 数值噪声；(b) 能定量报出"策略原始输出 → 实际下发"的形变幅度（每个关节的 RMS 差、速度谱差异）；(c) 方案 1 的离线指标切换到部署保真链路后，与真机成功率的相关性**不降低**（若显著提升，说明这条不一致确实是主要噪声源）。

---

### 方案 3：从全 topic 录制中提取行为指标体系

**缺口。** 真机侧只有人工数出的二值成功率。两个策略都是 40% 成功率时，无法区分哪个"接近成功"、哪个"完全乱来"；失败也无法归类。

**文献依据。** RoboEval（2507.00435）[~] 的三类行为指标（效率/协调/安全稳定）+ 分阶段进展指标，**在二值成功率饱和时仍有区分度**；Kress-Gazit et al.（2409.09491）[~] 的"使用与成功率互补的多个指标"；ROHE / 干预率作为连续人力指标 [~]。

**落地。** **不需要新增任何硬件或传感器**——`components/log_recorder_nodes_py/all_topic_log_recorder.py` 已经录了全部 topic。新增一个离线分析器（放 `robot_data_platform/tool/` 下，与 `rdp check-rosbag` 同一入口风格，例如 `rdp eval-rollout`），从 rollout bag 直接算：

| 类别 | 指标 | 从哪个 topic 算 |
|---|---|---|
| 效率 | 完成耗时、路径长度/最短路径比、停滞时长占比 | `/joint_states`、末端位姿 |
| 平滑 | jerk RMS、加速度谱高频能量、方向反转次数 | `/joint_states` 微分 |
| 协调 | 双臂时间重叠率、左右臂活动量不对称度 | 两臂 `joint_cmd` |
| 稳定 | 夹爪开合抖动次数、抓取后物体滑移代理量 | 夹爪反馈 / `end_effector` |
| 一致性 | chunk 重叠区分歧度（见方案 9） | 推理日志 |
| 进展 | 分阶段完成率（接近/对准/抓取/搬运/放置） | 方案 5 的 VLM 裁判 |
| 人力 | 干预次数、干预总时长、ROHE | `lerobot_rollout --strategy.type=dagger` 的干预记录 |

`lerobot_vlahost` 侧：`scripts/lerobot_rollout.py` 的 `dagger` 策略已经存在，把干预事件结构化落盘即可得到干预率。

**成本。** 1.5–2 人周。**前置**：需要确认 rollout 时 `all_topic_log_recorder` 确实开启（这应成为评估流程的硬性要求）。

**判定成功。** 在一组成功率相近（差异 < 10 pp）的策略对上，至少 3 个行为指标能给出统计上可区分的排序，且该排序与人类操作员的主观偏好一致。

---

### 方案 4：统计严谨的真机评估协议（盲测配对 A/B + 序贯检验）

**缺口。** 这是**当前最便宜、收益最直接**的一项。你现在"成功率没提升"的结论，在 10–30 次试验的样本量下**可能连"有没有变化"都没测出来**——15 pp 宽的置信区间会吞掉绝大多数真实改进。

**文献依据。** N-SCORE（2603.13616）[✓]：SAVI 序贯检验，支持连续/部分给分指标，减少 45–70% 评估负担；LBM（2507.05331）[~]：盲测随机化 A/B，1800 真机 rollout；Kress-Gazit et al.（2409.09491）[~]；TRI 统计方法论 [~]：CP 区间的样本量表；RoboArena（2506.18123）[~]：**双盲成对比较比集中式绝对成功率更准确地排序策略**。

**落地。**

- `robot_data_platform` 是这套协议的天然载体（它已经在管 Slurm 作业与 job 元数据）：新增 **eval job** 类型，强制记录：策略 checkpoint 哈希、**随机化的初始条件清单**（物体位姿由脚本随机生成而非人工摆）、盲测标签（操作员看不到当前跑的是 A 还是 B）、成功判据全文、每次试验的时间戳与操作员 ID。
- **配对设计优先于独立组**：同一初始条件下依次跑 A 和 B（顺序随机化），用配对检验——这能消掉"这次摆得比较好摆"的方差，是样本量效率最高的一步。
- 实现 N-SCORE 的序贯停止规则：每完成一个配对块就更新证据累加器，达到阈值即停。**接方案 3 的连续指标**（N-SCORE 的优势正在于此），而不只是二值成功。
- `Apex_Deploy_new`：`marvin_qp_controller` 可脚本化，用于**自动生成随机初始位姿**（把物体移到指定位置），保证初始条件真的随机且可复现。

**成本。** 1 人周（协议 + 记录）+ 1 人周（N-SCORE 实现）。**无需新代码就能开始的部分**：从下一轮实验起，立刻做盲测 + 配对 + 记录初始条件 + 报 Clopper-Pearson 区间。这一条今天就能执行。

**判定成功。** 每份实验报告都能回答：试验次数、初始条件如何随机化、成功判据、置信区间、检验统计量与 p 值/证据比。以及一个更硬的判据——**在已知有效的改动上，协议应能以少于旧方法的试验数检出它**。

---

### 方案 5：VLM 自动裁判——成功判定 + 子目标评分

**缺口。** 真机成功率靠人工数，因此不可能做到 AutoEval 式的规模；且"成功"判据在不同人不同天之间漂移。

**文献依据。** VLM success detectors 微调后 ≥95% 准确率 [~]；StepEval（2509.19524）[~] 的逐子目标成功率向量；AutoEval（2503.24278）[~] 的成功分类器是其无人值守评估的基石；Robometer（2603.16065）[~] 零样本输出稠密进度 + 二值成功；TOPReward（2602.19313）[~]。

**落地。** **`lerobot_vlahost/src/lerobot/rewards/topreward/` 已经是这个东西了**，只是当前定位为 RL 奖励。

- 把 TOPReward 从 `rewards/` 提升为**评估裁判**的一种后端，新增统一接口 `judge(video, instruction) -> {success, progress, subgoals}`，允许挂不同后端（TOPReward 零样本 / 微调分类器 / Robometer 式模型）。
- 子目标定义写成**数据而非代码**——沿用你们 `tool/robot_data/profiles/builtin/*.json` 的既有惯例（"profiles are data, never Python"），每个任务一个 `subgoals.json`（接近 → 对准 → 闭合 → 抬起 → 移动 → 释放）。
- 输入源：`Apex_Deploy_new` 的 `/quad_tile` 录制流直接就是裁判的视频输入，无需新采集。
- **裁判必须先被评估**：人工标注 ≥200 条 episode（含成功/失败/边缘各类），报告裁判 vs 人工的准确率、Cohen's κ、以及分子目标的混淆矩阵。**κ < 0.7 的裁判不要投入使用。**

**成本。** 2 人周 + 200 条人工标注（约 1–2 人日）。**前置**：`lelab-venv` 目前落后于 `train-venv` 且无法加载新模型——VLM 裁判建议跑在独立的推理环境/独立 Slurm 作业里，不要塞进 lelab 的 venv。

**判定成功。** κ ≥ 0.7 且成功检测准确率 ≥ 90%；子目标向量能对至少一类失败给出正确定位（如"抓取成功但放置失败"）。

---

### 方案 6：为你自己的场景建 real-to-sim 视觉匹配评估环境

**缺口。** 你说"仿真只评估动作质量，缺视觉监督"——这是**用别人的 benchmark** 的必然结果，不是仿真的固有属性。而 §2.3 [✓] 证明了公开 benchmark 的分数确实不可信（LIBERO 只有 19.8% 的进展统计显著）。

**文献依据。** SIMPLER（2405.05941）[~]：**视觉匹配**而非全保真孪生，1500+ 配对评估证明相关性；PolaRiS（2512.16881）[✓ 部分]：COLMAP + Gaussian Splatting → Isaac Sim 的自动化流水线；Real-is-Sim（2504.03597）[~]；GS 软体（2511.04665）[~]。

**落地。** 你的起点比大多数团队好：

- `Apex_Deploy_new/robot_node/marvin_description/` 已有 **mjcf + urdf + meshes**；`marvin_hardware_sim/` 已有仿真硬件接口；`marvin_qp_controller` 提供与真机同一套控制律——**控制差距这一半已经解决了大半**。
- 剩下的是**视觉差距**。关键是遵循 SIMPLER 的取舍：**不要追求物理与渲染的全保真**，只做视觉匹配——用真机背景图做环境纹理叠加、匹配相机内参/外参/曝光、复刻 `/quad_tile` 的四宫格拼接布局（含头部 tile 以 2× 放大存储、1280×960 降采样到 640×480 这一细节，`split_hero3_image` 的等价实现必须逐位一致，否则训练与评估的输入分布再次错位）。
- 场景重建：`/home/kewei/YING/colmap` 已存在，PolaRiS 路线（多视角拍摄 → COLMAP → GS → 仿真）是现成的。
- `lerobot_vlahost/src/lerobot/envs/` 下新增 `marvin_sim.py`，与已有的 libero/robocasa 封装同构，接入现有 `lerobot_eval.py`。

**成本。** 4–8 人周，是本文最重的一项。**建议先只做一个任务**（选一个当前失败模式最清楚的），验证相关性后再扩。

**判定成功。** **不要用"仿真成功率高不高"判定，要用相关性判定**：在 ≥6 个 checkpoint 上做配对 sim-real 评估，计算 **MMRV 与 Pearson/Spearman 相关**。SIMPLER 的标准是"低 MMRV + 高 Pearson"。**如果相关性不达标，这个环境就不能用于选型——但它仍可用于方案 10 的鲁棒性扫描。**

---

### 方案 7：仿真 + 少量真机的联合统计推断（prediction-powered inference）

**缺口。** 方案 6 建好后立刻会遇到的问题："仿真说 A 比 B 好，我该信几分？"这是本文中唯一**直接、定量**回答该问题的方法。

**文献依据。** SureSim（2510.04354）[~]：把"合并真机与仿真"形式化为 prediction-powered inference——用少量**配对**评估校正大规模仿真的偏差，用非渐近均值估计给出真机性能的置信区间，节省 20–25% 硬件评估。Sim2Real Predictivity [~] 提供更早的 SRCC 视角。

**落地。**

- 数据层面这几乎是免费的：方案 4 的 eval job 已经在记录真机试验，方案 6 的仿真环境在产出仿真试验，**只需要保证两侧的初始条件是配对的**（同一物体位姿在真机和仿真里各跑一次）。这是设计约束，不是额外工作量——但**必须在方案 6 动工前就定下来**，事后补配对的成本很高。
- 实现放 `robot_data_platform`（它已经是统计与作业元数据的归属地），产出"仿真估计 + 真机校正 → 真机性能置信区间"。
- 与方案 4 的 N-SCORE 是互补关系：N-SCORE 管**何时可以停**，SureSim 管**仿真的偏差怎么扣**。

**成本。** 1–2 人周（在方案 4 与 6 之后）。

**判定成功。** 对一个已有充分真机数据的策略，用"大量仿真 + 少量配对真机"给出的置信区间**覆盖**其真机真值，且区间宽度小于同等真机试验数下的纯真机区间。

---

### 方案 8：AutoEval 式自主评估工作站

**缺口。** 即使有了自动裁判（方案 5），**场景复位仍需要人**——这是真机评估无法规模化的最后一道人力瓶颈。

**文献依据。** AutoEval（2503.24278）[~]：自动成功判定 + 自动复位 + 作业队列 = 全天候评估，**人力 −99%**，且比照片级仿真和验证误差都更可靠。RoboArena（2506.18123）[~] 是另一条路（用组织方式而非自动化解决规模问题），若你未来有多个机位/多个站点可以考虑。

**落地。**

- **调度层你已经有了**：Slurm + `apps/lelab/lelab/jobs.py` + `runners/slurm.py`。新增 eval 队列，语义与训练作业完全一致（提交 → 排队 → 落 `log.jsonl`）。
- **复位策略**：两条路，建议先走第二条。
  1. 学习式 reset policy（AutoEval 的做法，需要额外采数与训练）；
  2. **脚本化复位**——`marvin_qp_controller` 已经能做位置控制，对"把物体放回随机起始位姿"这类任务，脚本化复位比学一个 policy 便宜得多且更可靠。它同时满足方案 4 对"初始条件可随机、可复现"的要求，**一份工作解决两个方案的需求**。
- **安全**：无人值守必须配方案 9 的在线监控作为急停条件，否则一次夜间失控会毁掉硬件。**这是方案 9 优先于方案 8 的原因。**

**成本。** 3–5 人周。**前置**：方案 5（裁判）+ 方案 9（安全网）必须先就位。

**判定成功。** 连续无人值守运行 8 小时以上，自动判定结果与人工复核的一致率 ≥ 90%，且期间无需人工干预硬件。

---

### 方案 9：在线一致性与失败监控（同时是评估信号与安全网）

**缺口。** 没有任何运行时信号能说明"策略在当前场景里是否失控/是否 OOD"。这既是评估盲区，也是方案 8 无人值守的安全前提。

**文献依据。** Sentinel（2410.04640）[~]：动作 chunk 分布的**时间一致性**检测错乱失败 + VLM 视频 QA 检测推进失败，两者结合多检出 **18%**，构建只需一组成功 rollout + 任务描述；SAFE（2506.09937）[~]：VLA 内部特征已含跨任务通用的成败知识，且**token 级不确定性无效、采样一致性太贵**——直接告诉你哪些路不用走；Foresight（2606.23085）[~]；VLAConf（2605.29605）[~]。

**落地。** **这一项在你的架构里近乎免费**，因为 chunk 机制天然提供了一致性信号：

- `Apex_Deploy_new/robot_node/vlahost/vlahost/vla_node.py` 的 `_check_need_new_chunk` 逻辑下，前后两次推理产生的 chunk 在时间轴上**存在重叠区**（尤其在 `_has_reached_target` 未满足而 5 秒超时触发重推理时）。**重叠区的逐点分歧度就是 Sentinel 的时间一致性统计量**，无需任何额外前向推理。
- 把该分歧度发布为一个 ROS topic 并落进 `all_topic_log_recorder`，它立刻成为：(a) 方案 3 的一个行为指标；(b) 方案 8 的急停触发条件；(c) 一个**不需要跑完整 episode 就能得到的连续策略质量代理量**——这点对早期筛选 checkpoint 很有价值。
- 阈值标定：按 Sentinel 的做法，用一组已知成功的 rollout 统计分歧度分布，取分位数作阈值。
- 推进失败检测复用方案 5 的 VLM 裁判（低频调用，如 1 Hz）。
- **不要**走 token 级不确定性或多次采样一致性——SAFE 已经证明前者无效、后者实时不可行。

**成本。** 1–1.5 人周。**这是本文性价比最高的一项**（成本最低，同时服务评估、安全、筛选三个目的）。

**判定成功。** 在一组已标注的成功/失败 rollout 上，一致性分数的 AUROC ≥ 0.75；作为急停条件时，误触发率低到不影响正常评估吞吐。

---

### 方案 10：预测式红队——离线鲁棒性画像

**缺口。** 你不知道模型对光照、干扰物、物体位置、**以及图像压缩质量**的敏感度分别有多大，因此不知道下一批数据该采什么。

**文献依据。** RoboART / Predictive Red Teaming（2502.06575）[~]：生成式图像编辑扰动标称观测 + **策略专属异常检测器**预测各条件下的性能下降，12 种非标称条件、500+ 硬件试验验证，预测误差 **< 0.19**；下游价值更大——**按预测的不利条件针对性采数微调，性能提升 2–7 倍**。

**落地。** **这条与你已有的工作直接接上**：

- `robot_data_platform/test_raw_jpeg` 与 `literature-review.zh.md` 已经建立了"压缩 → 像素保真 → DINO 特征"这条链（PSNR ↔ CLS 余弦 Spearman ρ=0.927），并明确留下了**"像素保真度 ≠ 策略成功率"**这个未解问题。RoboART 的异常检测器方法就是缺失的那一段桥——它把特征空间的偏移映射为**成功率下降的预测值**。
- 扰动因子清单应包含你已识别的真实风险：`--crf` / codec（你已有 h264 CRF 0/20/30 与 AV1 的真实数据集，**这是天然的、无需生成模型的扰动对**）、光照、`/quad_tile` 拼接的 tile 裁剪偏移、干扰物、物体初始位姿、gripper 尺度失配（方案 2 已确认的 1.25×）。
- 实现放 `lerobot_vlahost`（异常检测器需要访问策略内部特征），扰动数据生成放 `robot_data_platform/tool/`。
- 与方案 6 的关系：即使方案 6 的仿真环境**相关性不达标**（不能用于选型），它依然可以用于本方案的鲁棒性扫描——因为这里关心的是**相对下降幅度**而非绝对成功率。

**成本。** 2–3 人周。**前置**：需要一个已知真机成功率的基线策略做校准。

**判定成功。** 在 ≥5 种扰动条件上，预测的成功率下降与真机实测的平均绝对差 < 0.2（对齐 RoboART 的 0.19）；并据此产出一份"下一批数据应该采什么"的排序清单。

---

### 前沿观察（暂不投入）：世界模型作为评估器

**WorldEval / Policy2Vec**（2505.19017）[~]：用视频世界模型生成策略 rollout 视频并据此评估，不需要仿真器、不需要数字孪生、不占用机器人，且能排出策略与 checkpoint 优劣、可作安全检测器。天花板极高，但成熟度是本文所有路线中最低的。

**建议**：不投入实现，但**保留接口兼容性**——方案 5 的 `judge(video, instruction)` 接口如果设计得当，未来只需替换视频来源（真机录制 → 世界模型生成）即可接入，无需重构。这是一个几乎零成本的架构决策。

---

## 4. 优先级与路线图

### 4.1 排序依据

| 方案 | 成本 | 阻断了什么 | 无需硬件 | 优先级 |
|---|---|---|---|---|
| 4 统计协议 | 1–2 周 | 一切结论的可信度 | ✓ | **P0（今天就能开始一半）** |
| 2 部署保真回放 | 2–3 周 | 所有离线指标的有效性 | ✓ | **P0** |
| 1 验证集 + CI-MSE | 1–2 周 | checkpoint 选择、快速迭代 | ✓ | **P0** |
| 9 在线监控 | 1–1.5 周 | 方案 8 的安全前提 | ✓ | **P1（性价比最高）** |
| 3 行为指标 | 1.5–2 周 | 失败归因 | ✓ | **P1** |
| 5 VLM 裁判 | 2 周 | 方案 8 的规模化前提 | ✓ | **P1** |
| 10 预测式红队 | 2–3 周 | 数据采集方向 | ✓ | P2 |
| 6 real-to-sim | 4–8 周 | 视觉监督缺失 | 需拍摄 | P2 |
| 7 联合统计推断 | 1–2 周 | 仿真可信度的量化 | ✓ | P2（依赖 4+6） |
| 8 自主评估站 | 3–5 周 | 评估吞吐 | 占用机器人 | P3 |

**为什么方案 4 排第一而不是方案 1**：因为方案 4 的一半（盲测、配对、随机化初始条件、报置信区间）**不需要写任何代码**，从下一轮实验起就能执行，而它决定了此后所有实验结论是否可信。先修方法论，再修工具。

**为什么方案 2 排在方案 1 之前**：如果离线指标算在错误的对象上（策略原始输出而非实际下发指令），方案 1 做得再精细也是在优化一个错的量。**先确定"评什么"，再优化"怎么评"。**

### 4.2 三阶段路线

**第一阶段（第 1–6 周）：把 L1 建起来，让迭代周期从"一轮真机"降到"一次离线跑"**

方案 4（协议，立即）→ 方案 2（部署保真回放）→ 方案 1（验证集 + CI-MSE）→ 方案 9（在线监控）。

阶段结束时应该能回答的问题：
- 200k 步的 checkpoint 和 100k 步的哪个更好？（现在无法回答）
- 相对动作到底有没有效果？（`act_delta` 那轮实验本可以在这里就结束）
- 策略的原始输出被部署链路改变了多少？（现在完全未知）

**第二阶段（第 7–14 周）：把 L3 的信息量提上来**

方案 3（行为指标）→ 方案 5（VLM 裁判）→ 方案 10（红队画像）。

阶段结束时：每次真机 rollout 产出的不再是一个 0/1，而是一个包含分阶段进展、平滑度、协调度、一致性分数、干预率的向量；且有一份"下一批数据采什么"的证据支持的清单。

**第三阶段（第 15 周起）：把 L2 建起来，并规模化**

方案 6（real-to-sim）→ 方案 7（联合推断）→ 方案 8（自主评估站）。

**注意第三阶段的失败模式**：方案 6 有实质概率相关性不达标。**这不是灾难**——它仍然服务方案 10，且方案 7 的 prediction-powered inference 框架本身就是为"不完美仿真器"设计的。但**不要在方案 6 的相关性验证通过之前，把任何选型决策交给仿真**。

### 4.3 一条贯穿性的设计约束

方案 1/3/5/9 都在产出**每条 episode 的多维指标向量**，方案 4 的 N-SCORE 恰好**需要**连续指标才能发挥其样本效率优势，方案 7 需要**配对**的 sim-real 记录。因此从第一天起就应固定一个统一的 **episode 级评估记录 schema**（策略哈希、初始条件、部署链路版本、各指标值、裁判输出、一致性分数、干预事件），落在 `robot_data_platform` 的 job 元数据体系里。**这个 schema 定得早，后面九个方案都省事；定得晚，每加一个方案就要回填一次历史数据。**

---

## 5. 对你原始判断的三点修正

1. **"当前场景评估只评估训练数据自身的数值并计算 loss"** —— 准确，且比你说的更严重：连**验证集**都没有（`splits = {"train": "0:30"}`），所以现在的 loss 连"泛化"这个词都谈不上，它测的是记忆。修复成本极低（方案 1）。

2. **"真实测试缺乏合适的评估指标，目前只能评估准确率"** —— 部分准确。真正的瓶颈不在"缺指标"，而在**缺统计**（方案 4）与**缺自动化**（方案 5/8）。指标本身文献里现成的很多（RoboEval 那套），而且**你的全 topic 录制已经把算这些指标的原始数据全存下来了**——这是三个仓库里最被低估的资产。

3. **"仿真场景只评估动作本身的质量，缺乏视觉监督"** —— 这是**通用 benchmark 的性质，不是仿真的性质**。SIMPLER 的 visual matching 路线（不追求全保真，只对齐观测分布）在 1500+ 组配对评估上证明了仿真可以预测真机。代价是要为你自己的场景建环境（4–8 人周），而不是用 LIBERO——后者已被 2026 年的诊断研究证明只有 19.8% 的报告进展在统计上可证显著 [✓]。

**最后一点，也是最重要的一点**：你说"没有真实的评估，很难找到模型优化的方向"。文献支持这个判断，但顺序需要倒过来——**在建立真实评估之前，先建立可信的评估**。方案 4 的盲测 + 配对 + 置信区间不需要任何新技术、不需要新硬件、可以今天开始，而它决定了你此后每一次实验的结论是真的还是噪声。`act_delta` 那轮"成功率未提升"的实验，在当前样本量下**很可能连结论都不成立**——不是"相对动作没用"，而是"这个实验没有能力检出它有没有用"。

---

## 参考文献

**核实状态**：[✓] = 已抓取原文核实标题/作者/日期/摘要与引用数字；[~] = 仅来自检索结果元数据与摘要。

### 离线指标与策略选择
1. [✓] Huang, H., Zheng, T., Chen, Y., You, J., Gao, Y. (2026-06-29). *Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies*. arXiv:2606.29898. https://arxiv.org/abs/2606.29898
2. [~] Mandlekar, A. et al. (2021). *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*（robomimic）. https://robomimic.github.io/study/
3. [~] *Offline Policy Evaluation for Manipulation Policies via Discounted Liveness Formulation*. arXiv:2605.11479. https://arxiv.org/pdf/2605.11479
4. [~] *DataMIL: Selecting Data for Robot Imitation Learning with Datamodels*. arXiv:2505.09603（ICLR 2026）. https://arxiv.org/html/2505.09603

### 评估方法论与统计
5. [~] Kress-Gazit, H., Hashimoto, K., Kuppuswamy, N., Shah, P., Horgan, P., Richardson, G., Feng, S., Burchfiel, B. (2024). *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation*. arXiv:2409.09491. https://arxiv.org/abs/2409.09491
6. [✓] Snyder, D., Badithela, A., Matni, N., Pappas, G., Majumdar, A., Itkina, M., Nishimura, H. (2026-03-13). *Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison*（N-SCORE）. arXiv:2603.13616. https://arxiv.org/html/2603.13616
7. [~] Toyota Research Institute. *Statistical Thinking for Robot Policy Evaluation: From Rigorous A/B Testing to Effective Visualization*. https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d
8. [~] TRI LBM Team (2025). *A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*. arXiv:2507.05331 / Science Robotics. https://arxiv.org/abs/2507.05331
9. [~] NVIDIA. *How to Evaluate General-Purpose Robot Policies for Real-World Deployment*. https://developer.nvidia.com/blog/how-to-evaluate-general-purpose-robot-policies-for-real-world-deployment/

### Benchmark 可信度
10. [✓] Jiang, T., Tan, X., Wheeler, S., Sun, L., Ayalew, T. W., Walter, M. (2026-06-02). *What Are We Actually Benchmarking in Robot Manipulation?* arXiv:2606.04233. https://arxiv.org/html/2606.04233v1
11. [~] *Trustworthy Evaluation of Robotic Manipulation: A New Benchmark*. arXiv:2601.18723. https://arxiv.org/pdf/2601.18723

### 行为指标与细粒度评估
12. [~] *RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation*. arXiv:2507.00435. https://arxiv.org/abs/2507.00435 · https://robo-eval.github.io/
13. [~] *Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation*. arXiv:2605.19986. https://arxiv.org/html/2605.19986v1
14. [~] *Easy-IIL: Reducing Human Operational Burden in Interactive Imitation Learning via Assistant Experts*. arXiv:2603.12769. https://arxiv.org/html/2603.12769v1
15. [~] Liu, H. et al. *Robot Learning on the Job: Human-in-the-Loop Autonomy and Learning During Deployment*（Sirius）.
16. [~] Hoque, R. et al. *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning*. https://openreview.net/forum?id=KKBfrCzCVOn

### Real-to-Sim 与仿真评估
17. [~] Li, X. et al. (2024). *Evaluating Real-World Robot Manipulation Policies in Simulation*（SIMPLER）. arXiv:2405.05941, CoRL 2024. https://arxiv.org/abs/2405.05941 · https://simpler-env.github.io/
18. [✓ 标题/作者/日期] Jain, A., Zhang, M., Arora, K., Chen, W., Torne, M., Irshad, M. Z., Zakharov, S., Wang, Y., Levine, S., Finn, C., Ma, W.-C., Shah, D., Gupta, A., Pertsch, K. (2025-12-16). *PolaRiS: Scalable Real-to-Sim Evaluations for Generalist Robot Policies*. arXiv:2512.16881. https://arxiv.org/pdf/2512.16881
19. [~] Abou-Chakra, J. et al. (2025-04-04). *Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin*. arXiv:2504.03597. https://arxiv.org/html/2504.03597
20. [~] *Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions*. arXiv:2511.04665. https://real2sim-eval.github.io/
21. [~] *Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators*（SureSim）. arXiv:2510.04354. https://arxiv.org/abs/2510.04354 · https://suresim-robot-eval.github.io/
22. [~] Kadian, A., Truong, J. et al. *Sim2Real Predictivity: Does Evaluation in Simulation Predict Real-World Performance?*

### 真机评估自动化
23. [~] Zhou, Z. et al. (2025). *AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World*. arXiv:2503.24278, CoRL 2025. https://arxiv.org/abs/2503.24278 · https://auto-eval.github.io/
24. [~] Atreya, P., Pertsch, K., Lee, T., Kim, M. J., Jain, A., Kuramshin, A., Eppner, C., Neary, C. et al. (2025). *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*. arXiv:2506.18123, CoRL 2025. https://arxiv.org/abs/2506.18123

### VLM 裁判与奖励模型
25. [~] *Score the Steps, Not Just the Goal: VLM-Based Subgoal Evaluation for Robotic Manipulation*（StepEval）. arXiv:2509.19524. https://arxiv.org/html/2509.19524
26. [~] Luo, F. *Vision-Language Models for Robot Success Detection*. AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/30552/32714
27. [~] Chen, S., Harrison, C., Lee, Y.-C., Yang, A. J., Ren, Z., Ratliff, L. J., Duan, J., Fox, D., Krishna, R. *TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics*. arXiv:2602.19313. https://topreward.github.io/webpage/ （**已移植进 `lerobot_vlahost/src/lerobot/rewards/topreward/`**）
28. [~] *Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models*（Robometer）. arXiv:2603.16065
29. [~] *Progress Reward Modeling for Robotic Learning: A Comprehensive Survey*. arXiv:2607.21655. https://arxiv.org/html/2607.21655
30. [~] Baumli, K., Baveja, S. et al. *Vision-Language Models as a Source of Rewards*. arXiv:2312.09187

### 在线监控与失败检测
31. [~] Agia, C. et al. (2024). *Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress*（Sentinel）. arXiv:2410.04640, CoRL 2024. https://github.com/agiachris/sentinel
32. [~] *SAFE: Multitask Failure Detection for Vision-Language-Action Models*. arXiv:2506.09937, NeurIPS 2025. https://arxiv.org/abs/2506.09937
33. [~] *Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents*. arXiv:2606.23085
34. [~] *VLAConf: Calibrated Task-Success Confidence for Vision-Language-Action Models*. arXiv:2605.29605
35. [~] *Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning*. arXiv:2604.16683. https://arxiv.org/html/2604.16683

### 鲁棒性与红队
36. [~] Majumdar, A., Sharma, M., Kalashnikov, D., Singh, S., Sermanet, P., Sindhwani, V. (2025). *Predictive Red Teaming: Breaking Policies Without Breaking Robots*（RoboART）. arXiv:2502.06575, CoRL 2025. https://predictive-red-team.github.io/

### 世界模型评估器
37. [~] *WorldEval: World Model as Real-World Robot Policies Evaluator*. arXiv:2505.19017. https://arxiv.org/abs/2505.19017

### 本仓库内部前序工作
38. `paper/policy/literature-review.zh.md` —— 图像压缩 → 像素/特征保真度，明确留下"像素保真度 ≠ 策略成功率"的未解问题（对应方案 10）
39. `paper/policy/experiment_report/act/act_delta-batch5-failure-analysis-2026-08.md` §3 —— "没有任何离线评估信号"（对应方案 1、2）
40. `tool/CONVERSION.md` —— 已建立的数据侧审计范式，是模型侧审计的模板
