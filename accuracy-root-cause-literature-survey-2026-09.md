# 精度不足的成因：跨策略（ACT / ACT-DiT / patch_policy / VITA）文献综述

**范围。** 本项目四条策略线共同表现出的「精度不足」，在已发表文献中被归因到哪些原因；每条原因有哪些论文做过测量、用了什么方法、报了什么数字；以及这些结论与本仓库 `experiment_report/` 已产出的本地证据如何对齐。同时覆盖**离线 MAE** 与**真机成功率**两个口径，并把「两者不一致」本身作为一条独立的原因轴。

编撰于 2026-09-01。**核实方法：** 全部 30 篇均通过 WebFetch 抓取 `arxiv.org/abs/<id>` 摘要页逐篇核对（标题、作者、v1 日期、摘要原文），未采用任何来自模型记忆的引用。凡数字带引号者为摘要或正文原文；标注 *（作者列表未完整解析）* 或 *（已撤稿）* 者见 §8。

---

## 0. 结论先行

文献把「视觉模仿策略精度不足」拆成六条互不重叠的成因轴。按本地证据的**已确证强度**排序：

| 轴 | 名称 | 本地证据状态 | 文献支持强度 |
|---|---|---|---|
| **A** | 部署侧执行层扭曲动作 | **已实测确证**（Hermite fallback 100% 触发） | 强，2025–2026 密集 |
| **B** | 目标参数化（绝对 vs 相对 / 关节 vs EEF） | **已实测确证**（EEF 比值 2.97→1.41） | 强，有直接消融数字 |
| **D** | 执行窗口长度选错 | **已实测确证**（h=8 下 null 胜过三个权重） | 强，2026 集中爆发 |
| **C** | 条件融合 / 捷径学习 | **部分证伪**（state token 被读取，不是被淹没） | 中，结论分歧 |
| **E** | 评估口径与真机成功率脱节 | **已实测确证**（离线 MAE 与部署窗口结论相反） | 中，2026 才有量化 |
| **F** | 演示数据质量 | 未测 | 中，指标本身尚在争议 |

**对用户原问题的直接回答：** 「动作执行层算得不准」与「视觉-动作特征融合不够好」这两个假设中，**前者在本地已有决定性证据，后者已被本地数据部分证伪**——但真正的第三个答案（目标参数化）才是本地唯一一个「改了就见效」的杠杆。文献在这三点上的分布与本地证据一致。

---

## 1. 本地证据与文献成因轴的接口

先把本仓库已测量的事实列出，后续每条文献都回指这张表。

| 本地实测事实 | 出处 | 对应轴 |
|---|---|---|
| Hermite fallback 阈值 `threshold_rad = 0.0087`（0.5°），实际 `max\|action0 − state\|` 中位数 ~0.12 rad → **100% 的 chunk 被触发** | `record-chunk-is-filtered` | A |
| `chunk_len`(8 或 4) ≤ `K = 40` → 走 else 分支，**整条 chunk 被替换**为 `start_velocity = end_velocity = 0` 的三次 Hermite，只覆盖约 **65%** 的指令位移 | 同上 | A |
| 复现验证：回放滤波栈得到的 intra-chunk 速度峰/边比 1.89，实测 1.99；仅二项平滑只能到 1.14 | 同上 | A |
| `bridge_steps = min(40, chunk.shape[0])`，`n_action_steps: 8` → **桥覆盖全部 8 个执行点** | `patch-policy-deploy-window-is-all-bridge` | A + D |
| 部署窗口 h=8 实测 MAE：`hold_state` null **0.02825** / ACT **0.02432** / patch_policy 三个权重 0.03087–0.03110 → **每个 patch_policy 权重都输给「什么都不做」** | 同上 | D + E |
| 桥单独把 patch_policy 在 h=8 改善 **26–30%**，其余四个滤波器各 <3% | 同上 | A |
| `first_action_vs_state` 比值：EEF **1.41** vs 关节空间 2.97 / 2.81 / 3.05 / 3.95 | `eef-action-space-fixes-pose-anchoring` | **B** |
| EEF 是唯一在 step 1 胜过 null 的权重（0.02170 vs 0.02283） | 同上 | B |
| state 消融：`state_swapped` 移动输出 4.2%，`zero_state` 2.2%，`state_history_reversed` **0.02%** | 同上 | C |
| decoder cross-attention 落在 state 槽位 **1.12% = 8.65× 均分份额**；范数比 **0.76**；memory 共 3846 槽（1 + 5×769） | `patch-policy-state-token-is-read-not-drowned` | C |
| 遮挡头部相机**背景**比遮挡物体更伤性能 | `act-policy-leans-on-background-pixels` | C |
| 部署时夹爪 state 约为训练范围的 **1.25×** | `gripper-state-scale-mismatch` | A（归一化） |
| 单次推理 0.30–0.60 s vs 0.267 s 窗口 → duty > 1，机械臂饥饿 | `patch-policy-deploy-window-is-all-bridge` | A + D |

---

## 2. 轴 A —— 部署侧执行层扭曲动作

这是本地证据最硬的一条，也是 2025–2026 文献最密集的一条。**核心共识：chunk 边界与推理延迟造成的执行失真，其幅度足以掩盖模型本身的差异。** 本项目的特殊性在于，失真不是来自异步执行的固有难题，而是来自一个**阈值设错、对全部 chunk 无条件触发的手写后处理栈**——文献里没有人报告过这么极端的情形，但下列工作解释了为什么这类失真的代价如此之大。

### 2.1 问题的奠基与主线

**Real-Time Execution of Action Chunking Flow Policies (RTC)**
Kevin Black, Manuel Y. Galliker, Sergey Levine · arXiv:2506.07339 · v1 2025-06-09
提出问题的标准表述：action chunking 解决了时间一致性，但"does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries"。方法为推理期算法，"generates the next action chunk while executing the current one, 'freezing' actions guaranteed to execute and 'inpainting' the rest"，对任意 diffusion/flow VLA 免重训。在 Kinetix 12 个动态任务 + 6 个真机双臂任务上验证。
**接口：** LeRobot 已内置 RTC（`huggingface.co/docs/lerobot/rtc`）。本地 `record-chunk-is-filtered` 给出的建议「先修阈值/K，或改用 RTC 引擎」与此直接对应。注意 RTC 解决的是**延迟下的边界一致性**，不解决本项目「桥把整条 chunk 换掉」的问题——那是配置错误，不是算法缺口。

**Real-Time Robot Execution with Masked Action Chunking (REMAC)**
Haoxuan Wang, Gengyu Zhang, Yan Yan, Yuzhang Shang, Ramana Rao Kompella, Gaowen Liu · arXiv:2601.20130 · v1 2026-01-27
指出 RTC 一系工作只处理了 inter-chunk discontinuity，遗漏了另一个因素："intra-chunk inconsistency, where the robot's executed action chunk partially misaligns with its current perception"。方法是在预训练策略上学 corrective adjustments（masked action chunking）+ prefix-preserved sampling。
**接口：** 这是与本项目最贴近的诊断。本地的 Hermite 桥正是一种极端的 intra-chunk 失配——**执行的轨迹与模型基于该帧观测所规划的轨迹在整条 chunk 上都不一致**，而不仅在边界。

**DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors**
Pengcheng Wang, Kaiwen Hong, Chensheng Peng, Katherine Driggs-Campbell, Masayoshi Tomizuka, Chenfeng Xu, Chen Tang · arXiv:2604.25050 · v1 2026-04-27
论点：flow-matching + RTC 结构上次优，因为 inpainting 来自推理期修正而非基座策略本身。离散扩散策略（迭代 unmasking）天然就是异步执行器，inpainting 是其原生操作。报告真机 hockey defend 任务上比 flow-matching RTC **成功率高 65%**，比训练期 flow-matching RTC 高 30%，推理计算量约为从头生成的 **~0.7**。
**接口：** 若考虑替换 patch_policy 的扩散头，这是一条把「异步执行」从后处理下沉到架构的路线。

**Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection (PAINT)**
Trong-Bao Ho, Quang-Tan Nguyen, Thien-Loc Ha, Gia-Binh Nguyen, Viet-Thanh Nguyen, Long Dinh, Minh N. Vu, Duy M. H. Nguyen, An Thai Le, Ngo Anh Vien · arXiv:2606.19774 · v1 2026-06-18
把异步推理**重构为噪声选择问题而非轨迹引导问题**：通过 backward Euler inversion 选出恰当的初始噪声，让未经修改的 flow ODE 自然产出与已执行前缀一致的下一条 chunk。training-free，无梯度、无重训、不改策略。12 个仿真 benchmark + 6 个真机任务（单臂/双臂/人形）。
**接口：** 免重训、不动模型权重，是本项目在不重训任何 checkpoint 的前提下可先行验证的方案之一。

### 2.2 边界平滑与轨迹表示

**SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies**
Dijia Zhan, Xuemiao Xu, Jinyi Li, Jie Tang · arXiv:2607.04609 · v1 2026-07-06
命名了本项目也会遇到的机制——"multimodal bifurcation: a cross-chunk inconsistency in which adjacent chunks generated from independent Gaussian latents can converge to incompatible trajectory modes"。核心机制 Velocity-guided Loss Steering，利用上一条 chunk 未执行的尾巴作为解析一致性参考，每个 Euler 步后做闭式修正，不反传。LIBERO-10 + π₀.₅ 上 **boundary jerk −28%，chunk transition discontinuity −27%**，成功率持平，denoising 开销接近无引导基线。
**接口：** 数值上界得注意——最好的边界平滑方法也只把 jerk 降 28% 且**成功率持平**。这从反面支持本地结论：平滑类修补不是精度的主要杠杆。

**Smoother Action Chunking Flow Policy via Prior-Corrected Orthogonal Trust-Region Guidance (POTR)**
Kai Fang, Hailong Pei, Xuemin Chi · arXiv:2605.24433 · v1 2026-05-23
指出 RTC guidance 的两个缺陷：权重调度在中间时间步偏弱，且修正方向无约束会引入横向扰动。方法为引入 data-prior scale σd 增强中间时刻修正，并把引导向量分解为平行/垂直于去噪速度的分量、对垂直分量施加信赖域约束。LIBERO + π₀.₅ 上提升成功率并降低边界不连续、加速度与 jerk。消融显示**主要增益来自 prior-corrected weight**，正交信赖域进一步改善稳定性。

**LiPo: A Lightweight Post-optimization Framework for Smoothing Action Chunks Generated by Learned Policies**
Dongwoo Son, Suhan Park · arXiv:2506.05165 · v1 2025-06-05
三段式后处理：(1) inference-aware chunk scheduling，主动生成重叠 chunk 以避免推理延迟造成的停顿；(2) 重叠区线性混合；(3) 在有界扰动空间内的 jerk 最小化轨迹优化。在位置控制机械臂的动态操作任务上验证，显著降低振动与抖动。
**接口：** **这是与本项目 `send_next_action_chunk` 滤波栈最同构的一篇。** 关键差别：LiPo 的平滑被约束在 *bounded perturbation space* 内，而本地的 Hermite fallback 是无界替换——它可以把指令位移砍掉 35%。若要保留后处理路线，LiPo 的「有界扰动」是正确的约束形式。

**ABPolicy: Asynchronous B-Spline Flow Policy for Real-Time and Smooth Robotic Manipulation**
Fan Yang, Peiguang Jing, Kaihua Qu, Ningyuan Zhao, Yuting Su · arXiv:2602.23901 · v1 2026-02-27
明确列出原始动作空间同步推理的三个病症："intra-chunk jitter, inter-chunk discontinuities, and stop-and-go execution"。在 B 样条控制点空间做 flow matching：B 样条表示保证 chunk 内平滑，双向动作预测 + refitting 优化保证 chunk 间连续，异步推理提供实时更新。七个任务（含动态移动物体）上降低 jerk。
**接口：** "stop-and-go execution" 正是本地记录的 3 Hz judder 的文献名称。

**B-spline Policy: Accelerating Manipulation Policies via B-spline Action Representations (BSP)**
Xiaoshen Han, Haoyu Xiong, Haonan Chen, Chaoqi Liu, Antonio Torralba, Yuke Zhu, Yilun Du · arXiv:2607.09648 · v1 2026-07-10
不预测离散时间 chunk，而是预测 B 样条的 knots 与控制点，得到**时间连续、可时间缩放**的轨迹，底层控制器可用更高频率与速度执行。可直接接入标准策略学习流水线。
**接口：** 如果目标是让下位机拿到连续轨迹而非离散点再由手写滤波器补插值，这是把插值责任移进模型的路线——**本地当前的 Hermite 桥就是把插值责任留在了部署侧，且用了错误的边界条件（零速起停）。**

**ACG: Action Coherence Guidance for Flow-based Vision-Language-Action models**
Minho Park, Kinam Kim, Junha Hyung, Hyojin Jang, Hoiyeong Jin, Jooyeol Yun, Hojoon Lee, Jaegul Choo · arXiv:2510.22201 · v1 2025-10-25
把不连贯的来源上溯到**演示数据**："their high generative capacity makes them sensitive to noise in human demonstrations: jerks, pauses, and jitter which reduce action coherence"，并指出后果——"instability and trajectory drift during deployment, failures that are catastrophic in fine-grained manipulation where precision is crucial"。training-free 测试期引导。
**接口：** 连接轴 A 与轴 F：执行层抖动可能是模型忠实复现了演示里的抖动。

### 2.3 归一化与「可执行策略」的定义

**Same Weights, Different Robot: A Deployment Safety View of VLA Policies**
Jianwei Tai · arXiv:2606.03724 · v1 2026-06-02
论点：checkpoint 不等于策略。"the same normalized model output can become a different physical action after action unnormalization and controller conventions are applied"。给出 quantile 式动作归一化下的闭式 metadata 失配变换与 ExecSpec 证书（无需推理或 rollout 即可度量动作空间语义漂移）。LIBERO-Goal replay 上替换一个**貌似合理的同族 metadata key**，六个非夹爪维平均漂移 0.199，成功率从 **28/28 掉到 2/28**；LIBERO-Spatial 从 **26/26 掉到 0/26**；Object 全部四种替换均 0/28。
**接口：** 直接对应本地 `gripper-state-scale-mismatch`（部署夹爪 state ≈ 训练范围 1.25×）。这篇给出了该类错误的**破坏量级基准**——不是几个百分点，是成功率归零。任何精度归因在排除归一化/反归一化元数据错配之前都不成立。

**Deployment-Time Reliability of Learned Robot Policies**
Christopher Agia · arXiv:2603.11400 · v1 2026-03-12（博士论文）
把部署期可靠性拆为三类机制：运行时失败监测（不需失败数据或任务专用监督）、基于影响函数把部署期成败追溯到训练演示的可解释性框架、以及长时序任务的策略协调。
**接口：** 综述性质，适合作为轴 A/E/F 的统一框架引用。

---

## 3. 轴 B —— 目标参数化：本地唯一「改了就见效」的杠杆

**From Foundation to Application: Improving VLA Models in Practice (LingBot-VLA 2.0)**
Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, Kecheng Zheng · arXiv:2607.06403 · v1 2026-07-07

**这是本综述中与本地 EEF 发现最直接对应的定量证据。** §6.1（Action Space）与 Figure 11：

| 消融 | 平均成功率 |
|---|---:|
| 绝对关节动作（absQpos） | **33.7** |
| 相对关节动作（relQpos） | **55.0** |
| EEF 动作 | **56.0** |

原文机理解释："Across the four tasks, the standard deviation of relQpos is only 31%–37% of that of absQpos" —— 相对动作把预测目标从全局关节构型回归变成局部运动回归，目标更居中、方差更低。

关键补充：EEF 与关节的**平均**成功率接近（56.0 vs 55.0），但**任务级偏好差异极大**——Squeeze Ketchup（接触密集）EEF 81.7% vs 关节 41.7%；Barcode Scan 关节 58.7% vs EEF 24.0%。

**接口：**
- 本地 `first_action_vs_state` 比值 EEF **1.41** vs 关节 2.97–3.95，与「相对/EEF 目标方差更低、更居中」的机理完全一致，且本地是在**同一套数据、同一个头**上测出的自然实验。
- 但 LingBot 的任务级分化是一个**警告**：EEF 不是普遍优于关节空间。本地在 `tidy_up_stationery_le` 上的收益不能外推到接触特性不同的任务。建议按任务族分别验证，而不是全线切换。
- `act_delta` 的 `use_relative_actions` 移植（本地 §4.3 pre-gate 中位数 0.529）属于「相对关节」一列，对应 33.7→55.0 那一档；EEF 属于另一档。**两者是并列选项，不是递进关系**，本地报告应避免把它们叠加叙述。

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song · arXiv:2303.04137 · v1 2023-03-07
关于控制模式的原文结论："Diffusion Policy with a position-control action space consistently outperforms Diffusion Policy with velocity control"，机理为 "position control suffers less than velocity control from compounding error effects and is thus more suitable for action-sequence prediction"；且 "velocity control is more affected by latency than position control"。
**接口：** 本项目为位置控制，这一条已站在有利的一侧；但它也说明**动作表示的选择通过「复合误差」直接作用于精度**，与轴 B 同源。

---

## 4. 轴 D —— 执行窗口长度：本地 h=8 的配置在文献里是已知的坏区

**Diffusion Policy**（同上，arXiv:2303.04137）
原文："having an action horizon greater than 1 helps the policy predict consistent actions and compensate for idle portions of the demonstration, but too long a horizon reduces performance due to slow reaction time"，并 "found the action horizon of 8 steps to be optimal for most tasks that we tested"；"Diffusion Policy is able to maintain peak performance with latency up to 4 steps"。
**接口：** 注意口径差异。Diffusion Policy 的最优 8 步是在其**自身控制频率与 chunk 长度**下测得的；本地 patch_policy 的 chunk 长度为 50 而只执行 8，属于**训练目标与执行窗口严重不匹配**的配置，与「horizon=8 最优」不是同一件事。

**VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies (AutoHorizon)**
Haoxuan Wang, Gengyu Zhang, Yan Yan, Ramana Rao Kompella, Gaowen Liu · arXiv:2602.21445 · v1 2026-02-24
首先证明"varying the execution horizon leads to substantial performance deviations, with performance initially improving and then declining as the horizon increases"（先升后降的单峰）。机理分析给出两个现象："(i) intra-chunk actions attend invariantly to vision-language tokens, limiting adaptability to environmental changes; and (ii) the initial and terminal action tokens serve as stable anchors, forming latent centers around which intermediate actions are organized"。方法把 action self-attention 权重当作模型预测极限的代理，测试期动态估计每条 chunk 的执行窗口。
**接口：** 现象 (i) 与本地 `patch-policy-is-time-order-blind`（反转观测帧只移动 chunk 0.5%）指向同一件事的两面——chunk 内动作对观测变化不敏感。

**Dynamic Execution Horizon Prediction for Chunk-based Robot Policies (DEHP)**
Yuchi Zhao, Miroslav Bogdanovic, Arjun Sohal, Liyu Tao, Kourosh Darvish, Alán Aspuru-Guzik, Florian Shkurti, Animesh Garg · arXiv:2606.11408 · v1 2026-06-09
点破了固定窗口的隐含假设："During chunk execution, the policy operates open-loop, which is particularly problematic for fine-grained manipulation tasks that require frequent replanning. In practice, the execution horizon is typically chosen through empirical tuning and is highly task-dependent." 方法为在线 RL 训练一个轻量执行窗口预测分支，**保持预训练 chunk 策略完全冻结**，因而兼容黑箱策略。定性分析显示 DEHP 在精细阶段预测更短窗口、自由空间运动时更长。
**接口：** 「窗口靠经验调参且高度任务相关」正是本地 h=8 vs h=50 争议的文献表述。DEHP 的冻结基座设计意味着**可以在不重训任何 checkpoint 的前提下评估**。

**Spatial Attention: Adapting Execution Horizons for Diffusion Policies via Observation Sensitivity**
Che-Sang Park, Junsu Ha, Jianlong Fu, Frank C. Park · arXiv:2607.04739 · v1 2026-07-06
定义 Spatial Attention 为动作对数似然对观测梯度的期望平方范数，度量策略动作分布对观测变化的敏感度。理论结论：在固定 chunk 采样预算下，最小化扰动引起的累积似然下降的执行窗口**随 Spatial Attention 增大而减小**。据此预测未来 Spatial Attention 值并动态分配窗口，在保持平均窗口不变的前提下显著优于固定窗口基线。
**接口：** 提供了一个可离线计算的量，用于判断 patch_policy 在给定任务上「应该」用多长窗口——而不是继续在 8 与 50 之间二选一。

---

## 5. 轴 C —— 条件融合与捷径学习：本地数据与文献主流结论相左

这条轴上文献意见分歧最大，而**本地证据恰好落在少数派一侧**，值得单独说明。

**Do You Need Proprioceptive States in Visuomotor Policies?（State-free Policy）**
Juntu Zhao, Wenbo Lu, Di Zhang, Yufeng Liu, Yushen Liang, Tianluo Zhang, Yifeng Cao, Junyuan Xie, Yingdong Hu, Shengjie Wang, Junliang Guo, Dequan Wang, Yang Gao · arXiv:2509.18644 · v1 2025-09-23
结论明确：同时使用视觉与本体感受是常规做法，但"this common practice makes the policy overly reliant on the proprioceptive state input, which causes overfitting to the training trajectories and results in poor spatial generalization"。主张**直接移除 state 输入**，只用视觉观测预测动作。
**接口 —— 与本地证据的冲突：** 本地 `patch-policy-state-token-is-read-not-drowned` 与 `patch-policy-has-no-proprioception` 测得的是**相反的病**：state token 不是被过度依赖，而是被读取后不被使用（`zero_state` 只移动输出 2.2%，`state_history_reversed` 0.02%）。因此 State-free Policy 的处方对本项目**不适用**——patch_policy 事实上已经近似是 state-free 的了，移除 state 不会带来该论文所述的泛化收益。这一点应在任何引用该文的场合写明，否则容易误导。

**Causal Confusion in Imitation Learning**
Pim de Haan, Dinesh Jayaraman, Sergey Levine · arXiv:1905.11979 · v1 2019-05-28
奠基工作。"access to more information can yield worse performance" 的 causal misidentification 现象，及通过定向干预（环境交互或专家查询）确定正确因果模型的解法。

**Fighting Copycat Agents in Behavioral Cloning from Observation Histories**
Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, Yang Gao · arXiv:2010.14876 · v1 2020-10-28
copycat 问题的原始定义：在专家动作时间强相关的部分可观测设定下，"the imitator learns to cheat by predicting the expert's previous action, rather than the next action"。对抗式学习去除关于前一动作这一 nuisance correlate 的多余信息。
**接口：** 本地在示教数据上测得 `action_t ≈ state_t`（0.0140 rad）——这正是 copycat 的结构条件。但本地的表现形式是**恒等映射未被学出**（梯度信号太弱穿不过 `MLP → 加性进 memory → cross-attention → 去噪器`），而非被学成了抄袭。这是同一结构条件下的另一个失败分支，文献未见对称描述，**属于本项目可贡献的观察**。

**Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation**
Youguang Xing, Xu Luo, Junlin Xie, Lianli Gao, Hengtao Shen, Jingkuan Song · arXiv:2508.06426 · v1 2025-08-08
把捷径学习（依赖任务无关特征）定位为泛化受限的关键障碍，归因于两点：单个子数据集内部多样性不足，以及子数据集之间分布差异大导致的数据集碎片化。在无法获取新数据时，精选的数据增广策略可有效降低已有离线数据上的捷径学习（在 π₀ 上于仿真与真机验证）。
**接口：** 直接对应本地 `act-policy-leans-on-background-pixels`（遮背景比遮物体更伤）。「单个子数据集内多样性不足」正是 session block 成为捷径的机制。

**GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization**
Xiaosong Jia, Bowen Yang, Zuhao Ge, Xian Nie, Yuchen Zhou, Cunxin Fan, Yufeng Li, Yilin Chai, Chao Jing, Zijian Liang, Qingwen Bu, Haidong Cao, Chao Wu, Qifeng Li, Zhenjie Yang, Chenhe Zhang, Hongyang Li, Zuxuan Wu, Junchi Yan, Yu-Gang Jiang · arXiv:2605.12369 · v1 2026-05-12
"without explicit guidance, these models often overfit to spurious correlations, such as visual shortcuts or environmental noise"。方案是为物体 grounding、空间几何、时序技能逻辑设置专门化的注意力头。
**接口：** 与本地 memory 结构的接口很直接——本地已能测出 cross-attention 在 state 槽位上的质量分布（1.12%），同样的测量可以推广到「注意力落在背景 patch 上的比例」，把 `act-policy-leans-on-background-pixels` 从遮挡实验升级为注意力实验。

**Task-Relevant and Irrelevant Region-Aware Augmentation (DRAIL)**
Shun Hattori, Hikaru Sasaki, Takumi Hachimine, Yusuke Mizutani, Takamitsu Matsubara · arXiv:2603.04845 · v1 2026-03-05
显式把视觉观测切分为任务相关/无关区域：相关区按领域知识增广以保留关键视觉特性，无关区**激进随机化以抑制虚假背景相关**。在基于 diffusion policy 的视觉运动控制器上，于农业操作真机任务验证，注意力分析显示策略更依赖任务本质特征。
**接口：** 对 `act-policy-leans-on-background-pixels` 的直接处方，且是**唯一一篇把「背景激进随机化」与「前景保守增广」分开处理**的。

**DINOv3-Diffusion Policy: Self-Supervised Large Visual Model for Visuomotor Diffusion Policy Learning**
ThankGod Egbe, Peng Wang, Zhihao Guo, Zidong Chen · arXiv:2509.17684 · v1 2025-09-22
在三种模式（从头训练、冻结、微调）下评估自监督视觉骨干："(i) finetuned DINOv3 matches or exceeds ResNet-18 on several tasks, (ii) frozen DINOv3 remains competitive"；相比 ResNet18 在 Can 上最高 **+10%** 绝对成功率，Lift/PushT/Square 持平。
**接口：** 对 patch_policy 的冻结 DINOv2 设定是温和的支持证据（冻结仍有竞争力），但**注意其增益上界只有 10% 且任务相关**——不足以解释本地「输给 null」的量级。视觉骨干不是本项目精度问题的主因。本地已有的 `patch_policy-head-comparison` 与 `StereoPolicy`（arXiv:2605.09989，已在 `paper_note/`）覆盖同一议题。

---

## 6. 轴 E —— 离线 MAE 与真机成功率的脱节

用户明确要求覆盖「两个口径之间的差距」。这是本地证据最反直觉的部分：**同一批权重，在 horizon 50 的离线 MAE 上排序合理，在 horizon 8 的部署窗口上却全部输给「什么都不做」。**

**Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies**
Haoxu Huang, Tongsam Zheng, Yifan Chen, Jiacheng You, Yang Gao · arXiv:2606.29898 · v1 2026-06-29
问题陈述："A straightforward proxy for performance is validation loss on expert demonstrations, but this proxy is often poorly correlated with real-world performance." 方法 CI-MSE 把误差计算**限制在任务关键片段**，并配以更贴近 rollout 行为的动作对齐流程。跨仿真与真机：CI-MSE 与 rollout 表现的 **Spearman 秩相关 −0.87**，而原始 MSE 仅 **−0.61**（理想值 −1）。
**接口：** 这是本综述给本项目的**最可直接落地的一条**。本地 `run_eval_h8.sh` 已经做了「动作对齐贴近部署」这一半（deployment-faithful harness），但尚未做「限制在任务关键片段」那一半。补上后，`hold_state` null 之所以在全轨迹平均 MAE 上占优（大量自由空间、低位移片段被平均进来）这一伪像有望消除——**null 基线在关键片段上必然崩溃，而在全轨迹平均上却占便宜**。建议把 CI-MSE 作为 patch_policy 与 ACT 比较的主口径。

**Much Ado About Noising: Dispelling the Myths of Generative Robotic Control**
Chaoyi Pan, Giri Anantharaman, Nai-Chieh Huang, Claire Jin, Daniel Pfrommer, Chenyang Yuan, Frank Permenter, Guannan Qu, Nicholas Boffi, Guanya Shi, Max Simchowitz · arXiv:2512.01809 · v1 2025-12-01
系统评测生成式控制策略（GCP），结论："GCPs do not owe their success to their ability to capture multi-modality or to express more complex observation-to-action mappings. Instead, we find that their advantage stems from iterative computation, as long as intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity."。验证方式是提出 minimum iterative policy（MIP，两步回归式策略）基本追平 flow GCP，常优于蒸馏 shortcut 模型。
**接口：** 对本地 `act-dit-ignores-cameras`（「扩散目标吃掉了增益」）是一条重要的旁证——**扩散头的价值可能不在于它建模多模态的能力**。若 patch_policy 的扩散头收益主要来自迭代计算，则 MIP 式两步回归是一个成本低得多的对照组，值得作为消融基线。

**Train Offline, Test Online: A Real Robot Learning Benchmark (TOTO)**
Gaoyue Zhou, Victoria Dean, Mohan Kumar Srirama, Aravind Rajeswaran, Jyothish Pari, Kyle Hatch, Aryan Jain, Tianhe Yu, Pieter Abbeel, Lerrel Pinto, Chelsea Finn, Abhinav Gupta · arXiv:2306.00942 · v1 2023-06-01
共享硬件 + 开源离线数据集的 benchmark，用于「离线训练、在线测试」的直接比较。摘要未给出离线/在线相关性的具体数字（*该项 NOT FOUND*）。
**接口：** 作为「离线-在线差距需要专门 benchmark」的背景引用，不作为定量依据。

---

## 7. 轴 F —— 数据质量与纠错数据

**Learning from the Best: Smoothness-Driven Metrics for Data Quality in Imitation Learning (RINSE)**
Soham Kulkarni, Raayan Dhar, Yuchen Cui · arXiv:2604.23000 · v1 2026-04-24
基于轨迹平滑度打分（Spectral Arc Length 频域规整性 + Trajectory-Envelope Distance 接触感知几何偏差），策略架构无关。理论侧："smoothness filtering can reduce the conditional action variance of the retained data distribution, with downstream effects that can be amplified by action chunking and compounding error"。数值：RoboMimic 上 SAL 过滤用 **1/6 的数据**取得 **+16%** 成功率；真机 TED 过滤用**一半数据** **+20%**；作为 STRAP 检索阶段过滤器在 LIBERO-10 上 **+5.6%**；作为 Re-Mix 软权重时域分配与学到的分配 Spearman ρ ≥ 0.89。
**接口：** 「平滑度过滤的下游效应会被 action chunking 与复合误差放大」这一条把轴 F 直接连回轴 A 与 D。

**Auditing Demonstration Curation Metrics: Action-Only Scorers Fail on the Structural Defects That Degrade Imitation Policies**
Aarav Bedi 等 *（作者列表未完整解析，见 §8）* · arXiv:2606.05588 · v1 2026-06-04
受控测试台，注入已知类型缺陷，审计七种 curation 指标。两类缺陷区别对待：细微扰动（相关动作噪声、抖动、截断）可被多元离群打分检出，移除后可完全恢复下游差距；**结构性错误（演示在关键时刻执行了错误动作）对所有仅看动作的指标不可见，其中两个指标甚至是反向的**——把有缺陷的演示打成更高质量，用于筛选后策略表现等于或低于未筛选基线。只有检查状态轨迹的指标能检出结构性错误，且最好的也只恢复约三分之一的下游差距。结论："High detection accuracy does not guarantee downstream improvement."
**接口：** 对本地 `failure-data-naive-mixing-is-neutral`（「收益需要 conditioning/gating/weighting，而非比例调整」）是强支持，并给出了更强的警告：**基于动作的数据打分可能是负收益的**。在本项目引入任何自动数据筛选前应先读这篇。

**Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections (CR-DAgger)**
Xiaomeng Xu, Yifan Hou, Chendong Xin, Zeyi Liu, Shuran Song · arXiv:2506.16685 · v1 2025-06-20
两个组件：Compliant Intervention Interface（利用柔顺控制，让人在**不中断策略执行**的情况下给出温和精确的 delta 动作修正）与 Compliant Residual Policy（结合力反馈与力控从修正中学习）。用极少修正数据在四个接触密集任务（book flipping、belt assembly、cable routing、gear insertion）上把基座策略成功率**提升 64%**，优于从头重训与微调。
**接口：** 已出现在本地 `dagger-strategy-and-data-allocation-2026-08.md` 的候选列表中。注意其修正是 **delta 动作**——与轴 B 的相对目标同构。

**Residual Off-Policy RL for Finetuning Behavior Cloning Policies**
Lars Ankile, Zhenyu Jiang, Rocky Duan, Guanya Shi, Pieter Abbeel, Anusha Nagabandi · arXiv:2509.19301 · v1 2025-09-23
把 BC 策略当作黑箱基座，用样本高效的 off-policy RL 学习**轻量逐步残差修正**，只需稀疏二值奖励。报告为首次在带灵巧手的人形机器人上成功进行真机 RL 训练。*（摘要未给出具体成功率数字与交互时长，检索摘要中出现的 "~14–23% → ~64%、15–76 分钟" 未能在 abs 页核实，故不采信。）*
**接口：** 与 DEHP 共享「冻结基座 + 轻量外挂」的设计模式，是不动 checkpoint 的精度提升路线。

**From Reach to Insert: Tactile-Augmented Precision Assembly under Sub-Millimeter Tolerances**
Xinpan Meng, Siyao Huang, JingPu Yang, Muyuan Ma, Zhenghua Ma, Lijun Han, Gao Yuan, Houcheng Li, Long Cheng · arXiv:2605.04649 · v1 2026-05-06
两阶段：IL 学带位置泛化的 reaching 策略，RL 执行插入并支持接触中的失败恢复。tactile group sampling 提高关键接触片段覆盖，tactile critic 更准确评估策略价值。五种孔形 × 三种间隙；最难的 **0.05 mm 间隙下 67% 成功率**，最大交互力降 **60%**、力矩降 **44%**。
**接口：** 给出「视觉模仿的精度天花板在哪、超过之后需要什么模态」的参考点。若本项目任务需要亚毫米精度，纯视觉 + 关节位置的路线存在结构性上界。

---

## 8. 核实状态与例外

| 条目 | 状态 |
|---|---|
| 全部 30 篇 | 已通过 `arxiv.org/abs/<id>` 抓取核对标题 / 作者 / v1 日期 / 摘要原文 |
| arXiv:2506.23944 *Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning*（Fuhang Kuang, Jiacheng You, Yingdong Hu, Tong Zhang, Chuan Wen, Yang Gao, v1 2025-06-30） | **已被作者撤稿。** 其「proprioception shift」框架与本地 `gripper-state-scale-mismatch` 高度相关，但**不应作为引用依据**，本报告仅在此处记录其存在 |
| arXiv:2606.05588 | 作者列表在 abs 页仅解析出 Aarav Bedi（UC Berkeley）+ 1 位未显示姓名的合著者。正式引用前需补全 |
| arXiv:2509.19301 | 检索摘要中的成功率与交互时长数字未能在 abs 页核实，已在正文标注不采信 |
| arXiv:2306.00942 | 摘要中无离线-在线相关性数字，仅作背景引用 |
| arXiv:2607.06403 §6.1 数字 | 通过 `arxiv.org/html/2607.06403v1` 抓取正文核实，含节号与 Figure 11 |
| arXiv:2303.04137 引文 | 通过 `arxiv.org/html/2303.04137v5` 抓取正文核实 |
| arXiv:2605.09989 StereoPolicy | 已存在于 `paper_note/stereopolicy-2605.09989.md`，本报告未重复核实 |

---

## 9. 与本地证据的合流：文献未覆盖的部分

三处本项目已测量、而检索范围内**未见对应文献**的现象，可作为论文贡献点：

1. **无界后处理替换整条 chunk。** 文献处理的是边界不连续（RTC 系）与有界扰动内的平滑（LiPo），没有工作报告过「阈值设错导致后处理对 100% 的 chunk 触发、并把指令位移削减到 65%」这一失效模式。本地已有回放复现证据（峰/边比 1.89 vs 实测 1.99）。

2. **恒等映射未被学出，而非被抄袭。** copycat 文献（arXiv:2010.14876）描述的是 `action_t ≈ state_t` 条件下模型**学会了抄袭**；本地测得的是同一条件下模型**没能学出这条恒等映射**（梯度需穿过 MLP → 加性进 memory → cross-attention → 去噪器）。这是 copycat 的对偶失败分支。

3. **state token 被读取但不被使用。** State-free Policy（arXiv:2509.18644）主张移除 state 因为策略**过度依赖**它；本地测得 cross-attention 给 state 槽位 8.65× 均分份额（被充分读取）、范数 0.76（尺度正常），但 `zero_state` 只移动输出 2.2%。「注意力充分、影响为零」这一组合在检索范围内未见报告。

---

## 10. 建议的优先级（依据文献与本地证据的交集）

按「证据强度 × 改动成本」排序，仅列出有文献依据且本地可验证的项：

| 优先级 | 动作 | 依据 |
|---|---|---|
| 1 | 修 `threshold_rad` / `K`，并记录 post-filter chunk；在此之前**所有真机精度归因都不成立** | 本地 `record-chunk-is-filtered` + arXiv:2506.05165（有界扰动约束） |
| 2 | 审计所有归一化 / 反归一化元数据键，尤其夹爪通道 | arXiv:2606.03724（同族键替换 → 28/28 掉到 2/28）+ 本地 `gripper-state-scale-mismatch` |
| 3 | 把评测主口径从全轨迹 MAE 换成 CI-MSE（关键片段 + rollout 对齐） | arXiv:2606.29898（ρ −0.87 vs −0.61） |
| 4 | 按任务族分别验证 EEF vs 相对关节，不做全线切换 | arXiv:2607.06403 §6.1（平均持平但任务级 81.7 vs 41.7 / 24.0 vs 58.7） |
| 5 | 用免重训方法评估执行窗口，而非在 8 与 50 间二选一 | arXiv:2606.11408（冻结基座）、arXiv:2607.04739（可离线计算的敏感度）、arXiv:2602.21445 |
| 6 | 背景激进随机化 + 前景保守增广，验证 `act-policy-leans-on-background-pixels` | arXiv:2603.04845 (DRAIL)、arXiv:2508.06426 |
| 7 | 引入任何自动数据筛选前，先复核仅看动作的指标可能为负收益 | arXiv:2606.05588、arXiv:2604.23000 |
