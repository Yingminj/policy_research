# SFT 之后怎么办：四篇 RL 后训练论文精读 + 强化学习主线梳理

- **日期**：2026-08-21
- **对象**：用户指定的四篇文章及其代码仓库
  - `arXiv:2604.13016v2` — Rethinking On-Policy Distillation of LLMs
  - `arXiv:2602.12628v4` — Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models (RL-Co)
  - `arXiv:2506.15799v2` — Steering Your Diffusion Policy with Latent Space RL (DSRL)
  - `arXiv:2509.25756v3` — SAC Flow
- **配套阅读**：[`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md)、
  [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md)、
  [`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md)、
  [`team-division-strategy-2026-08.md`](./team-division-strategy-2026-08.md)

> **证据分级**（沿用本目录既有约定）：
> - **[✓]** = 本次直接抓取 arXiv abs / HTML 全文 或 GitHub API / raw 文件核实；
> - **[✓c]** = 代码仓库侧实抓（README / 文件树 / 配置文件原文），**未在本地安装或跑通**；
> - **[~]** = 仅来自检索结果元数据或摘要片段，未逐字核实，引用其数字前请自行复核。
>
> **一条必须先说的话**：本文四篇里有 **两篇（OPD、SAC Flow）的主结果只有图、没有表**。
> 我在写作过程中先后两次让抽取模型"转录表格"，两次都被明确告知"本文没有数值表，结果只在 Figure 里"。
> 因此本文**不给这两篇编造精确数字**——凡涉及它们的量化结论，只引用作者在摘要/正文里用文字写死的数（如 97–99%、+0.37→+0.02），
> 其余一律标为「曲线趋势，未逐点读数」。这本身也是这两篇的一条实实在在的缺点，见 §2.1.7 与 §2.4.7。

---

## 0. 一页结论

**这四篇是同一个问题的四种答案，问题是：「模仿学习/SFT 把能学的都学完了，然后呢？」**

| | 论文 | 学什么信号 | 动不动基座权重 | 落到你身上的成熟度 |
|---|---|---|---|---|
| A | **DSRL** (2506.15799) | 环境奖励 | **不动**（只改初始噪声） | 最高：黑盒、真机 40–150 episode 见效 |
| B | **SAC Flow** (2509.25756) | 环境奖励 | 动（改 velocity 网络的参数化） | 中：仿真扎实，**作者自陈未上真机** |
| C | **RL-Co** (2602.12628) | 仿真环境奖励 + 真机演示锚定 | 动（PPO 全量微调） | 中高：代码在 RLinf 里、配置全公开，但要建数字孪生 |
| D | **OPD** (2604.13016) | **教师模型的逐 token 分布** | 动（学生全量训练） | 低（LLM 域），但**方法论可迁移**，见 §5.4 |

**三句话总结这四篇的共同发现**：

1. **BC/SFT 的天花板不是数据量问题，是「没有闭环信号」问题。** 四篇的实验都出现同一个形状：
   基座策略卡在某个中等成功率（DSRL 真机 pick-place 2/10；π₀ 开烤箱 5/20；RL-Co 的 π₀.₅ 推方块 0/20），
   加多少演示都不动，一旦接上闭环奖励就跳到 90%+。**你现在 60–70% 的 rollout 成功率正好在这个形状的腰部。**

2. **这一代 RL 的主要矛盾是「策略类从高斯变成了生成模型」。** 高斯策略有解析的 `log π(a|s)`，
   diffusion / flow 策略没有，或者说要付出多步反传的代价。整个 2023–2026 的工作可以按「怎么绕开这件事」分成三派：
   **绕过**（DSRL：在噪声空间做 RL，基座当黑盒）、**修架构**（SAC Flow：把 flow rollout 认成残差 RNN 然后按 GRU/Transformer 重参数化）、
   **换目标**（FQL / ReinFlow / RL-Co：用代理目标或注入噪声把 flow 变回可算 log-prob 的东西）。见 §3.5。

3. **四篇里对你最有价值的不是任何一篇的方法，而是 OPD 的诊断范式。** 你在
   [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md) 里的核心痛点是「训练 loss 不预测真机成功率」。
   OPD 的做法是：**不看终点指标，看训练动态中「策略在自己访问过的状态上」与参考分布的重合度**（overlap ratio / entropy gap）。
   这个思路直接可以搬成你缺的那个闭环离线指标。见 §5.4。

**并且必须泼一盆冷水**：这四篇**全部**要求闭环——要奖励、要 rollout、要 reset。
而你的现状（`failure-data-in-imitation-2026-08.md` 开头的前置声明）是 **三个 job 的 `env_eval_freq` 都是 0，磁盘上没有任何 rollout 成功率记录**。
**在补上自动化闭环评测之前，这四篇一篇都用不了。** 这是本文最实用的一句话，排在所有技术方案之前。见 §5.0。

---

## 1. 为什么这四篇应该放在一起读

它们表面上分属三个圈子（LLM 后训练、VLA 机器人、离线/在线 RL 算法），但坐标是同一套。用两个轴就能摆开：

```
                     监督信号来自「另一个模型的分布」
                                  ▲
                                  │
                          [D] OPD │  （教师 token 分布 = 稠密监督）
                                  │
   不动基座权重 ◄───────────────────┼───────────────────► 动基座权重
                                  │
              [A] DSRL            │   [B] SAC Flow   [C] RL-Co
        （只改 denoiser 入口噪声）  │  （改 velocity  （PPO + 真机 SFT 锚）
                                  │    网络参数化）
                                  ▼
                     监督信号来自「环境奖励」（稀疏/闭环）
```

**左下（DSRL）与右下（SAC Flow / RL-Co）的分歧**，是这一代最值得注意的一条路线之争：

- **右下的赌注**：策略本身要变好，就必须让梯度流进策略。代价是 diffusion/flow 的多步采样让反传变成深层递归（SAC Flow 的核心诊断），
  以及 VLA 这种 3B–7B 规模模型的全量 RL 训练开销（RL-Co 用 FSDP + 64 并行环境）。
- **左下的赌注**：基座策略已经把「所有合理动作」都装在它的输出分布里了，缺的只是**在每个状态挑对哪一个**。
  所以不用动权重，只要控制那个决定"挑哪个"的隐变量（初始噪声 `w`）。代价是：**能挑的范围被基座锁死**（DSRL 自陈的第一条局限）。

**上半（OPD）的位置**：它换掉的是奖励本身。RLVR 的奖励是一条轨迹一个标量，OPD 的"奖励"是教师在每个 token 上的完整分布——
稠密到不能再稠密。这篇的价值在于它**证明了这顿免费午餐是有代价的**：教师的优势随轨迹深度单调衰减
（1K 前缀 +0.37 → 16K 前缀 +0.02 [✓]），所以稠密监督在长程上会自己失效。
**这条结论对机器人是直接适用的**——动作分块（action chunking）就是短程稠密监督，长程任务上同样会遇到"教师在后半段没优势"的问题。

---

## 2. 逐篇精读

### 2.1 Rethinking On-Policy Distillation of LLMs（arXiv:2604.13016v2）

#### 2.1.1 Metadata

| 项 | 内容 |
|---|---|
| 标题 | Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe [✓] |
| 作者 | Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, Ning Ding [✓] |
| 单位 | 清华 THUNLP 体系（通讯与仓库均在 `thunlp`） |
| 版本 | v1 2026-04-14，**v2 2026-04-15**（当前）[✓] |
| 篇幅 | **30 页 / 23 张图**（arXiv comment 原文）[✓] |
| 发表 | **ICML 2026 FoGen Workshop**（2026-05-27 仓库 News）[✓c] |
| 代码 | https://github.com/thunlp/OPD — 944 ★ / 69 fork，创建 2026-04-12，最后 push 2026-08-20，**无 LICENSE 文件** [✓c] |

#### 2.1.2 一句话总结

on-policy distillation（学生自己采样、用教师的逐 token 分布做稠密监督）什么时候有效、什么时候完全无效，
过去只有工程经验没有机理。本文给出**两个成立条件**（思维模式兼容 + 教师确有学生没见过的新能力）、
**一个 token 级机理**（成功的 OPD 表现为高概率重合 token 上的渐进对齐，这一小撮 token 集中了 97–99% 概率质量）、
**两条救活失败 OPD 的配方**（off-policy 冷启动 + 教师对齐的 prompt 选择），
以及**一条泼冷水的结论**：稠密奖励的优势随轨迹变长单调衰减，OPD 未必能扩展到长程。

#### 2.1.3 方法与实验设置

- **学生 / 教师** [✓]：
  - DeepSeek 系：R1-Distill-1.5B（学生）← R1-Distill-7B / JustRL-1.5B（教师）
  - Qwen 系：Qwen3-1.7B 系列（学生）← Qwen3-4B 系列（教师）
  - 跨族：DeepSeek-R1-Distill-1.5B ← Skywork-OR1-Math-7B
- **训练数据**：DAPO-Math-17K、OpenThoughts3-1.2M（math 子集）、DeepMath-57K 子集 [✓]
- **评测**：AIME 2024 / AIME 2025 / AMC 2023；每题采 16 条解（temperature 0.7, top-p 0.95），最大长度 31,744 token [✓]

#### 2.1.4 关键发现（按证据强度排序）

**(1) 反向蒸馏实验——最漂亮的一条。** [✓]
把 JustRL-1.5B（已经过 RL 的强模型）当学生，去蒸馏它自己 RL 之前的 checkpoint R1-Distill-1.5B，
学生**几乎精确地退回到 RL 之前的水平**。更狠的是：换成分数明显更高的 R1-Distill-**7B** 当教师，
学生**退回到同一个水平**。
→ 结论：从学生的视角看，同族的 1.5B 与 7B 教师**在分布上不可区分**（distributional indistinguishability）。
**OPD 学的是「思维模式」，不是「分数」。** 这解释了为什么"换更大的教师"经常没用。

**(2) 两个成立条件。** [✓]
- **思维模式兼容性**：GRPO 训过的 Qwen3-4B-Base-GRPO 当教师，效果**优于**基准分更高的 Qwen3-4B (Non-thinking)，
  原因是初始 overlap ratio 更高。**教师的 benchmark 分数不是选教师的正确指标，与学生的分布重合度才是。**
- **必须有新能力**：同一条 pipeline 训出来的教师带来的提升"有限"，经过后训练的教师则"显著更强"。

**(3) token 级机理。** [✓]
成功的 OPD 有三个同时出现的信号：overlap ratio 稳步上升（约 **72% → 91%**）、entropy gap 收窄、
overlap-token advantage 趋近 0。失败的 run 则"从一开始 overlap 就停滞、entropy 持续错配"。
高概率重合 token 集中了 **97–99%** 的概率质量（摘要原文写死的数）[✓]。
消融进一步表明：**只在 overlap 区域上算梯度，就能拿回标准 OPD 的几乎全部收益**；
只在 non-overlap 区域上算则"显著更弱"。

**(4) 两条救活配方。** [✓]
- **off-policy 冷启动**：先用 200K 条教师 rollout 做 SFT 再进 OPD，起点 overlap ratio 高得多、轨迹平滑稳定，优势贯穿全程。
- **教师对齐的 prompt 选择**：把 prompt 模板换成教师后训练时用的模板 → 全 benchmark 涨点；
  进一步用教师后训练数据集里的 prompt → overlap 增长更快，但**代价是学生熵显著降低**（作者自己标了这个代价）。

**(5) 稠密监督的代价（本文最有转载价值的一条）。** [✓]
- 教师的准确率优势随前缀长度**单调衰减**：**1K 前缀 +0.37 → 16K 前缀 +0.02**。
- 高熵**先出现在回答末尾，再逐步向前传播**。
- 长程推理出现"后期崩塌，overlap ratio 骤降"。
- 响应长度甜区在 **3K–7K token**，到 **10K–15K** 就平台或下降。
→ 作者的结论问句：OPD 能否扩展到长 CoT / 多轮 agent 场景，**存疑**。

#### 2.1.5 代码仓库审计（`thunlp/OPD`）[✓c]

实抓 GitHub tree API：**2145 个文件**，顶层分布如下。

| 目录 | 文件数 | 是什么 |
|---|---:|---|
| `verl/` | 1124 | **vendored 整份 verl v0.7.0**（不是 patch，是整棵树拷进来） |
| `LlamaFactory/` | 526 | **vendored 整份 LLaMA-Factory v0.9.5**（用于 SFT / 冷启动） |
| `datasets/` | 51 | DAPO-Math-17K、OpenThoughts3-opd、DeepMath-deduped 的 **parquet 直接入库**，外加 AIME24/25、AMC23、MATH-500、GPQA、MMLU-Pro、MedQA 等 13 套评测集 |
| `scripts/infer` | 14 | 推理脚本 |
| 顶层 | — | `grpo.sh`、`on_policy_distillation.sh` 两个入口 |

**优点**：
- **数据随仓库一起给了**（parquet 直接躺在 `datasets/`），这在 LLM 后训练类仓库里少见，省掉最容易对不齐的一环。
- **两个入口脚本**（`on_policy_distillation.sh` / `grpo.sh`）把 OPD 与 RL 基线并列，消融可直接跑。
- **真实的上游采纳**：作者的 top-k OPD overlap 诊断指标已经**合入 verl 主线**
  （[verl PR #6469](https://github.com/verl-project/verl/pull/6469)，新增 `distillation/overlap_ratio` 与
  `distillation/overlap_token_advantage` 两个 metric）[✓c]；结论被 **MiniCPM5-1B** 采用 [✓c]。
  **这是比 944 颗星更硬的证据**——指标进了别人的框架主线，说明它是可复用的诊断工具而不只是论文里的图。

**缺点**：
- **无 LICENSE 文件**（GitHub API `license: null`）[✓c]。vendored 了 Apache-2.0 的 verl 和 LLaMA-Factory 却不声明自身许可，商用引入前需要问作者。
- **vendored 而非 fork + patch**：1650 个文件是别人的代码，**没法一眼 diff 出「他们到底改了什么」**。
  要复现 OPD 的具体实现，得自己跟上游 verl v0.7.0 做 tree diff。这是本仓库最大的工程摩擦。
- 双 conda 环境（`verl` py3.12 / `sft` py3.11），SFT 与 OPD 不在一个环境里，跑全流程要来回切。

#### 2.1.6 亮点

1. **反向蒸馏是一个设计得极好的判决性实验。** 它把"教师更强所以学生更强"这条默认假设一刀切断，
   而且用的是同一模型 RL 前后两个 checkpoint，控制变量干净。
2. **overlap ratio 是可以立刻用起来的在线诊断量。** 不需要等 benchmark 跑完——训练前几百步看 overlap 涨不涨，
   就能预判这次 OPD 会不会失败。这是本文最有工程价值的产出（也是唯一进了 verl 主线的产出）。
3. **敢写负面结论。** §2.1.4(5) 那组数（+0.37 → +0.02）是在自我拆台，很多论文不会放。

#### 2.1.7 局限与批评

- **主结果只有图、没有表。** 30 页 23 张图，全篇没有一张可转录的数值结果表 [✓，两次独立抽取均确认]。
  后果很实际：**别人无法在不重跑的情况下把自己的数与本文对比**，也无法核验图里的曲线。
  对一篇以"phenomenology"（现象学）自我定位的论文，这是方法论上的自相矛盾——现象学要的恰恰是可复核的观测记录。
- **规模只到 1.5B ← 7B**。所有结论都在小模型上取得，"同族大小教师分布不可区分"这条在 7B ← 70B 上是否还成立，没验证。
- **只在数学推理域**（AIME/AMC/MATH）。overlap ratio 这个量在代码、agent、开放域对话上是否同样有判别力，未测。
- **"思维模式兼容"缺可操作定义**。文中用 overlap ratio 作为它的代理，但这是循环的：
  overlap 高 → 说明模式兼容 → 所以 overlap 会高。**缺一个独立于结果的先验判据**，
  也就是说你在开训之前仍然没法判断该选哪个教师（除非先跑一小段看 overlap，这是文中实际推荐的做法）。

---

### 2.2 Beyond Imitation: RL-Based Sim-Real Co-Training for VLA（arXiv:2602.12628v4）

#### 2.2.1 Metadata

| 项 | 内容 |
|---|---|
| 标题 | Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models [✓] |
| 作者 | Liangzhi Shi, Shuaihang Chen, Feng Gao, Yinuo Chen, Kang Chen, Tonghe Zhang, Hongzhi Zang, Jiakai Zhou, Weinan Zhang, Chao Yu, Yu Wang [✓] |
| 版本 | v1 2026-02-13 → v2 02-16 → v3 03-06 → **v4 2026-06-04**（当前）[✓] |
| 分类 / 许可 | cs.RO / CC BY 4.0 [✓] |
| 项目页 | https://rl-co-training.github.io/ [✓]（页面内容极简，实际资源都在 RLinf 里） |
| 代码 | **在 RLinf 主仓库内**：`RLinf/RLinf` → `examples/embodiment/config/maniskill_ppo_co_training_openpi_pi05.yaml` + `docs/.../embodied/co_training.rst` [✓c] |
| 权重/数据 | HF：`RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT`（SFT checkpoint）、`RLinf/RLCo-Example-Mix-Data`（50 真 + 1499 仿）、`RLinf/RLCo-Example-Real-Data`（50 真）、`RLinf/RLCo-maniskill-assets` [✓c] |

> 注意：arXiv abs 页**没有**列出代码/项目页链接 [✓]，靠 abs 页找不到代码。
> 代码是我从 RLinf 仓库文件树里搜 `co_train` 搜出来的。这是本文的一个信息可达性问题（§2.2.7）。

#### 2.2.2 一句话总结

现有 sim-real co-training 把仿真当成"一堆静态演示"（SFT），浪费了仿真真正独有的东西——**可以无限次闭环交互**。
本文提出两阶段的 RL-Co：**Stage I** 用真+仿混合数据做 SFT 热启动，**Stage II** 在仿真里跑 RL，
同时**在真机数据上加一个辅助监督损失把策略锚住**，防止灾难性遗忘。
在四个真机桌面任务 × 两种 VLA 架构（OpenVLA、π₀.₅）上，比 SFT co-training 分别再涨 **+24 / +20 个点**。

#### 2.2.3 方法

**Stage I（SFT 热启动）**：`L_SFT = α·L_SFT(D_sim) + (1−α)·L_SFT(D_real)`，实现上按概率 α 从 `D_sim` 采、
按 1−α 从 `D_real` 采 [✓]。作者不给固定 α，敏感性分析显示**最优 α 强烈依赖任务**（§2.2.5）。

**Stage II（真机正则化的 RL）**：`L_total = L_RL + β·L_SFT(D_real)` [✓]。

`L_RL` 具体是什么，论文正文写得含糊（OpenVLA 引用了他人协议，π₀.₅ 说"用 ReinFlow，在 RLinf 上实现"）[✓]，
**但配置文件把它写死了**——以下全部来自 `maniskill_ppo_co_training_openpi_pi05.yaml` 原文 [✓c]：

```yaml
algorithm:
  adv_type: gae            # 就是 PPO + GAE，不是 GRPO
  loss_type: actor_critic
  reward_type: chunk_level # 奖励/logprob/熵全部在 action-chunk 粒度上算
  logprob_type: chunk_level
  entropy_type: chunk_level
  clip_ratio_high: 0.2 ; clip_ratio_low: 0.2 ; value_clip: 0.2
  entropy_bonus: 0.005 ; gamma: 0.99 ; gae_lambda: 0.95
  update_epoch: 2 ; kl_beta: 0.0
actor:
  sft_loss_weight: 0.2     # 这就是论文里的 β
  enable_sft_co_train: True
  micro_batch_size: 40 ; global_batch_size: 2560
  optim: {lr: 4e-6, value_lr: 2e-4, clip_grad: 1.0}
  model.openpi:
    noise_method: "flow_noise"       # ReinFlow 式：给 flow 注噪以获得可算的 log-prob
    noise_params: [0.16, 0.12, 200]
    joint_logprob: True
    action_horizon: 8
  model: {num_action_chunks: 4, num_steps: 4, add_value_head: True, value_after_vlm: True}
env.train: {total_num_envs: 64, max_episode_steps: 80}
```

**几个只有读配置才知道的关键事实**：
- **β = 0.2**（论文正文只说"在一个范围内测过"，配置里给的默认值是 0.2）；
- **RL 算法是 PPO + GAE + value head**，`kl_beta: 0.0`（不加 KL 惩罚，靠 SFT 项做锚）；
- **一切在 chunk 级别**：reward / logprob / entropy 都是 `chunk_level`，`action_horizon: 8`，`num_action_chunks: 4`；
- **flow 策略的 log-prob 靠注噪拿到**（`noise_method: flow_noise`），这正是 §1 里说的"换目标"派做法；
- **learning rate 4e-6**，极小——VLA 全量 RL 微调不敢大步走。

**仿真侧** [✓]：ManiSkill3 数字孪生，与真机在布局 / 相机视角 / 任务逻辑 / 语言 / 动作空间上对齐，
**但刻意不追求照片级真实**（物体材质、纹理、光照、背景都不精确匹配）。每任务用 **MimicGen 从真机种子轨迹生成 1000 条成功轨迹**。

**真机侧** [✓]：Franka Emika Panda（7-DoF + 平行夹爪），末端 delta-pose 控制（7 维），固定 RGB 相机。
每任务 **20–50 条人工遥操作成功演示**。

**四个任务** [✓]：Pick and Place / Push Cube via Instruction（三个彩色方块中按语言指定推哪个）/ Open Drawer / Close Drawer。

#### 2.2.4 主结果（Table 1，真机成功率 %）[✓，两次独立抽取一致]

| 模型 | 方法 | Pick&Place | Push Cube | Open Drawer | Close Drawer | Avg |
|---|---|---:|---:|---:|---:|---:|
| OpenVLA | Real-Only | 6.3±0.0 | 20.0±13.3 | 0.0±0.0 | 10.0±10.0 | **16.5±13.3** ⚠ |
| OpenVLA | SFT Co-Train | 23.4±4.7 | 51.7±5.0 | 0.0±0.0 | 85.0±5.0 | 40.0±3.7 |
| OpenVLA | **RL-Co** | **58.8±10.0** | **68.3±11.7** | **35.0±15.0** | **95.0±5.0** | **64.0±0.7** |
| π₀.₅ | Real-Only | 71.9±9.4 | 0.0±0.0 | 0.0±0.0 | 35.0±15.0 | 26.7±1.4 |
| π₀.₅ | SFT Co-Train | 68.8±9.4 | 10.0±3.3 | 10.0±0.0 | 95.0±5.0 | 45.9±4.4 |
| π₀.₅ | **RL-Co** | **81.3±9.4** | **18.4±1.7** | **65.0±5.0** | **100.0±0.0** | **66.2±4.0** |

> ⚠ **我核出来的一处论文自身错误**：OpenVLA Real-Only 行的 Avg 印的是 **16.5±13.3**，
> 但该行四个任务的均值是 `(6.3+20.0+0.0+10.0)/4 = 9.1`。其余五行的 Avg 我都算过，**全部对得上**
> （26.7 ✓ / 45.9 ✓ / 66.2 ✓ / 40.0 ✓ / 64.3≈64.0 ✓）。而且 16.5 那个 ±13.3 与同行 Push Cube 的 ±13.3 一模一样，
> 像是一次复制粘贴事故。**引用这一格前请自行复核原文**——它影响的是"RL-Co 相对 Real-Only 涨了多少"这个说法。

> **摘要里的 +24% / +20% 指的是相对 SFT co-training，不是相对 real-only**：
> 64.0 − 40.0 = +24.0，66.2 − 45.9 = +20.3。相对 Real-Only 则是 +47.5 / +39.5。摘要写的是
> "over real-only fine-tuning and SFT-based co-training, including +24%..."，把两个基线并列后给一个数，**容易被读成前者**。

**其它实验** [✓]：
- **视觉多样性对照**（Pick&Place, π₀.₅）：重度域随机化零样本 10.9±7.8 / Cosmos 视频增广 67.2±1.6 / **RL-Co 81.3±9.4**。
  → 说明 RL-Co 的收益**不只是视觉多样性**，闭环交互带来了动作层面的额外增益。这是本文最有说服力的一个对照。
- **真机数据效率**（Open Drawer, π₀.₅）：真机数据从 20 条扫到 200 条，RL-Co 在少数据下就超过用更多演示的 SFT 方法。
- **泛化**：未见物体 / 未见机器人初始状态两种分布偏移下，RL-Co 掉点最少；Real-Only "急剧退化"。

**消融** [✓]：
- 去掉 Stage I 的仿真 SFT → **跑了三百万交互步策略仍近乎不动**。作者据此说 Stage I "不只是数据混合的热身，而是必要步骤"。
  **这是全文最重要的一条消融**：RL 在 VLA 上不是"加上就有用"，没有正确的热启动它根本不起飞。
- 去掉 Stage II 的真机 SFT 项 → 成功率"大幅下降"；两个 stage 的真机监督都去掉 → "性能崩溃"。
- α（co-train 比例）**强任务依赖**：Pick&Place（Real-Only 本来就强）里 α 高反而有害；Open Drawer（难）里混合 α 有益。

**RLinf 文档给的可复现数字** [✓c]：加载 Stage I checkpoint 后仿真零样本约 **35%**，co-training RL **100 步**后约 **50%**。

#### 2.2.5 代码仓库审计（`RLinf/RLinf`）[✓c]

不是一个独立的论文仓库，而是**并入了一个活跃的通用 RL 框架**。2450 个文件，`rlinf/` 758、`docs/` 498、
`examples/` 409、`tests/` 185。

**优点**：
- **配置即论文**：论文正文含糊的部分（RL 算法、β、lr、chunk 粒度、注噪参数）在 YAML 里全部写死。上面 §2.2.3 那段是直接抄的。
- **交付链条完整**：ManiSkill assets（HF dataset）→ 混合数据集（50 真 + 1499 仿，LeRobot 格式）→
  SFT checkpoint（HF model）→ RL 配置 → 一行 `bash examples/embodiment/run_embodiment.sh maniskill_ppo_co_training_openpi_pi05`。
  **允许你跳过 Stage I 直接跑 Stage II**，这大幅降低复现门槛。
- **数据是 LeRobot 格式** [✓c]——对你的仓库（`robot_data_platform/lerobot`）是直接可读的，不需要写转换器。
- 有 Docker 镜像（`rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`）、有国内镜像加速说明、有 e2e 测试
  （`tests/e2e_tests/embodied/maniskill_ppo_co_training_openpi_pi05.yaml`）。
- 文档里甚至给了**训练健康度告警**：`train/loss_ratio = β|L_SFT| / |L_RL|`，超过 1e5 会警告并建议调小 `sft_loss_weight`。
  这种"框架替你盯着调参"的细节，是真在生产里跑过才会有的。

**缺点**：
- **arXiv abs 页不给代码链接** [✓]，项目页 `rl-co-training.github.io` 内容极简、也没有直指仓库路径 [✓]。
  论文与代码之间**缺一条可发现的路径**——这是可以零成本修好却没修的问题。
- **只放了一个任务的完整链条**（Pick-and-Place 数字孪生）。论文里的另外三个任务（Push Cube / Open / Close Drawer）
  的 assets 与数据**没有对应发布**，Table 1 无法整表复现。
- **OpenVLA 分支的 RL 配置没在这条 example 里**——公开的是 π₀.₅ 那半。
- 门槛仍然高：`sharding_strategy: no_shard` + `gradient_checkpointing: False`（配置注释写明 openpi 不支持梯度检查点）[✓c]，
  意味着 **π₀.₅ 全量 PPO 需要相当大的显存**，且没有省显存的退路。64 并行环境 + FSDP，这不是单卡 4090 能跑的配置。

#### 2.2.6 亮点

1. **"仿真的价值是交互，不是数据"这句话被实验证成了。** 视觉多样性对照（域随机化 10.9 / Cosmos 67.2 / RL-Co 81.3）
   是把"你以为你缺的是视觉多样性，其实你缺的是闭环"讲得最清楚的一组数。
2. **辅助真机 SFT 损失是个廉价且有效的锚。** β=0.2、一行配置开关，解决 sim-only RL 过拟合到仿真的老问题。
   相比 KL 惩罚（`kl_beta: 0.0`，他们不用），用真机数据当锚更直接——**约束的是"在真机分布上别跑偏"，而不是"别离初始权重太远"**。
3. **两个架构、两种动作表示都验了**（OpenVLA 的 next-token 离散动作 vs π₀.₅ 的 flow matching），结论一致。
   这让"RL-Co 是架构无关的配方"这个说法站得住。
4. **数字孪生刻意不做照片级真实**——这条负面设计选择很有价值：它说明对齐**任务逻辑与动作接口**比对齐**像素**重要。

#### 2.2.7 局限与批评

- **Table 1 有一格算术不自洽**（见 §2.2.4 的 ⚠）。v4 了还没修。
- **评测 trial 数没写**。论文只说"相同的独立采样初始状态"，**没说每个条件每个任务跑多少次** [✓，专门问过]。
  从 `6.3` 这种数反推大概是 1/16 或 32 分之几，但这是猜。**在成功率 ±10 的方差下，没有 trial 数就没法判断显著性**——
  这是真机论文里最常见也最要命的省略。
- **作者自陈**：只做桌面操作、单一本体，没有异构 sim-real 设置；成功率仍低于 100%。
- **Push Cube 上 π₀.₅ 只有 18.4%**，RL-Co 也没救回来（OpenVLA 反而 68.3%）。这个反例论文没深究——
  语言条件的细粒度选择任务，flow-matching VLA 可能有结构性劣势，值得单独查。
- **Stage I 必要性的消融是把双刃剑**：它证明了方法的完整性，同时也说明 **RL 部分的收益高度依赖一个训好的热启动**，
  而热启动本身需要 1000 条仿真轨迹 + 数字孪生。**总成本没有论文标题暗示的那么低。**

---

### 2.3 Steering Your Diffusion Policy with Latent Space RL — DSRL（arXiv:2506.15799v2）

#### 2.3.1 Metadata

| 项 | 内容 |
|---|---|
| 标题 | Steering Your Diffusion Policy with Latent Space Reinforcement Learning [✓] |
| 作者 | Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, Sergey Levine [✓]（UC Berkeley / UW / Amazon 体系） |
| 版本 | v1 2025-06-18，**v2 2025-06-25**（当前）[✓] |
| 发表 | **CoRL 2025**（两个官方仓库的 citation 与 README 均写明）[✓c] |
| 项目页 | https://diffusion-steering.github.io [✓c] |
| 代码 | 两个官方仓库：`ajwagen/dsrl`（Robomimic/Gym，221★/30 fork，**无 LICENSE**）与 `nakamotoo/dsrl_pi0`（π₀，289★/53 fork，**MIT**）[✓c] |

#### 2.3.2 一句话总结

要用 RL 改进一个预训练好的 diffusion / flow 策略，不必动它一个参数——
**只要改「喂进去的那个初始噪声 `w`」**。把噪声空间当成新的动作空间，原 MDP 就变成一个标准的连续控制 MDP，
任何 RL 算法都能直接套上去。真机上 **40 个 episode 把 pick-place 从 2/10 拉到 9/10**，
并且做出了**第一个 π₀ 的真机 RL 微调**。

#### 2.3.3 方法

**MDP 改写** [✓]：原 MDP `M = (S, A, P, p₀, r, γ)`，构造噪声 MDP `M^W = (S, W, P^W, p₀, r^W, γ)`：

```
W  := R^d                              # 噪声空间就是新动作空间
P^W(·|s,w) := P(·|s, π_dp^W(s,w))      # 先把 w 去噪成动作，再给环境
r^W(s,w)   := r(s, π_dp^W(s,w))
```

于是"在 `M^W` 上做策略优化"就是一个标准问题。**只需要前向调用去噪过程，不需要反传穿过去噪链**，
基座策略是完全的黑盒（只要能控制初始噪声、且 diffusion 用 DDIM 采样）[✓]。

**DSRL-NA（noise-aliased，作者推荐的变体）** [✓]，双 critic：
- `Q^A(s,a)`：在**原动作空间**上做标准 TD 学习（可用离线数据）；
- `Q^W(s,w)`：从 `Q^A` 蒸馏 —— `min E_{s, w~N(0,I)} [ (Q^W(s,w) − Q^A(s, π_dp^W(s,w)))² ]`；
- `π^W(s)`：`max E_s [ Q^W(s, π^W(s)) ]`。

**"噪声混叠"（noise aliasing）**是这里的关键洞察：存在大量 `w' ≠ w` 使 `π_dp^W(s,w) ≈ π_dp^W(s,w')`。
`Q^W` 因此能通过基座策略把没见过的 `w` 映到见过的 `a` 上去推断价值——**样本效率就是从这来的**。

**天然的离线保守性** [✓]：`Q^A` 只会在离线数据里的动作和基座策略产出的动作上被查询，
所以**自动避开了 OOD 动作的价值查询**，同时 `π^W` 还能不受约束地自由优化。
这一条设计得很漂亮：离线 RL 的核心难题（分布外高估）在这个参数化下**不需要额外的保守项就消失了**。

#### 2.3.4 主结果 [✓，来自 v2 HTML 全文抽取；未逐格核对原始 PDF]

**在线适应（Robomimic + Gym）**：DSRL 在 Can / Square / Transport 等任务上达到近最优，
样本效率比 DPPO / IDQL / DQL / DIPO / QSM **好 5–10 倍**。

**离线适应（OGBench，10 任务）**，几个代表格：

| 任务 | 最好基线 | DSRL |
|---|---:|---:|
| cube-single-play | IDQL 96±2 | 93±14 |
| scene-play | 76±9 | **88±9** |
| puzzle-4x4-play | 11±3 | **37±13** |
| cube-double-play | 36±6 | **53±14** |

作者自陈"在约一半任务上达到 SOTA"——**没有夸大**，cube-single 上就是输的。

**真机单任务（Franka，pick-and-place）**：

| 方法 | 成功率 |
|---|---|
| 基座 diffusion 策略 π_dp | 2/10 |
| RLPD | 0/10 |
| RLPD + 人工干预 | 0/10 |
| **DSRL** | **9/10** |

**3500 个在线步 ≈ 40 个 episode** [✓]。注意 RLPD 两个变体都是 **0/10**——从零学的真机 RL 在这个预算下学不出任何东西，
这正是"站在 BC 肩膀上"的价值。

**真机多任务（WidowX 250 + Bridge V2，3 任务）**：pick-place / 关抽屉 / 叠方块，每任务 **100–150 episode** 显著提升。

**π₀ 引导（本文最重量级的结果）**：

| 场景 | π₀ 基座 | DSRL 引导后 | 预算 |
|---|---|---|---|
| Libero pick-place（仿真） | ~20% | ~100% | ≈10,000 在线样本 |
| 真机 · 打开烤箱 | 5/20 (25%) | **18/20 (90%)** | ~80 episode / ~11,000 步 |
| 真机 · 拿勺子放盘子 | 15/20 (75%) | **19/20 (95%)** | ~65 episode / ~10,000 步 |

作者原话："据我们所知，这是**第一次成功的 π₀ 真机 RL 微调**" [✓]。

**离线到在线**：加入离线数据把样本复杂度**改善约 2 倍** [✓]。

#### 2.3.5 代码仓库审计 [✓c]

**`ajwagen/dsrl`（Robomimic / Gym）**：**只有 20 个文件**——`train_dsrl.py`、`env_utils.py`、`utils.py`、7 个 cfg，
外加两个 submodule（他们 fork 的 DPPO 和 Stable-Baselines3）。

- **优点**：极小、极易读。README 里那段 **「Applying DSRL to new settings」** 是我见过写得最实用的迁移指南：
  明确说了 DSRL-SAC 只要写一个把动作空间换成噪声空间的 env wrapper（给了 `DiffusionPolicyEnvWrapper` 作范例），
  DSRL-NA 则是把策略传给 `SACDiffusionNoise` 然后当普通 gym 环境跑。
  **调参提示也给了具体数**：`action_magnitude ≈ 1.5`、`utd ≈ 20`、actor/critic 用 3 层 2048 宽的 MLP。
  这几个数省掉的试错时间，比论文本身还值钱。
- **缺点**：**最后一次 push 是 2025-08-05，创建当天，之后再没动过**；7 个 open issue 无人处理 [✓c]。
  **无 LICENSE**。checkpoint 放在 Google Drive（长期可用性存疑）。整个仓库只覆盖 Robomimic/Gym，
  论文里的真机与 π₀ 实验不在这里。

**`nakamotoo/dsrl_pi0`（π₀）**：67 个文件，JAX 实现（`jaxrl2/` 39 文件 + `examples/` 10）。

- **优点**：**MIT 许可**、最后 push **2026-04-27**（仍在维护）[✓c]。真机路径写得很实在：
  用 openpi 的 remote inference 在高配服务器上托管 π₀、机器人端只跑 DSRL，
  这直接解决了"机器人主机带不动 3B VLM"的现实问题。**并且公开了 W&B 训练日志**
  （`wandb.ai/mitsuhiko/DSRL_pi0_public`）——可以在自己复现前先看别人的曲线长什么样，这是很负责任的做法。
  覆盖 Libero / Aloha / 真机 Franka 三套。
- **缺点**：依赖链重（openpi + LIBERO + jax[cuda12]==0.5.0 + 一个 CPU-only 的 torch 2.6.0 用于 Libero），
  版本钉死得很紧，环境很脆。10 个 open issue。真机部分需要 DROID 硬件栈。

#### 2.3.6 亮点

1. **"不改权重"这条设计约束带来的连锁好处被吃干净了**：不用反传穿去噪链（省显存/省时间）、
   基座可以是任意黑盒（甚至是 API 后面的策略）、离线保守性免费获得、多任务可共享一个基座。
   这是那种**一个想法把四个问题一起解决**的工作。
2. **RLPD 0/10 vs DSRL 9/10 这个对照是全文的定海神针。** 它把"BC 先验值多少钱"量化了：
   在 40 个 episode 的预算下，先验的价值是"从完全学不会"到"90%"。
3. **π₀ 真机 RL 首例**，而且用的是 DROID 上预训练的通用策略。这是把"通用策略 + 少量在线适应"这条路线走通的第一个实证。
4. **诚实**：OGBench 上明说"约一半任务 SOTA"；局限一节写得很清楚（下面）。

#### 2.3.7 局限与批评

**作者自陈的（写得很坦白）** [✓]：
- **可引导性没有保证**："DSRL 的探索能力从根本上由底层 diffusion 策略决定，虽然实践中效果不错，但我们**不能保证所有 diffusion 策略都是可引导的**。"
- **分布过窄就没救**：在很窄的数据上训出来的策略，其动作分布高度集中，"可能没有提供足够的选项供我们挑选"。
- **没法提前判断能涨多少**：对给定的 diffusion 策略，方法**不提供任何事前的可改进量估计**。
- 以及所有 RL 的通病：要奖励、要在线 rollout、真机上要 reset。

**我补充的批评**：
- **第 1、3 条合起来是个实际的坑**：你没法在投入真机 40–150 个 episode **之前**知道这次能不能成。
  论文没给一个哪怕粗糙的**可引导性预检**。我认为这是可以补的，而且很便宜——见 §5.2 我给的 30 分钟预检方案。
- **`action_magnitude ≈ 1.5` 这个超参是把双刃剑。** 它允许采样超出 `N(0,I)` 典型半径的噪声，
  也就是说 DSRL **实际上会把基座策略推到它训练分布的尾部甚至外面**。这与"我们只是在基座的选项里挑"这个叙述有张力，
  论文没有讨论这个超参在语义上意味着什么，也没做它的消融。
- **要求 DDIM 采样**（README 明写）[✓c]——用随机 DDPM 采样的策略不能直接用，这是一条容易被忽略的适配前提。
- **两个仓库分裂**，主仓库停更，π₀ 仓库活着但只覆盖 π₀。想在别的 flow 策略（比如 π₀.₅ 或你的 VITA）上复现，
  得自己拼。

---

### 2.4 SAC Flow（arXiv:2509.25756v3）

#### 2.4.1 Metadata

| 项 | 内容 |
|---|---|
| 标题 | SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling [✓] |
| 作者 | Yixian Zhang, Shu'ang Yu, Tonghe Zhang, Mo Guang, Haojia Hui, Kaiwen Long, Yu Wang, Chao Yu, Wenbo Ding [✓]（清华 / CMU / 理想汽车 / 上海 AI Lab [~]） |
| 版本 | v1 2025-09-30 → v2 2025-10-26 → **v3 2026-01-14**（当前）[✓] |
| 发表 | **ICLR 2026**（OpenReview `id=zZvWj4JrYj`）[~] |
| 代码 | https://github.com/Elessar123/SAC-FLOW（注意仓库名全大写，默认分支 `master`）— 68★/8 fork，**无 LICENSE**，最后 push 2025-12-02 [✓c] |

> 与 RL-Co（2602.12628）**共享两位作者：Tonghe Zhang、Yu Wang、Chao Yu**（三位）。
> 这不是巧合——SAC Flow 是算法侧、RL-Co 是系统侧，出自同一个圈子的两条腿。

#### 2.4.2 一句话总结

用 off-policy RL 训 flow 策略为什么总炸？因为 **K 步 Euler 积分的 flow rollout 在代数上等价于一个残差 RNN**，
于是它继承了 RNN 的梯度消失/爆炸。既然是 RNN 问题，就用治 RNN 的办法治：
把 velocity 网络按 **GRU（Flow-G，门控速度）** 或 **Transformer（Flow-T，解码速度）** 重参数化，
再配一个**加噪 rollout** 让 log-prob 可算，就能直接端到端跑 SAC——
**不需要策略蒸馏，也不需要代理目标**。

#### 2.4.3 方法

**核心等价** [✓]：确定性 K 步 Euler 积分

```
A_{t_{i+1}} = A_{t_i} + Δt_i · v_θ(t_i, A_{t_i}, s)
```

令 `f_θ(·) = Δt_i · v_θ(·)`，它就**逐字**是一个残差 RNN 步 `A_{t_{i+1}} = A_{t_i} + f_θ(t_i, A_{t_i}, s)`。
用 off-policy 损失训练要"反传穿过 K 层递归栈"，梯度爆炸/消失是必然的。
Figure 2 里 naive SAC Flow 的梯度范数随反传步数**急剧攀升**，而两个重参数化版本"全程稳定，最大变化 0.29" [✓]。

**Flow-G（GRU 门控速度）** [✓]：

```
g_i = Sigmoid( z_θ(t_i, A_{t_i}, s) )
A_{t_{i+1}} = A_{t_i} + Δt_i · ( g_i ⊙ ( v̂_θ(t_i, A_{t_i}, s) − A_{t_i} ) )
```

门自适应地在"保持当前中间动作"和"施加新更新"之间插值——就是 GRU 的更新门，只是写在速度参数化里。

**Flow-T（Transformer 解码速度）** [✓]：动作-时间 token 各自独立走 L 层 decoder block（**保持马尔可夫性**），
用 cross-attention 让 action token 去查 state embedding：

```
Y_i^(l)   = Φ^(l-1) + Cross_l( LN(Φ^(l-1)), context = LN(Φ_S) )
Φ_i^(l)   = Y_i^(l) + FFN_l( LN(Y_i^(l)) )
A_{t_{i+1}} = A_{t_i} + Δt_i · W_o( LN(Φ_i^(L)) )
```

**加噪 rollout（让 log-prob 可算）** [✓]：把确定性 ODE 换成带漂移修正的 SDE，**保持终点边缘分布不变**：

```
A_{t_{i+1}} = A_{t_i} + b_θ(t_i, A_{t_i}, s)·Δt_i + σ_θ·√(Δt_i)·ε_i ,   ε_i ~ N(0, I_d)
b_θ = [ (1 − t_i + t_i σ_θ²/2) / (1 − t_i) ]·v_θ  −  [ t_i σ_θ² / (2(1−t_i)t_i) ]·A_{t_i}
```

于是路径密度可分解为逐步高斯的乘积（均值 `A_{t_i} + b_θ Δt_i`，协方差 `σ_θ² Δt_i I`），
再乘一个 `tanh` 的 Jacobian 修正项。**SAC 的熵项就有了。**

**损失** [✓]：
- from-scratch actor：`L = α·log p_c(A^θ|s) − Q_ψ(s, a^θ)`
- critic：标准 soft TD
- offline-to-online actor：额外加邻近正则 `+ β·‖a^θ − a‖²`

**超参** [✓]：flow 采样步 **K = 4**（diffusion 基线用 16 步），**σ_θ ≈ 0.10**，
熵温度 α 自动学（target entropy = 0），β 早期大、随在线数据增加退火。**所有结果 5 个随机种子，报 95% 置信区间。**

#### 2.4.4 主结果——以及一条必须说明的事

**本文的主结果全部在图里，没有数值表。** [✓，两次独立抽取均确认；arXiv comment 与 30 页 23 图的 OPD 类似]
具体位置：from-scratch 在 Figure 4（6 个 MuJoCo 任务 + 2 个 Robomimic），
offline-to-online 在 Figure 5（OGBench Cube 系列 + Robomimic 聚合），消融在 Figure 6–7。
附录 C 只有一张描述 Robomimic 环境的表，不是结果表。

因此**本节不给精确数字**。可以确证的只有这些：

- **摘要级断言** [✓]：在连续控制与机器人操作 benchmark 上达到 SOTA，
  "消除了对策略蒸馏或代理目标这类常见变通做法的需要"。
- **HumanoidStandup 上「相对基线最高提升 130%」**、**OGBench 上「最高高出 60 个百分点的成功率」**——
  这两个数出现在正文文字里 [✓]，但对应的是哪个基线、哪个 setting，只有看图才知道。
- **梯度稳定性消融**（Figure 6a）：naive SAC Flow 梯度范数随反传步骤急剧攀升；Flow-T/Flow-G **全程最大变化 0.29** [✓]。
  这是全文最干净的一条实证，直接验证了"这是个 RNN 病"的诊断。
- **对采样步数 K 鲁棒**（Figure 7）：K ∈ {4, 7, 10} 性能稳定 [✓]。
- 基线：from-scratch 对 FlowRL / DIME / SAC / PPO；offline-to-online 对 QC-FQL / FQL / ReinFlow [✓]。

> 关于我在起草过程中一度拿到的那份"SAC Flow-T Hopper ~3500 / Walker2D ~4800"之类的表格：
> 那是抽取模型**从图上读出来再排成表**的产物，**不是论文里的数字**。我没有把它写进本文。
> 如果你需要精确对比数，只能去 OpenReview 的 PDF 里逐图读点，或直接跑仓库。

#### 2.4.5 代码仓库审计（`Elessar123/SAC-FLOW`）[✓c]

99 个文件：`offline-to-online/` 44、`cleanrl_utils/` 27、`from_scratch_code/` 11，
外加 `README.md`、`sacflow-setup.md`、`requirements.txt`、`pyproject.toml`、`overview.png`。

**优点**：
- **命令行级别的可复现性**。README 给的是可以直接粘贴的完整命令，且**把三个消融并列**：
  ```bash
  python SAC_flow_transformer_jax.py     # Flow-T
  python SAC_flow_gru_jax.py             # Flow-G
  python Naive_sac_flow_jax.py           # 不重参数化的对照（论文的核心 claim 就靠它）
  ```
  offline-to-online 侧同样给了 Flow-T / Flow-G / Naive / QC-FQL / FQL **五条**命令，只差一个 `--agent` 参数。
  **把自己的负面对照（Naive）也做成一等公民的入口脚本，这个做法值得抄。**
- **老实交代血统**：from-scratch 基于 cleanrl，offline-to-online 基于 QC-FQL，README 明确请求一并引用。
- 单独的 `sacflow-setup.md` 装环境文档。

**缺点**：
- **无 LICENSE** [✓c]。
- **最后 push 2025-12-02**，而 arXiv v3 是 2026-01-14 —— **代码比当前论文版本旧**，v3 改了什么没有对应到代码上。
- **0 个 open issue / 68 星 / 8 fork**：热度和社区验证都很低（对比 DSRL 的 221+289 星）。0 issue 在这个星数下更可能意味着"没什么人真的跑起来"而不是"没 bug"。
- 文件名里带 `ablation`（`acfql_transformer_ablation_online_sac.py`）——**主方法的入口脚本叫 ablation**，
  说明这是研究期脚本原样发布，没有为发布整理过。
- **JAX 实现**。你的栈是 PyTorch（LeRobot），迁移不是零成本。

#### 2.4.6 亮点

1. **诊断本身就是贡献。** "flow rollout ≡ 残差 RNN"这个观察一旦说出来就是显然的，
   但在此之前整个领域的应对方式是绕（蒸馏成一步策略、用代理目标），没人指出病因。
   **这条等价对任何多步 flow/diffusion 策略都成立**——包括你的 VITA、RTC，也包括 π₀/π₀.₅。
2. **"既然是 RNN 病，就用治 RNN 的成熟药"** —— 门控与注意力都是被验证了十年的东西，
   不需要发明新机制。这是很好的品味。
3. **K = 4 就够**，且对 K 鲁棒。这对推理延迟敏感的机器人场景是关键——不用为了 RL 可训性去堆采样步。
4. **加噪 rollout 保持终点边缘分布不变**这个构造是干净的：得到 log-prob 不以改变策略本身为代价。

#### 2.4.7 局限与批评

**作者自陈** [✓]：
- **Robomimic 上没赢**：那组实验"用很大的 β 强正则化"，"导致其性能与 QC-FQL 的单步策略相近"。
  换句话说，**一旦施加强行为约束，flow 策略的表达力优势就被抹掉了**。
- **from-scratch 在大探索空间 + 稀疏奖励上全体失败**（Humanoid），所以还是要 offline-to-online。
- **未来工作明确列出"在真机上验证 SAC Flow"** —— 也就是说**本文没有任何真机结果**。

**我补充的批评**：
- **只有图、没有表**（§2.4.4）。对一篇 ICLR 论文而言这是相当反常的，也让后续工作无法引用它的具体数。
- **"稀疏奖励下 from-scratch 失败"这条与本文卖点有张力**：论文的主打是"表达型策略 + 稳定的 off-policy 训练"，
  但表达型策略最该发光的地方（多模态、稀疏奖励、长程）恰恰是它失败的地方；
  它赢得最漂亮的 HumanoidStandup 是**稠密奖励**任务。
- **计算开销没有报告**。Flow-T 在每个 Euler 步都跑 L 层 decoder，K=4 就是 4L 层的前向 + 反传。
  相对 naive flow 的 wall-clock 代价是多少，论文没给。对机器人部署这是必需信息。
- **没有真机、没有图像观测**。所有实验都在状态空间（MuJoCo / OGBench / Robomimic state）。
  从状态空间到像素空间，梯度病理的表现可能完全不同。

---

## 3. 强化学习：主线、核心进展与历史

> 你在需求里把 RL 写成了 "Research and Development"。四篇论文全部是 Reinforcement Learning，
> 本节按**强化学习**来写。如果你要的确实是研发方法论综述，说一声，那是另一篇。

### 3.1 一张时间线

```
1988  TD(λ)                Sutton               时序差分：不用等到回合结束就能学
1989  Q-learning           Watkins              off-policy 控制，收敛性证明
1992  REINFORCE            Williams             策略梯度，直接对策略参数求导
1992  经验回放              Lin                 后来成为 off-policy 深度 RL 的地基
─────────────────────────── 深度 RL 引爆 ───────────────────────────
2013  DQN                  arXiv:1312.5602 [✓]  CNN + 回放 + 目标网络，Atari 像素到动作
2015  DDPG                 arXiv:1509.02971 [✓] 连续动作的确定性策略梯度
2015  TRPO                 arXiv:1502.05477 [✓] 信赖域，第一次让策略梯度"不炸"
2015  AlphaGo              (Nature 2016)        RL + MCTS + 自博弈
2015  GAE                  arXiv:1506.02438 [✓] 优势估计的偏差-方差旋钮，至今仍是 PPO 默认件
─────────────────────────── 稳定化与规模化 ───────────────────────────
2017  PPO                  arXiv:1707.06347 [✓] clip 代替信赖域；十年过去仍是工业默认
2017  分布式 RL             arXiv:1707.06887 [✓] 学值分布而非期望
2017  DeepMind Parkour     arXiv:1707.02286 [✓] "环境多样性比奖励设计更重要"
2018  SAC                  arXiv:1801.01290 [✓] 最大熵 off-policy；本文四篇里两篇的地基
2018  IMPALA               arXiv:1802.01561 [✓] V-trace，把 actor-learner 分离做成规模化范式
─────────────────────────── 离线 RL 与世界模型 ───────────────────────────
2019  Dreamer              arXiv:1912.01603 [✓] 在学出来的世界模型里做想象式规划
2020  离线 RL 综述          arXiv:2005.01643 [✓] Levine 等把"不交互只用数据"立成一个子领域
2022  TD-MPC               arXiv:2203.04955 [✓] 隐空间模型 + 规划 + TD
─────────────────────────── RL 变成"后训练工具" ───────────────────────────
2022  InstructGPT/RLHF     arXiv:2203.02155 [✓] RL 的主战场从"控制"转向"对齐"
2024  GRPO / DeepSeekMath  arXiv:2402.03300 [✓] 去掉 value 网络，组内相对优势
2025  DeepSeek-R1          arXiv:2501.12948 [✓] RLVR：可验证奖励直接激发推理能力
2026  OPD 本文             arXiv:2604.13016 [✓] 把"奖励"换成教师的逐 token 分布，并指出其代价
─────────────────────────── 表达型策略 × RL（本文四篇的战场）───────────────────────────
2020  DDPM / 2020 DDIM     2006.11239 / 2010.02502 [✓]  生成模型侧的地基
2022  Flow Matching        arXiv:2210.02747 [✓]
2023  Diffusion Policy     arXiv:2303.04137 [✓] 动作生成从"回归一个动作"变成"采样一个分布"
2023  ACT / ALOHA          arXiv:2304.13705 [✓] CVAE + action chunking
2023  IDQL                 arXiv:2304.10573 [✓] diffusion 策略当 actor
2023  Q-Transformer        arXiv:2309.10150 [✓] 自回归 Q 函数，离线 RL 上大模型
2024  DPPO                 arXiv:2409.00588 [✓] 把去噪链当成 MDP，用 PPO 训
2024  π₀                   arXiv:2410.24164 [✓] flow matching VLA
2025  FQL                  arXiv:2502.02538 [✓] 蒸馏成一步策略以绕开多步反传
2025  ReinFlow             arXiv:2505.22094 [✓] 注噪让 flow 策略有可算的 log-prob
2025  DSRL 本文            arXiv:2506.15799 [✓] **不动权重，在噪声空间做 RL**
2025  Q-chunking           arXiv:2507.07969 [✓] 在动作块粒度上做 TD
2025  SAC Flow 本文        arXiv:2509.25756 [✓] **flow rollout ≡ 残差 RNN，按序列模型重参数化**
2025  RLinf / RLinf-VLA    2509.15965 / 2510.06710 [✓] VLA 的 RL 训练基础设施
2026  RL-Co 本文           arXiv:2602.12628 [✓] **PPO in sim + 真机 SFT 锚定**
```

### 3.2 三次范式转移

**第一次（2013–2018）：从表格到网络，从"能学"到"稳定地学"。**
DQN 之前，RL 是一个理论漂亮、实践只能做小网格世界的领域。DQN 之后五年的全部工作，
本质上是在回答同一个问题：**深度函数逼近 + bootstrapping + off-policy = 致命三角，怎么办？**
答案分两路：约束更新幅度（TRPO → PPO）、稳住值估计（目标网络、双 Q、分布式 RL、最大熵）。
**PPO 和 SAC 至今没被取代，说明这一路已经收敛。**

**第二次（2020–2024）：RL 的产品形态从"训练一个 agent"变成"后训练一个基座模型"。**
InstructGPT 是分水岭。此后 RL 的用法变了：
- **不再从零学**，而是在一个已经很强的预训练模型上做小幅调整（lr 4e-6，见 RL-Co 的配置）；
- **不再追求最优策略**，而是追求"不要偏离先验太远的同时改进某个指标"（KL 惩罚 / 锚定损失）；
- **奖励从环境里来变成从别处来**：人类偏好（RLHF）→ 可验证器（RLVR）→ 教师分布（OPD）。

**这一次转移的完整后果，正是本文四篇的共同前提**：所有四篇都不从零训，全部是"基座 + RL"。
DSRL 甚至把这条推到极致：基座连梯度都不给。

**第三次（2023–至今，进行中）：策略类从高斯变成生成模型。**
这是当前最活跃、也最没定论的一次。起因是机器人：人类演示天然多模态（同一个杯子可以从左边抓也可以从右边抓），
高斯策略会平均出一个两边都不是的动作。Diffusion Policy 和后来的 flow matching VLA 解决了表达力，
**但同时打断了 RL 的技术前提**——`log π(a|s)` 没了。

### 3.3 第三次转移的技术症结（本文四篇的真正战场）

把问题写清楚：flow/diffusion 策略是 `a = g_θ(w, s)`，其中 `w ~ N(0,I)`，`g` 是 K 步迭代。
RL 需要三样东西，它三样都不给：

| RL 要什么 | 高斯策略 | Flow/Diffusion 策略 | 后果 |
|---|---|---|---|
| `log π(a\|s)` | 解析 | **没有闭式** | SAC 的熵项、PPO 的重要性比都算不了 |
| `∇_θ log π` 或 `∇_θ a` | 一层 | **穿过 K 步递归** | 梯度爆炸/消失（SAC Flow 的诊断） |
| 高效采样 | 一次前向 | **K 次前向** | rollout 成本 ×K |

### 3.4 已知的四种解法（把四篇论文摆进去）

**解法 1 · 换掉动作空间——在噪声上做 RL。**
既然 `a = g_θ(w,s)` 里 `θ` 难动，那就固定 `θ`、把 `w` 当动作。
`w` 是标准高斯的，一切 RL 工具立刻可用。→ **DSRL**。
- **优点**：三个问题一次全绕开，基座黑盒，无需反传。
- **代价**：能达到的策略被 `g_θ(·,s)` 的值域锁死（DSRL 自陈的局限一、二）。
- **亲戚**：残差 RL（在基座输出上学一个增量）、拒绝采样（V-GPS）。DSRL 与它们的区别在于**改的是输入而不是输出**，
  因此产出的动作**永远在基座的流形上**——这既是它的安全性来源，也是它的天花板。

**解法 2 · 修架构——让 K 步反传变得可行。**
承认要反传，那就把递归栈修得能传。→ **SAC Flow**（Flow-G / Flow-T）。
- **优点**：不牺牲表达力、不引入代理目标，端到端。
- **代价**：改了策略架构，**不能直接套在已有的预训练策略上**（π₀ 的 velocity 网络不是 GRU）。
  这是它与 DSRL 最本质的适用性差别。

**解法 3 · 换目标——把 flow 变回可算 log-prob 的东西。**
- **蒸馏成一步**：FQL 把多步 flow 蒸成一步策略再做 RL。
- **注噪**：ReinFlow 给确定性 ODE 注入噪声变成 SDE，路径密度就可分解为高斯乘积。
  → **RL-Co 用的就是这条**（配置里的 `noise_method: flow_noise`）。SAC Flow 的"加噪 rollout"也属于这条，
  区别是 SAC Flow **同时**做了解法 2。
- **代价**：注噪改变了策略的随机性结构；蒸馏损失表达力（FQL 的一步策略就是多模态的损失）。

**解法 4 · 换监督信号——不要环境奖励。**
标量奖励太稀疏，那就用一个更强模型的完整分布当稠密监督。→ **OPD**。
- **优点**：监督密度提高几个数量级，样本效率极高。
- **代价**：OPD 这篇论文本身证明的——**优势随轨迹深度衰减**，长程失效；且需要一个真的更强、且思维模式兼容的教师。

### 3.5 我对当前局势的判断

1. **解法 1（DSRL 派）会先在工业上铺开**，因为它是唯一不要求你拥有基座模型训练权限的。
   在"通用策略 API 化"的趋势下（π₀ 系列、GR00T 都在往这个方向走），黑盒适应会变成刚需。
2. **解法 2 与解法 3 会合流**。SAC Flow 已经同时用了两条。我预期 2026–2027 的标准做法是
   "重参数化的 velocity 网络 + 注噪 rollout + chunk 级 TD"，三件套。
3. **真正没解决的是奖励，不是算法。** 四篇里三篇要环境奖励，而真机上的奖励要么靠人标、
   要么靠仿真（RL-Co 的做法）、要么靠精心设计的传感器判据。RL-Co 之所以要建数字孪生，
   根本原因就是**真机上拿不到便宜的奖励**。谁解决了"真机上的自动奖励"，谁就解开了这一整条链。
   这也是我认为你的 `real-world-policy-evaluation-2026-08.md` 那条线**比换算法更值钱**的原因。
4. **OPD 的负面结论会被反复重新发现。** "稠密监督在长程上失效"这个现象在机器人上的对应物是
   "action chunking 在长程任务上收益递减"。两边现在还没互相引用，但这是同一件事。

---

## 4. 四篇横向对照

| 维度 | OPD | RL-Co | DSRL | SAC Flow |
|---|---|---|---|---|
| 域 | LLM 数学推理 | VLA 真机操作 | Diffusion/Flow 策略、真机 | 连续控制 / Robomimic（**无真机**） |
| 基座权重 | 全量训练 | 全量 PPO 微调 | **完全冻结** | 需改架构后训练 |
| 监督信号 | 教师 token 分布 | 仿真奖励 + 真机 SFT 锚 | 环境奖励 | 环境奖励 |
| RL 算法 | OPD（KL 类）+ GRPO 对照 | **PPO + GAE**（chunk 级） | SAC / DSRL-NA 双 critic | SAC + 加噪 rollout |
| 处理 log-prob 的方式 | N/A（token 级本来就有） | **注噪**（ReinFlow 式） | **绕开**（噪声空间） | **注噪 + 重参数化** |
| 关键超参 | top-k、prompt 模板 | β=0.2, lr 4e-6, K_chunk=4, H=8 | `action_magnitude≈1.5`, `utd≈20` | K=4, σ≈0.10, target entropy=0 |
| 真机预算 | — | 20–50 演示/任务 + 数字孪生 | **40–150 episode** | — |
| 结果呈现 | **只有图** | 有表（且有一格算错） | 有表 | **只有图** |
| 代码许可 | **无** | Apache-2.0（RLinf） | dsrl 无 / dsrl_pi0 **MIT** | **无** |
| 代码活跃度 | push 2026-08-20，944★ | 活跃框架，2450 文件 | 主仓停更 2025-08；π₀ 仓 2026-04 | push 2025-12（**旧于论文 v3**），68★ |
| 复现门槛 | 中（数据已随仓库给） | **高**（需数字孪生 + 多卡） | **低**（20 个文件 + 迁移指南） | 中（JAX，命令齐全） |
| 上游采纳 | **指标合入 verl 主线** | 合入 RLinf 主线 | 有第三方 fork（lasgroup / arayabrain） | 无 |

**如果只能读一篇**：DSRL。理由是它的想法最简单、代码最小、迁移指南最具体、真机证据最硬，
而且**它的适用前提（有一个能控制初始噪声的生成式策略）你已经满足**。

**如果只能抄一段代码**：RL-Co 的 `maniskill_ppo_co_training_openpi_pi05.yaml`。它是四篇里唯一一份
"经过真机验证的 VLA RL 超参全集"。

---

## 5. 对你自己研究的启示

> 本节所有 `file:line` 均为本次实际 grep 所得（`/home/kewei/YING/robot_data_platform/lerobot`），
> 涉及你既有结论处均链接到本目录下的对应报告。

### 5.0 先说唯一的硬前提：你现在一篇都用不了

四篇论文都要求闭环：要 rollout、要奖励、要 reset。而你的现状是——

- [`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md) 的前置声明：
  **三个 job 的 `env_eval_freq` 都是 0，`/mnt/robot_platform/` 下没有评测目录，磁盘上没有任何 rollout 成功率记录。**
- [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md)：训练侧只有 loss，真机侧只有二值成功率。

**结论很硬：在自动化闭环评测跑起来之前，本节 5.2–5.5 的所有方案都是纸上谈兵。**
不是"最好先有"，是"没有就动不了"——RL 的每一次梯度更新都要吃一个 rollout 回报。
好消息是这件事你本来就排在第一优先级（那份报告的 §6.4），四篇论文只是又给了四个理由。

**并且注意 DSRL 的真机预算给出了闭环需要多自动化的量化答案**：40–150 个 episode。
如果你的 reset 是人工摆放，40 个 episode 大约是一下午；如果是 150 个且要跑多次实验，
**自动 reset 就从"锦上添花"变成了瓶颈**。这个数值应该直接进你的评测台设计需求。

### 5.1 你现在在哪

按 §1 的坐标：**你在原点以下、还没上任何一条轴**。纯 BC（ACT / diffusion / VITA），
rollout 成功率 60–70%（`team-division-strategy-2026-08.md` 白板记录）。

而 60–70% 正是这四篇论文的实验里**基座策略最典型的位置**——
DSRL 的真机 pick-place 是 2/10、π₀ 开烤箱 5/20、RL-Co 的 π₀.₅ 平均 26.7%。
**这些数字全都在"加多少演示都不太动"的区间里**，而闭环一接上就跳到 90%+。
这是四篇论文给你的最强信号：**你剩下那 30–40% 的失败，大概率不是数据量问题。**

这也直接回答了 `team-division-strategy-2026-08.md` §2 的"失败预算：那 30–40% 归谁"——
文献的答案是：**归"没有闭环信号"这一项**，而不是归数据、归主干、归架构。

### 5.2 最高优先级的具体点子：DSRL 用在 ACT 的 VAE latent 上

**这是本文最有行动价值的一条，也是我在读的过程中真正觉得可做的一个新东西。**

DSRL 的 MDP 改写只需要一件事：策略是 `a = g_θ(w, s)`，**给定 `w` 时确定性**，且 `w` 可控。
论文和 README 都写成"diffusion 或 flow 策略"（并要求 DDIM 采样），但那个 DDIM 要求
**本质上只是为了拿到"给定 `w` 的确定性"**。ACT 满足这个条件——而且入口就在你手里：

```
lerobot/src/lerobot/policies/act/configuration_act.py:113  use_vae: bool = True
                                                     :114  latent_dim: int = 32
                                                     :123  kl_weight: float = 10.0
lerobot/src/lerobot/policies/act/modeling_act.py:451  latent_sample = mu + log_sigma_x2.div(2).exp() * randn_like(mu)   # 训练时
                                            :456  latent_sample = torch.zeros([batch_size, latent_dim])                # 推理时全零
                                            :461  encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
```

**推理时 `z` 被写死成全零**（`modeling_act.py:456`），也就是说 ACT 部署时**永远走 CVAE 先验的均值**。
把这个全零换成一个由 RL 学出来的 `w ∈ R³²`，就是 DSRL 在 ACT 上的直译版本：

- 噪声空间 `W = R³²`（**32 维，比 diffusion 策略的 `w` 小一到两个数量级**——
  DSRL 的 `w` 与动作 chunk 同维，ACT 的 `z` 只有 32 维）。**维度小是巨大的样本效率优势**，
  DSRL 论文自己也说噪声空间的探索是样本效率的来源。
- 基座 ACT 完全冻结、黑盒调用，改动只有 `modeling_act.py:456` 这一行 + 一个 env wrapper。
- 可直接照抄 `ajwagen/dsrl` 的 `DiffusionPolicyEnvWrapper` 与 `SACDiffusionNoise`，
  以及 README 给的超参起点（`action_magnitude≈1.5`、`utd≈20`、3×2048 MLP）。

**但先做这个 30 分钟的预检——可引导性测试。**
DSRL 自陈的第一条局限是"不保证所有策略都可引导"，而它**没给检测方法**。
对 ACT 这个检测特别必要，因为 `kl_weight = 10.0` 是个不小的值，
**CVAE 在这个权重下很可能后验坍塌（posterior collapse）**——`z` 根本没编码任何东西，
那么在 `z` 上做 RL 就是在优化一个对输出没有影响的变量，投入的每一个真机 episode 都会白费。

预检做法（不需要真机，用你已有的权重和数据集）：

```
对若干真实观测 s：
  采 N=64 个 z ~ N(0, I) 与 z ~ 1.5·N(0, I)         # 1.5 对应 action_magnitude
  前向得到 64 个动作 chunk a_i ∈ R^{100×7}
  指标 1：动作 chunk 在 z 上的方差 / 在 s 上的方差    # 前者接近 0 = 坍塌，DSRL-on-ACT 死路
  指标 2：64 条 chunk 的两两 L1 距离分布 vs 该数据集里动作本身的尺度
  指标 3：末端轨迹画在一张图上，肉眼看是否出现不同模式（比如左手抓 / 右手抓）
```

**这个测试的价值不止于 DSRL**：如果 ACT 的 `z` 确实坍塌了，那意味着你的 ACT
**根本没有在建模多模态**，只是在做带 chunk 的确定性回归。那会同时解释
[`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md) 里的现象——
一个单模态的回归器，加入失败数据就是在往回归目标里掺噪声，**当然会变差**。
所以无论 DSRL 走不走得通，**这个预检本身就是一次高价值的诊断**。

如果确实坍塌，有两条路：(a) 降 `kl_weight` 重训一版专供 DSRL 的 ACT；
(b) 放弃在 ACT 上做，转到 diffusion / VITA 策略上做（它们的噪声入口天然可控）——这就是 §5.3。

### 5.3 SAC Flow → 你的 flow matching 线（VITA / RTC / DiT）

[`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md) 和
[`act-diffusion-integration-2026-08.md`](./act-diffusion-integration-2026-08.md) 已经把你的 flow 路线铺开了。
SAC Flow 对这条线有两个直接可用的产出，**而且不需要你做 RL 就能用**：

1. **"flow rollout ≡ 残差 RNN"这条等价对你的 VITA / RTC 一样成立。**
   即使你短期不做 RL，这条也解释了一类你可能已经遇到的现象：
   **flow 策略在采样步数 K 变大时训练变难**。如果你在 `vita-arch-ablation` 里见过这个，
   现在有了病因名字，也有了两个现成的药（Flow-G 的门控 / Flow-T 的 cross-attention 解码）。
2. **K = 4 就够，且对 K ∈ {4,7,10} 鲁棒**（Figure 7）。这条对你的推理延迟预算是直接的好消息——
   参照 [`vision-backbone-why-resnet18-2026-08.md`](./vision-backbone-why-resnet18-2026-08.md) 里的延迟量级分析，
   K 从 16 降到 4 省下的时间比换主干省的多得多。

**但要认清 SAC Flow 的适用边界**（§2.4.7）：**没有真机、没有像素观测、Robomimic 上没赢**、
JAX 实现、代码比论文旧。**它不是一个可以照搬的方案，是一个诊断 + 两个架构建议。**
按这个定位用它，收益是实的；当成"上 RL 的路线图"用，会踩坑。

### 5.4 OPD 的诊断范式 → 你缺的那个闭环离线指标（我认为这是四篇里最被低估的一条）

你的核心痛点（`real-world-policy-evaluation-2026-08.md` §0）：**训练 loss 不预测真机成功率**。
那份报告已经找到了一半答案（CI-MSE 把 Spearman 从 −0.61 提到 −0.87）。OPD 提供了另一半，而且是正交的：

**OPD 的关键结构不是"用教师分布"，而是"在策略自己访问过的状态上，度量它与参考分布的重合度"。**
拆开看有三个要素：
1. **on-policy 的状态分布**（不是数据集的状态分布）——这是它区别于 teacher-forcing 指标的根本；
2. **分布级的比较**（overlap ratio / entropy gap），不是点估计的误差；
3. **在训练过程中连续测**，用来**预判**成败，而不是事后总结。

搬到你身上就是：

| OPD 里的量 | 机器人侧的对应物 | 怎么算 |
|---|---|---|
| student-visited states | **rollout 中策略自己到达的观测** | 从闭环评测里录 |
| teacher token distribution | 专家演示在**相似观测**上的动作分布 | 在演示集里做最近邻检索，取邻域内的动作集合 |
| overlap ratio | 策略动作落在专家动作邻域内的比例 | 阈值化的动作距离 |
| entropy gap | 策略动作分布的熵 − 专家邻域动作的熵 | 策略侧多采样 z/w 得到分布 |
| overlap-token advantage | 重合动作 vs 非重合动作的回报差 | 需要成功/失败标签 |

**为什么这比你现在的 loss 强**：teacher-forcing 的 chunk 误差（你在 failure-data 报告 §4 里用的）
是在**专家状态**上测的，而策略失败恰恰发生在**它自己漂出去以后到达的状态**上。
你那份报告的结论"离线指标上'变差'几乎看不出来，所以伤害必然发生在闭环"——
**OPD 的范式正是为这句话准备的工具**：把度量点从专家状态搬到策略访问状态上。

**而且这个指标不需要 RL，只需要 rollout。** 一旦 §5.0 的闭环评测建起来，它是几乎免费的附加产出。
另外，OPD 的 overlap 指标已经合入 verl 主线（`distillation/overlap_ratio`）[✓c]，
实现可以直接读它们的 PR 找参考。

**顺带一条方法论警告**（OPD §2.1.4(5)）：教师优势随前缀长度单调衰减（+0.37 → +0.02）。
机器人上的对应物是 **action chunk 越长，后半段的监督越没信息量**。
你的 ACT 是 `chunk_size = 100`（`configuration_act.py:85`）——这是个相当长的 chunk。
**如果按 OPD 的方式测一下"专家动作在 chunk 第 1 步 vs 第 100 步上相对策略的优势"，很可能看到同样的单调衰减。**
这是一个便宜、可做、而且直接指向 chunk 长度选择的实验。

### 5.5 RL-Co → 失败数据的正确用法，以及数字孪生要不要建

**(1) 它顺带回答了你的失败数据问题。**
[`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md) 的问题是
"361 成功 + 36/72 失败，加了失败数据反而更差"。
RL-Co 给出的框架性答案是：**失败数据在 BC 目标下没有正确的位置**——
BC 的损失函数只会让策略去拟合它见到的动作，你把失败轨迹塞进去，就是在让它学会失败。
**失败数据的正确位置是 RL 的负回报样本**，那里有一个符号可以把它翻过来用。

这不是我的推测，是 RL-Co 的两阶段结构在说的事：Stage I 的 SFT 只吃**成功**演示
（真机 20–50 条成功 + MimicGen 生成的 **1000 条成功**轨迹），失败只在 Stage II 的 RL 里以低回报的形式出现。
**所以短期的建议很明确：把失败数据从 BC 训练集里拿出去，存起来，等 RL 阶段再用。**

**(2) 数字孪生：值得建，但要按 RL-Co 的标准建，不是按 SimFoundry 的标准。**
RL-Co 的一条重要负面设计选择是：数字孪生**刻意不做照片级真实**——
只对齐布局 / 相机视角 / 任务逻辑 / 语言 / 动作空间，物体材质、纹理、光照、背景**都不精确匹配** [✓]。
这大幅降低了建孪生的成本。对照
[`paper_note/simfoundry-2606.28276.md`](./paper_note/simfoundry-2606.28276.md) 里 SimFoundry 那条
"自动重建高保真数字孪生"的路线，RL-Co 说的是：**如果你的目的是 RL 训练（不是仿真评测），保真度要求低得多。**
两者目标不同，不要把 SimFoundry 的成本预算套到 RL-Co 上。

**(3) 可以直接抄的超参**（§2.2.3 那段 YAML）。特别是：
- `sft_loss_weight: 0.2`（真机锚定权重）、`kl_beta: 0.0`（**不用 KL 惩罚，用真机数据当锚**）；
- chunk 级的 reward / logprob / entropy —— 这与你的 ACT/VITA 的 chunk 结构天然对齐；
- `lr: 4e-6`；`entropy_bonus: 0.005`；
- 监控 `train/loss_ratio = β|L_SFT|/|L_RL|`，RLinf 在它超过 1e5 时告警。
  **这个健康度指标你在自己实现锚定损失时也该加**——两个损失量级失衡是这类方法最常见的失效模式。

**(4) 但别低估成本。** RL-Co 的消融显示：**没有 Stage I 的仿真 SFT 热启动，跑了三百万交互步策略仍近乎不动**。
也就是说要拿到 RL 的收益，你得先有：数字孪生 + MimicGen 生成的 1000 条仿真轨迹 + 一次混合 SFT。
再加上 `no_shard` + 不支持梯度检查点的显存需求——**这是四篇里工程量最大的一条路。**

### 5.6 优先级

| # | 动作 | 依赖 | 成本 | 预期收益 | 出处 |
|---|---|---|---|---|---|
| **0** | **把自动化闭环评测跑起来**（含自动 reset，目标 ≥150 episode/次不需人守） | 无 | 高 | **一切的前提** | §5.0 |
| **1** | **ACT 的 `z` 可引导性预检**（30 分钟，不用真机） | 已有权重+数据 | 极低 | 高：同时诊断多模态坍塌 | §5.2 |
| **2** | **on-policy overlap 指标**（在 rollout 上测策略 vs 专家的分布重合） | #0 | 低 | 高：补上你缺的闭环离线指标 | §5.4 |
| **3** | **失败数据移出 BC 训练集** | 无 | 极低 | 中：止损 | §5.5(1) |
| **4** | **chunk 位置 × 优势衰减实验**（chunk 第 1 步 vs 第 100 步） | #2 | 低 | 中：直接指向 `chunk_size` 选择 | §5.4 |
| **5** | **DSRL-on-ACT 真机试点**（单任务，40–150 episode） | #0, #1 通过 | 中 | **高：可能是最快把 60–70% 推上去的一条** | §5.2 |
| **6** | flow 线用 Flow-G/Flow-T 重参数化 velocity | flow 策略成熟 | 中 | 中 | §5.3 |
| **7** | RL-Co 全套（数字孪生 + 两阶段） | #0 + 多卡 + 建孪生 | **高** | 高但慢 | §5.5 |

**#1 到 #4 加起来不到一周，而且四项都不需要真机、不需要 RL、不需要新硬件。** 建议先把这四项做完再谈 #5–#7。

### 5.7 反对意见与风险（我自己给自己挑刺）

1. **§5.2 的 DSRL-on-ACT 是我的推演，不是论文的结论。** DSRL 从未在 CVAE 策略上测过。
   风险点有三：(a) `z` 可能坍塌（所以有 #1 预检）；(b) ACT 的 `z` 是**每条轨迹一个**、
   而 DSRL 的 `w` 是**每步一个**，语义不完全对应——把 `z` 变成每步可变会偏离 ACT 的训练分布；
   (c) 32 维虽然样本效率好，但也可能表达力不足以覆盖需要的行为修正。
   **(b) 是最实质的一条**，试点时应先固定"每个 chunk 换一次 `z`"，与 chunk 的语义对齐。
2. **DSRL 真机的 40 个 episode 是在一个"基座 2/10"的任务上取得的。** 你的基座是 60–70%，
   起点更高意味着**剩下的失败可能是基座分布里根本没有的行为**——那正好落进 DSRL 自陈的第二条局限
   （分布过窄 / 没有足够选项）。**这个风险不能靠预检排除，只能靠试点回答。**
3. **§5.4 的 overlap 指标我没有验证过它与真机成功率的相关性。** OPD 证明的是它在 LLM OPD 训练里有预判力，
   搬到机器人上是类比。**上线前应该按 `real-world-policy-evaluation-2026-08.md` 里 CI-MSE 那套做法，
   先测它与 rollout 成功率的 Spearman 相关**，别直接当指标用。
4. **四篇没有一篇解决"真机上从哪来奖励"。** 这是整条链上真正的瓶颈（§3.5.3），
   而我在本节给的所有方案都默认你能拿到奖励。如果你的任务成功判据本身要人看，那 #5 和 #7 的实际成本要乘以人力系数。

---

## 6. 参考文献

### 6.1 本文四篇主体

| # | 文献 | 核实 |
|---|---|---|
| [1] | Li, Zuo, He, et al. *Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.* arXiv:2604.13016v2, 2026-04-15. ICML 2026 FoGen Workshop. 代码 https://github.com/thunlp/OPD | [✓] abs 页 + [✓c] README/tree |
| [2] | Shi, Chen, Gao, et al. *Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models.* arXiv:2602.12628v4, 2026-06-04. 项目页 https://rl-co-training.github.io/ ；代码在 https://github.com/RLinf/RLinf | [✓] abs + HTML 全文 + [✓c] YAML/docs 原文 |
| [3] | Wagenmaker, Nakamoto, Zhang, Park, Yagoub, Nagabandi, Gupta, Levine. *Steering Your Diffusion Policy with Latent Space Reinforcement Learning.* arXiv:2506.15799v2, 2025-06-25. CoRL 2025. 项目页 https://diffusion-steering.github.io ；代码 https://github.com/ajwagen/dsrl 与 https://github.com/nakamotoo/dsrl_pi0 | [✓] abs + HTML 全文 + [✓c] 两份 README |
| [4] | Zhang, Yu, Zhang, et al. *SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling.* arXiv:2509.25756v3, 2026-01-14. ICLR 2026 [~]。代码 https://github.com/Elessar123/SAC-FLOW | [✓] abs + HTML 全文 + [✓c] README/tree |

### 6.2 §3 时间线中引用的文献（全部本次批量核对标题与编号）

深度 RL 主线：DQN `1312.5602` [✓] · DDPG `1509.02971` [✓] · TRPO `1502.05477` [✓] ·
GAE `1506.02438` [✓] · PPO `1707.06347` [✓] · 分布式 RL `1707.06887` [✓] ·
Emergence of Locomotion `1707.02286` [✓] · SAC `1801.01290` [✓] · IMPALA `1802.01561` [✓] ·
Dreamer `1912.01603` [✓] · 离线 RL 综述 `2005.01643` [✓] · TD-MPC `2203.04955` [✓]

LLM 后训练：InstructGPT `2203.02155` [✓] · DeepSeekMath/GRPO `2402.03300` [✓] · DeepSeek-R1 `2501.12948` [✓]

生成模型与表达型策略：DDPM `2006.11239` [✓] · DDIM `2010.02502` [✓] · Flow Matching `2210.02747` [✓] ·
Diffusion Policy `2303.04137` [✓] · ACT/ALOHA `2304.13705` [✓] · IDQL `2304.10573` [✓] ·
Q-Transformer `2309.10150` [✓] · DPPO `2409.00588` [✓] · OpenVLA `2406.09246` [✓] ·
π₀ `2410.24164` [✓] · π₀.₅ `2504.16054` [✓] · FQL `2502.02538` [✓] · ReinFlow `2505.22094` [✓] ·
Q-chunking (RL with Action Chunking) `2507.07969` [✓]

机器人系统与数据：RLPD `2302.02948` [✓] · SERL `2401.16013` [✓] · HIL-SERL `2410.21845` [✓] ·
MimicGen `2310.17596` [✓] · ManiSkill3 `2410.00425` [✓] · RLinf `2509.15965` [✓] ·
RLinf-VLA `2510.06710` [✓] · 模仿学习数据 scaling law `2410.18647` [✓]

> 核对方式：`export.arxiv.org/api/query?id_list=...` 批量取回，逐条比对标题。
> 上述编号与标题**全部对上**。1988/1989/1992 的经典工作（TD(λ)、Q-learning、REINFORCE、经验回放）
> 早于 arXiv，未做编号核对。AlphaGo (Nature 2016) 同理。

### 6.3 本目录内被引用的既有报告

- [`failure-data-in-imitation-2026-08.md`](./failure-data-in-imitation-2026-08.md)
- [`real-world-policy-evaluation-2026-08.md`](./real-world-policy-evaluation-2026-08.md)
- [`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md)
- [`act-diffusion-integration-2026-08.md`](./act-diffusion-integration-2026-08.md)
- [`vision-backbone-why-resnet18-2026-08.md`](./vision-backbone-why-resnet18-2026-08.md)
- [`team-division-strategy-2026-08.md`](./team-division-strategy-2026-08.md)
- [`paper_note/simfoundry-2606.28276.md`](./paper_note/simfoundry-2606.28276.md)

---

## 7. 本文的局限（必读）

1. **两篇论文（OPD、SAC Flow）我没能拿到精确数值**，因为它们的主结果只存在于图里。
   本文对这两篇的量化陈述**只限于作者在文字里写死的数**（97–99%、72%→91%、+0.37→+0.02、0.29、K∈{4,7,10}、130%、60%）。
   起草过程中抽取模型曾"从图上读数并排成表"，我**没有采用**那批数字。要精确数只能读 PDF 原图或跑代码。
2. **DSRL 与 RL-Co 的表格数字来自 arXiv HTML 全文的模型抽取，未逐格与 PDF 原文核对。**
   RL-Co 的 Table 1 我做了两次独立抽取且结果一致，并对每一行的 Avg 做了算术自检（发现一格不自洽，已在 §2.2.4 标注 ⚠）。
   DSRL 的表只做了一次抽取。
3. **所有代码仓库均为远端实抓（GitHub API + raw 文件），未在本地 clone、安装或跑通任何一个。**
   凡涉及运行时行为处，本文陈述的是"README/配置文件是这么写的"，不是"我验证过它能跑"。
4. **§5.2（DSRL-on-ACT）与 §5.4（overlap 指标搬到机器人）是我的推演，不是任何一篇论文的结论。**
   §5.7 已列出我能想到的反对意见。这两条在投入资源前应先做小规模验证。
5. **ICLR 2026 接收（SAC Flow）标 [~]**：来自 OpenReview 检索结果元数据，未打开 OpenReview 页面核实。
6. **RL-Co 的评测 trial 数论文未给**，因此其成功率的显著性无法判断；本文引用其数字时保留了原文的 ±。
7. **§3 的时间线是主线梳理，不是完整综述。** 有意省略了 model-based RL、多智能体、探索理论、
   RL 理论（regret bound）等分支——它们与本文四篇的关系较远。

---

## 8. AI 使用声明

本文由 Claude (Opus 5) 在 2026-08-21 编写。四篇论文的元数据、摘要、正文事实与全部 arXiv 编号
经 `arxiv.org` abs / HTML 页与 `export.arxiv.org` API 实时核实；代码仓库信息经 GitHub API 与
raw 文件实抓；`file:line` 经本地 grep 所得。未核实项已按 §0 的分级标注。
§5 的方案建议与 §3.5、§7 的判断为模型生成的分析，不构成已验证结论。
