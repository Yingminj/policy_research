# 本体感觉这条论证站不站得住：先从代码，再从你自己的数字

**撰写日期** 2026-09-03
**回答对象** 三个问题：(1) `act` 训练时是否显式使用数据集的 state / action；(2) `patch_policy` 的
`use_robot_state=false` 是否等于"只用图像预测未来动作"，扩散头到底需要收到什么才能解码；
(3) 在你已有的实验证据下，"本体感觉缺失"这条论证成立到什么程度。
**代码基线** `/home/kewei/YING/robot_data_platform/lerobot/src/lerobot/policies/{act,patch_policy}`
**证据基线** `experiment_report/patch_policy/*`、`experiment_report/act/act_delta-rel100-precision-analysis-2026-08.md`、
`eval_policy/runs/scripts_patch_policy_eval_fix/*.json`
**本报告新算的数字** §4.2 的 anchor share 与 §4.3 的损失权重分配，是从
`scripts_patch_policy_eval_fix/{new_state5,new_obs2,prev_act_head,act_baseline}.json`
的 `mae_per_horizon` 现场算出来的，之前的报告里没有。

---

## 0. 结论先行

**对三个问题的直接回答：**

1. **是。** `act` 显式用 `observation.state` 和 `action`，而且用在**三个不同的位置**（§2）。
   但它**不是**"(图像, 当前 state) → 未来动作"这么干净的映射：训练时解码器还额外拿到一个
   **由未来真值动作算出来的 VAE latent**，推理时这个 latent 被置零。训练与推理的输入不同构。
   另外 `act` 被硬性限制为 `n_obs_steps=1`（`configuration_act.py:148`），
   **它拿到的是单帧位置，没有任何速度**。

2. **是，而且比"只用图像"更严格。** `use_robot_state=false` 时
   `encode_observations` 返回的 token 里一个 state 都没有（`modeling_patch_policy.py:431-433`）。
   扩散头 `TransformerForDiffusion.forward` 的全部入参只有三样：**加噪动作块 `x_k`、扩散步 `k`、
   patch token memory**。没有 state、没有上一次动作、没有 episode 内时间、没有语言。
   **它要解码的目标是 16 维绝对关节角，而"手臂现在在哪"这个锚点必须从像素里回归出来。**

3. **这条论证的方向对，但它现在的表述是错的，而且你自己的数据已经把它证伪了一次。**
   正确的表述不是"策略缺少本体感觉"，而是：
   > **网络被迫在内部重建绝对锚点 `s_t`，而它重建不准。**
   > 消除这个负担有三条路——**当作输入给它**（本体感觉）、**从目标里减掉**（相对动作 / EEF）、
   > **在执行侧还回来**（Hermite 桥）。你三条都跑过了，**只有后两条有效**。

**一句话的取舍：** 继续写"patch_policy 差是因为没有本体感觉"这句话，会被
`new_state5` 这一个 checkpoint 当场推翻（打开开关，锚定误差 2.81× → 2.97×，没有改善）。
**要论证的应该是"目标参数化决定了本体感觉有没有用"，而这一点你有正反两组实验、
外加 2025–2026 两篇直接相关的文献支持。**

---

## 1. 三个问题的坐标：一张图

```
                       训练时喂进去的                        解码器实际收到的
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ ACT                                                                          │
  │   image(1 帧) ──ResNet18(可训)──► ~900 token ─┐                              │
  │   state(1 帧) ──Linear(16→512)──► 1 token ────┼──► TransformerEncoder ──► K/V │
  │   latent z ────Linear──────────► 1 token ────┘         ▲                     │
  │      ▲                                                 │                     │
  │      └─ 训练: VAE([cls, state, a_{t..t+H-1}]) ；推理: z := 0                  │
  │   50 个 learned query ──► ACTDecoder(cross-attn) ──► Linear ──► â (绝对关节角) │
  │   loss = L1(a, â) + β·KL                                                     │
  └──────────────────────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │ patch_policy, use_robot_state=false, head=diffusion                          │
  │   image(5 帧×3 相机) ──冻结 DINOv2 ViT-S/14──► 5×768 patch token ──► memory   │
  │   ( state: 不存在 )                                                          │
  │   x_k = 加噪的 54 步绝对动作块 ─┐                                             │
  │   k  = 扩散步 ──sinusoidal────┼──► TransformerDecoder(memory=patch) ──► ε̂    │
  │   loss = MSE(ε̂, ε)  ← 均匀覆盖全部 54 步                                     │
  └──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ACT：state 和 action 用在哪三个位置

全部在 `act/modeling_act.py`。

| # | 位置 | 用什么 | 何时 |
|---|---|---|---|
| 1 | `:465` `encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))` | `observation.state` → 1 个 encoder token（有专属 `encoder_1d_feature_pos_embed`） | 训练 + 推理 |
| 2 | `:413-418` `vae_encoder_input = [cls_embed, robot_state_embed, action_embed]` | `observation.state` + **未来真值动作序列** → VAE latent `z` | **仅训练** |
| 3 | `:145` `abs_err = F.l1_loss(batch[ACTION], actions_hat, ...)` | `action` 作为 L1 回归目标 | 训练 |

三件必须说清楚的事：

**2.1 ACT 没有 `use_robot_state` 这种开关。** 它读的是 `config.robot_state_feature`，
而这是 `configs/policies.py:131-138` 的一个**派生属性**——数据集里有 `observation.state` 就自动接上，
配置层面关不掉。所以你的全部 ACT 权重都带本体感觉，这不是一个可比较的变量。

**2.2 它不是 "(image, state) → future action"，中间还有一个 latent。**
训练时 `z ~ q(z | state, a_{t:t+H})`——**由答案算出来的**；推理时 `:456` 把它置零。
这是 CVAE 的标准结构：`z` 吸收演示的多模态，推理走先验均值。
后果是**推理时 ACT 输出的是"模态平均"的动作**。这一点在写论文时值得单列，
因为它和"L1 把稀疏夹爪跳变抹成斜坡"是同一个失效族。

**2.3 ACT 是无记忆的单帧策略。** `configuration_act.py:148-151` 在 `n_obs_steps != 1` 时直接 raise，
`observation_delta_indices` 返回 `None`。**它没有任何速度信息**——和 patch_policy 的
"5 帧被当成无序集合"（时间倒序只值 0.5–0.8%）在信息量上是同一个结局，
只是一个是设计如此，一个是学出来的。

> **顺带一个免费的发现：** `lerobot/utils/constants.py` 里根本没有 `OBS_VELOCITY` 这个 key。
> 你的数据集**带着 `observation.velocity`**（`eef-independent-eval` §3.1 就是用它做的指纹），
> 而**树里没有任何一个策略读它**。四条策略线全部没有速度输入，
> 这一条到今天为止没有被任何一份报告记录过。

---

## 3. patch_policy：`use_robot_state=false` 时，扩散头到底收到什么

### 3.1 开关改的三处（约 10 行）

| 位置 | 改动 |
|---|---|
| `configuration_patch_policy.py:256` | `use_robot_state: bool`；`:378` 只是一个校验 |
| `modeling_patch_policy.py:363-369` | `tokens_per_frame += int(use_robot_state)`；建 `state_projector = MLP(16 → 384)` |
| `modeling_patch_policy.py:431-433` | `torch.cat([patch_tokens, state_token], dim=2)`，每帧 768 → 769 |

**没有改 loss、没有改 mask、没有改 decoder。** 关掉时 `encode_observations` 返回
纯视觉的 `(B, 5, 768, 384)`，`forward` 里 `cond = rearrange(patch_tokens, "b t p d -> b (t p) d")`
就是全部条件。

### 3.2 扩散头的完整签名

```python
# modeling_patch_policy.py:270
def forward(self, sample: Tensor, timestep: Tensor | int, cond: Tensor) -> Tensor:
    # sample: (B, 54, 16)  加噪的动作块
    # timestep: 扩散步 k
    # cond:   (B, 5*768, 384)  patch token
```

memory 的布局是 `[1 个 timestep token] + [T*P 个 patch token]`，
decoder 是 54 个 learned position 的 action token，带因果自注意力 + block-causal memory mask。
**没有第四个入口。** 所以要回答你的问题——扩散头需要收到什么才能解码未来动作：

> **形式上它需要能表示 `∇ log p(A_{t..t+53} | O)`。**
> 而 `A` 是绝对关节角，可以逐维分解成
> **`A_{t+k} = s_t + Δ_k`**（锚点 + 位移）。
> 于是 `E[A_{t+k} | O] = E[s_t | O] + E[Δ_k | O]`。
> **第一项是一个与 k 几乎无关的常数**，它的误差原封不动地加到 chunk 的每一帧上。
> 关掉本体感觉，网络就必须自己算 `E[s_t | O]`；打开它，这一项**可以**是恒等映射，
> 但**没有任何东西强迫它是**——这是 §5 的全部内容。

### 3.3 一个之前没被提过的结构细节：54 步里有 4 步是过去

`configuration_patch_policy.py:347-354`：`horizon = n_obs_steps + action_chunk_size - 1`。
`action_delta_indices = range(1 - n_obs_steps, action_chunk_size)`，
所以 `n_obs_steps=5, chunk=50` 时目标是 **54 步，下标 0–3 是 t-4 … t-1，即已经执行过的动作**。
`predict` 里 `start = n_obs_steps - 1 = 4`，只取下标 4 之后的 50 步；
`predict_action_chunk` 再截到 `n_action_steps=8`，也就是**下标 4–11**。

而 `forward` 里 `loss = F.mse_loss(noise_pred, target)`（`:472`）是**对全部 54 步均匀求和的**。
量化后果见 §4.3。

---

## 4. 数字：锚点到底占多少

### 4.1 量纲（引自 `no-proprioception` §2，本报告不重测）

| 量 | 弧度 |
|---|---:|
| `\|action_t − state_t\|`（演示动作与当前位姿的距离） | 0.0140–0.0156 |
| `\|action_{t+8} − state_t\|`（一个部署 waypoint 的全部运动量） | **0.032** |
| 纯视觉线性读出 `s_t` 的误差（ridge, mean-pool） | 0.121 |
| 策略自己学出来的非线性读出 `s_t` 的误差 | **0.066**（远留出）/ 0.040（近留出） |

**要求 0.032，能做到 0.040–0.066。误差比要发出的指令本身还大。** 这一条是整条论证的基石，
而且它**不依赖任何架构假设**——它是"从这套相机里能不能看出手臂在哪"的物理上限。

### 4.2 锚点占 chunk 误差的多少（**本报告现算**）

用 `scripts_patch_policy_eval_fix/*.json` 的 `mae_per_horizon`（50 步，2007 anchor，
`batch_success_53_eval_data`）。记 `b = mae_per_horizon[0]`，在"锚点是一个跨全 chunk 的常数偏置"
这个假设下，锚点占平方误差的份额 = `50·b² / Σ_k mae_k²`：

| run | 本体感觉 | `b` = MAE@1 | MAE@50 | **anchor share（上界）** | `b` / 演示自身的 0.01556 |
|---|---|---:|---:|---:|---:|
| `new_state5` | **✓ 开** | 0.03961 | 0.08263 | **39.6 %** | **2.55×** |
| `new_obs2` | ✗ 关 | 0.04022 | 0.08489 | 39.9 % | 2.58× |
| `prev_act_head` | ✗ 关 | 0.04107 | 0.08566 | 39.7 % | 2.64× |
| **`act_baseline`（ACT）** | **✓ 恒开** | **0.02529** | 0.07180 | **22.6 %** | **1.63×** |
| *null* `hold_state` | — | 0.01556 | — | — | 1.00× |

> **这张表的正确读法（三条，第三条最重要）：**
>
> 1. **39.6 % vs 39.9 %。** 打开 `use_robot_state` 之后，锚点占误差的份额**一点没变**。
>    这不是"收益小"，是"零"。
> 2. **ACT 的 22.6 % 是这套硬件上本体感觉能买到的上界。** ACT 有两条 state 通路、
>    backbone 可训、mean/std 归一化——它把锚点份额从 40% 压到 23%，**但没有压到 0**。
>    **一个全程带本体感觉的策略，它 chunk 的第一步仍然比演示动作偏 1.63 倍。**
>    所以"给它本体感觉"这条路的天花板是 40%→23%，不是 40%→0。
> 3. **39.6% 是上界，不是实测值。** 实测的完美重锚收益（`policy_reanchored` / Hermite 桥）
>    是 **18–30%**（`no-proprioception` §4：0.1137→0.0928；`state-and-window` §6：h=8 上 −26~−30%）。
>    差额是因为锚点误差与形状误差不严格正交。**写论文时用 18–30%，
>    §4.2 这一列只用来做"开关前后没变"的对照。**

### 4.3 损失权重落在哪（**本报告现算**）

| | `n_obs_steps=5` | `n_obs_steps=2` |
|---|---:|---:|
| 训练目标步数 `horizon` | 54 | 51 |
| 其中**已经执行过的过去动作**（下标 0..T-2） | 4 步 = **7.4 %** | 1 步 = 2.0 % |
| **部署真正执行的窗口**（下标 T-1 … T-1+8） | 8 步 = **14.8 %** | 8 步 = 15.7 % |
| 训练完全没有机会影响机器人的步数 | 46 步 = **85 %** | 43 步 = 84 % |

**85% 的梯度花在机器人永远不会执行的动作上，7.4% 花在已经过去的动作上。**
与之对照，在真实部署窗口（`deploy_config_patch_policy.yaml`，`n_action_steps: 8`，
桥接 `min(40,8)=8`）里，**策略的全部贡献只是 `chunk[7]` 这一个端点**，
而那一个点的正确性几乎完全由锚点决定（§4.1：一个 waypoint 的运动量 0.032 < 锚点误差 0.040）。

> 这是训练目标与部署口径的直接错配，**和本体感觉是两回事，可以叠加**。
> 它也解释了 `state-and-window` §5 那张最刺眼的表：h=8 上四个 patch_policy 权重全部输给
> `hold_state`，而 h=50 上它们都赢——**你在优化一个不被执行的 horizon。**

---

## 5. 反驳：这条论证哪里不成立

我把你的隐含命题写成可证伪的形式，逐条对照你自己的数据。

### 命题 A：「patch_policy 精度差是因为 `use_robot_state=false`」 —— **已被你自己的实验证伪**

`new_state5` 打开了这个开关。锚定比值 **2.97×**，而关掉它的 `new_obs2` 是 **2.81×**、
`prev_act_head` 是 **3.05×**（`state-and-window` §7.1）。按 §4.2 的口径是 2.55× vs 2.58×。
**打开之后没有变好。** 单 checkpoint 内的干预探针更直接：
换掉整个 16 维 state → 输出只移动 **6.7%**；置零 → **5.1%**；而换掉图像 → **136%**。

> 必须同时说清的方法学限制：`new_state5` ↔ `new_obs2` **不是受控消融**
> （`n_obs_steps` 同时从 5 变到 2）。**但 §7 的干预探针是单 checkpoint 内部做的，那部分归因是干净的**，
> 而它给出的 6.7% 已经足以否掉命题 A。

### 命题 B：「state token 被 768 个 patch token 淹没了」 —— **已被实测证伪**

`measure_state_token.py`：state 槽位拿到 **1.12% 的 cross-attention 质量 = 均分份额的 8.65 倍**；
范数比 **0.76**，与 patch token 同量级。**模型主动多看它 8.65 倍，然后不用它。**
推论：把 state token 复制成 K 份、或者放大它的尺度，都不会有用。

### 命题 C：「本体感觉是纯粹的增益，加上总没坏处」 —— **你的 ACT 数据里有反例**

`act_delta-rel100-precision-analysis-2026-08.md` §结论：

| | in-sample | 真留出集 | 泛化倍数 |
|---|---:|---:|---:|
| ABS（绝对动作，**有** state） | **0.00583** | 0.15379 | 26.4× |
| REL（相对动作，**有** state） | 0.00794 | **0.08417** | 10.6× |

**绝对动作在 in-sample 上更准（0.0058 < 0.0079），在留出集上差 1.83 倍。**
这是 copycat / causal confusion 的教科书形状：演示里 `a_t ≈ s_t`（差 0.014 rad），
`state` 因此是 `action` 的一个近乎完美的**混淆相关量**；
监督学习会优先抓这条捷径，而它在分布移动下崩掉。
**本体感觉在绝对动作参数化下不是免费的，它是一个陷阱。**（文献见 §6.1。）

### 命题 D：「真正有效的杠杆是把锚点从目标里拿掉」 —— **成立，而且是你唯一被证实有效的干预**

同一把正运动学尺子（`eef-independent-eval` §5、§6）：

| | 锚定比值 `\|chunk[0]−s\|` / 演示 | 位置误差 |
|---|---:|---:|
| `pp_joint_state5`（关节，开 state） | 2.663 | 15.55 mm |
| `pp_eef`（EEF） | **1.399** | **11.28 mm（−27%）** |
| `act_joint_361`（关节，开 state） | 1.916 | 11.28 mm |
| `acteef_505`（EEF） | 1.345 | 10.23 mm（−9%） |

**换动作空间对 patch_policy 值 −27%，对 ACT 只值 −9%（姿态还变差 4%）。**
再加上 §5 命题 C 的 ACT 相对动作 1.83×——**两条独立证据都指向目标参数化，而不是输入。**

> 但同一张表里还有一行必须写进论文，否则会被审稿人抓住：
> **`pp_eef` 的 11.28 mm 与 `act_joint_361` 的 11.28 mm 完全相同。**
> 换动作空间把 patch_policy 抬到了 ACT 在关节空间早就在的位置，**没有抬过去**。
> 所以正确的论断是"目标参数化解释了 patch_policy 相对 ACT 多出来的那部分劣势"，
> **不是**"目标参数化解决了精度问题"。剩下的 1.27× 天花板是三条策略共有的，还没被触碰。

### 综合：应该写进论文的那句话

> 决定策略精度的不是"有没有本体感觉"这个二值输入，而是
> **"绝对锚点 `s_t` 是否必须在网络内部被重建"**。
> 在绝对动作参数化下，`s_t` 既是必须重建的量（视觉重建上限 0.066 rad > 一个 waypoint 的
> 0.032 rad），又是一个会诱发记忆化的混淆相关量（in-sample 0.0058 / held-out 0.1538）。
> 把它作为输入加进去，两个问题都不解决：梯度不要求网络走这条恒等映射，
> 实测注意力找得到它（8.65× 均分份额）却不用它。
> 把它从目标里减掉（相对动作 / EEF），锚定比值 2.66 → 1.40，
> 而在执行侧把它还回来（Hermite 桥）值 26–30%。

---

## 6. 文献

**核实方式：** 以下每一篇都用 WebFetch 逐篇抓取 `arxiv.org/abs/<id>` 摘要页核对
（标题、完整作者列表、v1 日期、摘要原文），未使用任何来自模型记忆的引用。
`Qwen-VLA (2605.30280)` 的"本体感觉注入消融只值 +0.7~1.3 pp"这条在摘要页**核不到**，
**因此不引用**。

### 6.1 本体感觉作为混淆相关量（支持 §5 命题 C）

**Causal Confusion in Imitation Learning** · Pim de Haan, Dinesh Jayaraman, Sergey Levine ·
arXiv:1905.11979 · v1 2019-05-28
> "it leads to a counter-intuitive 'causal misidentification' phenomenon: **access to more
> information can yield worse performance**."

直接对上你的 ABS in-sample 0.0058 / held-out 0.1538。**"多给一路输入"本身不是增益论证。**

**Fighting Copycat Agents in Behavioral Cloning from Observation Histories** ·
Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman, Yang Gao · arXiv:2010.14876 · v1 2020-10-28
> "a common instance of this causal confusion occurs in partially observed settings when
> **expert actions are strongly correlated over time**: the imitator learns to cheat by
> predicting the expert's previous action, rather than the next action."

你的语料里 `|a_t − s_t| = 0.014`，`|a_{t+8} − s_t| = 0.032`——**动作的时间自相关极强**，
正是这篇描述的情形。

### 6.2 去掉本体感觉反而更好（对你的论证是**反例**，必须处理）

**Do You Need Proprioceptive States in Visuomotor Policies?** ·
Juntu Zhao, Wenbo Lu, Di Zhang, Yufeng Liu, Yushen Liang, Tianluo Zhang, Yifeng Cao, Junyuan Xie,
Yingdong Hu, Shengjie Wang, Junliang Guo, Dequan Wang, Yang Gao · arXiv:2509.18644 · v1 2025-09-23
> "this common practice makes the policy **overly reliant on the proprioceptive state input**,
> which causes overfitting to the training trajectories and results in poor spatial generalization."
> 去掉本体感觉后真机成功率 "from 0% to 85% in height generalization and from 6% to 64% in
> horizontal generalization"。

**这篇是你的论证最大的威胁，也是最好的支撑，取决于你怎么用它。**
它的两个前提被摘要明确写死：**(a) 相对 EEF 动作空间；(b) 双广角腕部相机提供完整任务可观测性。**
你的设定**两条都不满足**：绝对关节角目标，顶视相机几乎看不到手臂。
**所以这篇预测的正是你观测到的失败——但归因不是"缺 state"，而是"绝对动作 + 视觉不充分"。**
这恰好就是 §5 的综合论断，可以直接引来支撑；如果论文写成"缺本体感觉导致精度差"，
这篇会被审稿人拿来当反例。

### 6.3 本体感觉的**编码方式**才是变量

**When Absolute State Fails: Evaluating Proprioceptive Encodings for Robust Manipulation** ·
Maxime Alvarez, Ryo Watanabe, Paul Crook, Afshin Zeinaddini Meymand, Suvin Kurian,
Pablo Ferreiro, Genki Sano · arXiv:2605.13067 · v1 2026-05-13
> "Through a systematic study of joint representations, we find that a simple
> **episode-wise relative frame** provides the best trade-off between task performance and
> robustness, outperforming the baselines in extensive real-robot experiments."

**结论与本报告 §5 完全同向：变量是参数化，不是有无。** 而且它给了一个你还没试过的第三档——
**episode 级相对帧**（以 episode 起始位姿为参考系），介于绝对与逐帧相对之间。

### 6.4 两个基线的原始出处（用于论文的方法学描述）

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)** ·
Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn · arXiv:2304.13705 · v1 2023-04-23

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion** ·
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel,
Russ Tedrake, Shuran Song · arXiv:2303.04137 · v1 2023-03-07
> 摘要点名的三项技术贡献之一是 "**receding horizon control**"。

这一条值得单引：**Diffusion Policy 明确把"预测长 horizon、只执行前一小段"当作设计**，
而你的 §4.3 显示 patch_policy 的实现把 85% 的损失权重放在了不执行的段上，
**且训练损失对全部 54 步均匀加权**——原文的 receding horizon 是一个执行策略，
这里被当成了一个损失定义。

**Patch Policy: Efficient Embodied Control via Dense Visual Representations** ·
Gaoyue Zhou, Zichen Jeff Cui, Ada Langford, Bowen Tan, Yann LeCun, Lerrel Pinto ·
arXiv:2607.18236 · v1 2026-07-20（元数据取自 `paper_note/patch-policy-2607.18236.md`，本次未重抓）

---

## 7. 建议

按「证据强度 × 代价」排序。前两条不需要训练。

### 7.1 【0 GPU·h，今天就能做】把 §4.2 的上界换成实测的分解

现有 harness 已经有 `--trace` 导出（`eval_policy/runs/2026-09-03_metric_cdf/cdf_probe.py` 用的就是它）。
加一个 **有符号** 的逐步偏置：`bias_k = mean(pred_k − gt_k)`，以及去掉 `bias_0` 之后的残差曲线。
这给出论文需要的那张图：**误差 = 常数锚点 + 随 horizon 增长的形状项**，两项分开画。
现在报告里全部是 MAE（无符号），**证明不了"这是一个偏置"而不是"这是方差"**——
审稿人会问这一句，而你手上的 npz 已经够回答。

### 7.2 【~9 GPU·h，2 个 run】补齐 2×2，这是论文真正缺的那次实验

现有的两格：绝对×state开（`new_state5`）、绝对×state关（`new_obs2`）。
缺的两格：**相对×state开、相对×state关**。移植 `policies/act_delta/` 的
`use_relative_actions` 处理器（`prepare_relative_stats.py` 已经算过 pre-gate：
`batch_success_361` / chunk 50 上中位收缩 **0.529**，J7 收缩最多，夹爪维 ≈0.99 → 保留
`--exclude-joints gripper`）。基座用 `new_obs2` 的配置（`n_obs_steps=2`，4.5 h/run）。

**这次实验的判别力：**

| 结果 | 结论 |
|---|---|
| 相对×关 ≈ 相对×开 ≫ 绝对×任意 | 锚点是唯一问题，本体感觉确实冗余 → 引 §6.2 |
| 相对×开 ≫ 相对×关 | 本体感觉有用，但只在相对参数化下被激活 → 这是最强的论文故事 |
| 两格都没动 | 瓶颈不在参数化，回到数据侧（§8） |

**三种结果都是可发表的，这才是值得跑的实验。** 而且它是 `patch_policy-matrix.md` §4
那张"空的格子"表里唯一由理论预测会动的一格。

### 7.3 【0 GPU·h，一个配置项】先修 h=8 这个窗口，否则前两条的结论无法上机验证

`deploy_config_patch_policy.yaml` 的 `n_action_steps: 8` + 桥接 `min(40,8)=8`
= **执行窗口内每一步都是 S 曲线，策略只贡献一个端点**，同时推理 duty 是 1.11–2.23×（跑不动）。
`state-and-window` §10.5 已经建议改成 50。**在这个窗口没修之前，任何离线改善都不会在机器人上显现**，
因为你测的量（chunk 的形状）和执行的量（一个端点）不是同一个东西。

### 7.4 【0 GPU·h，一个配置项，标记为**未验证假设**】`prediction_type: "sample"`

`configuration_patch_policy.py` 的 `prediction_type` 默认 `"epsilon"`。
锚点是动作块里的**直流分量**（跨 54 步近似常数）。ε-参数化下，
`x̂_0 = (x_k − √(1−ᾱ_k)·ε̂)/√ᾱ_k`，在高噪声步 `√ᾱ_k → 0` 时对 `ε̂` 的误差有巨大放大，
而采样轨迹的直流分量恰恰是在早期（高噪声）步确定的、且不可恢复。
**这是一个"为什么锚点特别难学"的结构性候选解释，但本报告没有测它**，
只值一个配置项 + 一次重训，可以和 7.2 的某一格搭车。**不要在论文里当成机制写，除非测了。**

### 7.5 【独立于以上全部，而且它才是"抓取准确率低"的直接来源】夹爪通道

`no-proprioception` §8：夹爪**在训练集内**就比"不改变握持状态"差 **2.8 倍**
（0.0225 vs 0.0081），held-out 差 3.3 倍。**这一列没有泛化问题，它在模型背下来的数据上就是错的。**
原因是量纲：跳变只占 4.5% 的帧，L1/MSE 的最优解就是把阶跃抹成斜坡；
而夹爪是**唯一不被 Hermite 桥接、原样发给机器人的通道**。

这条 2026-08 就提了，`patch_policy-matrix.md` 显示至今**一个权重都没跑**。
两个做法（并行，不冲突）：给夹爪列单独的损失权重；或把夹爪从回归改成开/合二分类。
**如果你要回答的问题是"为什么抓不准"，这一条的因果链比本体感觉短得多。**

### 7.6 免费的、树里已经有数据的一条：喂速度

§2 的发现：数据集有 `observation.velocity`，`lerobot/utils/constants.py` 里没有对应的 key，
**没有任何策略读它**。而两份探针都显示策略**没有速度信息**（时间倒序值 0.2–0.8%）。
patch_policy 加一个 velocity token 是 §3.1 那三处改动的复制粘贴（约 10 行）。
**但按 §5 命题 A/B，我预期它同样不会有用**——除非先做了 7.2 的相对动作。
所以：**7.2 之后再考虑，不要现在做。**

### 7.7 不要再花时间的地方（已实测无效，引自现有报告）

| 旋钮 | 实测 |
|---|---|
| `action_head`（act ↔ diffusion ↔ vqbet） | 差 <5%，被锚点支配 |
| `use_robot_state` 再翻一次 | 6.7% 干预位移，锚点份额 39.6% vs 39.9% |
| 复制 state token / 放大它的尺度 | 注意力 8.65× 均分份额、范数 0.76，不是"淹没" |
| 提高 `n_obs_steps` | 时间顺序只值 0.2–0.8% |
| 给 ACT 基线加训练数据 | 363/505/535 非单调，波动 3–7% > patch_policy 与 ACT 的全部差距 |

---

## 8. 我需要你确认的三件事

1. **这份论证是给论文的哪一节？** 如果是"方法动机"，§5 的综合论断可以直接用；
   如果是"实验分析"，那么 §7.2 的 2×2 是必须补的——现在的证据链里
   "相对动作对 patch_policy 有效"只有 EEF 这一条间接证据，而 EEF 同时换了三个变量
   （动作空间、`n_obs_steps`、训练集，见 `matrix.md` §2）。

2. **1.27× 那个共有天花板，是否在本文的范围内？** 三条策略在远留出集上都只比
   `hold_state` 好 20–30%。如果论文要下"我们解决了 X"的结论，
   审稿人会问剩下的 70% 是什么。`no-proprioception` §11 猜是数据侧
   （363 episode、单场景、`act-policy-leans-on-background-pixels`），**但没有测过**。
   要不要把它列成 limitation，还是要补一次场景多样性的测量？

3. **有没有真机成功率？** 全部现有数字都是开环片段指标。§5 命题 D 那句
   "EEF 把 patch_policy 抬到 ACT 早就在的位置"如果有一组真机成功率佐证，
   论证强度会完全不同；没有的话，`state-and-window` §9 那条边界
   （"输给 null 不等于机器人会静止"）必须原样写进论文。

---

## 9. 这份报告的边界

- **§4.2 的 anchor share 是一个上界**，建立在"锚点误差跨 chunk 恒定且与形状误差正交"这个假设上。
  实测的重锚收益是 18–30%（§4.2 读法第 3 条）。**论文里用 18–30%。**
- **§4.3 的损失权重是纯代码算术**（`horizon = n_obs + chunk − 1`、均匀 MSE、
  `start = n_obs − 1`、`n_action_steps=8`），没有实验支持"改变加权会有用"这一步。
- **§7.4 是未验证假设**，明确标注了。
- 本报告**没有跑任何新的模型推理**，全部数字要么引自现有报告，要么从现有 JSON 现算。
- 引用的 5 篇论文（1905.11979 / 2010.14876 / 2509.18644 / 2605.13067 / 2303.04137 / 2304.13705）
  均已逐篇 WebFetch 核对；2607.18236 的元数据取自本仓库既有笔记，本次未重抓。
