# `patch_policy` 抓取准确率低的原因：策略没有本体感觉，连"手臂现在在哪"都答错

**Checkpoint** `/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-20_21-31-19-689756/run/checkpoints/200000/pretrained_model`
**Policy** `patch_policy` — `action_head=diffusion`，`vision_encoder=dino_patch`（冻结 DINOv2 ViT-S/14），
`n_obs_steps=5`，`action_chunk_size=64`，`n_action_steps=32`，**`use_robot_state=False`**，
200 000 步，batch 16（10.64 epoch），常数 LR 5.5e-5，无 scheduler，最终 loss 0.014
**对照 checkpoint** `..._2026-08-20_21-21-58-821581` — 同数据集、`action_head=act`、`action_chunk_size=50`、
**同样 `use_robot_state=False`**
**训练集** `tidy_up_stationery_le/batch_success_361`（363 episodes，300 689 帧，无验证集划分）
**部署配置** `lerobot_vlahost/workflows/robot_interaction/deploy_config_patch_policy.yaml`，`n_action_steps: 8`
**测量日期** 2026-08-24，mgmt01（RTX 4090），`/opt/robot-platform/train-venv`（训练该 checkpoint 的解释器）
**脚本** `../test_scripts/scripts_patch_policy_probe/`

---

## 1. 结论

这个 checkpoint **在训练集内就比"什么都不动"差**。它 chunk 第一帧的误差是 0.0293 rad，
而 `hold_state`（整段输出当前实测关节角）是 0.0135 rad——在它最该确定的那一帧上差 2.2 倍。
held-out 上是 0.0667 vs 0.0155，差 4.3 倍。

误差曲线几乎是**平的**（0.0293 → 0.0398，横跨 64 帧），这是**常数位姿偏置**的形状，
不是轨迹外推误差。把 chunk 平移到实测位姿上（`policy_reanchored`），偏置就消失了。

原因是配置，不是架构：**两个 patch_policy checkpoint 都是 `use_robot_state=False`**。
这是 reference 的默认值，port 的 README 把它列为 *Deviation 5*。策略因此**完全没有本体感觉**，
必须从三路相机回归 16 个绝对关节角。而这套硬件上视觉单独定位位姿的上限大约就是 0.07 rad
（§5）——顶视相机几乎看不到手臂，两个腕部相机是自我中心视角。

**致命的是量纲对比**：一个部署 waypoint 覆盖的全部运动量
（`|action_{t+8} − state_t|`）只有 **0.032 rad**，而定位误差是 **0.067 rad**。
**误差比要发出的动作本身还大一倍。**

第二个独立缺陷（§6）：`n_obs_steps=5` 的历史被当成**无序集合**读。把 5 帧在时间上倒过来，
输出只变 0.8%。模型没有任何速度信息，所以即使去掉位姿偏置，它在执行窗口里预测的位移
也不比"位移为零"更好（0.0436 vs `hold_state` 0.0411）。

**用户已排除的两项，排除得不对**：换 action head 只改变 <5%（§7），
调 `n_action_steps` 8/16/32 更不可能有用——一个在第 1 帧就存在的常数偏置，
和执行多少帧无关。两个 head 之所以表现一样，正是因为**它们共用同一个 `use_robot_state=False`**。

---

## 2. 量纲参照：读下面任何一个数字之前先看这张表

在本语料上实测（弧度；两路夹爪通道是 0–1）：

| 量 | 数值 |
|---|---:|
| `\|action_t − state_t\|`（动作基本是当前位姿的副本） | 0.012 |
| `\|action_{t+8} − state_t\|`（一个部署 waypoint 覆盖的全部运动） | **0.032** |
| episode 内 `std(state)` | 0.324 |
| 夹爪发生跳变的帧占比 | 4.5 % |
| 每 episode 每个夹爪的开合次数 | 2.49 |

**waypoint 误差一旦超过 0.032 rad，这条指令就不携带任何运动信息**——误差比要求的位移还大。

---

## 3. 离线指标：训练集内就输给"什么都不做"

`eval_patch_chunk.py`，与 ACT / ACT-DiT 报告同一套指纹去污染和同一个误差累加器，
所以下面的数字和那两份报告可直接比较。`hold_state` = 整段 chunk 输出当前实测关节角。

### 3.1 训练集内（`batch_success_361`，15 035 anchors）

| horizon | 1 | 8 | 10 | 25 | 64 |
|---|---:|---:|---:|---:|---:|
| **policy** | **0.0293** | 0.0314 | 0.0306 | 0.0329 | 0.0398 |
| `hold_state`（什么都不做） | **0.0135** | 0.0362 | 0.0280 | 0.0759 | 0.1062 |

policy 的逐帧曲线：`0.0293 0.0294 0.0297 0.0300 0.0303 0.0307 0.0311 0.0314 …`
`hold_state` 的逐帧曲线：`0.0135 0.0168 0.0200 0.0232 0.0265 0.0297 0.0330 0.0362 …`

**读法**：`hold_state` 线性增长（手臂在动，不动就越来越错），这是正常的。policy 几乎是水平线——
它第 1 帧的误差（0.0293）已经是第 64 帧误差（0.0398）的 74%。**误差的绝大部分与 horizon 无关，
是一个常数。** 这台模型对"下一时刻手臂在哪"的回答，在它背下来的数据上都比直接读取当前位姿差 2.2 倍。

### 3.2 held-out（`batch_1–4` 指纹过滤后 263 episodes，8 897 anchors）

| horizon | 1 | 8 | 10 | 64 | vs. null |
|---|---:|---:|---:|---:|---:|
| **policy** | 0.0667 | 0.0768 | 0.0730 | 0.1137 | **1.04×** |
| **policy_reanchored** | 0.0155\* | 0.0436 | 0.0350 | 0.0928 | 1.27× |
| `hold_state` | 0.0155 | 0.0411 | 0.0320 | 0.1178 | — |
| `train_mean` | 0.3080 | — | 0.3081 | 0.3089 | 0.38× |

\* 按构造相等：`policy_reanchored` 把第 0 帧强行设成实测位姿，所以 horizon 1 的对比是空的，
从 horizon 2 起才有信息。

**整段 chunk 上，policy 只比"什么都不做"好 4%。** 作为对照，同数据集上
ACT 是 1.21×、ACT-DiT 是 1.27×（见 `../scripts_act_eval_test/README.md` 和
`../act_dit/act_dit-encoder-collapse-2026-08.md`）。**patch_policy 是三者里最差的。**

---

## 4. 机制：误差是一个常数位姿偏置，去掉它就回到 ACT 的水平

`policy_reanchored` 把预测 chunk 整体平移到实测位姿上：`state + (pred − pred[0])`。
它只去掉常数偏置，完全保留 chunk 的形状。

| | 整段 MAE | vs. null |
|---|---:|---:|
| policy（held-out） | 0.1137 | 1.04× |
| **policy_reanchored（held-out）** | **0.0928** | **1.27×** |
| ACT baseline（held-out，另一份报告） | 0.1355 | 1.21× |
| ACT-DiT（held-out，另一份报告） | 0.1289 | 1.27× |

**去掉位姿偏置之后，patch_policy 精确落回 ACT-DiT 所在的 1.27× 档。**
也就是说：patch_policy 相对 ACT / ACT-DiT 多出来的那部分劣势，**全部**由缺失的本体感觉解释；
剩下的 1.27× 是这三个策略共有的、与架构无关的天花板。

训练集内的同一对比给出一个补充信号（3 007 anchors）：

| horizon | 1 | 10 | 64 |
|---|---:|---:|---:|
| policy | 0.0292 | 0.0306 | 0.0395 |
| policy_reanchored | 0.0136 | 0.0253 | 0.0472 |
| `hold_state` | 0.0136 | 0.0282 | 0.1065 |

训练集内 re-anchor 在短 horizon 有用、在长 horizon **反而更差**（0.0472 > 0.0395）——
背下来的绝对轨迹在远处比实测位姿更好用。held-out 上正好相反。这是过拟合的形状。

---

## 5. 视觉单独能把位姿定位到多准？约 0.07 rad

`probe_state_decodability.py`，冻结 DINOv2 的 patch token → 当前 `observation.state` 的线性读出，
3 000 训练帧拟合、1 000 held-out 帧评测：

| 读出方式 | held-out MAE (rad) |
|---|---:|
| `train_mean`（预测训练集均值） | 0.3069 |
| **每相机 mean-pool + ridge** | **0.1209** |
| 全 patch 线性层（3×256×384 特征） | 0.3854 —— **未收敛，不作为证据** |

全 patch 那一档在 n=3 000 对 294 912 维特征上欠定，比均值预测还差，是拟合失败而不是发现；
这里如实列出并排除。

结论用两个数就够：**冻结编码器里确实带着位姿信息**（线性 mean-pool 就比均值好 2.5 倍），
但线性读出只到 0.121 rad；而**策略自己训练出来的非线性读出达到 0.0657 rad**
（§6 的 `first_action_vs_state`）。所以 0.066 rad 就是这套相机布置上、这个编码器上，
视觉单独定位绝对关节角的实测水平。

对照 §2：需要的是 0.032 rad。**差一倍。这个缺口不可能靠调 head 或调 chunk 长度补上。**

---

## 6. 第二个独立缺陷：5 帧历史被当成无序集合

`probe_patch_conditioning.py`，held-out `batch_3`，48 个 anchor，**每次干预前重新播同一个随机种子**，
所以扩散采样的初始噪声和每一步的噪声都完全一致，测到的位移全部来自干预本身。
分母 `interframe_scale` = 两个不相干 anchor 的 chunk 差异 = 0.310 rad。

| 干预 | Δ chunk (rad) | 占 interframe |
|---|---:|---:|
| 三路相机全换成另一 anchor 的图 | 0.3760 | **121 %** |
| 全部像素置 0.5（该模型的无条件先验） | 0.3354 | 108 % |
| 只换 `wrist_R` | 0.2001 | 65 % |
| **patch 网格塌成每帧 1 个 token**（推理时，Table 4 的 n=1 档） | 0.1963 | 63 % |
| 只换 `wrist_L` | 0.1428 | 46 % |
| 只换 `top` | 0.0983 | 32 % |
| 四分之一分辨率（224 → 56 → 224） | 0.0574 | 19 % |
| **冻结历史**（前 4 帧 := 最新帧） | 0.0160 | **5.2 %** |
| **时间倒序**（5 帧顺序翻转） | 0.0024 | **0.8 %** |

三点读法：

1. **编码器是健康的**，这不是 ACT-DiT 那种塌缩。换图移动 121%，视觉通路完全活着。
   `act_dit` 的对应数字是 0.0%（`../act_dit/act_dit-encoder-collapse-2026-08.md` §3）。
   **两份报告诊断的是两个完全不同的故障。**
2. **稠密网格确实在被用**（塌成 1 token 移动 63%），所以 patch_policy 的前提在这个任务上没有失效。
   但四分之一分辨率只移动 19%，说明精细空间细节的贡献比预期小得多。
3. **历史是死的。** 时间倒序只移动 0.8%——模型把 5 帧读成一个**无序集合**，
   因此**没有任何速度信息**。这解释了 §3.2 里那个现象：即使 re-anchor 掉位姿偏置，
   执行窗口内的预测位移仍然不比"位移为零"更好（0.0436 vs 0.0411）。
   `cond_pos_emb` 在 `1 + T·P` 个 memory slot 上是存在的，顺序**可以**被表示——只是没被用上。

同一次运行直接测了位姿定位：

| | rad |
|---|---:|
| policy chunk 第一帧 vs 实测位姿 | **0.0657** |
| 示教动作第一帧 vs 实测位姿（应有的水平） | 0.0182 |
| 比值 | **3.61×** |

---

## 7. 换 action head 没有用（控制实验）

用户已试过 act 和 diffusion 两个 head。两个 checkpoint 在同一 held-out 集（`batch_3`）上：

| | @1 | @10 | 整段 | vs. null |
|---|---:|---:|---:|---:|
| `action_head=diffusion`（chunk 64） | 0.0667 | 0.0730 | 0.1137 | 1.04× |
| `action_head=act`（chunk 50） | 0.0702 | 0.0768 | 0.1116 | 1.03× |
| diffusion，re-anchored | — | 0.0350 | 0.0928 | 1.27× |
| act，re-anchored | — | 0.0322 | 0.0824 | 1.31× |

**两个 head 的位姿偏置一模一样（0.067 vs 0.070，差 5%）。**
换 head 改变的量级，比偏置本身小一个数量级。这不是"head 都不行"，
而是**两次实验都带着同一个 `use_robot_state=False`，被同一个偏置支配**。

---

## 8. 部署侧：机器人实际执行的轨迹，比站着不动还差

前面所有数字量的都是**模型发出的 chunk**。机器人执行的不是它。

`deploy_config_patch_policy.yaml` 设 `n_action_steps: 8`，于是
`lerobot_vlahost/src/lerobot/rollout/strategies/core.py::send_next_action_chunk` 里
`bridge_steps = min(40, 8) = 8`：**前 14 个手臂关节的每一帧都被改写成一条三次 Hermite 曲线**，
起点是实测位姿、起始速度 0，终点是 `chunk[7]`。模型每次推理只贡献**一个路标点**。
**两路夹爪不在桥接范围内**（`joint_count=14`），原样发出。

`eval_deploy_bridge.py` 直接 import 部署仓库自己的 `cubic_hermite_segment`（逐位导入，
不污染 `sys.path`，否则会把训练该 checkpoint 的 lerobot 顶掉），重建这条真实轨迹。
held-out，8 897 anchors，只统计被执行的那 8 帧：

| 条件 | 手臂 MAE (rad) | 夹爪 MAE (0–1) |
|---|---:|---:|
| **`policy_bridge`（机器人真正执行的）** | **0.0471** | **0.0358** |
| `hold_state`（站着不动） | 0.0307 | 0.0109 |
| `oracle_bridge`（同一条桥，瞄准示教 waypoint） | **0.0110** | 0 |
| `raw_chunk`（不桥接，直接发模型输出） | 0.0763 | 0.0358 |

同一张表在另外两个条件下（训练集内 3 007 anchors；act head held-out 3 638 anchors）：

| | 手臂 policy | 手臂 hold | 手臂 oracle | 夹爪 policy | 夹爪 hold |
|---|---:|---:|---:|---:|---:|
| diffusion，**held-out** | 0.0471 | 0.0307 | 0.0110 | 0.0358 | 0.0109 |
| diffusion，**训练集内** | 0.0227 | 0.0273 | 0.0096 | 0.0225 | 0.0081 |
| act head，**held-out** | 0.0512 | 0.0346 | 0.0124 | 0.0384 | 0.0116 |

手臂在训练集内勉强比站着不动好 1.20 倍，到 held-out 就变成差 1.53 倍；act head 差 1.48 倍——
和 §7 一致，**换 head 在部署层面同样没有区别**。

**而夹爪在训练集内就比"不改变握持状态"差 2.8 倍**（0.0225 vs 0.0081），held-out 差 3.3 倍。
这一列没有过拟合与否的问题：它在模型背下来的数据上也是错的。

三条结论：

1. **执行出来的手臂轨迹比站着不动差 1.53 倍**（0.0471 vs 0.0307）。
2. **夹爪比"不改变握持状态"差 3.3 倍**（0.0358 vs 0.0109）。夹爪是唯一不被桥接、
   原样发给机器人的通道，而它的跳变只占 4.5% 的帧（§2）——L1/MSE 会把这种稀疏跳变抹平，
   于是手在错误的时刻开合。**这就是"抓取准确率低"的直接来源。**
3. **部署侧的桥接设计没有问题，反而在救模型。** `oracle_bridge` 说明：同一条 Hermite，
   只要 waypoint 是对的，就能比站着不动好 2.8 倍（0.0110 vs 0.0307）。
   而 `raw_chunk`（0.0763）比 `policy_bridge`（0.0471）差 62%——**桥接正是靠"从实测位姿起步"
   替模型抹掉了 §4 那个常数位姿偏置。** 缺口全部在模型给出的 waypoint 上，不在轨迹改写上。

> 本节只重建了 Hermite 桥接这一级。真实管线在桥接前还跑
> `remove_small_rollbacks` / `remove_open_gripper_loops` / `smooth_action_chunk` /
> `smooth_large_excursions`（见 memory `record-chunk-is-filtered`）。前三者对手臂的影响会被桥接
> 整段覆盖，但 `remove_open_gripper_loops` 会改夹爪列，所以上表的夹爪数字是**未经该级处理**的模型输出。

---

## 10. 建议

按预期收益排序。前两条改配置，第三条改损失，后两条是"不要再花时间的地方"。

### 10.1 打开 `use_robot_state`（必做，单字段）

`--policy.use_robot_state=true`。这一条同时针对两个缺陷：每帧多一个投影后的 state token，
既给出精确位姿（消掉 §4 的常数偏置），也给出精确速度（5 帧 state 之间的差分，
可能连带修好 §6 的时间顺序失明）。§9 是它的受控测量。

一个必须提前说清的风险：state token 是 3 841 个 memory token 里的**一个**，
加性地进入一个 2 层 MLP 编码的 memory，再被 cross-attention 读取。
这比 `act_dit` 那种**乘性** adaLN 通路弱得多，模型完全可能学不会重视它。
§9 如果显示增益有限，下一步不是放弃本体感觉，而是加强它的注入方式
（每个 patch token 上拼接 state、或者走 FiLM/adaLN），而不是回到 `False`。

### 10.2 改用相对动作（强烈建议，与 10.1 叠加）

树里已经有成熟实现：`lerobot/policies/act_delta/`（`use_relative_actions`、
`relative_exclude_joints`、`prepare_relative_stats.py`，以及 `processor_act_delta.py` 里
"raw → relative → normalize" 的正确顺序）。让模型预测 `action − state` 而不是绝对关节角，
**位姿定位问题在结构上消失**，而不是靠数据统计去压。

§8 已经给出这条路的实测上界：部署侧的 Hermite 桥接本来就是从实测位姿起步的，
`raw_chunk`（0.0763）→ `policy_bridge`（0.0471）那 62% 的改善，就是"把绝对量换成相对量"
在推理侧的效果。把同一件事放到训练目标里，模型的全部容量就都用在预测位移上。

> 注意与 memory `act-delta-phase2-gate-express` 的差别：那次相对动作在 ACT 上"两个都落在
> 不确定区间"，但 **ACT 本来就有 state 输入**，相对化只是换个参数化。
> patch_policy 没有 state，相对化解决的是一个它现在根本无法解决的问题，两者不可类比。

### 10.3 单独处理夹爪通道

§8 显示夹爪**在训练集内**就比"不改变握持"差 2.8 倍，这与泛化无关。原因是量纲：
跳变只占 4.5% 的帧，L1/MSE 的最优解就是把阶跃抹成斜坡。而它是唯一不被桥接、
原样发给机器人的列——直接决定手在什么时刻开合。

两个可选做法：给夹爪列单独的损失权重；或者把夹爪从回归改成开/合二分类
（`relative_exclude_joints` 的存在说明这套代码已经预期夹爪要区别对待）。
**这一条独立于 10.1/10.2，应该并行做。**

### 10.4 不要再调的旋钮（已实测无效）

| 旋钮 | 实测 |
|---|---|
| `action_head`（act ↔ diffusion） | 差 <5%，被偏置支配（§7、§8） |
| `n_action_steps`（8/16/32） | 偏置在第 1 帧就存在，与执行帧数无关（§3.1） |
| `num_inference_steps` / 采样平均 | ACT-DiT 上实测分别值 0% 和 3%，此处同理 |
| 提高 `n_obs_steps` | 现有 5 帧里时间顺序只值 0.8%（§6），加更多帧没有意义 |

反过来，§6 给了一个**省算力**的机会：既然历史几乎不被使用，
`n_obs_steps` 从 5 降到 2（reference 的 diffusion preset 就是 2）能把 memory token
从 3 841 降到 1 537，训练和推理都快 2.5 倍，而按现有测量精度不会掉点。
把省下来的算力用在 10.1/10.2 上。

---

## 11. 这份报告的边界

**修好本体感觉是必要条件，不是充分条件。** §4 已经给出这个上界：
把位姿偏置去掉之后，patch_policy 落在 1.27×，和 ACT（1.21×）、ACT-DiT（1.27×）**同一档**。
也就是说，同一数据集上三个不同架构、不同 head、不同编码器健康状况的策略，
在 held-out 上全部只比"什么都不做"好 20–30%。

这说明还有一个**与架构无关的、共有的**天花板，本报告没有触碰它。它更可能在数据侧：
363 个 episode、单一场景、batch 目录是累积合并的、
而 memory `act-policy-leans-on-background-pixels` 记录过 ACT 会去抓背景像素当捷径。
那需要另一次测量（场景多样性、物体位置分布、session 泄漏），不在本报告范围内。

与 `../act_dit/act_dit-encoder-collapse-2026-08.md` §3.1 的结论一致，
只是那份报告修的是"编码器被关掉"，这份修的是"没有本体感觉"——
**两个不同的故障，同一个天花板。**

另外三点没有测：

- 真机成功率。本报告全部是离线开环指标，`policy_bridge` 是对部署轨迹的**重建**，
  不是实机回放。桥接前的 `remove_open_gripper_loops` 会改夹爪列，未纳入。
- 部署相机的实际 FOV / 分辨率是否与训练一致。`deploy_config_patch_policy.yaml` 声明 640×480 rgb，
  与数据集一致，但未在实机上核对过。
- `all_patches_linear` 那一档（§5）欠定未收敛，已排除；
  "稠密 patch 网格对本任务的真实价值"因此只有推理侧的 63%（§6）这一个证据，
  不足以下"该不该用 patch_policy"的结论。
