# 09-03 两个新 EEF 权重：关掉本体感觉要价 5.2%，换 act head 把这笔钱赚回来一半并省掉 5 倍算力；而部署重写让四个权重全部变差

**评测集** `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data_eef`
（53 ep / 40 132 帧 / 2007 anchor / 97 090 个有效动作步 / stride 20 / 被判为污染而丢弃的 episode：**0**）

**待评权重** 全部 `run/checkpoints/200000/pretrained_model`（两个新 arm 另评 `100000`）

| 名字 | job | head | n_obs | robot_state | 训练集 | 说明 |
|---|---|---|---:|---|---|---|
| `pp_eef_state` | `patch_policy_..._505_eef_2026-08-31_13-14-33-857400` | diffusion | 2 | **on** | `batch_success_505_eef` | 基线，09-02 报告的主角 |
| `pp_eef_nostate` | `patch_policy_..._505_eef_2026-09-03_09-44-14-723349` | diffusion | 2 | off | `batch_success_505_eef` | **新**，相对基线只改一件事 |
| `pp_eef_act5` | `patch_policy_..._505_eef_2026-09-03_09-33-43-303120` | **act** | **5** | off | `batch_success_505_eef` | **新**，相对基线改了三件事 |
| `acteef_533` | `act_eef_..._533_eef_2026-08-27_14-48-42-880941` | (ACT) | 1 | — | `batch_success_533_eef` | `deploy_config_eef.yaml` 现在指着的权重 |

**测量日期** 2026-09-03，本机 mgmt01 RTX 4090（49 GB），解释器
`/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的同一环境）
**脚本与原始结果** `eval_policy/runs/20260903_pp_eef_state_head/`
（`results/*.json` 按仓库 `.gitignore` 约定只留在本地，不进 git）
**前置报告** `patch_policy-eef-independent-eval-2026-09.md`。本报告**不推翻它的任何精度结论**
（那份报告的两个数字在这里逐位复现），但**订正它的一条方法学前提**：见 §3.4。

---

## 1. 结论

1. **`job.json` 看不出这两个权重的区别，`config.json` 才能。** 三个 patch_policy 的
   `job.json` 逐字段相同（200 k 步 / bs 16 / seed 1000 / lr 5.5e-5 常数），
   65 个策略配置字段里动了 5 个。其中 **`gpt_block_size` 2→25 和
   `n_vqvae_training_steps` 5000→20000 是死字段**：前者只被 `BlockCausalGPT` 读，
   后者只在 `action_head == "vqbet"` 下读，两个新权重都不是 vqbet
   （`probes/config_diff.txt` §4 给了行号）。**真正生效的改动只有三条**：
   `use_robot_state` true→false、`action_head` diffusion→act、`n_obs_steps` 2→5。
2. **关掉本体感觉是纯亏损，要价 5.2%。** `pp_eef_nostate` 相对基线唯一的改动是
   `use_robot_state: true→false`：MAE 0.03721 → **0.03916（+5.2%）**，
   @1 0.02254 → 0.02613（**+15.9%**），位置误差 11.28 mm → **12.27 mm（+8.7%）**，
   null 比值 1.95x → 1.86x。该权重自身的采样噪声只有 1.58%，**差距是它的 3.3 倍**，
   不是噪声。null 基线 `hold_state` 0.07266 / `train_mean` 0.22729。
3. **换 act head + n_obs 5 把这笔亏损赚回一半，并且省掉 5 倍算力。**
   `pp_eef_act5` 0.03796：比基线差 2.0%，但比同样关掉 state 的 `pp_eef_nostate`
   **好 3.1%**（deployed 好 4.4%）。位置误差回到 11.38 mm（比基线只差 0.9%）。
   代价侧是它真正的卖点：**墙钟 0.19x**（每 anchor 13.9 ms vs 71.7 ms），
   而且 **head 是确定性的**——两个种子的 MAE 逐位相同（0.03796 = 0.03796），
   扩散头两个权重则分别有 0.97% 和 1.58% 的采样抖动。
4. **它已经不动了。** `pp_eef_act5` 100 k → 200 k：0.03778 → 0.03796（**+0.5%，在原地**），
   而 `pp_eef_nostate` 同期 0.04649 → 0.03916（**−15.8%，还在降**）。
   "再训久一点"这条路对 act5 已经关上，对 nostate 还开着——但 nostate 降完 15.8% 之后
   仍然是三个里最差的。
5. **一个谁都没测过的坏消息：EEF 部署重写让四个权重全部变差。**
   `deploy_config_eef.yaml` 的确会跑 chunk 重写（§3.4），过完真实部署栈之后
   `pp_eef_state` +8.4%、`acteef_533` +7.8%、`pp_eef_nostate` +4.5%、`pp_eef_act5` +3.1%，
   **全部损失来自那个 K=40 的实测状态桥**（回退与平滑合计只有 −0.3%，夹爪 clip −0 至 −1.6%）。
   今天应该上机的仍然是 `acteef_533`：部署窗口 60 步下 deployed 0.03975，
   对自己那把尺子的 null（0.08216）是 **2.07x**，四个权重里最高。

**一句话的取舍：** **两个新权重都没有超过 08-31 的基线，但 `pp_eef_act5` 用 2.0% 的精度
换来了 5 倍速度和确定性输出，是三个 patch_policy 里唯一值得继续投的方向；
真正该修的不是权重，是那个让所有权重都变差 3–8% 的部署桥。**

---

## 2. 到底改了哪几个参数

`probes/config_diff.py` 直接读四个 `config.json` 与 `job.json`，不依赖任何文档。

**训练层面（`job.json`）——没有任何差别可归因：**

| 字段 | `pp_eef_state` | `pp_eef_nostate` | `pp_eef_act5` |
|---|---|---|---|
| `dataset_repo_id` | `505_eef` | `505_eef` | `505_eef` |
| `steps` / `batch_size` / `seed` | 200000 / 16 / 1000 | 200000 / 16 / 1000 | 200000 / 16 / 1000 |
| 末步 loss | 0.017 | 0.018 | 0.033 |
| 末步 grad_norm | 0.216 | 0.222 | 0.82 |
| 墙钟 | 16 108 s | 16 440 s | 29 557 s |
| 节点 | mgmt01 | gpu03 | gpu04 |

**策略层面（`config.json`）——65 个字段里差 5 个：**

| 字段 | `pp_eef_state` | `pp_eef_nostate` | `pp_eef_act5` | 生效吗 |
|---|---|---|---|---|
| `use_robot_state` | True | **False** | **False** | **是** |
| `action_head` | diffusion | diffusion | **act** | **是** |
| `n_obs_steps` | 2 | 2 | **5** | **是** |
| `gpt_block_size` | 2 | 2 | 25 | **否** |
| `n_vqvae_training_steps` | 5000 | 5000 | 20000 | **否** |

对照项（三者全同）：`action_chunk_size=50`、`n_action_steps=50`、
`vision_encoder=dino_patch`、`freeze_vision_encoder=True`、`resize_shape=[224,224]`、
`dim_model=512`、`optimizer_lr=5.5e-5`、`beta_schedule=squaredcos_cap_v2`、
`normalization_mapping={VISUAL: IDENTITY, STATE: MIN_MAX, ACTION: MIN_MAX}`。

**两个死字段的证据。** `gpt_block_size` 的全部读取点在
`modeling_patch_policy.py:157/163/170/171`，都在 `class BlockCausalGPT` 内；
该类只在 `modeling_patch_policy.py:372` 的 `if config.action_head == "vqbet"` 分支里被构造。
`n_vqvae_training_steps` 唯一的读取点是 `modeling_patch_policy.py:598`，
同样在 `action_head == "vqbet"` 的守卫下。
**所以 `pp_eef_act5` 是一个三变量 arm，不是五变量 arm**——但它仍然是三变量，
下面任何"act head 值多少"的说法都必须带上这句。

> **归因边界。** `pp_eef_nostate` ↔ `pp_eef_state` 只差一件事，§4 里它的数字可以直接读作
> "本体感觉的价格"。`pp_eef_act5` ↔ 任何一个都差两件以上（head + n_obs），
> **它的收益不可分离到 head 或窗口长度中的任何一个**。要拆开，需要再训一个
> (act, n_obs 2, state off)。

---

## 3. 评测集的体检

### 3.1 来源与污染：0 个 episode 被训练集见过

`splits.py`（复用 `runs/scripts_patch_policy_eval_0902/`，未改）用两个指纹判定，
本轮重跑逐位复现 09-02 的结果：

```
observation.velocity 指纹与关节评测集共享: 53 / 53
EEF episode 没有关节对应物的: []
episode 序号原样保留: True
53 集出现在 361 / 505 / 361_eef / 505_eef / 533_eef 中的: velocity 0, action 0
```

harness 侧的独立确认：四次评测都传了 `--train-root batch_success_505_eef`
**和** `--train-root batch_success_533_eef`（572 个唯一训练 episode 的动作指纹），
`episodes_dropped_as_contaminated = 0`，53 个 episode 全部进表。
两个训练集都传，是为了让四次评测的 anchor 集合**完全相同**——
即使它实际一个都不丢。

### 3.2 越界率 0.24%

落在 `batch_success_505_eef` 的 MIN_MAX 盒子外的帧：**98 / 40 132（0.24%）**，
全部集中在 `eef_r_x` 一个通道，最大越界 0.0301（原始单位）。其余 13 个通道 0 帧。
（对照：08-31 报告用的 67-episode 随机留出集是 731 / 56 278 = 1.30%。）
**下面的差距不能推给分布外。**

### 3.3 历史索引自检：四个 checkpoint 全过

`patch_policy` 的 `action_delta_indices` 从 `1 - n_obs_steps` 开始，
从索引 0 开始打分就是拿 chunk 去比**过去**。`check_alignment.py` 对四个权重各跑一次：

| 权重 | n_obs | action 窗口 | 计算出的偏移 | 负对照 | 结果 |
|---|---:|---|---:|---|---|
| `pp_eef_state` | 2 | −1..49 | 1 | 8/8 触发 | OK |
| `pp_eef_nostate` | 2 | −1..49 | 1 | 8/8 触发 | OK |
| `pp_eef_act5` | **5** | **−4..49** | **4** | 8/8 触发 | OK |
| `acteef_533` | 1 | 0..99 | 0 | 不适用 | OK |

`pp_eef_act5` 是本仓库第一个 `n_obs_steps=5` 的 EEF 权重，偏移 4 是新出现的取值，
所以这条自检这轮不是形式主义。

### 3.4 harness 分叉了——以及它订正了 09-02 报告的哪一句

顶层 `offline_chunk_eval.py` **一个字节没动**。本轮在 run 目录下用了一个副本，
四处改动，完整 diff 在 `runs/20260903_pp_eef_state_head/fork.diff`（72 行）。

**订正 09-02 报告的一条方法学前提。** 那份报告的 `run_eval.sh` 全程 `--filters none`，
理由写在脚本注释里：EEF 部署路径"hands the chunk to VlaHost verbatim"。
**代码不是这样**（`probes/deploy_stack.txt` §1）：

- `deploy_config_eef.yaml`：`inference.type: chunk`、`strategy: base`；
- `strategies/factory.py` 把 `base` 映射到 `BaseStrategy`；
- `strategies/base.py:69` 在 `engine.produces_chunks` 时调用 `send_next_action_chunk`；
- 该函数就是做 chunk 重写的那一个。

**重写会发生。** 因此 09-02 报告只有 `policy_raw` 一列这件事不是保守，是漏了一整列；
本报告 §6 把那一列补上。**它的精度结论不受影响**（`policy_raw` 与 filters 无关，
两个已发表数字在本轮逐位复现，见 §4 开头）。

分叉的原因是另一件事：**顶层 harness 的部署重写把动作空间写死成 16-D 关节**。
`send_next_action_chunk` 是按名字分的
（`[k for k in ordered_keys if "gripper" not in k.lower()][:14]`），
在 16-D 关节里是 14 臂 + 第 14/15 列夹爪，**在 14-D EEF 里是 12 位姿 + 第 12/13 列夹爪**：

| # | 分叉改了什么 | 不改会怎样 |
|---|---|---|
| 1 | 从数据集动作名推出臂/夹爪划分（沿用 deploy 自己那条规则） | — |
| 2 | 桥只改写非夹爪列 | 桥会改写夹爪列，而部署不碰；合成 chunk 上 max\|Δ\| = 1.43 |
| 3 | 夹爪 clip 作用在真实夹爪列 | 宽度 14 时整个 clip 是空操作，而驱动实际会 clip；max\|Δ\| = 0.50 |
| 4 | `--filter-ablation` 在宽度 < 16 时跳过 `gripper_loops` | **直接崩**：`ValueError: Gripper indices [14, 15] are outside action width 14`。该级在机器人上本来就是 `ENABLE_REMOVE_OPEN_GRIPPER_LOOPS = False` |

**分叉在 16-D 关节空间下与顶层逐位相同**（`probes/deploy_stack.txt` 末行断言 True），
关节侧历史数字不受影响。

### 3.5 刻度选择

**horizon 60**，取自 `deploy_config_eef.yaml` 的 `inference.n_action_steps`，不是习惯值。
harness 夹到策略自己的 chunk：patch_policy `action_chunk_size = 50` → **实际 50**，
`acteef_533` `chunk_size = 100` → **实际 60**。
**这个夹取本身是部署事实：配置要 60 步，三个 patch_policy 都只能给 50。**

**filters `rollbacks,smoothing,bridge,gripper_clip`**，不是 `none` 也不是 `all`。
`strategies/core.py:52-55` 四个开关里 `gripper_loops`、`excursions` 是 False，
`smoothing` 是无条件调用，加上驱动的夹爪 clip。`--filters all` 会多算两级机器人已关掉的。

**`--batch-size` 全程 8**：扩散采样种子按 batch 重置，同一 anchor 的噪声取决于它在
batch 里的位置，换 batch size 数字就变。

---

## 4. 主表

> **复现性前置。** 同一把尺子重跑两个已发表的数：
> `pp_eef_state` policy_raw **0.03721**（09-02 报告 §4：0.03721），
> `acteef_533` @h50 policy_raw **0.03443**（同 §4：0.03443）。
> 两个都在 5e-5 容差内逐位对上，`summarise.py --selftest` 把它写成断言。
> 所以下面的新数字和那份报告是同一把尺子量出来的。

### 4.1 执行窗口（`--n-action-steps 60`，按各自 chunk 夹取）

| run | head | n_obs | state | horizon | raw | deployed | @1 | @10 | @25 | @50 | raw vs null | deployed vs null | 墙钟 (s) | 每 anchor (ms) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_state` (08-31) | diffusion | 2 | on | 50 | **0.03721** | 0.04034 | **0.02254** | **0.02564** | **0.02994** | **0.03721** | **1.95x** | 1.80x | 286（2 draws） | 71.2 |
| `pp_eef_nostate` (09-44) | diffusion | 2 | off | 50 | 0.03916 | 0.04090 | 0.02613 | 0.02877 | 0.03271 | 0.03916 | 1.86x | 1.78x | 288（2 draws） | 71.7 |
| `pp_eef_act5` (09-33) | **act** | **5** | off | 50 | 0.03796 | **0.03912** | 0.02564 | 0.02840 | 0.03222 | 0.03796 | 1.91x | **1.86x** | 56（2 draws） | **13.9** |
| `acteef_533`（部署中） | (ACT) | 1 | — | **60** | 0.03689 | 0.03975 | 0.01694 | 0.02129 | 0.02713 | 0.03443 | **2.23x** | **2.07x** | 16 | **8.2** |

**null 基线**（每个 horizon 一行；同 horizon 的 run 共用）：

| null | horizon | mae | @1 | @10 | @25 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| `hold_state` | 50 | 0.07266 | 0.02098 | 0.03138 | 0.04757 | 0.07266 |
| `train_mean` | 50 | 0.22729 | 0.22632 | 0.22650 | 0.22676 | 0.22729 |
| `hold_state` | 60 | 0.08216 | 0.02098 | 0.03138 | 0.04757 | 0.07266 |
| `train_mean` | 60 | 0.22685 | 0.22570 | 0.22587 | 0.22614 | 0.22668 |

> **`acteef_533` 那一行是 horizon 60，其余三行是 horizon 50。** 标量不可直接相减，
> null 也不同（0.08216 vs 0.07266）。跨行只读 `vs null` 比值；同 horizon 的对照见 §7。

### 4.2 相对基线的差值（同 anchor 集、同 seed、同 batch size）

| run | raw Δ | deployed Δ | @1 Δ | 位置 Δ | 该权重自身的采样噪声 |
|---|---:|---:|---:|---:|---:|
| `pp_eef_nostate` | **+5.2%** | +1.4% | **+15.9%** | +8.7% | 1.58% |
| `pp_eef_act5` | +2.0% | **−3.0%** | +13.8% | +0.9% | **0（确定性）** |
| `acteef_533` *(h60，跨 horizon，不可直接读)* | −0.9% | −1.5% | −24.9% | −6.9% | 0（确定性） |

`pp_eef_act5` vs `pp_eef_nostate`（两者 `use_robot_state` 都是 off，差的是 head 与 n_obs）：
raw **−3.1%**、deployed **−4.4%**、墙钟 **0.19x**。

### 4.3 rmse 与尾部

| run | raw rmse | deployed rmse | norm_mae | norm_rmse | tail ratio |
|---|---:|---:|---:|---:|---:|
| `pp_eef_state` | 0.07869 | 0.08944 | 0.18789 | 0.30994 | 1.65 |
| `pp_eef_nostate` | 0.08025 | 0.08922 | 0.20095 | 0.32147 | 1.60 |
| `pp_eef_act5` | **0.07607** | **0.08555** | 0.19238 | **0.30859** | 1.60 |
| `acteef_533` (h60) | 0.07867 | 0.08788 | **0.18406** | 0.31150 | 1.69 |

`pp_eef_act5` 的 rmse 是四个里最低的，而它的 MAE 不是最低——它的错误更均匀，
没有基线那样的长尾（tail 1.60 vs 1.65）。这是确定性 head 该有的样子。

### 4.4 分组误差（14 个维度不共享单位）

| run | 位置 MAE (mm) | 姿态 MAE (°) | 夹爪 MAE (0–1) | 夹爪 clip 后 | 位置 deployed (mm) | 姿态 deployed (°) |
|---|---:|---:|---:|---:|---:|---:|
| `pp_eef_state` | **11.28** | **3.86** | 0.0245 | 0.0245（未越界） | 12.52 | 4.21 |
| `pp_eef_nostate` | 12.27 | 4.05 | 0.0255 | 0.0255（未越界） | 13.00 | 4.24 |
| `pp_eef_act5` | 11.38 | 3.93 | 0.0260 | **0.0217** | **12.27** | **4.11** |
| `acteef_533` (h60) | 10.51 | 3.87 | 0.0243 | 0.0214 | 11.98 | 4.22 |
| *null* `hold_state` (h50) | 23.62 | 7.06 | 0.0681 | — | — | — |
| *null* `train_mean` (h50) | 45.43 | 18.64 | 0.4789 | — | — | — |

关掉本体感觉在**位置**上最贵（+0.99 mm / +8.7%），姿态只贵 0.19°（+4.9%）。
换 act head 把位置几乎全部拉回（11.38 vs 11.28 mm，+0.9%）。

**夹爪那一列要读两次。** 未经处理时 `pp_eef_act5` 是三个里最差的（0.0260），
但过一遍部署的夹爪 clip 之后它变成**最好的**（0.0217，扩散头两个仍是 0.0245 / 0.0255）。
两个扩散头 clip 前后**逐位不变**——它们的夹爪输出从来没有越出 [0, 1]；
act 类 head 会越界，而越界部分被夹回去正好是净收益。§6.2 第 3 点是同一件事。

### 4.5 采样噪声：哪些差距是真的

| run | seed 0 | seed 1 | Δ | 确定性 |
|---|---:|---:|---:|---|
| `pp_eef_state` | 0.03721 | 0.03757 | +0.363e-3 | 否（0.97%） |
| `pp_eef_nostate` | 0.03916 | 0.03854 | −0.617e-3 | 否（1.58%） |
| `pp_eef_act5` | 0.03796 | 0.03796 | ±0 | **是（逐位相同）** |
| `acteef_533` | 0.03689 | — | — | 未测（ACT 无采样） |

**判读线：两个扩散权重的抖动是 0.97% 与 1.58%，所以本报告里小于 ~1.6% 的差不作数。**
`+5.2%`（本体感觉）是它的 3.3 倍，`−3.1%`（act5 vs nostate）是它的 2 倍，两条都成立；
`+2.0%`（act5 vs 基线）只有 1.3 倍，**是弱结论，不要单独引用**。

`pp_eef_act5` 两次抽样逐位相同，这既是 `config.json` 说它是 `act` head 的独立验证
（扩散头在 `modeling_patch_policy.py:494` 有 `torch.randn`，act 头没有），
也意味着**它在机器人上重复执行同一段任务不会有分布抖动**。

---

## 5. 训练是不是不够久

| run | 100 k raw | 200 k raw | Δ | 100 k deployed | 200 k deployed |
|---|---:|---:|---:|---:|---:|
| `pp_eef_nostate` | 0.04649 | 0.03916 | **−15.8%** | 0.04445 | 0.04090 |
| `pp_eef_act5` | 0.03778 | 0.03796 | **+0.5%** | 0.03909 | 0.03912 |

两条完全不同的曲线：

- **`pp_eef_act5` 在 100 k 就到位了**，后 100 k 步（约 4 小时机时）什么也没换来，
  +0.5% 在它自己的确定性尺度上仍然是零。**继续训练不是这个 arm 的杠杆。**
- **`pp_eef_nostate` 还在降**，100 k→200 k 降了 15.8%。但它降完之后
  （0.03916）仍然是三个 patch_policy 里最差的，而且比 100 k 的 act5（0.03778）还差。
  **它不是训得不够，是配置本身更差。**

对照：09-02 报告记录 `pp_eef_state` 从 100 k 到 200 k 降了 15.3%——
与 `pp_eef_nostate` 的 15.8% 同量级，说明扩散头这条曲线的形状没被 state 开关改变，
只是整体抬高了一档。

---

## 6. 机制：损失发生在哪一级

### 6.1 部署栈是哪几级

`probes/deploy_stack.txt` 从部署 checkout 直接读：

| harness filter | 部署开关 | 活着 |
|---|---|---|
| `rollbacks` | `ENABLE_REMOVE_SMALL_ROLLBACKS` | **是** |
| `gripper_loops` | `ENABLE_REMOVE_OPEN_GRIPPER_LOOPS` | 否 |
| `smoothing` | （无条件调用） | **是** |
| `excursions` | `ENABLE_SMOOTH_LARGE_EXCURSIONS` | 否 |
| `bridge`（K=40 Hermite） | `ENABLE_FIXED_K_REAL_STATE_BRIDGE` | **是** |
| `gripper_clip` | 驱动 `_prepare_action` | **是** |

### 6.2 逐级归因（累积，按部署顺序，相对 `policy_raw`）

| run | clip only | +rollbacks | +gripper_loops | +smoothing | +excursions | **+bridge（= 部署值）** | bridge 单独 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pp_eef_state` | +0.0% | −0.1% | −0.1% | −0.3% | −0.3% | **+8.4%** | +9.7% |
| `pp_eef_nostate` | +0.0% | −0.0% | −0.0% | −0.3% | −0.3% | **+4.5%** | +5.8% |
| `pp_eef_act5` | **−1.6%** | −1.6% | −1.6% | −1.6% | −1.6% | **+3.1%** | +3.6% |
| `acteef_533` (h60) | **−1.1%** | −1.1% | −1.1% | −1.2% | −1.2% | **+7.8%** | +8.0% |

（`gripper_loops` 与 `excursions` 两级在机器人上是关的，本表里按定义重复上一级。）

三件事：

1. **全部损失来自桥。** 回退删除 + 平滑合计只有 −0.1% 到 −0.3%（轻微有益），
   而 K=40 的实测状态桥单独就要 +3.6% 到 +9.7%。**部署栈里其他四级都不是问题。**
2. **桥为什么伤 EEF 特别重。** patch_policy 的 chunk 只有 50 步，桥固定改写前 40 步
   （`DEPLOY_BRIDGE_STEPS = 40`，部署侧同样写死），**50 步里只有 10 步是策略原始输出**。
   而这些权重的强项恰恰在前段：`pp_eef_state` 的 @1 是 0.02254、@10 是 0.02564，
   远好于全窗口的 0.03721。桥用一条零初速 Hermite 覆盖掉的，正是策略最准的那一段。
   `acteef_533` 的 chunk 是 100 步、执行 60 步，桥只盖掉 40/60，损失却也有 +7.8%——
   同一个原因，同样是盖掉了最准的前段。
3. **夹爪 clip 只对 act 类 head 有用。** 它给 `pp_eef_act5` −1.6%、`acteef_533` −1.1%，
   给两个扩散头 **+0.0%**。含义很直接：**扩散头的夹爪输出从来没有越出 [0, 1]，
   act 类 head 会越界**，而越界的部分被 clip 掉正好是净收益。
   数量上：`pp_eef_act5` 夹爪误差 0.0260 → **0.0217**，`acteef_533` 0.0243 → 0.0214，
   两个扩散头 0.0245 / 0.0255 → **逐位不变**。这与 §4.4 的夹爪两列是同一件事的两面。

> **这一节的数字 09-02 报告没有。** 那轮用 `--filters none`，
> 所以四个 EEF 权重的部署值在本报告之前**从未被测过**（§3.4）。

### 6.3 部署配置要 60 步，patch_policy 只有 50 步

`deploy_config_eef.yaml` 写 `inference.n_action_steps: 60`；
三个 patch_policy 的 `action_chunk_size` 都是 50。**策略给不满这个窗口。**
`acteef_533`（`chunk_size=100`）能给满，它 h50→h60 的代价见 §7。

---

## 7. 同 horizon 对照：`acteef_533` 的两把尺子

| run | horizon | chunk | raw | deployed | hold_state | raw vs null | deployed vs null | 位置 (mm) | 姿态 (°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `acteef_533` | 50 | 100 | 0.03443 | 0.03794 | 0.07266 | 2.11x | 1.92x | 9.78 | 3.61 |
| `acteef_533` | **60** | 100 | 0.03689 | 0.03975 | 0.08216 | **2.23x** | **2.07x** | 10.51 | 3.87 |

**在与三个 patch_policy 完全相同的 horizon 50 上：** `acteef_533` raw 0.03443、
deployed 0.03794，都优于任何一个 patch_policy（最好的 raw 是 0.03721、
最好的 deployed 是 0.03912）。**新权重没有改变 09-02 报告的排序结论。**

窗口从 50 拉到 60 时标量变差（0.03443 → 0.03689）但 null 比值变好
（2.11x → 2.23x），因为 null 自己变差得更快（0.07266 → 0.08216）——
**这是窗口变长的正常现象，不是权重变差。**

---

## 8. 建议

1. **上机仍然用 `acteef_533`，`deploy_config_eef.yaml` 不用改权重。**
   它在部署窗口 60 步下 deployed 0.03975 / 2.07x null，
   在与 patch_policy 同尺的 50 步下 deployed 0.03794 / 1.92x，两条都排第一。
2. **停止在 `use_robot_state=false` 的扩散 arm 上投入。** `pp_eef_nostate` 是本轮
   唯一一个单变量对照，结论干净：关掉本体感觉在 EEF 空间里是 **+5.2% 的纯损失**
   （位置 +8.7%），而它换来的是零——推理时间与基线相同（71.7 vs 71.2 ms/anchor）。
   它的 100 k→200 k 还在降 15.8%，但降完仍是最差的一个，不值得再补训。
3. **`pp_eef_act5` 是唯一值得继续的 patch_policy 方向，但下一步不是再训，是拆变量。**
   它把速度做到 13.9 ms/anchor（扩散头的 0.19x）、输出确定、rmse 最低，
   代价只有 +2.0%（而且这个数在采样噪声的 1.3 倍处，是弱结论）。
   它在 100 k 就收敛了（200 k 只 +0.5%），所以**加步数没有意义**。
   该做的是它自己引入的三变量：训一个 **(act head, n_obs 2, state off)** 拆掉窗口长度，
   再训一个 **(act head, n_obs 2, state on)** —— 后者正是 `patch_policy-matrix.md` §4
   列为空格、head-comparison §1 点名建议而至今没有任何权重覆盖的组合，
   而 §4.2 已经证明 state 在 EEF 空间里值 5.2%。
4. **真正的杠杆在部署侧，不在权重侧：K=40 的桥让四个权重全部变差 3.1%–8.4%。**
   任何一个权重再训 100 k 步能拿到的收益（15.8%）与这条重写的损失同量级，
   而改一个常数不需要 4 小时机时。三个可查的方向，按代价排序：
   ① 把 `DEPLOY_BRIDGE_STEPS` 从 40 调小（对 chunk 只有 50 步的 patch_policy 尤其），
   ② 只在首帧偏差超过阈值时才启用桥（`send_next_action_chunk` 已经算了 `max_diff`，
   目前只用于日志），③ 让桥长度随 chunk 长度缩放而不是写死。
   **本报告没有测过其中任何一个**——只测出了 K=40 的代价。
5. **顺手可以关掉的一件事：** `deploy_config_eef.yaml` 要 60 步而 patch_policy 只有 50 步
   （§6.3）。若哪天真把 patch_policy 放上 EEF 部署，这个 10 步的空缺会以配置 bug 的
   形式再出现一次——08-31 报告 §1.6 已经在另一个配置文件上踩过同样的坑。

### 还没测的

- **闭环。** 全部是 teacher-forced 开环 chunk 评测，每个 anchor 从示教状态出发。
- **孤立推理延迟。** 表里的 ms 是整轮墙钟（含取数、含两次抽样）除以 anchor 数，
  是相对代价，不是单次推理延迟。09-02 报告用另一种测法给过 297 ms / 6.6 ms，
  两者不可混用。
- **50 k / 150 k。** 两个新 arm 只评了 100 k 与 200 k。
- **桥的任何替代参数。** 见建议 4。
- **act head 的三变量拆分。** 见建议 3。
- **`pp_eef_act5` 在关节空间的对应权重。** 本轮四个权重全在 EEF 空间。
