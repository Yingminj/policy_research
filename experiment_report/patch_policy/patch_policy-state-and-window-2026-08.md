# 打开 `use_robot_state` 之后：本体感觉被模型忽略，而真正的部署窗口里两个新权重都输给"什么都不做"

**Checkpoint `new_state5`** `/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-27-35-272413`
— `action_head=diffusion`，`n_obs_steps=5`，**`use_robot_state=true`**，`action_chunk_size=50`，Slurm 146 @ **gpu05**
**Checkpoint `new_obs2`** `/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-31-30-522146`
— `action_head=diffusion`，**`n_obs_steps=2`**，`use_robot_state=false`，`action_chunk_size=50`，Slurm 147 @ **gpu03**
**训练集** `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_361`（363 episodes）
**评测集** `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data`
（53 episodes / 40 132 帧，本次重新核查：**53 个 episode 无一被指纹过滤器判为训练集内容**，§3.3）
**测量日期** 2026-08-31，本机 mgmt01 RTX 4090（49 GB），`/opt/robot-platform/train-venv`（训练这些 checkpoint 的同一解释器）
**脚本与原始结果** `../test_scripts/scripts_patch_policy_eval_fix/`
**前置报告** `patch_policy-head-comparison-2026-08.md`（§1 的三条改进建议正是本次两个权重要验证的对象）、
`patch_policy-no-proprioception-2026-08.md`
**评测方法来源** `../test_scripts/scripts_act_eval_test_fix/`（部署忠实的 chunk 评测），本次为其加了观测历史支持，见 §3.1

---

## 1. 结论

**五句话：**

1. **`use_robot_state=true` 生效了，但模型没在用它。** 把另一个 anchor 的本体感觉换进去，
   输出只移动 **6.7%**（`zero_state` 5.1%）；同一把尺子下换掉图像移动 **136%**。
   前置报告诊断的"策略不知道手臂在哪"**没有被修好**：chunk 首帧与实测位姿的偏差
   **0.0417 rad**，真实动作只有 0.0140 rad——**仍然偏 2.97 倍**，与关掉本体感觉的
   `prev_act_head`（3.05 倍）、`new_obs2`（2.81 倍）在同一水平。
2. **`n_obs_steps` 5→2 不花钱。** 两个新权重在 horizon 50 上的差距是 **1.1%**
   （0.06121 vs 0.06190），而同一 checkpoint 换一个扩散采样种子就能移动 **3.3%**——
   **差距落在采样噪声以内，不可判别**。但 `new_obs2` 的推理只要 **296 ms**，`new_state5` 要 **595 ms**；
   训练也从 11.1 h 降到 4.45 h。**同样的精度，一半的代价。**
3. **在机器人真正执行的窗口里，两个新权重都比站着不动更差。**
   `deploy_config_patch_policy.yaml` 的 `inference.n_action_steps` 是 **8**，
   而部署侧的 Hermite 桥接是 `min(40, len(chunk))`——**8 步窗口里每一步都是桥接**。
   在 horizon 8 上：`hold_state` 0.02825，`new_state5` 部署后 0.03110，`new_obs2` 0.03087。
   **两个都输给 null**，唯一赢的是 ACT 基线（0.02432）。
4. **误差的主体仍是常数位姿偏置，而部署桥接正是它的补救。** 桥接把 horizon 8 的误差
   压低 **26–30%**（把 chunk 的起点强行拉回实测位姿），其余四个滤波器合计影响 <3%。
   换句话说：**这些权重在机器人上能动，靠的是部署侧的 S 曲线，不是策略本身。**
5. **对照组仍然是 ACT。** 同一评测集、同一 harness、同一 anchor：ACT 基线 horizon 50 上
   0.05112（部署后 0.05088），比最好的 patch_policy 好 **17%**，而推理只要 **6.5 ms**（91 倍快）。

**一句话的取舍：** 两个新权重都不应上机。`new_obs2` 的方向是对的（省掉的算力是白捡的），
但它证明的是"多出来的观测帧本来就没用"，而不是"精度提高了"。**下一次改动应当只做一件事：
让策略真正读到本体感觉**——当前的做法（把 1 个 state token 拼进 768 个 patch token 里）
被实测证明无效。§7.3 列了三条可查的假设，其中**两条已在同日被后续测量证伪**
（state token 拿到 8.65 倍于均分份额的注意力、范数正常），剩下的只有"目标参数化让它冗余"，
所以下一步要改的是**训练目标**（相对动作），不是条件化方式。§7.4 解释这个开关在代码里到底改了什么。

---

## 2. 两个新 checkpoint 到底差在哪

`job.json` 的 `config` 段**逐字节相同**，`job.sbatch` 除 `--output_dir` 外**逐字节相同**，
`seed` 都是 1000。真正的差异只在各自保存的 `pretrained_model/config.json` 里：

```
$ diff new_state5/config.json new_obs2/config.json
<  "n_obs_steps": 5             >  "n_obs_steps": 2
<  "use_robot_state": true      >  "use_robot_state": false
<  "gpt_block_size": 55         >  "gpt_block_size": 2      # 惰性字段，见下
```

`gpt_block_size` 只在 `action_head="vqbet"` 时才会被用到（`BlockCausalGPT` 只在那一支里构造），
两个 checkpoint 都是 `diffusion` 头，**这个字段对模型没有任何影响**，不必解释它的取值。

可训练参数差 598 400，几乎全部是**条件位置编码表**：
`cond_pos_emb` 的大小是 `(1 + n_obs_steps × tokens_per_frame) × 256`，
5 帧 ×(768 patch + 1 state) 得 984 576，2 帧 ×768 得 393 472，差 591 104；
再加 `state_projector`（16→384）的 6 528。**去噪器本身的容量两者相同。**

> ⚠️ **这不是一次受控消融。** 两个字段同时变了。下文一切"A 与 B 差 X%"的说法，
> 归因到 `n_obs_steps` 与归因到 `use_robot_state` 是**无法分离的**。
> 好在 §7 的条件化探针是在**单个 checkpoint 内部**做干预，那一部分的归因是干净的。

与 2026-08-20 的前一代（`prev_diffusion`）相比还多变了一个字段：
`action_chunk_size` 64→50、`n_action_steps` 32→50。所以"新权重比 prev_diffusion 好 9–10%"
里也混着"chunk 更短、容量摊得更薄的位置更少"这一项。

命令行里从头到尾没有指定过 `n_obs_steps`、`use_robot_state`、`action_head` 中的任何一个——
它们全部来自各自节点上 `train-venv` 的默认值。这与 memory `act-dit-train-venv-defaults-drift`、
以及前置报告 §2 记录的是同一个现象：**同一份 sbatch 落在不同节点上得到不同的模型。**
本次两个权重的差异同样是被观察到的，不是被设计的。

### 2.1 训练侧代价（`slurm.out`）

| | `new_state5` | `new_obs2` |
|---|---:|---:|
| 节点 / Slurm | gpu05 / 146 | gpu03 / 147 |
| 总参数 / 可训练参数 | 32.12 M / **10.07 M** | 31.52 M / **9.47 M** |
| 每步耗时 `updt_s` | 0.187 s | **0.072 s** |
| 吞吐 `smp/s` | 80 | **205** |
| 显存 | 4.28 GB | **1.91 GB** |
| **墙钟总时长** | **11 h 06 m** | **4 h 27 m** |
| 最终 loss（MSE-ε） | 0.016 | **0.014** |
| 最终 grad norm | 0.219 | 0.221 |

> 两个 checkpoint 的 loss 这次**是可比的**（同为扩散头的 ε-MSE）。
> `new_obs2` 的训练 loss **更低**，而离线精度**略差**（差异在噪声内）。
> 又一次：**训练 loss 不能用来给这两个权重排序。**
> 另外两个 job 跑在不同型号的节点上，墙钟时长的 2.5 倍差距里含有节点差异，
> 不能全部归给 `n_obs_steps`；§8 的单机延迟测量才是干净的算力对照。

### 2.2 loss 轨迹

| step | 1 k | 10 k | 50 k | 100 k | 150 k | 200 k |
|---|---:|---:|---:|---:|---:|---:|
| `new_state5` | 0.193 | 0.057 | 0.032 | 0.022 | 0.019 | 0.016 |
| `new_obs2` | 0.188 | 0.058 | 0.033 | 0.024 | 0.020 | 0.014 |

两条曲线几乎重合，200 k 时都仍在缓慢下降。**打开本体感觉没有让训练 loss 下得更快**——
这本身就是 §7 结论的一个早期迹象。

---

## 3. 评测方法

### 3.1 用的是"部署忠实"的 harness，不是原始 chunk

本次没有沿用 `scripts_patch_policy_compare/eval_patch.py`（它评的是策略的原始输出），
而是按要求改用 `scripts_act_eval_test_fix/offline_chunk_eval.py` 的方法：
**评的是机器人真正被命令去执行的那串动作**。每个 anchor 都同时给出四个数：

| 名称 | 含义 |
|---|---|
| `policy_raw` | 策略输出的 chunk，截断到执行 horizon |
| `policy_deployed` | 同一 chunk 经过 `send_next_action_chunk` 的完整改写：去小回退 → 去开夹爪环路 → 二项式平滑 → 大波峰线性化 → **K=min(40, len) 的三次 Hermite 桥接** → 夹爪 clip 到 [0,1] |
| `hold_state` | null 基线：整段输出当前实测关节角，即"什么都不做" |
| `train_mean` | null 基线：输出训练集动作均值 |

改写代码不是重写的，而是**按文件路径从跑在机器人上的那个 checkout
（`/home/kewei/YING/lerobot_vlahost/src/lerobot/rollout/trajectory.py`）直接 import** 的。

**本次为该 harness 新增的唯一功能是观测历史支持**（`scripts_patch_policy_eval_fix/offline_chunk_eval.py`），
因为 patch_policy 的 `n_obs_steps > 1`，三处索引与 ACT 不同：

1. `observation.*` 的形状是 `(B, n_obs_steps, ...)`，anchor 是**最后一帧**——
   部署桥接的起点和 `hold_state` 都必须锚在这一帧上。
2. `action` 的形状是 `(B, n_obs_steps - 1 + action_chunk_size, A)`，因为
   `action_delta_indices` 从负数开始；**delta 0 在下标 `n_obs_steps - 1` 上**。
   从下标 0 开始评分就是拿 chunk 去比**过去**的动作，而且看上去完全合理。
3. `PatchPolicy.predict_action_chunk` 读的是 `select_action` 的部署队列而不是 batch，
   所以批量路径改调 `policy.model.predict`，图像按 `select_action` 的方式堆叠。

第 2 点是唯一能悄无声息出错的地方，因此单独留了一个可运行检查
（`check_alignment.py`，与不带 `delta_timestamps` 的同一数据集逐帧比对）：

```
patch_policy: n_obs_steps=5, action window -4..49, computed action offset 4
alignment OK: observation anchor = newest frame, action offset = n_obs_steps - 1
```

### 3.2 两个 horizon，因为部署配置只用 8 步

`deploy_config_act_dit.yaml` 用 `n_action_steps: 50`，但
**`deploy_config_patch_policy.yaml` 用的是 `n_action_steps: 8`**（`chunk_interval_s: 0.26667`）。
而 `send_next_action_chunk` 里 `bridge_steps = min(40, chunk.shape[0])`——
**chunk 只有 8 步时，桥接覆盖它的全部 8 步**。因此本报告给两套数：

* **horizon 50**：与前置报告、与 ACT-DiT 的部署配置可比，是"如果把窗口开大"的假设情形。
* **horizon 8**：**patch_policy 今天在机器人上的真实执行窗口**。§5。

指标定义：对 anchor *t*，取 chunk 的前 H 步与真值 `action[t … t+H-1]` 逐元素比 MAE，
按 `action_is_pad` 屏蔽超出 episode 末尾的步。表中 `@k` 一律是**前 k 步的累计均值**。

> 📐 **与前置报告的一处口径差异**：前置报告 §4 的 `hold_state` 一行是**逐步瞬时值**
> （0.0154 / 0.0386 / 0.0642 / 0.1128 / 0.1667），而策略各行是累计均值。
> 本报告所有行统一为累计均值。用本 harness 复算同一 null 的逐步瞬时值是
> 0.0156 / 0.0407 / 0.0678 / 0.1199 / 0.1752 @ step 1/8/16/32/50——**与前置报告一致**，
> 两套 harness 在同一数据上互相验证通过。

抽样 stride 20 → **2 007 个 anchor**（覆盖全部 53 个 episode）。

### 3.3 去污染核查

`--train-root` 打开时，harness 会对训练集每个 episode 的 `action` 数组做 SHA1 指纹，
凡指纹命中的评测 episode 一律丢弃。五次运行的输出都是：

```
contamination filter: 363 training episodes fingerprinted from batch_success_361
episodes_evaluated: 53   episodes_dropped_as_contaminated: 0
```

**评测集干净**，与 `scripts_patch_policy_compare/contam.json`（MD5 + 近似重复双重核查）一致。

### 3.4 采样噪声的地板

扩散头每次采样都会给出不同的 chunk。同一 anchor 换一个种子重跑一遍：

| run | seed 0 | seed 1 | 差 |
|---|---:|---:|---:|
| `new_state5` | 0.06121 | 0.06321 | **+3.3%** |
| `new_obs2` | 0.06190 | 0.06272 | +1.3% |
| `prev_diffusion` | 0.06773 | 0.06762 | −0.2% |

**任何小于约 3% 的差距都不可判别。** 下文凡在此范围内的比较一律标注为"噪声内"。

---

## 4. 精度：horizon 50（假设把部署窗口开大到 50 步）

N = 2 007，单位弧度，越小越好。`@k` = 前 k 步累计 MAE。

### 4.1 `policy_raw`（策略原始输出）

| run | @1 | @10 | @25 | @50 | RMSE | norm_MAE |
|---|---:|---:|---:|---:|---:|---:|
| `new_state5` | **0.03961** | **0.04277** | **0.04968** | **0.06121** | 0.10561 | 0.1886 |
| `new_obs2` | 0.04022 | 0.04342 | 0.04998 | 0.06190 | 0.10713 | 0.1932 |
| `prev_diffusion` | 0.04455 | 0.04810 | 0.05542 | 0.06773 | 0.11757 | 0.2081 |
| `prev_act_head` | 0.04107 | 0.04489 | 0.05192 | 0.06349 | 0.10542 | 0.1938 |
| **`act_baseline`（ACT）** | **0.02529** | **0.03115** | **0.03972** | **0.05112** | 0.09132 | 0.1575 |
| *null* `hold_state` | 0.01556 | 0.03174 | 0.05721 | 0.09705 | 0.20211 | 0.2988 |

### 4.2 `policy_deployed`（机器人真正收到的那串）

| run | @1 | @10 | @25 | @50 | RMSE | norm_MAE |
|---|---:|---:|---:|---:|---:|---:|
| `new_state5` | 0.01838 | 0.03138 | 0.04490 | **0.05827** | 0.10842 | 0.1801 |
| `new_obs2` | 0.01781 | 0.03068 | 0.04384 | 0.05838 | 0.10897 | 0.1814 |
| `prev_diffusion` | 0.01883 | 0.03188 | 0.04580 | 0.06206 | 0.11712 | 0.1909 |
| `prev_act_head` | 0.01774 | 0.03040 | 0.04365 | 0.05844 | 0.10509 | 0.1803 |
| **`act_baseline`** | **0.01652** | **0.02910** | **0.04013** | **0.05088** | 0.09535 | 0.1577 |
| *null* `hold_state` | 0.01556 | 0.03174 | 0.05721 | 0.09705 | 0.20211 | 0.2988 |

**四条读法：**

1. **两个新权重不可判别。** `policy_raw` 差 1.1%，`policy_deployed` 差 0.2%，
   都远小于 §3.4 的 3.3% 采样噪声。**`n_obs_steps=5` + 本体感觉，与 `n_obs_steps=2` 无本体感觉，
   打成平手。**
2. **相对上一代提升 9–10%，但混着 chunk 长度的变化。** `prev_diffusion` 0.06773 →
   新权重 0.0612–0.0619。这一项超过噪声，是真的；但 §2 已说明它同时包含
   `action_chunk_size` 64→50 的效应，**不能整条记到本体感觉或观测帧数头上**。
3. **对 `prev_act_head`（上一代最好的 patch_policy）只赢 2.5–3.6%，落在噪声边缘。**
   把扩散头 + 新配置换上去，相对上一代冠军**没有确定的收益**。
4. **ACT 基线赢 17%，且赢在每一个 horizon。** 而且注意 `@1` 一列：ACT 0.0253，
   patch_policy 全部在 0.0396–0.0446。**差距在第一帧就已经拉开**，见 §7。

### 4.3 逐关节：误差集中在腕转

`policy_raw` 整段 MAE：

| run | 14 个臂关节均值 | Joint2_L（最好） | **Joint7_L** | **Joint7_R** | grip_L | grip_R |
|---|---:|---:|---:|---:|---:|---:|
| `new_state5` | 0.06514 | 0.0302 | **0.1011** | **0.0958** | 0.0310 | 0.0365 |
| `new_obs2` | 0.06664 | 0.0304 | 0.0968 | 0.0989 | **0.0283** | **0.0292** |
| `prev_diffusion` | 0.07207 | 0.0320 | 0.1114 | 0.1123 | 0.0310 | 0.0436 |
| `prev_act_head` | 0.06778 | 0.0288 | 0.1092 | 0.1156 | 0.0328 | 0.0340 |
| `act_baseline` | **0.05518** | **0.0273** | **0.0882** | **0.0878** | **0.0222** | **0.0232** |

误差单调向远端关节集中，J7 是 J2 的 3.2–3.5 倍，与前置报告 §4.1 的形状一致。
**打开本体感觉没有改变这个分布**——如果模型真的在用关节角，最先受益的应该正是这些关节。

---

## 5. 精度：horizon 8——patch_policy 今天的真实部署窗口

`deploy_config_patch_policy.yaml` 下发 8 个 waypoint，桥接 `min(40, 8) = 8`。
**执行窗口内的每一步都是从实测位姿出发的 S 曲线，策略的唯一贡献是这条曲线的终点。**

| run | raw @1 | raw @4 | raw @8 | **deployed @1** | **deployed @4** | **deployed @8** |
|---|---:|---:|---:|---:|---:|---:|
| `new_state5` | 0.03961 | 0.04115 | 0.04215 | 0.01838 | 0.02249 | **0.03110** |
| `new_obs2` | 0.04022 | 0.04100 | 0.04253 | 0.01781 | 0.02198 | **0.03087** |
| `prev_diffusion` | 0.04455 | 0.04530 | 0.04738 | 0.01883 | 0.02332 | 0.03355 |
| `prev_act_head` | 0.04107 | 0.04222 | 0.04401 | 0.01774 | 0.02175 | 0.03106 |
| **`act_baseline`** | 0.02529 | 0.02717 | 0.02983 | **0.01652** | **0.01881** | **0.02432** |
| ***null* `hold_state`** | **0.01556** | **0.02106** | **0.02825** | **0.01556** | **0.02106** | **0.02825** |

**这是本报告最重要的一张表。**

* **四个 patch_policy 权重，在部署改写之后，全部劣于"什么都不做"**
  （0.0309–0.0336 vs null 0.0283）。两个新权重也不例外。
* **唯一赢过 null 的是 ACT 基线**（0.0243，比 null 好 14%）。
* 原始输出（`raw @8`）更糟：patch_policy 是 null 的 1.5–1.7 倍。
  **部署改写把它们从"明显更差"救到了"略差"**，但没有救过线。
* 前置报告 §4 用另一套口径得出过同一结论（"在部署窗口里两个模型都比站着不动更差"）。
  **本次用部署忠实的 harness、在打开本体感觉之后复现了它。**

> 这不等于"机器人执行这些权重会静止不动"。null 基线的含义是
> **在这个开环片段评测里，与示教轨迹的一致性**；真实任务成功率没有测（§9）。
> 但它确实说明：8 步窗口内，策略提供的信息量低于"保持当前位姿"这一条先验。

---

## 6. 部署改写阶梯：桥接就是全部

五个滤波器按部署顺序累加，每一级是上一级的输出加一个阶段（Δ 相对 `policy_raw`）。

**horizon 50：**

| 阶段 | `new_state5` | `new_obs2` | `prev_act_head` | `act_baseline` |
|---|---:|---:|---:|---:|
| 夹爪 clip | 0.06121 (+0.0%) | 0.06190 (+0.0%) | 0.06280 (−1.1%) | 0.05096 (−0.3%) |
| + 去小回退 | 0.06123 (+0.0%) | 0.06190 (−0.0%) | 0.06281 (−1.1%) | 0.05096 (−0.3%) |
| + 去开夹爪环路 | 0.06123 (+0.0%) | 0.06190 (−0.0%) | 0.06281 (−1.1%) | 0.05095 (−0.3%) |
| + 二项式平滑 | 0.06039 (−1.3%) | 0.06178 (−0.2%) | 0.06280 (−1.1%) | 0.05094 (−0.4%) |
| + 大波峰线性化 | 0.06039 (−1.3%) | 0.06178 (−0.2%) | 0.06280 (−1.1%) | 0.05094 (−0.4%) |
| **+ K=40 Hermite 桥接（完整）** | **0.05827 (−4.8%)** | **0.05838 (−5.7%)** | **0.05844 (−8.0%)** | **0.05088 (−0.5%)** |
| 只有桥接 | 0.05938 (−3.0%) | 0.05924 (−4.3%) | 0.05879 (−7.4%) | 0.05098 (−0.3%) |

**horizon 8（真实部署窗口）：**

| 阶段 | `new_state5` | `new_obs2` | `prev_act_head` | `act_baseline` |
|---|---:|---:|---:|---:|
| 前四级合计 | 0.04096 (−2.8%) | 0.04232 (−0.5%) | 0.04336 (−1.4%) | 0.02966 (−0.6%) |
| **+ 桥接（= 完整）** | **0.03107 (−26.3%)** | **0.03084 (−27.5%)** | **0.03103 (−29.5%)** | **0.02430 (−18.5%)** |

**两条读法：**

1. **四个滤波器合计影响 < 3%，桥接一项 −26% 到 −30%。** 与 ACT-DiT 报告
   （`scripts_act_eval_test_fix/README.md` §4）的结论形状相同：桥接不是滤波器，是替换。
2. **桥接对 patch_policy 的收益（−26～−30%）远大于对 ACT（−18.5%）。**
   桥接做的事就是"把 chunk 的起点强行拉回实测位姿"——**它是一次重锚定**。
   它能救回这么多，正说明 patch_policy 的误差里有多大一块是起点锚错了。
   前置报告 §5.2 用显式 re-anchor 量到 −18.8%（act 头）；这里的桥接是同一件事在部署侧的实现，
   而**打开本体感觉之后它依然能救回 26%**，即锚定问题原封不动。

---

## 7. 机制：本体感觉接进来了，但模型不看它

`probe_conditioning.py`，96 个 anchor（stride 401，跨全部 episode），
每次干预都**重置扩散采样种子**，所以两次输出之间的差异全部归因于干预本身。
`interframe_scale` = 两个不相关 anchor 的 chunk 之间的平均绝对差（≈0.35 rad），
即"两个完全不同的情境相差多少"——这是判读所有百分比的尺子。

| 干预 | `new_state5` | `new_obs2` | `prev_act_head` |
|---|---:|---:|---:|
| 三个相机全换成别的 anchor | 135.7% | 139.6% | 139.7% |
| 只换 `top` | 22.8% | 25.1% | 35.5% |
| 只换 `wrist_L` | 59.7% | 60.6% | 51.4% |
| 只换 `wrist_R` | 81.2% | 83.3% | 57.3% |
| 全灰图（无条件先验） | 108.4% | 92.9% | 89.9% |
| patch 网格 → 每帧一个 token | 55.6% | 66.2% | 35.9% |
| 1/4 分辨率 | 16.3% | 18.8% | 25.4% |
| **历史帧全部替换为最新帧** | **2.8%** | **1.5%** | 3.5% |
| **历史帧时间顺序倒转** | **0.5%** | **0.2%** | 1.0% |
| **换成别的 anchor 的本体感觉** | **6.7%** | — | — |
| **本体感觉置零** | **5.1%** | — | — |
| **本体感觉历史倒转** | **0.1%** | — | — |

### 7.1 本体感觉：接上了，但几乎不参与

`new_state5` 是这批权重里唯一 `use_robot_state=true` 的。把整个 16 维关节角
换成另一个 anchor 的，输出只移动 **6.7%**；直接置零，移动 **5.1%**。
作为对照，换掉图像移动 **136%**。

**这个 token 的影响力大约是图像的 1/20。**

更直接的证据是 chunk 首帧与实测位姿的距离：

| | chunk 首帧 vs 实测位姿 | 示教动作 vs 实测位姿 | 比值 |
|---|---:|---:|---:|
| `new_state5`（**有**本体感觉） | 0.04168 | 0.01405 | **2.97×** |
| `new_obs2`（无） | 0.03953 | 0.01405 | 2.81× |
| `prev_act_head`（无） | 0.04291 | 0.01405 | 3.05× |

**打开本体感觉之后，策略对"手臂现在在哪"的回答依然偏了 3 倍，与关掉时没有区别。**
前置报告把这一条列为"两个 head 共享的致命缺陷"，并建议打开 `use_robot_state`；
**这条建议被执行了，而缺陷没有消失。**

### 7.2 时间维度：依旧完全没用上

* 图像历史倒转：`new_state5` **0.5%**，`new_obs2` **0.2%**，`prev_act_head` 1.0%。
* 本体感觉历史倒转：**0.1%**。
* 历史帧全部换成最新帧的副本：**2.8% / 1.5%**。

与 memory `patch-policy-is-time-order-blind` 记录的 0.8% 是同一个数量级。
**`n_obs_steps` 从 5 降到 2 之后，剩下那一帧多余的历史仍然只值 1.5%。**
按这批测量，`n_obs_steps=1` 的代价上界约为 1.5%——远在采样噪声（3.3%）以内。

### 7.3 为什么 state token 没被用起来（三条假设，其中两条已于 2026-08-31 证伪）

三条可查的假设，按可验证性排序：

1. **淹没在 token 里。** 每帧 3 相机 × 256 patch = 768 个视觉 token，state 只占 1 个。
   可查：把 `state_projector` 的输出复制成 K 个 token，或给它单独一路加性条件。
2. **训练时是冗余特征。** 在示教数据里 `action_t ≈ state_t`（差 0.0140 rad），
   而图像已经足以定位到 0.04 rad——梯度可能从未要求模型去用这条更精确的通路。
   可查：训练一个只有 state、没有图像的对照，看它能到多少。
3. **归一化尺度。** state 走 `MIN_MAX` 归一到 [−1, 1]，与 DINOv2 的 patch 特征
   在同一个线性层里相加，两者的动态范围未做匹配。可查：看 `state_projector`
   输出的激活幅度与 patch token 的比值。

**这三条本报告都没有测**，列出来是为了下一次改动不再只翻一个布尔开关。

#### Update 2026-08-31：假设 1 和 3 已被证伪

`scripts_patch_policy_eval_fix/measure_state_token.py`，同一 checkpoint `new_state5`，
16 个 anchor，`batch_success_53_eval_data`：

| 量 | 实测 |
|---|---:|
| memory 槽位总数（1 timestep + 5 × 769） | 3 846 |
| 其中 state 槽位 5 个 → **均分份额** | 0.130 % |
| decoder cross-attention 落在 state 槽位上的质量 | **1.12 % = 8.65× 均分份额** |
| `‖state 槽位‖ / mean ‖patch 槽位‖`（memory 空间） | **0.76** |

* **假设 1（被 768 个 patch token 淹没）证伪。** 模型主动多看它 8.65 倍，不是看不见。
  推论：**把 state token 复制成 K 份不会有用**——只是把同一份 softmax 质量拆开。
* **假设 3（归一化尺度失配）证伪。** 范数 0.76，与 patch token 同量级，没有被压扁或炸开。
* **剩下假设 2。** token 被找到了、幅度正常、换掉它却只移动 6.7%——说明模型读了之后
  没有把它用在输出上。在示教数据里 `action_t ≈ state_t`（0.0140 rad），而图像已经能
  定位到 0.04 rad；在**绝对动作**参数化下，"把 state 抄进输出"这条恒等映射要穿过
  `MLP → 加性进 memory → cross-attention → 去噪器` 才能实现，梯度信号太弱，从未被学出来。

**结论换向：修 target，不是修 conditioning。** 把 `lerobot/policies/act_delta/` 的
`use_relative_actions` 移植进 patch_policy，让模型预测 `action − state`：恒等映射变成"输出 0"，
位姿锚定问题在结构上消失，而不是指望模型自己学会去重视那一个 token。
在 `batch_success_361`、chunk_size 50 上的 pre-gate 中位数 **0.529**（逐维 0.376–0.722，
§4.3 里最差的 J7 收缩最多），夹爪维 ≈0.99，所以保留 `--exclude-joints gripper`。
记录在 memory `patch-policy-state-token-is-read-not-drowned`。

### 7.4 `use_robot_state` 在代码里究竟改了什么（以及 ACT 为什么没有这个开关）

三处，加起来约 10 行。它是**条件输入开关，不是数据开关**：`observation.state` 一直在数据集里、
一直被 normalize，只是模型看不看它。

| 位置 | 改动 |
|---|---|
| `configuration_patch_policy.py:256` | `use_robot_state: bool = False`（reference 默认）；`:378` 只做一个校验：开了但数据集没有 state feature 就报错 |
| `modeling_patch_policy.py:363-368` | `tokens_per_frame += int(use_robot_state)`；建 `state_projector = MLP(16 → feature_dim)`，注释原文 `# The reference has no state pathway at all; this is a lerobot-side addition.` |
| `modeling_patch_policy.py:429-434` | `encode_observations` 里把投影后的 state token 拼到 patch token 后面 |

```python
if self.config.use_robot_state:
    state_token = self.state_projector(batch[OBS_STATE]).unsqueeze(2)  # (b, s, 1, e)
    patch_tokens = torch.cat([patch_tokens, state_token], dim=2)       # 每帧 768 -> 769
```

**没有改 loss、没有改 block-causal mask**（帧内 token 天然双向可见，README *Token layout* 一节
明说 mask 不需要变）、**没有改 decoder**。本报告配置下的量级：

| | `False`（`new_obs2` 等） | `True`（`new_state5`） |
|---|---:|---:|
| 每帧 token（DINOv2 ViT-S/14, 224², 3 相机） | 3 × 256 = 768 | **769** |
| diffusion memory（1 timestep + `n_obs_steps` × 每帧） | 3 841 @ 5 帧 | **3 846** @ 5 帧 |
| state 占 memory 的比例 | 0 | **0.130 %** |

唯一的隐藏副作用在 `vqbet` 头上（README *Deviation 5*）：VQ-BeT 读每帧 block 的**最后一个 token**，
开了之后读出点从"最后一个 patch"变成 state token。本报告全部是 `diffusion` / `act` 头，不受影响。

**ACT 没有这个开关。** `act` 用的是 `config.robot_state_feature`，而它是
`configs/policies.py:133` 的一个**派生属性**——数据集里有 `observation.state` 就自动接上，
没有配置项能关掉。所以前置报告 §7 那句"两个 head 表现一样是因为共用同一个
`use_robot_state=False`"能成立，前提正是 ACT 侧不存在这种失配。ACT 还**用了两次**
（`modeling_act.py`）：

1. `:412-421` **VAE encoder（仅训练时）**：输入序列是 `[cls, robot_state, *action_sequence]`,
   所以风格 latent 天然编码的是"相对于当前位姿的动作模式"。
2. `:461-476` **Transformer encoder**：token 列 `[latent, robot_state, (env_state), *image_patches]`,
   state 走 `nn.Linear(16, 512)` 并有**专属的 `encoder_1d_feature_pos_embed`**。

token 比例上 ACT 并不比 patch_policy 宽裕：ResNet-18 在 640×480 上给出 20×15 = 300 token/相机,
三相机 ≈900，state 同样是 1/900 左右。**所以差别不在稀释度**（上面的 Update 也独立证实了这点）,
而在三处：ACT 的视觉 backbone 端到端训练，梯度可以塑造图像特征去和 state 互补；VAE latent 把
state 与动作序列绑在一起编码；ACT 每次只看一帧，没有 5 × 768 个 patch 的时间维。

**为什么关掉本体感觉照样能训练、loss 照样降到 0.014。** 监督回归不要求输入充分，只要求输入与
目标相关。目标是 16 个绝对关节角，"从三张图回归当前位姿 + 未来 50 帧"是个有解的问题，梯度下降
一定会收敛到某个东西——§2.2 两条 loss 曲线几乎重合，打开本体感觉没让它下得更快。模型学到的是
"看画面像任务的哪个阶段，就输出那个阶段的平均位姿"：在 loss 上是好解（363 episode、单场景，
平均轨迹本身信息量就大），作为控制指令则封顶在前置报告 §5 量到的 **0.066 rad**,
而一个部署 waypoint 要表达的全部运动量只有 **0.032 rad**。
**能训练 ≠ 能训练好。** reference 敢默认关掉，是因为它跑的是 PushT 之类的俯视仿真基准,
末端位置在画面里一目了然、视觉本身就是充分观测；搬到顶视几乎看不见手臂、腕部相机是自我中心
视角的这套真机上，同一个默认值就从"合理简化"变成"关键输入缺失"。

---

## 8. 延迟：扩散头仍然跑不进部署窗口

batch 1，GPU 独占，预热 3 次后取 20 次的中位数。窗口 = `n_action_steps / 30`。

| run | 中位延迟 | p10–p90 | 编码器图像数 | 去噪步数 | 自身窗口 | 自身窗口 duty | **部署窗口 0.267 s 的 duty** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `new_state5` | **0.595 s** | 0.593–0.596 | 15 | 100 | 1.667 s | 0.36 | **2.23×** |
| `new_obs2` | **0.296 s** | 0.295–0.298 | 6 | 100 | 1.667 s | 0.18 | **1.11×** |
| `prev_diffusion` | 0.608 s | 0.607–0.609 | 15 | 100 | 1.067 s | 0.57 | 2.28× |
| `prev_act_head` | **0.0105 s** | 0.0104–0.0117 | 15 | — | 1.667 s | 0.006 | 0.04× |
| `act_baseline`（ACT） | **0.0065 s** | 0.0065–0.0077 | 3 | — | 3.333 s | 0.002 | 0.02× |

* **按今天的 `deploy_config_patch_policy.yaml`（`n_action_steps: 8` → 0.267 s 窗口），
  两个新权重都跑不动**：`new_state5` 要 2.23 个窗口，`new_obs2` 要 1.11 个。
* `n_obs_steps` 5→2 把延迟**减半**（0.595→0.296 s）：编码器图像数与去噪器的
  cross-attention memory 长度都随之减半。这是本次唯一确定为真的收益。
* **扩散头本身是主要成本**：同样 15 张图，`prev_act_head` 只要 10.5 ms，
  `new_state5` 要 595 ms——**57 倍**，全部来自 100 步 DDPM。
* 若把部署窗口改成 50 步（1.667 s），`new_obs2` 的 duty 是 0.18，**能跑**；
  但那时它的部署后精度是 0.0584，仍不如 ACT 的 0.0509，而 ACT 只要 6.5 ms。

---

## 9. 这份评测不是什么

* **不是闭环 rollout。** 每个 anchor 都是从**示教状态**开环打分。真机上第 N+1 个 chunk
  从第 N 个 chunk 把手臂留在的地方开始，Hermite 桥接会重锚在那个漂移后的位姿上。
  这是教师强制的片段评测，它给出行为的界，不模拟行为。
* **不是任务成功率。** 量的是与一条示教轨迹的一致性，不是东西有没有被收好。
  §5 的"输给 null"应当读作"在这个片段指标上"，不是"机器人会静止"。
* **不是部署的观测。** 图像走的是数据集的视频编解码，不是真机的 JPEG q90 + `INTER_AREA`
  分块；夹爪 `observation.state` 训练侧是命令回显、部署侧是过标定的真实反馈
  （见 memory `gripper-state-scale-mismatch`）。这两条离线数据集都无法复现。
* **评测集偏乐观。** `batch_success_53_eval_data` 与训练集零重叠，但分布很近：
  前置报告在更远的 held-out（`batch_1–4` 指纹过滤）上量到的位姿偏置是 0.066–0.070 rad，
  这里是 0.040–0.043 rad。**所有绝对数字都应理解为近分布 held-out 上的乐观值。**
* **两个新权重的对比不是受控消融**（§2）。§7 的探针是单 checkpoint 内部干预，那部分干净。

---

## 10. 建议

1. **两个新权重都不上机。** `new_obs2` 可作为后续实验的基座（同精度、一半算力），
   但它相对上一代冠军 `prev_act_head` 的优势（2.5–3.6%）落在噪声边缘。
2. **下一次只改一件事：把训练目标换成相对动作。**（2026-08-31 修订，原文建议先试
   "复制 state token"，已被同日的注意力测量否掉——见 §7.3 Update：token 拿到 8.65 倍
   均分份额的注意力、范数 0.76，**没有被淹没，复制它只会拆分同一份 softmax 质量**。）
   移植 `lerobot/policies/act_delta/` 的 `use_relative_actions`，让模型预测 `action − state`,
   保留 `--exclude-joints gripper`。再翻 `use_robot_state` 这个开关不会有任何变化。
3. **`n_obs_steps` 设为 1 或 2，不要 5。** 按 §7.2，第 5 帧到第 2 帧的信息量 ≤2.8%，
   第 2 帧到第 1 帧 ≤1.5%，都在噪声内；而算力是线性的。
4. **扩散头需要减去噪步数或换头。** 100 步 DDPM 占了延迟的 98%。
   先试 `num_inference_steps: 10`（DDIM），这是一个纯推理侧改动，不必重训，
   可以直接用本 harness 复测。
5. **如果要继续用 patch_policy，把部署窗口从 8 改到 50。** 8 步窗口里
   每一步都是桥接的 S 曲线，策略只贡献一个端点——那等于花 0.3–0.6 s 推理去决定一个点。
6. **重跑时固定所有策略超参。** 本次和上次的差异都来自节点默认值漂移，
   而不是设计。sbatch 里显式写死 `--policy.n_obs_steps` / `--policy.use_robot_state` /
   `--policy.action_head` / `--policy.action_chunk_size`。

---

## 11. 复现

```bash
S=/home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_patch_policy_eval_fix
cd $S

# 0) 历史索引自检（唯一能悄无声息出错的新逻辑）
/opt/robot-platform/train-venv/bin/python check_alignment.py

# 1) 精度，horizon 50（5 个 checkpoint，~35 min）
./run_eval.sh                    # -> new_state5.json / new_obs2.json / ... + run_eval.log

# 2) 精度，horizon 8 = 真实部署窗口（~15 min）
./run_eval_h8.sh                 # -> *_h8.json

# 3) 条件化探针：模型到底看什么（~6 min）
./run_probe.sh                   # -> probe_*.json

# 3b) state token 的注意力份额与范数（§7.3 Update，~1 min）
/opt/robot-platform/train-venv/bin/python measure_state_token.py \
  --checkpoint <new_state5>/run/checkpoints/200000/pretrained_model \
  --dataset-root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data \
  --out state_token_new_state5.json

# 4) 延迟（跑之前确认 GPU 无其他进程，~2 min）
/opt/robot-platform/train-venv/bin/python latency.py --out latency.json

# 5) 报告里的表
/usr/bin/python3 summarise.py .        # horizon 50
/usr/bin/python3 summarise.py . _h8    # horizon 8
```

单个 checkpoint 的最小命令：

```bash
D=/mnt/robot_platform/datasets/tidy_up_stationery_le
/opt/robot-platform/train-venv/bin/python offline_chunk_eval.py \
  --checkpoint <job>/run/checkpoints/200000/pretrained_model \
  --dataset-root $D/batch_success_53_eval_data --train-root $D/batch_success_361 \
  --stride 20 --batch-size 8 --num-workers 8 \
  --n-action-steps 8 --filter-ablation --seed-repeat 1 --out out.json
```

解释器必须是 `/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的同一环境）。
跑之前 `ulimit -n "$(ulimit -Hn)"` 并 `export LEROBOT_VIDEO_DECODER_CACHE_SIZE=400`
（见 memory `decoder-cache-eviction-leak`）。
