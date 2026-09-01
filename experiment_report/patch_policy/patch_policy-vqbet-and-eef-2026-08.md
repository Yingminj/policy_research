# 换动作空间比换条件化管用：EEF 权重是第一个真正能用的 patch_policy，而 `vqbet` 头被自己的码本卡死

**Checkpoint `vqbet`** `/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_361_2026-08-30_11-19-40-151329`
— `action_head=vqbet`，`n_obs_steps=5`，`use_robot_state=false`，16-D 关节动作，Slurm 145 @ **gpu04**，
**仍在训练：测量时 119 k / 200 k 步，评的是 100 k checkpoint**
**Checkpoint `eef`** `/mnt/robot_platform/jobs/patch_policy_tidy_up_stationery_le_batch_success_505_eef_2026-08-31_13-14-33-857400`
— `action_head=diffusion`，`n_obs_steps=2`，**`use_robot_state=true`**，**14-D 末端位姿动作**，Slurm 153 @ mgmt01，200 k 步已完成
**训练集** `batch_success_361`（363 ep）/ `batch_success_505_eef`（505 ep）
**评测集** 关节侧 `batch_success_53_eval_data`（53 ep / 40 132 帧，2007 anchor，与 08-30 报告同一份）；
EEF 侧 `batch_success_533_eef` 中未被训练集覆盖的 **67 个 episode**（2814 anchor，§3.2 解释为什么只能这样）
**测量日期** 2026-08-31，本机 mgmt01 RTX 4090（49 GB），`/opt/robot-platform/train-venv`（训练这些 checkpoint 的同一解释器）
**脚本与原始结果** `../test_scripts/scripts_patch_policy_eval_0831/`
**前置报告** `patch_policy-state-and-window-2026-08.md`（§7.3 留下的那条未证伪假设，本次被证实）、
`patch_policy-optimization-proposals-2026-08.md`、`patch_policy-head-comparison-2026-08.md`

---

## 1. 结论

**六句话：**

1. **EEF 权重是这一系列里第一个越过 ACT 的 patch_policy。** 在两个 checkpoint 都没见过的
   18 个 episode（742 anchor）上，horizon 50 的 MAE：patch_policy-EEF **0.03588**，
   ACT-EEF 0.05453，null `hold_state` 0.06933——**比 ACT 好 34%**。
   在完整的 67-episode 留出集上是 0.03323，**2.00 倍于 null**；
   位置误差 **10.6 mm**、姿态 0.058 rad（3.3°）、夹爪 0.026。
   关节侧的 patch_policy 从来没有超过 1.67 倍，而且一直输给 ACT 17%。
2. **前一份报告诊断的"策略不知道手臂在哪"被修好了，但不是靠 `use_robot_state`。**
   chunk 首帧与实测位姿的偏差 / 真实动作的同一比值：`new_state5` 2.97、`new_obs2` 2.81、
   `vqbet` **3.95**，而 EEF 权重只有 **1.41**。
   同时本体感觉**仍然是死的**：换掉另一个 anchor 的 state 只让输出移动 **4.2%**，
   置零 2.2%，把 state 历史倒序 **0.02%**——与 `new_state5` 的 6.7% 同一水平。
   **真正起作用的是把预测目标换成末端位姿本身**，这正是 08-30 报告 §7.3 三条假设里
   唯一没被当日测量证伪的那条（"目标参数化让本体感觉冗余"）。
3. **`vqbet` 头不必等它训完就可以判死刑。** 把**真值** chunk 送进冻结的
   编码器 / 量化器 / 解码器，重建误差是 **0.0769**——比这个策略自己的输出误差
   0.06723 **还大**。这不是硬下界（连续 offset 头可以把码本的错误补回来，实测正是如此），
   但它说明离散码本在这批数据上**不是有用的先验而是负担**。
   256 种码组合里只用到 **112** 种，前 10 种占 45.6%。
   **"离散行为基元"这个前提在这批数据上不成立。**
4. **`vqbet` 在机器人真正执行的窗口里是测过的所有权重里最差的一个。**
   `deploy_config_patch_policy.yaml` 的执行窗口是 8 步，桥接后
   `vqbet` 0.03572，null 0.02822——**0.79 倍，输给站着不动 27%**；
   同一尺子下 `new_state5` 0.91 倍、ACT 基线 1.16 倍。
   它 100 k 步仍在快速下降（50 k → 100 k 降了 15.7%），但要追上 `new_state5` 的 0.06121
   还需要再降 9%，而**训练成本是扩散头的 16 倍**（§2.1）。
5. **EEF 链路上没有 Hermite 桥接，而它不需要。** 关节侧的权重靠部署侧那条 K=40 的 S 曲线
   把误差压低 26–30%，EEF 侧 `policy_deployed` 与 `policy_raw` 完全相等——
   因为 EEF 权重的 chunk 首帧本来就落在实测位姿附近（第 2 条）。
   反证：在**训练集**上给 `vqbet` 加桥接反而让误差涨 7.8%。
6. **`deploy_config_eef.yaml` 有一个配置 bug。** `inference.n_action_steps: 60`，
   而 checkpoint 的 `action_chunk_size` 只有 50——多要的 10 步被静默截断，
   真实执行窗口是 1.67 s 不是 2.0 s。

**一句话的取舍：** **EEF 方向应当继续推进并准备上机**，`vqbet` 这条线可以停。
但在 EEF 权重上机之前有一个必须补的窟窿：**它至今没有在独立场次上被评过**
（§3.2）——现有的 67-episode 留出集是随机交错划分，比 53-episode 独立集容易。
下一步优先级：① 在部署机上把 53-episode 评测集的 rosbag 转成 EEF 版；
② 修 `n_action_steps`；③ `n_obs_steps` 降到 1（第二帧只贡献 1.9%，见 §6），推理直接减半。

---
## 2. 两个权重到底是什么

这一次的"两个 patch_policy 权重"**不是一次受控对比**：动作头、动作空间、训练集、
观测帧数、本体感觉开关全都不同，连训练完成度都不同。它们是两条独立的线索，
必须分开读。

| | `vqbet` | `eef` |
|---|---|---|
| job | `patch_policy_..._batch_success_361_2026-08-30_11-19-40-151329` | `patch_policy_..._batch_success_505_eef_2026-08-31_13-14-33-857400` |
| Slurm / 节点 | 145 @ **gpu04** | 153 @ **mgmt01** |
| `action_head` | **`vqbet`** | `diffusion` |
| 动作空间 | 16-D 关节（弧度 + 2 夹爪） | **14-D 末端位姿**（米 + 弧度 + 2 夹爪） |
| 训练集 | `batch_success_361`（363 ep / 301 k 帧） | `batch_success_505_eef`（505 ep / 419 k 帧） |
| `n_obs_steps` | 5 | 2 |
| `use_robot_state` | false | **true** |
| `action_chunk_size` | 50 | 50 |
| 可训练 / 总参数 | **49.03 M / 71.09 M** | 9.47 M / 31.53 M |
| 状态 | **仍在训练：119 k / 200 k** | 已完成 200 k |
| 可用 checkpoint | 50 k、100 k | 100 k、200 k |

命令行里同样没有指定 `action_head`、`n_obs_steps`、`use_robot_state` 中的任何一个
（`job.json` 的 `config` 段两者只差 `dataset_repo_id` 和 `save_freq`），
它们仍旧来自各自节点上 `train-venv` 的默认值——与 memory `act-dit-train-venv-defaults-drift`
和前两份报告记录的是同一个现象。**这一次连动作头都漂了。**

### 2.1 `vqbet` 的训练代价：第 20 k 步之后慢了 108 倍

`n_vqvae_training_steps=20000`，前 20 k 步只训练残差 VQ-VAE，之后才开始训练 BeT。
这个切换在 `slurm.out` 里是一个断崖：

| | step 1 k–20 k（VQ-VAE） | step 21 k 起（BeT） |
|---|---:|---:|
| `updt_s` | 0.011 s | **1.186 s** |
| `smp/s` | 115 | **13** |
| `mem_gb` | 4.69 | **21.58** |
| `loss` | 0.108（L1 重建） | 3.848 → 0.498（分类 + offset） |

两段的 `loss` 是**两个不同的目标函数**，不能连起来看，也不能和扩散头的 ε-MSE 比。

对照 `eef` 权重：全程 `updt_s` 0.073 s、`smp/s` 202、`mem_gb` 1.92，200 k 步
**4 h 27 m** 跑完。`vqbet` 到 119 k 步已经跑了 **34 h**，按当前速度还要 **27 h**。
**同一份 sbatch，因为默认值漂到了 `vqbet`，训练成本涨了约 16 倍。**

### 2.2 loss 轨迹

| step | 1 k | 10 k | 50 k | 100 k | 150 k | 200 k |
|---|---:|---:|---:|---:|---:|---:|
| `eef`（ε-MSE） | 0.195 | 0.061 | 0.036 | 0.027 | 0.020 | **0.017** |
| `vqbet`（VQ-VAE L1 → BeT） | 0.163 | 0.110 | 0.982 | 0.568 | — | — |

`vqbet` 在 119 k 步时 loss 0.498 仍在下降。**训练没跑完，下面所有 `vqbet` 的数字都是中途快照。**

---

## 3. 评测方法

### 3.1 harness

沿用 `../test_scripts/scripts_patch_policy_eval_fix/offline_chunk_eval.py`，
**逐字节复制**，只改了一处两行：`--train-root` 变成可重复（`action="append"`），
排除集取各 root 的并集。EEF 的公平对照需要同时排掉两个训练集，单个 root 做不到。
因此 `vqbet` 的数字与 08-30 报告的表**直接可比**（同一评测集、同一 stride、同一 2007 个 anchor）。

`check_alignment.py` 对两个 checkpoint 都通过（观测锚点 = 窗口最新帧，动作偏移 = `n_obs_steps - 1`）。
它在 EEF 集上先失败了一次：原版断言"`action[t-1]` 必定不等于 `action[t]`"，
而 EEF 录制里 **2.7%** 的相邻帧命令逐位相同（`held.py`，关节集是 2.55–2.80%，量级相同，
不是 EEF 特有的问题）。断言改成"8 个探针里至少有 1 个触发"，语义不变而不再误报。

### 3.2 EEF 权重没有属于自己的评测集

这是本次最大的方法学问题。53-episode 独立评测集
（`batch_success_53_eval_data`）**只有关节版**：它从未被转成 EEF，
而 `meta/conversion_manifest.json` 指向的原始 rosbag 在部署机 `snorlax` 上，不在本集群
（`/mnt/robot_platform/raw` 是空的）。EEF 的位姿来自 `/tj/info/eef_left|right` 这两个
**录制时就由 teleop_manager 算好的 FK 话题**，不是从关节角推出来的，所以也无法用现有数据补算。

退而求其次，用指纹划一个留出集（`splits.py`）：

```
batch_success_505 vs batch_success_505_eef，observation.velocity 指纹相同的 episode: 505 / 505
```

`observation.velocity` 在两个动作空间里都是 16 维，是唯一能跨空间认同一条 episode 的键。
它证明 `batch_success_505_eef` 是 `batch_success_505` 的**同一批录制的重投影**，
而不是另一次采集。据此：

| | episodes |
|---|---:|
| `batch_success_533_eef` 总数 | 535 |
| 其中 action 指纹**不在** `batch_success_505_eef` 里的 | **67** ← EEF 留出集 |
| 其中同时**不在** `batch_success_361_eef` 里的 | **18** ← 与 ACT-EEF 的公平对照集 |
| 53-episode 评测集出现在上述任何一个训练集里的 | **0** |

> ⚠️ **这是一个随机交错划分，不是独立场次。** 留出的 67 个 episode 序号散布在
> 0–506 之间，与训练 episode 同场次交错；53-episode 集则是一次独立采集。
> **随机划分更容易。** 因此本报告的 EEF 绝对数字**不能**和关节侧的绝对数字并排读，
> 只有各自"策略 / null 基线"的比值是跨空间可读的，而且这个比值本身也受划分难度影响。
> §5.3 用同一 checkpoint 的训练集内 / 留出集对比来给这件事定量。

### 3.3 留出集有多"越界"

checkpoint 用 `batch_success_505_eef` 的 MIN_MAX 统计做归一化，留出 episode 里
落在这个盒子外的帧（`ood.py`）：**56 278 帧中 731 帧（1.30%）**至少有一个通道越界，
最大越界 0.22 rad（`eef_r_roll`）。**可以忽略**，留出集与训练集基本同分布。

### 3.4 EEF 的部署链路上没有桥接

关节链路上，`send_next_action_chunk` 会用一条 K=40 的三次 Hermite 曲线把 chunk 的
前 40 步换成"从实测位姿出发"的 S 曲线——08-30 报告测出这条桥接承担了 26–30% 的误差修正。
**EEF 链路上没有这个东西。** `marvain_m6_eef_ros_robot.send_action_chunk` 把
waypoints 原样打包发给 VlaHost；关节侧的 ROS 机器人在
`marvain_m6_ros.py:611` 的 `chunk_blend_from_current` 也只是**在队首插入一个当前位姿的
waypoint**（一段 1/30 s 的插值），不是 40 步的 Hermite。VlaHost 服务端还做了什么，
不在这个 checkout 里，无法测量。

因此 EEF 侧一律用 `--filters none`，`policy_deployed` 与 `policy_raw` 相等，
表里两行数字相同**不是 bug**。关节侧的滤波阶梯照旧全测。

### 3.5 `deploy_config_eef.yaml` 要了 60 步，checkpoint 只有 50 步

`inference.n_action_steps: 60`，而 checkpoint 的 `action_chunk_size` 是 50。
harness 里 `horizon = min(n_action_steps, chunk_size)`，所以 `eef_200k_h60.json` 与
`eef_200k.json` **逐位相同**。真实执行窗口是 50 步（1.67 s），不是配置里写的 2.0 s。
对照的 ACT-EEF（`chunk_size=100`）不受影响，它的 h60 是真的 60 步。

---

## 4. 关节侧：`vqbet` 与 08-30 的五个权重同台

同一评测集、同一 harness、同一 2007 个 anchor，`new_*` / `prev_*` / `act_baseline`
五行直接取自 `../test_scripts/scripts_patch_policy_eval_fix/` 的原始 JSON，未重跑。

### 4.1 horizon 50，策略原始输出

| run | @1 | @10 | @25 | @50 | rmse | norm_mae | vs null |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`vqbet` @100 k** | 0.05352 | 0.05545 | 0.05901 | **0.06723** | 0.10857 | 0.2084 | 1.44x |
| `vqbet` @50 k | 0.06538 | 0.06787 | 0.07158 | 0.07971 | 0.12355 | 0.2453 | 1.22x |
| `new_state5` | 0.03961 | 0.04277 | 0.04968 | 0.06121 | 0.10561 | 0.1886 | 1.59x |
| `new_obs2` | 0.04022 | 0.04342 | 0.04998 | 0.06190 | 0.10713 | 0.1932 | 1.57x |
| `prev_diffusion` | 0.04455 | 0.04810 | 0.05542 | 0.06773 | 0.11757 | 0.2081 | 1.43x |
| `prev_act_head` | 0.04107 | 0.04489 | 0.05192 | 0.06349 | 0.10542 | 0.1938 | 1.53x |
| `act_baseline` | 0.02529 | 0.03115 | 0.03972 | 0.05112 | 0.09132 | 0.1575 | **1.90x** |
| *null* `hold_state` | 0.01556 | 0.03174 | 0.05721 | 0.09705 | 0.20211 | 0.2988 | — |

`vqbet` 在 100 k 步排在倒数第二，只比 `prev_diffusion` 好 0.7%。
**注意 @1 那一列**：0.05352，比所有扩散 / ACT 头都差 30% 以上。
chunk 的**第一步**是离散化最藏不住的地方——§8 给出这个数字的来源。

### 4.2 部署重写之后

| run | @1 | @50 | rmse | vs null |
|---|---:|---:|---:|---:|
| `vqbet` @100 k | 0.01806 | 0.05897 | 0.10591 | 1.65x |
| `new_state5` | 0.01838 | 0.05827 | 0.10842 | 1.67x |
| `new_obs2` | 0.01781 | 0.05838 | 0.10897 | 1.66x |
| `act_baseline` | 0.01652 | 0.05088 | 0.09535 | **1.91x** |

桥接把 `vqbet` 的 12.3% 误差抹掉（其余权重 4.8–8.4%），于是部署后它看起来和
`new_state5` 差不多。**这不是模型变好了，是桥接替它做的工作更多。**

### 4.3 horizon 8：机器人真正执行的窗口

`deploy_config_patch_policy.yaml` 的 `inference.n_action_steps` 是 8，
而桥接是 `min(40, len(chunk))`——**8 步里每一步都是桥接**。

> 这个配置文件不在 `lerobot_vlahost` 当前签出的分支（`cfw`）上，它在 `dev-yw` 的
> `f55ec43 add patch policy deploy`。本报告用的滤波器实现来自 `cfw` 的
> `src/lerobot/rollout/trajectory.py`（harness 按路径加载的就是它），
> 执行窗口 8 来自 `dev-yw` 的那份 YAML。两者若在部署时不配套，§4.3 整节的前提都要重看。

| run | policy_raw | policy_deployed | vs null（部署后） |
|---|---:|---:|---:|
| `vqbet` @100 k | 0.05501 | **0.03572** | **0.79x** |
| `new_state5` | 0.03961 | 0.03107 | 0.91x |
| `new_obs2` | 0.04022 | 0.03084 | 0.92x |
| `prev_act_head` | 0.04107 | 0.03103 | 0.91x |
| `act_baseline` | 0.02529 | 0.02430 | **1.16x** |
| *null* `hold_state` | — | 0.02822 | 1.00x |

结论与 08-30 报告一致且更极端：**在真实执行窗口里，唯一赢过 null 的仍然只有 ACT 基线**，
而 `vqbet` 是输得最多的一个。

### 4.4 采样噪声：`vqbet` 是确定性的

| run | seed 0 | seed 1 | spread |
|---|---:|---:|---:|
| `vqbet` @100 k | 0.06723 | 0.06725 | **+0.0%** |
| `new_state5` | 0.06121 | 0.06321 | +3.3% |
| `new_obs2` | 0.06190 | 0.06272 | +1.3% |

VQ-BeT 走的是 argmax 取码 + 确定性 offset，换种子不动。
好处是**它与其他权重的差距不需要跟采样噪声地板比**：0.06723 vs 0.06121 的 9.8% 是真差距。

### 4.5 训练进度：还在降，但降不到位

| step | 50 k | 100 k | 需要达到 |
|---|---:|---:|---:|
| `vqbet` horizon-50 MAE | 0.07971 | 0.06723 | 0.06121（`new_state5`）|

50 k → 100 k 降了 15.7%。**这是一个中途快照，不是终值**，200 k 时很可能落在
0.060–0.065，即"与 `new_state5` 打平上下"。§8 说明为什么不该为这个可能性再花 27 h。

---

## 5. EEF 侧：第一个越过 ACT 的 patch_policy

单位提醒：这 14 维里 xyz 是**米**（训练集全幅约 0.28–0.51 m），rpy 是**弧度**
（全幅 1.4–2.7 rad），夹爪是 [0, 1]。标量 MAE 把三种量纲平均在一起，
**分组表才是该读的那张**；标量只用来和同一 anchor 上的 null 比。

### 5.1 67-episode 留出集，horizon 50

| run | @1 | @10 | @25 | @50 | rmse | norm_mae | vs null |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`eef` @200 k** | **0.02170** | 0.02443 | 0.02816 | **0.03323** | 0.07548 | 0.1699 | **2.00x** |
| `eef` @100 k | 0.03441 | 0.03524 | 0.03701 | 0.04105 | 0.08687 | 0.2048 | 1.62x |
| *null* `hold_state` | 0.02283 | 0.03131 | 0.04518 | 0.06649 | 0.16578 | 0.3342 | — |
| *null* `train_mean` | 0.22910 | 0.22915 | 0.22942 | 0.22988 | 0.33833 | 0.8601 | — |

`policy_deployed` 与 `policy_raw` 逐位相同（§3.4：EEF 链路上没有滤波器栈）。
第二个扩散种子给出 0.03362（+1.2%），**所有差距都在采样噪声之外**。

**@1 那一列是本次最重要的一个数字。** 0.02170 < null 的 0.02283——
EEF 权重的 chunk **第一步就已经比"站着不动"好**。
关节侧的每一个 patch_policy 在 @1 上都比 null 差 2.5 倍以上
（`new_state5` 0.03961 vs 0.01556）。这就是"常数位姿偏置"这个老毛病的直接体检，
而它在这里**不存在**。

### 5.2 分组误差

| run | 位置（m） | 姿态（rad） | 夹爪（0–1） |
|---|---:|---:|---:|
| `eef` @200 k | **0.01060** | 0.05817 | 0.02632 |
| `eef` @100 k | 0.01258 | 0.07089 | 0.03697 |
| *null* `hold_state` | 0.02154 | 0.11413 | 0.05843 |

**位置 10.6 mm、姿态 3.3°**，都是 null 的一半。

### 5.3 与 ACT-EEF 的公平对照（18 个双方都没见过的 episode，742 anchor）

| run | @1 | @50 | rmse | norm_mae | vs null | 推理 |
|---|---:|---:|---:|---:|---:|---:|
| **`eef` @200 k** | **0.02439** | **0.03588** | 0.08264 | 0.1805 | **1.93x** | 297 ms |
| `act_eef`（361_eef） | 0.03267 | 0.05453 | 0.11366 | 0.2697 | 1.27x | 6.6 ms |
| *null* `hold_state` | 0.02744 | 0.06933 | 0.16756 | 0.3514 | — | — |

分组：位置 0.01087 vs 0.01558（好 30%），姿态 0.06319 vs 0.10051（好 37%），
夹爪 0.02897 vs 0.03340（好 13%）。**每一个横向切面 patch_policy 都赢。**

> ⚠️ **这不是干净的架构消融。** `act_eef` 训练在 `batch_success_361_eef`（363 ep），
> patch_policy 训练在 `batch_success_505_eef`（505 ep）——**多 39% 的数据**；
> 骨干（resnet18 vs 冻结 DINOv2 patch）、学习率（1e-5 vs 5.5e-5）也都不同。
> 这一栏说的是"当前手上最好的 EEF 权重是哪个"，不是"patch 架构优于 ACT"。
> 集群上确实有一个 `act_eef` 训练在 `batch_success_533_eef` 上（08-27），
> 但它覆盖了全部 67 个留出 episode，**没有任何干净的评测集**，所以无法入表。

### 5.4 记住了多少 vs 学会了多少

同一 checkpoint、同样大小的 anchor 预算，训练集内 vs 留出：

| | 训练集内 | 留出 | 比值 |
|---|---:|---:|---:|
| `eef` @200 k（EEF，同批次交错划分） | 0.02511 | 0.03323 | **1.32x** |
| `vqbet` @100 k（关节） | 0.03320 | 0.06723 | 2.02x |

EEF 的 1.32 倍是一个**干净的记忆量**：训练与留出来自同一批录制、同一场次，
差别只有"见过没见过"。**记忆带来的优势只有 32%**，其余是真的学到了。

`vqbet` 的 2.02 倍不能这样读：它的训练集是 `batch_success_361`，留出集是另一次独立采集
（`batch_success_53_eval_data`），**这个比值里混着记忆和场次漂移，两者无法分离**。

顺带一个反证：在训练集内给 `vqbet` 加部署桥接，误差**上升 7.8%**
（`filt_bridge_only` +13.7%）。桥接只在策略的起点错的时候才有用——
它是补丁，不是改进。

### 5.5 越界程度

留出 episode 里落在训练集 MIN_MAX 盒子外的帧：56 278 帧中 **731 帧（1.30%）**，
最大越界 0.22 rad。留出集与训练集基本同分布，上面的数字没有被归一化截断污染。

---

## 6. 条件化探针：本体感觉仍然是死的，但已经不重要了

96 个 anchor，每次干预都重置采样种子，所以整个位移都归因于干预本身。
`frac_of_interframe` 是"两个不相干 anchor 的输出差"这把尺子下的比例：
接近 1 表示这条通路被完全使用，接近 0 表示它是死的。

| 干预 | `eef` @200 k | `vqbet` @100 k |
|---|---:|---:|
| `all_cams_swapped` | 150.1% | 138.4% |
| `gray_images` | 101.3% | 101.7% |
| `swap_wrist_R` | 89.6% | 85.0% |
| `swap_wrist_L` | 56.2% | 72.4% |
| `patch_avg_pool` | 47.4% | 67.8% |
| `swap_top` | **23.7%** | **19.1%** |
| `quarter_resolution` | 14.7% | 25.0% |
| `history_frozen` | **1.9%** | **5.1%** |
| `history_reversed` | **0.3%** | **3.6%** |
| `state_swapped` | **4.2%** | —（`use_robot_state=false`）|
| `zero_state` | **2.2%** | — |
| `state_history_reversed` | **0.02%** | — |

三件事：

1. **本体感觉还是没被读。** EEF 权重打开了 `use_robot_state`，换掉整份 state 只让输出
   移动 4.2%，置零 2.2%，倒序 0.02%；同一把尺子下换掉图像移动 150%。
   与 08-30 报告在 `new_state5` 上测到的 6.7% / 5.1% 是同一结论，
   **打开这个开关依旧不解决问题**。
2. **但"策略不知道手臂在哪"这个后果消失了。** chunk 首帧与实测位姿的偏差，
   除以真实动作的同一偏差：

   | | `eef` | `new_state5` | `new_obs2` | `prev_act_head` | `vqbet` |
   |---|---:|---:|---:|---:|---:|
   | ratio | **1.41** | 2.97 | 2.81 | 3.05 | **3.95** |

   EEF 权重是唯一接近 1 的。它的定位信息来自**图像**（150% 的敏感度）
   和**目标本身就是位姿**这件事，不来自 state token。
   这正是 08-30 报告 §7.3 留下的那条假设——"目标参数化让本体感觉冗余"——
   的正面证据：**换目标有效，换条件化无效。**
3. **多出来的观测帧仍然一文不值。** `history_frozen` 把历史帧全部替换成最新帧，
   EEF 权重只动 1.9%，`vqbet`（5 帧）只动 5.1%；时间倒序更是 0.3% / 3.6%。
   与 memory `patch-policy-is-time-order-blind` 一致。
   **EEF 权重的 `n_obs_steps` 可以直接降到 1**，编码器开销减半（§7）。

摄像头的分工两个权重一致：**腕部相机主导，头部相机贡献最小**（19–24%）。
与 memory `act-policy-leans-on-background-pixels` 记录的"头部相机被当作场次线索"
不矛盾——那说的是 ACT，这里是"头部相机的信息量本来就低"。

---

## 7. 推理延迟（独占 GPU，batch 1，RTX 4090）

| run | 中位延迟 | 窗口 | duty | 编码图像 / 次 | 去噪步数 |
|---|---:|---:|---:|---:|---:|
| `vqbet` @100 k | **30.0 ms** | 1.667 s | 0.018 | 15 | — |
| `eef` @200 k | 296.7 ms | 1.667 s | 0.178 | 6 | 100 |
| `act_eef` | **6.6 ms** | 3.333 s | 0.002 | 3 | — |
| （参考）`new_state5` | 594.6 ms | 1.667 s | 0.357 | 15 | 100 |
| （参考）`act_baseline` | 6.5 ms | 3.333 s | 0.002 | 3 | — |

两点：

- **`vqbet` 推理快得多**（30 ms，比扩散头快 20 倍），但这是它唯一的优势，
  而代价是训练慢 16 倍（§2.1）。
- **EEF 权重的 duty 0.178 有充裕余量。** 把 `n_obs_steps` 从 2 降到 1（§6 证明第二帧无用）
  会让编码图像从 6 张降到 3 张，延迟大约减半，duty 降到 0.09。

---

## 8. `vqbet` 的天花板：码本重建比策略输出还差

VQ-BeT 的 chunk 是 `decode(code) + offset`。code 从
`vqvae_n_embed² = 16² = 256` 种组合里选一个（`ResidualVQ` 的层数在
`modeling_vqbet.py:779` 里写死为 2），由一个 256 类分类头挑。
所以在问"策略挑得准不准"之前，先问"码本能不能表达"：
把**真值** chunk 送进冻结的编码器 / 量化器 / 解码器，看回来的是什么
（`vq_floor.py`，2007 个 anchor，与 §4 同一批）。

| | 值 |
|---|---:|
| 码本大小 × RVQ 层数 | 16 × 2 = **256 种组合** |
| 评测集上实际用到的组合 | **112**（43.8%） |
| 前 10 种组合的占比 | 45.6% |
| 真值 chunk 的重建 MAE（归一化） | 0.0853 |
| **真值 chunk 的重建 MAE（原始关节单位）** | **0.0769** |
| 对比：策略自己的 horizon-50 输出误差 | 0.0672 |

**重建误差比策略的输出误差还大。** 这句话的含义要说清楚：
它**不是**一个策略无法跨越的硬下界，因为 offset 头是连续的、可以把码本的错误补回来——
实测正是如此（0.0672 < 0.0769）。它说明的是：

> 在这批数据上，离散码本不但没有提供有用的先验，反而是一个负担，
> 现在的精度是**连续 offset 头把码本的错误修回来之后**的结果。
> "把动作 chunk 离散成行为基元"这个前提在这里不成立。

再加上 §4.1 的 @1 误差（0.05352，比扩散头差 35%）——chunk 第一步是离散化最藏不住的位置——
和 §2.1 的 16 倍训练成本，**不必等 200 k 步就可以停掉这条线**。
如果一定要救，能改的是 `vqvae_n_embed`（16 → 64 以上）和 RVQ 层数（写死在 lerobot 里，
需要改上游），而不是训练更久。

---

## 9. 这次没能做到的事

1. **EEF 权重没有在独立场次上被评过。** 这是最大的窟窿。53-episode 评测集只有关节版，
   原始 bag 在部署机 `snorlax` 上（`/mnt/robot_platform/raw` 是空的），
   EEF 位姿又来自录制时的 `/tj/info/eef_left|right` 话题而非关节角 FK，**无法在本集群补算**。
   现有的 67-episode 留出集是随机交错划分，比独立场次容易。
   §5.3 的 1.93 倍和 §5.4 的 1.32 倍记忆比值都受这一点影响。
   **修复办法只有一个：在部署机上用同一条转换链把 53-episode 的 bag 转成 EEF 版。**
2. **`vqbet` 只有中途快照。** 100 k / 200 k。200 k 的结论可能与本报告不同——
   但 §8 的码本测量是**与训练进度无关的**，它对 100 k 和 200 k 同样成立。
3. **与 ACT-EEF 的对比混着数据量差异**（505 vs 363 episode），见 §5.3 的警告框。
4. **仍然是开环、teacher-forced 的分段评测。** 每个 anchor 都从**演示的**状态出发，
   机器人上 chunk N+1 从 chunk N 实际留下的位置出发。这个 harness 的既有局限，未变。
5. **VlaHost 服务端对 EEF chunk 做了什么，测不到。** 它不在这个 checkout 里（§3.4）。
   如果它也做了某种 blend，§5.1 的 `policy_deployed` 就还不是机器人上的真实曲线。

---

## 10. 下一步

按性价比排序：

| # | 动作 | 理由 | 成本 |
|---|---|---|---|
| 1 | 在部署机上转出 EEF 版的 53-episode 评测集 | §9.1，现在所有 EEF 结论的前提 | 一次转换 |
| 2 | 修 `deploy_config_eef.yaml` 的 `n_action_steps` 60 → 50 | §3.5，配置与权重不符 | 一行 |
| 3 | EEF 权重 `n_obs_steps` 2 → 1 重训 | §6 第 3 点，第二帧贡献 1.9%；延迟减半 | 一次训练（~4 h）|
| 4 | 停掉 `vqbet` 那条线 | §8 | 省 27 h |
| 5 | 关节侧权重也换成相对/末端目标参数化 | §6 第 2 点：换目标有效，换条件化无效 | 一次训练 |
| 6 | 若要继续 `use_robot_state`，先解决"state token 被读到但没用" | §6 第 1 点 + memory `patch-policy-state-token-is-read-not-drowned` | — |

第 5 条是本报告对前两份报告的直接回应：
`patch_policy-state-and-window-2026-08.md` §7.3 列了三条假设解释本体感觉为什么无效，
其中两条（注意力份额、范数）当日已被证伪，只剩"目标参数化让它冗余"。
**EEF 权重是这条假设的一次自然实验，而且它通过了**：
目标一变成末端位姿，`first_action_vs_state` 就从 2.97 掉到 1.41，
而 state token 的贡献仍然是 2–4%。要改的确实是**训练目标**，不是条件化方式。
