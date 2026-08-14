# `act_delta` @ tidy_up_stationery_le/batch_5：成功率未提升的归因分析

> **审计对象**
> - 权重：`/mnt/robot_platform/jobs/act_delta_tidy_up_stationery_le_batch_5_2026-08-13_03-52-51-314668`
> - 数据：`/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_5`
> - 实现：`/home/kewei/YING/lerobot_vlahost/src/lerobot/policies/act_delta/`
> - 部署日志：`/home/kewei/YING/robot_data_platform/record_chunk.txt`（21 个 chunk × 80 步，2 次 rollout）
> - 复现脚本：[`../scripts_act_delta_audit/audit_act_delta_batch5.py`](../scripts_act_delta_audit/audit_act_delta_batch5.py)
> - 撰写日期：2026-08-14。配套：[`ACT-experiment-plan-2026-08.md`](./ACT-experiment-plan-2026-08.md)（§2 相对动作方案）

---

## 结论先行

**成功率没提升，是因为这次训练根本没有启用相对动作。** checkpoint 里
`use_relative_actions=false`，训练命令里也没有这个参数。按 `modeling_act_delta.py:40-42`
自己的说法，此配置下 `ACTDeltaPolicy` 就是 **bit-for-bit 的上游 ACT**。也就是说这是一次
把 `--policy.type` 从 `act` 改成 `act_delta` 的、结果完全等价的重跑——**没有任何自变量被改变，
成功率不变是预期结果，不是 bug**。

在此之上还有三层问题，按修复优先级排列：

| # | 问题 | 等级 | 后果 |
|---|---|---|---|
| **A** | `use_relative_actions=false`——相对动作从未开启 | **决定性** | 实验自变量为空，本次运行 ≡ 基线 ACT |
| **B** | `meta/stats.json` 未在相对空间重算 | **阻断性** | 即使打开开关，下一次跑仍会失败（详见 §2） |
| **C** | 无验证集 / 无离线指标 / 无 eval | **方法论** | 106 epoch、30 条示教，无法区分"过拟合"和"进步"，也无法离线比较 R0/R1/R2 |
| **D** | 夹爪观测越界、`observation.velocity` 死配置、三份代码副本不同步 | **卫生** | 单独不致命，但会污染后续任何对照实验 |

另外一个必须先说清楚的量化结论：**在这个数据集上，相对动作即使完全正确地实现，预期收益也很有限。**
手臂 14 维的方差增益只有 **1.12–2.06×（平均 1.59×）**——见 §2.2。别把它当成能大幅拉高成功率的杠杆。

---

## 1. 根因：相对动作从未开启

### 1.1 三份证据

**(a) checkpoint 配置** ——
`run/checkpoints/last/pretrained_model/config.json`：

```json
"type": "act_delta",
"use_relative_actions": false,
"relative_exclude_joints": ["gripper"],
"relative_consistency_check": "warn",
"chunk_size": 100, "n_action_steps": 100
```

**(b) 训练命令** —— `job.sbatch` 里全部的 `--policy.*` 参数：

```
--policy.type act_delta   --policy.device cuda
--policy.use_amp false    --policy.push_to_hub false
```

没有 `--policy.use_relative_actions true`。

**(c) dataclass 默认值** —— `configuration_act_delta.py:123`：

```python
use_relative_actions: bool = False
```

三者一致：这次 200k step 训练出来的是**绝对动作 ACT**。

### 1.2 为什么没有任何报错

这是设计使然，但代价是它静默通过：

- `validate_relative_setup()`（`processor_act_delta.py:72-73`）**第一行就 `if not config.use_relative_actions: return`**。
  开关关着的时候，所有一致性检查——包括"stats 看起来是绝对空间的"这一条——**全部跳过**。
- `make_act_delta_pre_post_processors()` 照常构造 `RelativeActionsProcessorStep(enabled=False)` 和
  `AbsoluteActionsProcessorStep(enabled=False)`，两个 step 都退化成恒等映射
  （`relative_action_processor.py:133-134`）。
- 部署端 `deploy.py` / `rollout/context.py` 只在 `use_relative_actions=True` 时拒绝 `sync`；
  关着的时候什么都不检查。

所以从训练到部署，整条链路对"你以为开了但其实没开"这件事**没有任何一处会告诉你**。

### 1.3 修复

```bash
--policy.type act_delta \
--policy.use_relative_actions true \
--policy.relative_exclude_joints '["gripper"]' \   # R2；R1 用 '[]'
--policy.relative_consistency_check error          # 关键：别用默认的 warn
```

`relative_consistency_check` 必须设成 `error`。默认 `warn` 只会在 log 里打一行字，
在 200k step 的日志里根本不会有人看见——B 类问题就是这么漏过去的。

**校验方式**：训完后直接读 checkpoint，别信 log：

```bash
python -c "import json;c=json.load(open('<ckpt>/pretrained_model/config.json'));print(c['use_relative_actions'])"
```

排除掩码本身是对的：`relative_exclude_joints=["gripper"]` 对
`action_feature_names` 里的 `gripper_L` / `gripper_R` 命中，14/16 维走相对、2 维夹爪保持绝对
（即方案里的 R2 臂）。这一条不用改。

---

## 2. 阻断性问题：stats 未在相对空间重算

打开 §1.3 的开关后，下一次运行会**在另一个地方**失败。

### 2.1 当前 stats 是绝对空间的

`meta/stats.json` 里 `action` 的统计量是绝对关节角：

| | J1_L | J2_L | J4_L | J5_R |
|---|---|---|---|---|
| mean | 1.5856 | −0.9830 | −1.9005 | −0.1702 |
| std | 0.2125 | 0.1427 | 0.1621 | 0.6944 |
| **\|mean\|/std** | **7.46** | **6.89** | **11.73** | 0.25 |

`validate_relative_setup()` 在 `processor_act_delta.py:117-130` 就是用这个比值判断的
（对 14 个相对维取 `|mean|` 均值 / `std` 均值），当前数据算出来是 **3.19 > 1.0**，会触发告警。相对动作的目标均值应该在 0 附近
（实测 `|mean|/std` 全维 ≤ 0.082，见下表），绝对统计量则远大于 1。

后果如果不修：用绝对 std（≈0.21）去归一化本该用相对 std（≈0.14）归一化的目标，
再叠加均值偏移 ≈1.59 —— 网络学到的是一个被平移和缩放错了的目标空间。

### 2.2 相对动作在本数据集上能省多少方差（重要）

对全部 30,134 帧 × `chunk_size=100` 个时间偏移（3,013,400 个样本）实测：

| dim | abs_std | rel_std | **方差增益** | \|mean\|/std (abs) | \|mean\|/std (rel) |
|---|---|---|---|---|---|
| J1_L | 0.2126 | 0.1415 | 1.50 | 7.46 | 0.064 |
| J2_L | 0.1427 | 0.0846 | 1.69 | 6.89 | 0.049 |
| J3_L | 0.1570 | 0.1046 | 1.50 | 7.37 | 0.082 |
| J4_L | 0.1621 | 0.1311 | **1.24** | 11.73 | 0.059 |
| J5_L | 0.3033 | 0.1677 | 1.81 | 0.49 | 0.022 |
| J6_L | 0.1624 | 0.0990 | 1.64 | 4.28 | 0.054 |
| J7_L | 0.4307 | 0.2162 | 1.99 | 1.17 | 0.027 |
| J1_R | 0.3284 | 0.2894 | **1.13** | 5.11 | 0.001 |
| J2_R | 0.1690 | 0.1467 | **1.15** | 6.51 | 0.018 |
| J3_R | 0.2931 | 0.1463 | 2.00 | 3.69 | 0.023 |
| J4_R | 0.3408 | 0.3050 | **1.12** | 4.86 | 0.009 |
| J5_R | 0.6944 | 0.3368 | **2.06** | 0.25 | 0.004 |
| J6_R | 0.3566 | 0.2208 | 1.62 | 1.20 | 0.038 |
| J7_R | 0.3868 | 0.2121 | 1.82 | 0.28 | 0.004 |

**手臂 14 维平均方差增益 = 1.59×。** 这是相对动作在这个数据集上的收益上限。
对比一下：文献里相对动作（OpenPI 的 `DeltaActions`）真正见效的场景，
是底座会移动、绝对坐标漂移大的任务；这里是**固定底座 + 每条 episode 从同一个位姿起步**
（30 条 episode 首帧的手臂位姿标准差 ≤0.002 rad），绝对空间本身就已经很集中，
所以相对化能榨出来的东西不多。

### 2.3 还有一个容易忽视的副作用

实测 `k=0` 的 `|a_t − s_t|` 均值（rad）：

```
[0.0079, 0.0051, 0.0056, 0.0071, 0.0074, 0.0067, 0.0099,
 0.0124, 0.0073, 0.0070, 0.0133, 0.0121, 0.0178, 0.0098]
```

**action ≈ state**（这是主从遥操录制的典型特征：action 是指令位、state 是跟随位，跟得很紧）。
在相对表示下，chunk 前几步的回归目标 ≈ 0，模型输出常数 0 就能拿到近乎完美的 loss。
而 `NormalizerProcessorStep` 对每一维只用**一个跨所有 k 的 std**（≈0.14），
于是前几步的误差在归一化后被压得极小、后几步被相对放大——
**相对表示实际上把 loss 权重从 chunk 头部搬到了尾部**。

这跟本仓库为了抑制 chunk 间抖动而加的 `loss_front_weight`（加重第 0 帧）**方向相反**。
如果后续要同时用这两个机制，必须明确它们互相抵消，并重新调 `loss_front_weight`。
（注：本次训练两个字段都不存在，见 §4.3。）

### 2.4 修复

```bash
python -m lerobot.policies.act_delta.prepare_relative_stats \
    --root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_5 \
    --chunk-size 100 --exclude-joints gripper
```

`--chunk-size` **必须**等于 policy 的 `chunk_size=100`，`--exclude-joints` 必须和
`relative_exclude_joints` 完全一致；两者任一不匹配，统计量和实际目标分布就对不上。
然后按脚本打印的 stats view 路径训练。

---

## 3. 方法论问题：没有任何离线评估信号

这是为什么"成功率没提升"这个现象**花了一整轮真机实验才被发现**的原因。

### 3.1 现状

| 项 | 值 | 来源 |
|---|---|---|
| episodes | **30** | `meta/info.json` |
| frames | 30,134（25–43 s/条） | 同上 |
| splits | `{"train": "0:30"}` —— **无验证集** | 同上 |
| `env_eval_freq` | **0** —— eval 从未跑过 | `job.json` |
| `image_transforms.enable` | **false** —— 无图像增广 | `train_config.json` |
| steps / batch | 200,000 × 16 = 3.2M 样本 = **106.19 epoch** | `log.jsonl` 末行 |
| scheduler | `null`，lr 恒定 1e-5 | `train_config.json` |

训练 loss 曲线（**训练集**，不是泛化指标）：

```
step=  250  epoch=  0.13  loss=5.845
step=   6K  epoch=  3.45  loss=0.160
step=  22K  epoch= 11.95  loss=0.062
step=  62K  epoch= 33.19  loss=0.036
step= 126K  epoch= 67.17  loss=0.026
step= 200K  epoch=106.19  loss=0.022   ← 部署用的就是这个
```

30 条示教、106 个 epoch、关闭增广、没有验证集。loss 从 0.026 降到 0.022 的那 74k step
里发生了什么，**现有数据无法回答**——可能是继续学，也可能纯粹是在背这 30 条轨迹。

### 3.2 这直接卡死了实验本身

方案 §0.3 要求的离线 MAE 指标（用 `predict_absolute_chunk` 在同一物理空间比较 R0/R1/R2）
**一次都没有建立**。没有它：

- 无法在烧真机时间之前判断 `use_relative_actions` 有没有效果；
- 无法定位最佳 checkpoint（现在默认拿 `last`，即最过拟合的那个）；
- 无法把"相对动作没用"和"训练本身就过拟合了"这两个假设分开。

### 3.3 修复（建议在跑任何新 arm 之前先做）

1. **切出验证集**：30 条留 5 条（`--dataset.episodes` 指定训练用的 25 条），
   或直接用现成的 `eval_steps` 机制。
2. **建离线 MAE**：`inference_act_delta.py:52` 的 `predict_absolute_chunk()` 就是为此写的——
   它走 `predict_action_chunk`（latent=0，与部署一致）+ 完整 postprocessor，
   所以相对臂和绝对臂输出的是**同一个物理空间**的绝对动作，可直接比 MAE。
   按 chunk 内位置 k 分桶统计（k=0 / k=20 / k=50 / k=79），能同时看出 §2.3 的权重搬移效应。
3. **打开图像增广**：30 条示教 + ResNet18 + 106 epoch，`image_transforms.enable=true`
   基本是必需的。
4. **扫 checkpoint**：至少在 {50k, 100k, 150k, 200k} 上算离线 MAE 再决定部署哪个。

---

## 4. 卫生问题

### 4.1 夹爪观测越界（唯一确认的部署侧 train/deploy 失配）

21 个 chunk 起点观测中，**左夹爪 12 次、右夹爪 11 次超出训练范围**：

| dim | 部署观测范围 | 训练 state 范围 | 部署 max z | 训练 max z |
|---|---|---|---|---|
| gL | [0.012, **1.228**] | [0.000, 1.000] | **1.37** | 0.905 |
| gR | [0.019, **1.165**] | [0.000, 1.000] | **1.35** | 0.905 |

训练数据里夹爪 state 严格落在 [0,1]，部署时最高读到 1.228——**超出训练上界 23%**。
归一化后 z=1.37，而模型在训练中见过的最大值是 z=0.905。

原因是**动作路径有 clip、观测路径没有**：

- 发送侧 `marvain_m6_http.py:793` 显式 `np.clip(raw_gripper_values, 0.0, 1.0)`，
  注释也写明了 VlaHost 只接受 [0,1]；
- 接收侧 `marvain_m6_http.py:533` / `:542` 直接 `obs[...] = float(gripper_value)`，**无任何 clip**。

这也解释了日志里一个反复出现的现象：`|a0 − state|` 在 dim14 上稳定是 **0.18–0.20**——
模型正确预测 ≈1.004（它的训练天花板），而机器人回报 1.187。两者的差就是越界量。

**修复**：在 `_extract_gripper_pos` 之后对夹爪观测同样 clip 到 [0,1]，
或者查清服务端为什么会回报 >1（更可能是标定/量程问题，值得先查这个）。

> 顺带：预测动作里 gL 有 496/1680 低于 0、825/1680 高于 1（范围 [−0.132, 1.020]）。
> 这些**会被发送侧 clip 掉，不构成问题**。根因是夹爪在训练数据里是二值的
> （gL：44.0% <0.05，54.5% >0.95，中间态仅 **1.6%**），却被当作 MEAN_STD 连续量做 L1 回归。
> 不致命，但值得记一笔：这个表示方式让夹爪的开合时机天然模糊。

### 4.2 `observation.velocity` 是四方不一致的死配置

| 环节 | 状态 |
|---|---|
| 数据集 | `meta/info.json` 里有 `observation.velocity`，16 维，parquet 里真实存在 |
| checkpoint config | 声明为 input feature，`"type": "STATE"`，shape [16] |
| **ACT 模型** | **从不读取**。`ACTConfig.robot_state_feature`（`configs/policies.py:137`）只匹配 `ft_name == OBS_STATE`，即精确的 `"observation.state"`；`modeling_act.py:490-494` 只用 `OBS_STATE` 和 `OBS_ENV_STATE` |
| **部署机器人** | **从不提供**。`marvain_m6_http.observation_features`（`:400-406`）只产出 16 个 `<joint>.pos` + 相机 |

归一化器对缺失 key 静默跳过（`normalize_processor.py:281`：`... and key in new_observation`），
所以不会崩，**净效果为零**。但它会污染任何配置对比——看 config.json 的人会以为模型用了速度信息。

**修复**：要么从 input_features 移除，要么真的接进 encoder（作为独立的 state token）。
在做相对动作实验期间，建议先移除以冻住变量。

### 4.3 三份 lerobot 代码副本不同步

| 副本 | 路径 | `loss_time_decay` | `rollout/inference/chunk.py` | `marvain_m6_http` |
|---|---|---|---|---|
| **训练 venv** | `/opt/robot-platform/train-venv/.../lerobot` | ✗ | ✗ | ✗ |
| **部署 checkout** | `/home/kewei/YING/lerobot_vlahost/src/lerobot` | ✓ | ✓ | ✓ |
| **平台 checkout** | `/home/kewei/YING/robot_data_platform/lerobot/src/lerobot` | ✗ | ✗ | ✗ |

`act_delta` 目录下训练副本与部署副本有 **5 处文件差异**。已确认的实质差异：
训练侧的 `ACTConfig` / `ACTDeltaConfig` **没有** `loss_time_decay` / `loss_front_weight`，
`ACTPolicy.forward` 里也没有时间加权 loss 那段代码（`modeling_act.py:150-170`）。

这与 checkpoint 的 `config.json` 里查不到这两个字段完全吻合——**交叉验证了训练确实跑的是旧副本**。

由于两个字段的默认值（0.0 / 1.0）是恒等的，**这次的权重没有被影响**。但是：

- 本仓库为抑制 chunk 间抖动新加的时间加权 loss，**在这次训练里根本不存在**；
- 更普遍地说，"训练用的代码"和"部署/审计看的代码"不是同一份，
  任何"改了 X 之后效果变了"的结论都不可信。

**修复**：把训练 venv 改成指向 `lerobot_vlahost` 的 editable 安装（`pip install -e`），
或者在 `job.json` 里记录训练侧的 git commit。目前 `job.json` 里没有任何代码版本字段。

### 4.4 配置与实际不符

- **`n_action_steps`**：`deploy_config_act_delta.yaml` 写 100，`deploy_config_chunk.yaml` 写 50，
  而 `record_chunk.txt` 里每个 chunk **实测 80 步**。三个值都不一样，说明真机跑的时候有 CLI 覆盖，
  但覆盖值没有回写到任何 YAML。CLAUDE.md 说"YAML 是唯一真相"——这里破了。
- **`chunk_interval_s: 8`** vs 80 步 @30fps = **2.67 s**。只有走 `need_new_chunk`
  事件驱动路径（`rollout/inference/chunk.py:170-200`）才不会有问题；一旦服务端不发这个字段，
  会退化成"动 2.67 s、停 5.3 s"。日志确认当前走的是事件驱动路径，但这个配置本身是个雷。

### 4.5 相机切分后不做 resize（**需要现场确认**）

`observation_features`（`:400-406`）**声明**每个相机是 `(480, 640, 3)`，
但 `_split_quad_image`（`:246-282`）在 1+2 布局下返回的是 `h/3 × w/2`，
且 `top` 是把整条全宽大图 `cv2.resize` 压到 `(w_half, h_bottom)` 得到的——
**切分之后没有任何一步把图像 resize 到配置里声明的 480×640**。

训练数据的三路相机都是 480×640。如果运行时 quad_image 是 1280×960，
那么送进模型的就是 320×640，且 `top` 的视场比例与训练时不同。
ACT 的 ResNet18 backbone 不会因此报错（feature map 尺寸自适应），
所以这同样是一个**静默的 train/deploy 视觉域偏移**。

**没有实测运行时的 quad_image 尺寸，所以这一条我不下结论**。确认方法：

```python
# 在 marvain_m6_http.py:585 附近临时打印
logger.info("quad=%s → %s", quad_image.shape, {k: v.shape for k, v in camera_images.items()})
```

如果不是 (480, 640, 3)，就需要在切分后补一次 resize。

---

## 5. 排除掉的假设（这些**不是**问题）

为了不把后续调试引到错的方向，把已经量化排除的记下来：

| 假设 | 实测 | 结论 |
|---|---|---|
| 复位位姿不对 | 部署起始位姿 → 训练 episode 首帧最小 L2 = **0.0062 rad**（手臂 14 维） | 初始条件正确 |
| 关节观测跑出训练分布 | 手臂 14 维全部落在训练 min/max 内，\|z\| ≤ 2.30 | 观测正常（夹爪除外，见 §4.1） |
| 预测动作越界 | 手臂仅 J6_R 有 56/1680 轻微超上界（1.058 vs 1.047） | 可忽略 |
| chunk 边界跳变导致抖动 | 前 8 个 chunk 手臂跳变 ≤ 0.13 rad（中位数 0.054） | 开环衔接良好 |
| 动作被时间缩放 / 下发速率不对 | 每步 \|Δaction\| 部署/训练 = **1.17×**；80 步窗口弧长比 **1.14×** | 节奏与训练一致 |
| 相对/绝对转换写反了 | `use_relative_actions=false`，两个 step 都是恒等映射 | 这次根本没走这条路径 |
| exclude 掩码没命中 | `["gripper"]` 命中 `gripper_L`/`gripper_R`，14/16 维相对 | 掩码正确 |
| `ACTDeltaConfig` 漏字段 | AST 对比：`ACTConfig` 的字段 `ACTDeltaConfig` **一个不缺** | parity 完好 |

值得单独提一句：日志里 chunk 1–10 和 chunk 11–21 是**两次独立 rollout**
（chunk 1 与 chunk 11 的观测状态 L2 差仅 0.037 rad，都是 home 位姿），不是策略在打转。

另外，**chunk 9/10 的手臂边界跳变退化到 0.20 / 0.51 rad**（J5_R / J6_R），
明显高于前 8 个 chunk 的 ≤0.13 rad。这是 episode 后段策略已经跟丢的信号——
但在 §1 修好之前，追这个没有意义。

---

## 6. 建议的执行顺序

先把实验做成"能测出差异"的形态，再谈相对动作有没有用。

**第 0 步（先于一切）：建立离线评估**
- 切 5 条验证 episode；
- 用 `predict_absolute_chunk()` 实现按 k 分桶的绝对空间 MAE；
- 在现有这个 checkpoint 上跑一遍，拿到 **R0 基线数**。
  这一步不需要真机，也不需要重新训练。

**第 1 步：重算相对空间 stats**（§2.4）

**第 2 步：冻住变量**
- 移除 `observation.velocity`（§4.2）；
- 训练 venv 改 editable 指向 `lerobot_vlahost`，或在 job.json 记 commit（§4.3）；
- 打开 `image_transforms.enable`（§3.3）；
- 确认相机切分尺寸（§4.5）。

**第 3 步：重跑 R1 / R2**
- `--policy.use_relative_actions true --policy.relative_consistency_check error`；
- R2：`--policy.relative_exclude_joints '["gripper"]'`；R1：`'[]'`；
- 训完**先读 config.json 确认开关**，再算离线 MAE 与 R0 对比。

**第 4 步：只有当离线 MAE 显示出差异时，才上真机。**
部署必须用 `--inference.type=chunk` 或 `rtc`（`sync` 会漂移，`context.py` 会拒绝）；
先修 §4.1 的夹爪观测 clip。

**关于预期**：§2.2 的 1.59× 方差增益意味着，相对动作在这个数据集上大概率是个
**小幅改善而非质变**。如果目标是把成功率拉上去，§3 里"30 条示教 + 106 epoch + 无增广 + 无验证集"
这个组合，可能才是比动作表示更值得动的地方。

---

## 附：复现

```bash
/home/kewei/anaconda3/envs/lerobot/bin/python \
    /home/kewei/YING/paper/policy/experiment_report/scripts_act_delta_audit/audit_act_delta_batch5.py
```

脚本只读，输出六段分别对应本文 §1–§4 与 §5 的每一个数字。
注意必须用 `envs/lerobot` 这个环境——base env 的 pandas 与 numpy 2.x 不兼容。
