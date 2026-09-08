# 09-08 `patch_new_policy` 视觉骨干对：**ImageNet 冻结与随机可训练在 held-out 上不可区分（raw 差 +0.04 %，判读线实测为 0）**；可训练骨干多出来的 11 186 132 个可训练参数全部换成了记忆（seen −28.5 %，seen→unseen 3.63× vs 2.60×）；而这两个权重是整条 `patch_new_policy` 线上离线最好的两格（部署列 −6.3 %）

**评测集** `/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_success_53_eval_data_eef`
（53 ep / 40 132 帧 / **2007 anchor** / horizon 50 下 **97 090** 个有效动作步 /
stride 20 / 被判为污染而丢弃的 episode：**0**）
"见过的一半"对照集 `batch_success_505_eef`（505 ep / 419 330 帧 / stride 209 /
**2007 anchor** / **96 799** 个有效动作步 / 保留 505 集、丢弃 0 集）

**待评权重** —— 两份权重相对彼此有 **2 个** `config.json` 字段不同
（`probes/config_diff.txt` §3：55 个共有字段里其余 53 个逐位相同）

| 短名 | job | 相对彼此动的字段 | 视觉骨干 | 可训练参数 / 总参数 | 训练集 |
|---|---|---|---|---:|---|
| `imnet_frozen` | `patch_new_policy_..._505_eef_2026-09-07_19-36-54-349768` | —（本轮参照） | ResNet18 **ImageNet**，**冻结** | **9 120 526** / 20 306 658（44.9 %） | `batch_success_505_eef` |
| `rand_trained` | `..._2026-09-07_19-50-23-601027` | `vision_encoder` `resnet18_imagenet`→**`resnet18_random`** **且** `freeze_vision_encoder` `true`→**`false`** | ResNet18 **随机初始化**，**可训练** | **20 306 658** / 20 306 658（100 %） | 同上 |

两份全部 `run/checkpoints/{050000,100000,150000,200000}/pretrained_model`。
训练侧完全一致：同一个缓存目录 `/cache/datasets/batch_success_505_eef-c7a7cd44`、
seed 1000、200 000 步、batch 16、lr 5.5e-5 常数、无 scheduler、`image_transforms` 关闭；
两份归一化器**逐位相同**、两份 `model.safetensors` md5 互不相同
（`probes/config_diff.txt` §1/§2/§7）。两个 job 跑在**不同节点**（gpu04 / gpu03），
所以训练 wall-clock（4.33 h / 3.97 h）**不能横比**，本报告不据此下任何结论。

**测量日期** 2026-09-08，本机 mgmt01 RTX 4090（47.4 GiB），解释器
`/opt/robot-platform/train-venv/bin/python`（3.12.13 / torch 2.11.0+cu130，
训练这些 checkpoint 的同一环境）
**脚本与原始结果** `eval_policy/runs/20260908_pp_vision_encoder/`
（`results/*.json` 按仓库 `.gitignore` 约定只留在本地；本报告全部数字由
`summarise.py` 从这些 JSON 生成，见 `tables.md`；表号引用即 `tables.md` 的表号）

**与前置报告的关系**

- **填上** `MASTER-LEDGER-2026-09.md` §4.3 / §5.2 点名的头号空格：
  这两个权重"已训完、四个 checkpoint 齐全、**从没评过**、一次评测就能填"。现在评过了。
- **更正** `MASTER-LEDGER-2026-09.md` §1.5 / §5.2 对这两个权重的字段计数。
  账本记的是"A 动了 2 个字段、B 动了 3 个"（相对 09-04 `base`），这是对的；
  但它没有记下 **A 与 B 之间**只差 2 个字段，且这 2 个字段是**绑在一起翻的**
  （`vision_encoder` 与 `freeze_vision_encoder`）。账本据此判断"A 可以和 `base` 直接比"
  ——本报告 §2.1 说明为什么**A 与 B 之间**的比较才是本轮唯一受控的那一条，
  而它恰恰**无法**把差异归给初始化或冻结中的任何一个。
- **不推翻、并再次确认** `patch_policy-param-ablation-2026-09.md`（下称"09-07 报告"）§1.5：
  K=40 桥仍然是这张表上唯一的大数。本轮实测 **+16.4 %**（两臂相同），
  比 09-07 那轮的 12.0 %–15.5 % **还高**，且是两臂之间全部差距的 **377 倍**（raw）/ **152 倍**（部署），见本报告 §5.1 与 `tables.md` §12。
- **不推翻、并把证据从 4 个点扩到 6 个点** 09-07 报告 §1.4（记忆倍数与泛化收益反向）。
  本轮两个新点完全落在那条关系上（§6.3）。
- **修正一条方法学惯例，不是修正某份报告的数**：09-07 报告 §3.3 论证过
  `--seed-repeat 1` 会给出一条估得过紧的判读线。**那个论证对采样 head 成立，对本轮不成立**
  ——本轮两个权重的 `action_head` 是 `act`，确定性，判读线是**实测的 0**（§3.3）。
  两份报告不冲突：它们说的是两类不同的 head。

---

## 1. 结论

1. **两个视觉骨干在 held-out 上不可区分，而且这一次"不可区分"是有判读线撑着的最强版本。**
   `policy_raw` MAE `imnet_frozen` **0.03100** vs `rand_trained` **0.03102**（**+0.04 %**）；
   上机口径 `policy_deployed` **0.03608** vs **0.03612**（**+0.11 %**）。
   本轮判读线是**实测的 0.0000 %**——两次抽样逐位相同，因为 `act` head 不碰 `torch.randn`
   （§3.3）。所以这 +0.04 % 是**真的**，只是它小到没有工程意义。
   null 同批 anchor：`hold_state` **0.07266**、`train_mean` **0.22729**；
   两臂 raw 都是 hold 的 **2.34×**、train_mean 的 **7.33×**，部署后都是 **2.01×** / **6.30×**。
2. **flat MAE 平，不代表下面没动——而且动的方向是反的。**
   分组之后（§4.2）：`rand_trained` 位置 **9.32 mm** vs **9.72 mm**（**−4.2 %**）、
   夹爪 **0.0202** vs **0.0224**（**−9.8 %**），但朝向 **3.23°** vs **3.16°**（**+2.1 %**）。
   朝向维（弧度，量级 0.05）独占 flat MAE 的 **76.2 %–77.8 %**，位置维（米，量级 0.01）只占
   **12.9 %–13.4 %**，所以位置和夹爪的收益被朝向的退化按权重抹平成 +0.04 %。
   null：`hold_state` 位置 23.62 mm / 朝向 7.06° / 夹爪 0.0681。
3. **可训练骨干那 11 186 132 个参数（+123 % 可训练量）买到的全部是记忆，held-out 收益是 0。**
   见过的一半：`rand_trained` **0.00854** vs `imnet_frozen` **0.01194**（**−28.5 %**）、
   acc@0.25σ **0.990** vs 0.968、位置 **2.79 mm** vs 4.00 mm；
   而 seen→unseen 倍数 **3.63×** vs **2.60×**。
   **训练集上好 28.5 %，held-out 上差 0.04 %。** null 在 seen 侧是 `hold_state` 0.06750。
4. **上机口径下真正的大数仍然只有 K=40 桥，而且本轮比 09-07 那轮更大。**
   `policy_raw → policy_deployed`：两臂**都是 +16.4 %**（09-07 四臂是 +12.0 %~+15.5 %）。
   部署栈其余三级累积是 **−1.1 %~−0.9 %**（小幅改善），**全部代价都在桥这一级**（§5.1）。
   损伤集中在窗口前段：@1 **+58.1 %** / **+47.9 %**，@10 **+64.6 %** / **+55.7 %**，
   到 @50 收敛回 +16.4 %（§5.2）。**桥的代价是两臂之间全部差距的 377 倍（raw 口径）/ 152 倍（部署口径）。**
5. **这两个权重是整条 `patch_new_policy` 线上离线最好的两格——但这条排名不是归因。**
   同一批 2007 anchor、同一套 flag（§3.4 逐位验证）下跨轮读（§7）：
   raw **0.03100 / 0.03102** vs 09-04 `base` **0.03462**（**−10.5 % / −10.4 %**）；
   部署列 **0.03608 / 0.03612** vs 09-07 那轮最好的 `depth1` **0.03849**（**−6.3 % / −6.2 %**）。
   **但它们相对 `base` 各动了 2 个和 3 个字段**（`action_head` flow→act 也在里面），
   所以这 −10.5 % 里有多少是 ResNet18、多少是 `act` head，**本轮测不出来**。

**一句话：** **换视觉骨干这件事在 held-out 上没有分出胜负（+0.04 %，判读线 0），
"随机初始化 + 可训练"多花的 11 186 132 个可训练参数（+123 %）只买到了 28.5 % 的训练集记忆；
两个权重同时是这条线上离线最好的两格（−10.5 % vs 09-04 基线），但那是
`act` head 与 ResNet18 两件事绑在一起的结果，本轮拆不开。**

---

## 2. 评测集与权重审计（在任何精度表之前）

### 2.1 这不是单变量消融：两个字段是绑在一起翻的

`probes/config_diff.txt` §3 的逐字段比对：`rand_trained` 相对 `imnet_frozen`
**有 2 个字段不同**，55 个共有字段里其余 53 个逐位相同：

| 字段 | `imnet_frozen` | `rand_trained` |
|---|---|---|
| `vision_encoder` | `resnet18_imagenet` | **`resnet18_random`** |
| `freeze_vision_encoder` | `true` | **`false`** |

**"ImageNet vs 随机初始化"与"冻结 vs 可训练"在这一对权重里是混淆的。**
本报告任何一个数都不能归给其中单独一个。要拆开需要第三个权重
（`resnet18_imagenet` + `freeze=false`，或 `resnet18_random` + `freeze=true`），**它不存在**。
这一条写在所有表之前，因为它限制的是**全部**表的读法。

两个 preset 拖进来的东西（`probes/config_diff.txt` §4）：

| | preset | `pretrained` | `feature_key` | `output_dim` | `n_patches` | 冻结 |
|---|---|---|---|---:|---:|---|
| `imnet_frozen` | `resnet18_imagenet` | True | `x_norm_clstoken` | 512 | **1** | True |
| `rand_trained` | `resnet18_random` | False | `x_norm_clstoken` | 512 | **1** | False |

两个 preset 的 `n_patches` **都是 1**：`ResNet18Encoder` 强制 `feature_key = CLS_TOKEN`，
每台相机只回一个全局池化的 512 维向量（`patch_encoders.py:273-292`）。
所以**两臂都没有走 patch 网格那条路**——`dino_patch` 给的是 256 个 patch token，
ResNet18 给的是 1 个。这一点对读 §6.1 很重要。

还有一条不在字段 diff 里：`ResNet18Encoder.encode` 在**两臂上都**用 ImageNet 的 mean/std
做归一化，包括那个随机初始化的骨干——对它来说这两个常数没有任何意义。

### 2.2 冻结是真的冻结，可训练是真的训了（不是读 flag，是读权重）

`probes/config_diff.txt` §6 把两个 checkpoint 的骨干张量与本机缓存的
torchvision `ResNet18_Weights.IMAGENET1K_V1`（`resnet18-f37072fd.pth`，
md5 `e0b1c919e74f9a193d36871d9964bf7d`）逐张量比对：

| checkpoint | 比对张量数 | 与 ImageNet 逐位相同 | 最大绝对偏差 |
|---|---:|---|---:|
| `imnet_frozen` | 100 | **YES** | **0** |
| `rand_trained` | 100 | no | **23.6549** |

**`freeze_vision_encoder: true` 做到了它说的事**——200 000 步之后骨干仍然逐位是 ImageNet 权重。
另一臂偏差 23.65，说明它确实从随机初始化训起来了，而不是被静默留在初始化状态。
这两条都是从权重读出来的，不是从 config 的布尔值推出来的。

### 2.3 来源与污染：0 个 episode 被丢

`probes/splits.txt`：评测集 `batch_success_53_eval_data_eef` 的 53 个 episode，
按 `observation.velocity` 与 `action` 双指纹去查**全部**训练集
（361 / 505 / 361_eef / 505_eef / 533_eef），命中数**全部为 0**。
harness 侧独立复核：`contamination filter: 505 training episodes fingerprinted from
batch_success_505_eef`，8 次 held-out 评测**每一次**都是
`episodes_dropped_as_contaminated = 0`（`tables.md` §0、§13 断言）。

`probes/splits.txt` 同时确认 EEF 评测集与关节评测集是**同一批录制**
（53/53 速度指纹共享、episode 索引恒等置换）。

### 2.4 OOD 率：0.24 %，只有一个通道

`probes/ood.txt`：评测集 40 132 帧里，落在训练集 `batch_success_505_eef` 的 MIN_MAX 盒子
之外的有 **98 帧（0.24 %）**，最坏越界量 **0.0301**，全部集中在 `eef_r_x` 一个通道
（该通道 0.24 %，其余 13 个通道全是 0.00 %）。
**评测集基本落在训练分布内**，本报告的差异不是外推造成的。

### 2.5 对齐自检：两个 checkpoint 各一次，全过

`probes/check_alignment.txt`，两个 checkpoint 结果相同：

```
patch_new_policy: n_obs_steps=2, action window -1..49, computed action offset 1
negative control fired at 8 of 8 probes (a held command cannot fire it)
alignment OK: observation anchor = newest frame, action offset = n_obs_steps - 1
```

`action_delta_indices` 从 −1 起，所以 `batch["action"]` 前面带 1 行**过去**的动作；
从 index 0 打分会把 chunk 和过去比。偏这一行，每个数都还"看着合理"。
负对照 8/8 全部触发，说明这个检查本身不是恒真的。

---

## 3. harness：分叉、口径、判读线、跨轮可比性

### 3.1 分叉六处，顶层一个字节没动

顶层 `eval_policy/offline_chunk_eval.py` **未修改**（`git status` 干净，md5 `04e7f95d...`，
`--selftest` 四组全过）。本轮目录里的副本是
`runs/20260907_pp_new_param_ablation/offline_chunk_eval.py` 的**逐字节副本**
（md5 `792b7e58...`），相对顶层 6 处改动，完整 diff 见 `fork.diff`（+40 / −5）。
对本轮**必须**的是：

- `ACTION_LAYOUT`（4 处）：14-D EEF 空间里桥接 **12** 个位姿维、夹爪在第 **12/13** 列。
  不打这个补丁，桥会 Hermite 改写夹爪、`gripper_clip` 变成空操作
  （`probes/deploy_stack.txt` §4 实测 max|Δ| 分别是 **1.43375** 与 **0.49834**）。
  运行日志里 `action layout: {'n_bridge': 12, 'grippers': (12, 13)}` 是它生效的证据。
- `is_patch` 认 `patch_new_policy`：顶层的精确字符串比较会把这两个权重路由进 ACT 分支，
  在空 deque 上炸。
- 宽度 < 16 跳过 `gripper_loops`：否则 `--filter-ablation` 直接
  `ValueError: Gripper indices [14, 15] are outside action width 14`。该级在机器人上本就是 `False`。

分叉在 16-D 关节空间下与顶层**逐位相同**（`probes/deploy_stack.txt` 末行断言 `True`）。

### 3.2 部署配置正处在迁移中间，所以本轮报两列而不是一列

这是本轮相对 09-07 最重要的口径变化。`probes/deploy_mode.txt` 从部署 checkout 本身实测：

| 版本 | `inference.type` | 引擎 | `produces_chunks` | `send_next_action_chunk` | 对应 harness 列 |
|---|---|---|---|---|---|
| **工作区**（未提交修改） | **`sync`** | `SyncInferenceEngine` | **False**（继承 `InferenceEngine`） | **不跑** | **`policy_raw`** |
| **HEAD**（`751056e`） | **`chunk`** | `ChunkInferenceEngine` | **True**（自己覆盖） | **跑** | **`policy_deployed`** |

`strategies/base.py:69` 只有一行：`engine.produces_chunks` 为真才调 `send_next_action_chunk`。
三个引擎里只有 `ChunkInferenceEngine` 把它覆盖成 True，`sync` 和 `rtc` 都继承 False。
工作区那版还把 `policy.temporal_ensemble_coeff: 0.01` / `policy.n_action_steps: 1` 解注释了，
即 `runs/20260908_temporal_ensembling/` 正在研究的那条路。

**所以：`policy_raw` 是今天机器人上真正执行的东西，`policy_deployed` 是 HEAD 那版执行的东西。**
本报告两列都给，且每一处都注明是哪一列。

**注意**：复用的 `probes/deploy_stack.txt` 的 **§1 结论句是写死的字符串**
（它会正确打出 `inference.type = 'sync'`，然后照旧打 "**The rewrite runs.**"）。
读那份探针只信它的 §2/§3/§4；§1 以 `deploy_mode.txt` 为准。

`--filters` 因此取 `rollbacks,smoothing,bridge,gripper_clip` + `--filter-ablation`：
HEAD 那版栈里活着的正是这四级（`core.py:53-55` 中 `gripper_loops` 与 `excursions` 是 `False`，
`smoothing` 是无条件调用，加上驱动 `_prepare_action` 的夹爪 clip）。
一次跑同时拿到两个口径和中间逐级。`--filters none` 只答得了 sync 一半；
`--filters all` 会多算两级机器人已经关掉的。

**horizon = 60**，取自 `deploy_config_eef.yaml:90` 的 `inference.n_action_steps`
（工作区与 HEAD 都是 60，`deploy_mode.txt` §1）。两个权重 `action_chunk_size` 都是 50，
harness 夹到 **50**——**部署要 60 步、策略只给得出 50**，这个夹取是部署事实，不是评测选择。

### 3.3 判读线：本轮是实测的 0，而不是估出来的

`tables.md` §1：

| checkpoint | head | 抽样次数 | mae（第 0 次） | mae（第 1 次） | 极差 | acc@0.25σ 极差 | eef 位置极差 |
|---|---|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | `act` | 2 | 0.03100421 | 0.03100421 | **0.0000 %** | 0.0000 % | 0.0000 % |
| `rand_trained` | `act` | 2 | 0.03101767 | 0.03101767 | **0.0000 %** | 0.0000 % | 0.0000 % |

两个权重的 `action_head` 都是 **`act`**：`PatchNewPolicyModel.predict` 直接
`return self.trunk(patch_tokens, state=state)`，全程不经过 `torch.randn`
（`modeling_patch_new_policy.py:536-538`）。重抽得到的是**逐位相同**的 chunk。

**这与 09-07 报告 §3.3 不矛盾。** 那轮四个 head 全从 `torch.randn` 起步，
`--seed-repeat` 在**估计**一条非零噪声线，估紧了会把噪声读成信号；
本轮 `--seed-repeat` 在**证明**噪声为 0。两份报告说的是两类 head。
本轮所有差异按面值读——包括 +0.04 % 这种小到没有工程意义的差。

### 3.4 跨轮可比性：不是假设，是断言

`summarise.py --selftest` 的 11 条断言全过（`tables.md` §13），其中 5 条是
**与 09-07 那一轮逐位对齐**的证据——这些量只取决于评测集和 flag，与 checkpoint 无关：

| 量 | 本轮 | 09-07 报告 | 一致 |
|---|---:|---:|---|
| `hold_state` null MAE | **0.07266** | 0.07266 | ✓ |
| `train_mean` null MAE | **0.22729** | 0.22729 | ✓ |
| held-out anchor 数 | **2007** | 2007 | ✓ |
| held-out 有效动作步 | **97 090** | 97 090 | ✓ |
| seen 侧 anchor / 动作步 | **2007 / 96 799** | 2007 / 96 799 | ✓ |

另外断言：两个 null 在本轮 8 次 held-out 评测之间**逐位相同**（它们本就不该依赖 checkpoint）；
chunk / horizon / batch_size / stride / seed / filters / latency / n_obs_steps / action offset
在每一次 held-out 评测之间逐位相同；两次重抽逐位相同；
harness 从每个 checkpoint 自己的 config 读出的 `policy_vision_encoder` 是预期的那两个值。

**§7 的跨轮表因此是可读的**，而不是"看着像同一个尺度"。

---

## 4. 主表

### 4.1 `policy_raw`（= 今天工作区 `sync` 配置下机器人真正执行的东西）

| run | 配置 | mae | vs 参照 | rmse | norm_mae | tail | @1 | @10 | @25 | @50 | vs `hold_state` | vs `train_mean` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | ImageNet，冻结 | **0.03100** | +0.0 % | 0.07009 | 0.15946 | 1.74 | 0.01354 | 0.01752 | 0.02341 | 0.03100 | 2.34× | 7.33× |
| `rand_trained` | 随机，可训练 | **0.03102** | **+0.04 %** | 0.06970 | 0.15714 | 1.75 | 0.01442 | 0.01862 | 0.02422 | 0.03102 | 2.34× | 7.33× |
| *null* `hold_state` | — | 0.07266 | — | 0.17191 | 0.36402 | 1.81 | 0.02098 | 0.03138 | 0.04757 | 0.07266 | — | — |
| *null* `train_mean` | — | 0.22729 | — | 0.33229 | 0.84181 | 1.17 | 0.22632 | 0.22650 | 0.22676 | 0.22729 | — | — |

判读线 **0.0000 %**。+0.04 % 因此是真实差异，但桥的代价是它的 **377 倍**（`tables.md` §12）。
注意 `rand_trained` 的 **rmse 更低**（0.06970 vs 0.07009）而 **mae 略高**——
即它的大误差更少、小误差更多；这与 §4.2 的分组结果一致。

### 4.2 分组误差：14 个动作维不共享单位

位置维是米、朝向维是弧度、夹爪维是 [0, 1]，flat MAE 把三种单位平均在一起。
`mae_per_joint` 重新分组：

| run | 口径 | 位置（mm） | vs 参照 | 朝向（度） | vs 参照 | 夹爪（0-1） | vs 参照 |
|---|---|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | `policy_raw` | 9.72 | +0.0 % | 3.16 | +0.0 % | 0.0224 | +0.0 % |
| `rand_trained` | `policy_raw` | **9.32** | **−4.2 %** | 3.23 | **+2.1 %** | **0.0202** | **−9.8 %** |
| `imnet_frozen` | `policy_deployed` | 11.68 | +0.0 % | 3.77 | +0.0 % | 0.0201 | +0.0 % |
| `rand_trained` | `policy_deployed` | **11.39** | **−2.5 %** | 3.83 | **+1.5 %** | **0.0182** | **−9.4 %** |
| *null* | `hold_state` | 23.62 | — | 7.06 | — | 0.0681 | — |
| *null* | `train_mean` | 45.43 | — | 18.64 | — | 0.4789 | — |

**为什么 flat MAE 是平的**（`tables.md` §4）：

| run | 口径 | flat mae | 位置占比 | 朝向占比 | 夹爪占比 |
|---|---|---:|---:|---:|---:|
| `imnet_frozen` | `policy_raw` | 0.03100 | 13.4 % | **76.2 %** | 10.3 % |
| `rand_trained` | `policy_raw` | 0.03102 | 12.9 % | **77.8 %** | 9.3 % |
| `imnet_frozen` | `policy_deployed` | 0.03608 | 13.9 % | **78.2 %** | 7.9 % |
| `rand_trained` | `policy_deployed` | 0.03612 | 13.5 % | **79.3 %** | 7.2 % |

**朝向维独占 flat MAE 的四分之三以上。** `rand_trained` 在位置（−4.2 %）和夹爪（−9.8 %）
上的收益，被朝向（+2.1 %，权重 76 %）的退化按权重抵消成 +0.04 %。
**只看 flat MAE 会把这一对读成"完全一样"，那是错的读法。**

逐维（`policy_raw`，`tables.md` §4）：

| run | `eef_l_x` | `eef_l_y` | `eef_l_z` | `eef_l_roll` | `eef_l_pitch` | `eef_l_yaw` | `eef_r_x` | `eef_r_y` | `eef_r_z` | `eef_r_roll` | `eef_r_pitch` | `eef_r_yaw` | `gripper_L` | `gripper_R` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | 0.0100 | 0.0099 | 0.0079 | 0.0601 | 0.0382 | 0.0816 | 0.0104 | 0.0106 | 0.0096 | 0.0502 | 0.0445 | 0.0563 | 0.0234 | 0.0214 |
| `rand_trained` | **0.0090** | **0.0091** | **0.0078** | 0.0614 | 0.0414 | **0.0767** | **0.0103** | **0.0098** | 0.0099 | 0.0562 | 0.0446 | 0.0575 | **0.0187** | 0.0217 |
| *null* `hold_state` | 0.0208 | 0.0212 | 0.0176 | 0.1157 | 0.0989 | 0.1688 | 0.0310 | 0.0271 | 0.0240 | 0.1470 | 0.1117 | 0.0973 | 0.0681 | 0.0680 |

`rand_trained` 在 **6 个位置维里赢 5 个**、`gripper_L` 赢 20 %，
在 **6 个朝向维里输 5 个**（只有 `eef_l_yaw` 赢）。**方向是系统性的，不是抖动。**

### 4.3 末端位姿误差（欧氏位置 / 测地朝向）

| run | 口径 | 左位置(mm) | 左朝向(°) | 右位置(mm) | 右朝向(°) | 平均位置(mm) | vs 参照 | 平均朝向(°) | vs 参照 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | `policy_raw` | 18.82 | 5.23 | 20.22 | 5.41 | 19.52 | +0.0 % | 5.32 | +0.0 % |
| `rand_trained` | `policy_raw` | 17.68 | 5.30 | 20.04 | 5.58 | **18.86** | **−3.4 %** | 5.44 | **+2.3 %** |
| `imnet_frozen` | `policy_deployed` | 22.65 | 6.24 | 24.37 | 6.01 | 23.51 | +0.0 % | 6.13 | +0.0 % |
| `rand_trained` | `policy_deployed` | 22.06 | 6.27 | 24.06 | 6.09 | **23.06** | **−1.9 %** | 6.18 | +0.9 % |
| *null* | `hold_state` | 41.10 | 11.51 | 55.01 | 12.75 | 48.06 | — | 12.13 | — |
| *null* | `train_mean` | 76.51 | 28.42 | 101.55 | 49.21 | 89.03 | — | 38.81 | — |

同一个形状：位置好、朝向差。判读线 0，所以 −3.4 % 与 +2.3 % 都是真实的。

绝对容差下的命中率（`tables.md` §5b）：

| run | 口径 | L 5mm | L 10mm | L 25mm | L 50mm | R 5mm | R 10mm | R 25mm | R 50mm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | `policy_raw` | 0.102 | 0.316 | 0.773 | 0.950 | **0.123** | **0.362** | 0.725 | 0.927 |
| `rand_trained` | `policy_raw` | **0.104** | **0.336** | **0.784** | **0.964** | 0.110 | 0.326 | **0.732** | **0.937** |
| `imnet_frozen` | `policy_deployed` | 0.088 | 0.250 | 0.682 | 0.912 | **0.123** | **0.316** | 0.641 | 0.874 |
| `rand_trained` | `policy_deployed` | 0.088 | 0.254 | 0.682 | **0.924** | 0.106 | 0.295 | **0.654** | **0.881** |
| *null* | `hold_state` | 0.063 | 0.153 | 0.447 | 0.709 | 0.074 | 0.167 | 0.401 | 0.662 |

左臂 `rand_trained` 全线略好，右臂在紧容差（5 mm / 10 mm）上反而略差。
**两臂不同向**，这也是"没有一个统一赢家"的一个侧证。

### 4.4 `acc@τ`（动作向量空间，τ 是每维自身动作 σ 的倍数）

| run | 口径 | @0.1σ | @0.25σ | @0.5σ | @1σ | @0.25σ 第 1 步 → 窗口末 |
|---|---|---:|---:|---:|---:|---|
| `imnet_frozen` | `policy_raw` | **0.565** | 0.803 | 0.929 | 0.987 | 0.961 → 0.803 |
| `rand_trained` | `policy_raw` | 0.559 | **0.808** | **0.935** | 0.987 | 0.956 → 0.808 |
| `imnet_frozen` | `policy_deployed` | **0.527** | 0.760 | 0.902 | 0.977 | 0.900 → 0.760 |
| `rand_trained` | `policy_deployed` | 0.523 | **0.763** | **0.907** | 0.978 | 0.900 → 0.763 |
| *null* | `hold_state` | 0.428 | 0.625 | 0.775 | 0.892 | — |
| *null* | `train_mean` | 0.052 | 0.131 | 0.288 | 0.662 | — |

**紧容差（@0.1σ）`imnet_frozen` 赢，松容差（@0.25σ / @0.5σ）`rand_trained` 赢。**
与 §4.1 的 "rmse 更低但 mae 更高" 一致：可训练骨干的误差分布更集中、尾巴更短，
但中心附近的精度略差。两者都远在 `hold_state` 之上（@0.25σ 0.625）。

---

## 5. 部署窗口：重写掉了多少，掉在哪一级、哪几步

**只在 HEAD 那版配置（`inference.type: chunk`）下读这一节。**
工作区当前是 `sync`，这一整套重写**根本不跑**（§3.2）。

### 5.1 逐级归因：全部代价都在桥这一级

| run | `raw` | `clip_only` | `rollbacks` | `gripper_loops` | `smoothing` | `excursions` | `bridge` | `bridge_only` | `deployed` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | 0.03100 | 0.03067 | 0.03067 | 0.03067 | 0.03065 | 0.03065 | 0.03608 | 0.03611 | 0.03608 |
| `rand_trained` | 0.03102 | 0.03073 | 0.03073 | 0.03073 | 0.03072 | 0.03072 | 0.03612 | 0.03609 | 0.03612 |

相对 `policy_raw` 的百分比：

| run | `clip_only` | `rollbacks` | `gripper_loops` | `smoothing` | `excursions` | `bridge` | `bridge_only` | `deployed` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | −1.1 % | −1.1 % | −1.1 % | −1.1 % | −1.1 % | **+16.4 %** | +16.5 % | **+16.4 %** |
| `rand_trained` | −0.9 % | −0.9 % | −0.9 % | −1.0 % | −1.0 % | **+16.4 %** | +16.4 % | **+16.4 %** |

（`gripper_loops` 在机器人上是 `False`，且在动作宽度 14 下被分叉跳过，所以该级按构造重复上一级。）

**夹爪 clip + rollbacks + smoothing 三级合计是 −0.9 %~−1.1 %，即小幅改善；
K=40 桥单独 +16.4 %。** `bridge_only`（只上桥、不过前三级）与 `deployed` 几乎相同
（+16.5 % vs +16.4 %），说明前三级的那点改善在桥面前可以忽略。

**两臂的 +16.4 % 精确到小数点后一位相同**——桥的代价与用哪个视觉骨干**无关**。

### 5.2 损伤落在被桥覆盖的前 40 步上

| run | 口径 | @1 | @10 | @25 | @50 | acc@0.25σ @1 | acc@0.25σ @50 |
|---|---|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | `policy_raw` | 0.01354 | 0.01752 | 0.02341 | 0.03100 | 0.961 | 0.803 |
| `imnet_frozen` | `policy_deployed` | 0.02140 | 0.02884 | 0.03330 | 0.03608 | 0.900 | 0.760 |
| | *deployed vs raw* | **+58.1 %** | **+64.6 %** | +42.2 % | +16.4 % | | |
| `rand_trained` | `policy_raw` | 0.01442 | 0.01862 | 0.02422 | 0.03102 | 0.956 | 0.808 |
| `rand_trained` | `policy_deployed` | 0.02132 | 0.02899 | 0.03390 | 0.03612 | 0.900 | 0.763 |
| | *deployed vs raw* | **+47.9 %** | **+55.7 %** | +40.0 % | +16.4 % | | |

桥换掉的是执行窗口 50 步里的前 40 步，损伤也就集中在那里：**@10 处最坏（+55.7 %~+64.6 %）**，
到 @50 稀释回 +16.4 %。**注意 acc@0.25σ 在第 1 步被拉到 0.900——两臂完全相同**，
因为桥的第 1 步是从**实测关节状态**出发的零速起点，与策略输出无关。

**部署窗口一句话：在 HEAD 那版配置下，两个权重执行的前 10 步里有一半以上的误差
不是策略造成的，是桥造成的；而在今天工作区那版配置下，这一整块代价不存在。**

---

## 6. 机制：为什么是这些数

### 6.1 两个骨干都被压成了 1 个 token，所以它们能差的地方本来就不多

`probes/config_diff.txt` §4：`resnet18_imagenet` 与 `resnet18_random` 两个 preset 的
`n_patches` **都是 1**，`feature_key` 都是 `x_norm_clstoken`，`output_dim` 都是 512。
`ResNet18Encoder` 强制 `feature_key = CLS_TOKEN` 并返回一个全局池化向量
（`patch_encoders.py:273-292`）。

也就是说：`patch_new_policy` 的"patch 记忆"trunk，在这两臂上每个时刻只看到
**3 个 token（每台相机 1 个）**，而 09-04 `base` 用的 `dino_patch` 每台相机给 **256 个**。
**空间信息在进 trunk 之前就已经被池化掉了。**

这解释了本轮最中心的那个"没差别"：两个骨干的差异（ImageNet 特征 vs 从任务学到的特征）
必须先穿过一个 512 维全局平均池化的瓶颈，才能到达策略。
瓶颈之后，"这张图里哪儿有东西"已经没了，只剩"这张图整体像什么"。
**在这个瓶颈下，预训练特征和任务特征区分不出来是可预期的，而不是意外。**

### 6.2 为什么位置好而朝向差：可训练骨干学到的是任务相关的量，朝向不在里面

§4.2 的方向是系统性的：`rand_trained` 在 6 个位置维里赢 5 个、在 6 个朝向维里输 5 个。

可训练骨干的梯度来自动作回归损失。该损失是 14 维上的平均，而**朝向维在数值上占 76 %**
（§4.2）——按理它应该优先优化朝向。实测是反的。一个与数据一致的解释：
位置误差与图像里物体位置**强相关**（东西在哪，手就该去哪），而末端朝向在这个整理任务里
**大部分时间由轨迹阶段而不是由视觉决定**（`hold_state` 在朝向上只有 7.06°，
相对它自己在位置上的 23.62 mm 要"容易"得多——即朝向本来就更可预测）。
可训练骨干把容量花在了视觉能帮上忙的那一维上，代价是牺牲了它本来可以从 ImageNet
先验里白拿的那部分平滑性。

**这是一个与数据一致的解释，不是一个被本轮实验证明的因果。** 要验证需要
分维度的梯度归因或一个只训位置头的对照，**本轮都没做**。

### 6.3 容量与记忆：09-07 的那条关系又多了两个点

09-07 报告 §1.4 给出四个点：seen→unseen 倍数越大，held-out 越差或持平。
本轮两个新点（同一批 anchor、同一套 flag，§3.4 已证可比）落在同一条线上：

| 权重 | 轮次 | seen mae | seen→unseen | held-out raw mae |
|---|---|---:|---:|---:|
| `rand_trained` | 09-08 | **0.00854** | **3.63×** | 0.03102 |
| `imnet_frozen` | 09-08 | 0.01194 | 2.60× | **0.03100** |
| `width512` | 09-07 | 0.01617 | 2.14× | 0.03461 |
| `base` | 09-07 | 0.01830 | 1.89× | 0.03462 |
| `head_diff` | 09-07 | 0.02203 | 1.52× | 0.03357 |
| `depth1` | 09-07 | 0.02314 | 1.46× | 0.03376 |
| *null* `hold_state` | — | 0.06750 | — | 0.07266 |

`rand_trained` 现在是**六个权重里记忆倍数最高的**（3.63×，比 09-07 的冠军 `width512`
的 2.14× 还高 **70 %**，`tables.md` §12），而它的 held-out 比同轮对照差 0.04 %。
**多给的 11 186 132 个可训练参数（+123 %）全部转化成了训练集拟合。**

注意这条比值**不依赖任何判读线**：它是同一个权重两侧之比，
两臂之差 40 %（3.63 vs 2.60）远超任何噪声尺度——何况本轮噪声尺度是 0。

### 6.4 训练曲线：`imnet_frozen` 50k 就到顶，`rand_trained` 要 150k 才追上

| checkpoint | 50k raw | 100k raw | 150k raw | 200k raw | 100k→200k | 50k dep | 100k dep | 150k dep | 200k dep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | **0.03078** | 0.03103 | 0.03118 | 0.03100 | **−0.1 %** | 0.03606 | **0.03577** | 0.03641 | 0.03608 |
| `rand_trained` | 0.03209 | 0.03111 | **0.03099** | 0.03102 | **−0.3 %** | 0.03673 | 0.03608 | **0.03592** | 0.03612 |

**`imnet_frozen` 的最优点在 50 000 步**（0.03078），之后 150 000 步的训练把它**弄差了 0.7 %**。
**`rand_trained` 从 0.03209 单调改善到 150k 的 0.03099**，然后持平。

这与 §2.2 的权重证据互相印证：`imnet_frozen` 的骨干是冻的，
50k 步之后 trunk 已经把这个固定表示能榨的东西榨完了；
`rand_trained` 的骨干要从随机初始化学起，所以前 100k 步都在补课，
补完之后停在**同一个地方**。

**两条曲线终点相同（0.03100 vs 0.03102），路径不同。**
工程含义：如果只跑 50k 步，`imnet_frozen` 领先 **4.3 %**（0.03078 vs 0.03209）；
跑满 200k，这个优势归零。**冻结的 ImageNet 骨干买到的是收敛速度，不是终点质量。**

### 6.5 桥的代价与策略无关，因为它换掉的不是策略的值，是策略的速度剖面

两臂的桥代价 **+16.4 % 完全相同**，@1 的 acc@0.25σ 都被拉到 **0.900**。
`send_next_action_chunk` 的固定 K=40 三次 Hermite 桥用**实测关节状态**做零速起点、
用 chunk 第 40 步做终点，中间 39 步整个被 S 曲线替换——**策略在这 40 步里的输出被丢弃了**，
只有第 40 步的值被用作边界条件。所以无论哪个骨干，被换掉的都是同一段。

这也解释了 §5.1 里 `bridge_only` 与 `deployed` 几乎相同：前三级过滤器修的是
policy 输出里的小毛病，而桥把那段输出整个扔了。

---

## 7. 跨轮：这两个权重在整条 `patch_new_policy` 线上的位置

**先看可比性**（`tables.md` §11 自己先断言，三条全 True 才出表）：
flags 逐位相同 **True**、anchor / 动作步数相同 **True**（2007 / 97 090）、
`hold_state` 空模逐位相同 **True**。

| 权重 | 轮次 | 相对 09-04 `base` | raw mae | vs base | 部署 mae | vs base | 位置(mm) | seen mae | seen→unseen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `imnet_frozen` | 09-08 | `action_head` flow→**act** + `vision_encoder` dino_patch→**resnet18_imagenet**（2 字段） | **0.03100** | **−10.5 %** | **0.03608** | **−7.4 %** | 9.72 | 0.01194 | 2.60× |
| `rand_trained` | 09-08 | 上面 2 个 + `freeze_vision_encoder` true→**false**（3 字段） | **0.03102** | **−10.4 %** | **0.03612** | **−7.3 %** | **9.32** | 0.00854 | 3.63× |
| `base` | 09-07 | —（flow / dino_patch / 8L） | 0.03462 | +0.0 % | 0.03897 | +0.0 % | 10.37 | 0.01830 | 1.89× |
| `head_diff` | 09-07 | `action_head` flow→diffusion（1 字段） | 0.03357 | −3.0 % | 0.03876 | −0.5 % | 10.01 | 0.02203 | 1.52× |
| `depth1` | 09-07 | `n_decoder_layers` 8→1（1 字段） | 0.03376 | −2.5 % | 0.03849 | −1.2 % | 10.29 | 0.02314 | 1.46× |
| `width512` | 09-07 | `dim_model` 256→512（1 字段） | 0.03461 | −0.0 % | 0.03876 | −0.5 % | 9.92 | 0.01617 | 2.14× |
| *null* `hold_state` | 两轮 | — | 0.07266 | — | 0.07266 | — | 23.62 | 0.06750 | — |
| *null* `train_mean` | 两轮 | — | 0.22729 | — | 0.22729 | — | 45.43 | — | — |

**本轮两个权重在 raw 和 deployed 两列上都是六个里最好的两个**：
raw 比 09-04 `base` 好 **10.4 %–10.5 %**（09-07 那轮三个旋钮最好只做到 −3.0 %）；
部署列 **0.03608 / 0.03612** 比 09-07 的冠军 `depth1`（0.03849）好 **6.3 % / 6.2 %**。

**但这条排名不是归因。** 这两个权重相对 `base` 各动了 **2** 个和 **3** 个字段，
`action_head` flow→**act** 也在里面。**这 −10.5 % 里有多少是 ResNet18、多少是 `act` head，
本轮测不出来。** 要拆开需要 "`act` head + `dino_patch` + 原 trunk" 这个中间权重，
和 09-05 报告当年要的那个中间权重是同一类需求，**它同样不存在**。

---

## 8. 建议

### 该部署什么

- **两个都可以，选 `imnet_frozen`。** held-out 上两者差 +0.04 %（判读线 0），
  部署列差 +0.1 %——**从精度上说没有理由偏好任何一个**。
  选 `imnet_frozen` 的理由是**训练成本**：它只训 44.9 % 的参数，
  并且 **50 000 步就到达最优点**（0.03078，§6.4），而 `rand_trained` 要 150 000 步。
  在"再训一版"这件事上，它便宜三倍。
- **如果部署路径要回到 `chunk` 模式，先修桥再换骨干。** 桥的代价 **+16.4 %**
  是本轮两臂部署口径差距（+0.11 %）的 **152 倍**、raw 口径差距（+0.04 %）的 **377 倍**，也比这两个权重相对 09-04 基线的全部改善
  （−10.5 %）还要大。§5.2 显示它把执行窗口前 10 步的误差抬高了 **55.7 %–64.6 %**。
- **不要仅凭本轮的排名（§7）就把 `act` head + ResNet18 定为新基线。**
  排名是真的（−10.5 %，同批 anchor），但归因不干净（2/3 个字段同时动）。

### 该停止投入什么

- **停止在"随机初始化 + 可训练骨干"这条路上加训练预算。** 多出来的 11 186 132 个可训练参数
  （**+123 %**，`tables.md` §12）换到的是 **seen −28.5 %、held-out +0.04 %**，seen→unseen 从 2.60× 涨到 **3.63×**
  ——六个权重里最高。这是 09-07 报告 §1.4 那条关系的第五、第六个点，方向一致。
- **停止用 flat MAE 单独判断这类对比。** 本轮 flat MAE 差 +0.04 %，
  下面是位置 −4.2 %、夹爪 −9.8 %、朝向 +2.1 %（§4.2）。
  朝向维吃掉 flat MAE 的 76 %+，任何位置侧的改善都会被它稀释到读不出来。
- **停止把 ResNet18 preset 当作"换了个 patch 骨干"。** 它的 `n_patches` 是 **1**（§6.1），
  空间信息在进 trunk 前就被池化掉了。想测"patch 表示"要用 `dino_patch` / `dinov3_patch`
  那一族，不是这两个。

### 还没测的（写"没测"，不写猜测）

- **init 与 freeze 拆不开。** 需要第三个权重（`resnet18_imagenet` + `freeze=false`
  或 `resnet18_random` + `freeze=true`）。**不存在，没测。**
- **`act` head 与 ResNet18 骨干拆不开。** §7 的 −10.5 % 无法归因。
  需要 "`act` head + `dino_patch` + 原 trunk"。**不存在，没测。**
- **temporal ensembling 口径没测。** 工作区那版部署配置是 `sync` + `n_action_steps: 1`
  + `temporal_ensemble_coeff: 0.01`（§3.2）。本轮的 `policy_raw` 是它的
  **无 ensembling 上界**，不是它本身。真正的口径要用
  `runs/20260908_temporal_ensembling/ensemble_eval.py`，**本轮没跑**。
- **闭环成功率没测。** 全部是 teacher-forced 开环 chunk 评测。
  本报告任何一个数都不是成功率，也不能换算成成功率。
- **部署的图像路径没测。** 训练读视频帧，部署读 JPEG，且部署的头部图用 `INTER_AREA`
  缩放而训练用 `INTER_LINEAR`（harness 文档字符串 "What it still cannot reproduce" 第 2 条）。
  **这一条对本轮格外重要**：本轮的主角就是视觉骨干，而训练与部署之间唯一没被复现的差异
  也正好在视觉侧。一个对图像预处理更敏感的骨干（随机初始化那个，它没有 ImageNet 先验
  提供的缩放不变性）**可能**在真机上表现与本报告不同——**这是一个没测的风险，不是一个发现。**
- **`num_flow_steps` / trunk 形状与骨干的交叉项没测。** 本轮只有骨干这一格在动。
