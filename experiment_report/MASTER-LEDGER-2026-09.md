# 全局实验总账：`tidy_up_stationery_le` 上跑过的一切（2026-08-07 → 2026-09-08）

**整理日期** 2026-09-08 · 权重根目录 `/mnt/robot_platform/jobs/` ·
数据根目录 `/mnt/robot_platform/datasets/tidy_up_stationery_le/` ·
评测仓库 `/home/kewei/YING/paper/eval_policy/`

**用途** 一份把三处（报告 / 数据 / 评测）串起来的索引：跑过哪些模型、每个动了什么参数、
用哪把尺子比的、结论在哪份报告里、以及**哪些格子是空的**。
本文件**不重新解释任何结论**，每个数字后面都注明出处报告；明细以各报告为准。

已有的两份分策略矩阵（`patch_policy/patch_policy-matrix.md`、`act_dit/act_dit-matrix.md`）
仍然有效，本文件是它们之上的一层，并补上它们不覆盖的 `act` / `act_eef` / `act_delta` /
`vita` 三条线，以及 09-03 之后的五轮实验。

---

## 0. 一页速览 — 一个月下来真正站得住的七条

按"证据强度 × 可操作性"排序，每条都带出处。

1. **上机现役仍然是 `acteef_533`。** 三个月里没有任何一个权重在**部署口径**上超过它。
   09-05 新架构差 2.7 %、09-07 三个旋钮差 ≤1.2 %、09-06 的 616 差 2.3 %——
   全部落在或压在判读线上。（new-arch §1.3、param-ablation §1.1、616-baseline §1.1）
2. **K=40 Hermite 桥是这一个月里测到的最大单一效应，比任何架构改动大 4–12 倍。**
   `policy_raw → policy_deployed` 稳定 +10 %~+15.5 %，而部署栈其余四级合计
   −1.8 %~−0.3 %（略微改善）。**全部代价都在桥这一级**，且几乎与用哪个策略无关。
   **【09-09 更正】"全部代价在桥这一级"成立（17/17 权重证实）；"+10~15.5 %、
   与策略无关"不成立。** 桥代价是**评测集的属性**：同一个 `pn_flow_base`
   在 53 集上 +12.57 %、在 100 集上 −1.55 %，**17 个权重无一例外**地在新集上更划算
   （低 9~14 个百分点）；五个关节权重在两个集上都是**负的**（−4.81 %~−22.96 %，桥大幅帮忙）。
   桥代价与 lag-1 计划抖动 Pearson **r = −0.628**——桥是低通滤波器，
   **越抖的策略它帮得越多**。原区间是"EEF patch 权重 × 53 集"这一格的读数。
   见 dual-evalset §5。
   （new-arch §5.1、param-ablation §5.1、eef-state-head §1.5；VITA 那边同一根桥
   把每块 chunk 整条换成零速 S 曲线，vita-deploy-inference-code-audit §0 P0-1）
3. **两条一致的"记忆 vs 泛化"曲线，四个独立实验一个例外都没有。**
   加宽度 / 加步数 / 加参数买到的是 seen→unseen 比值变大，held-out 一分不变：
   `width512` 2.14× vs `depth1` 1.46×（param-ablation §1.4）、
   `pp_new` 1.89× vs `pp_old` 1.16×（new-arch §1.4）、
   ACT 后 100 k 步"全部是记忆、泛化收益为零"（act_delta-rel100 §结论②）。
4. **越小越好，至今没有反例。** trunk 8 层 → 1 层：参数 −22.9 %、推理快 26 %、
   部署 MAE 还是四者最低（param-ablation §1.2）；`pp_eef_act5` 用 2.0 % 精度换
   5 倍速度 + 确定性输出（eef-state-head §1.3）。
5. **数据配方在这条曲线上不是杠杆。** 363 → 616 集，离线跨度只有 7.4 %，
   其中四分之一是训练噪声；而它们相对空基线的领先是 2.07×–2.23×（616-baseline §1.5）。
   加失败数据反而更差，原因是场次捷径不是"失败数据有毒"（failure-data §0）。
6. **训练数据里有三个没人记过的关节硬钳位，评测集不覆盖其中之一。**
   `J6_L @ +π/3`（12.5 % 指令帧被钉死）、`J7_L @ −π/2`（1.8 %，撞上后从臂失速数百帧）、
   `J6_R @ +π/3`（7.5 %）。它把 `mean|action − state|` 在关节空间抬高 3.2 倍，
   经 FK 放大后在 EEF 空间抬高 **7.5 倍**（`act_eef` 线对它的敏感度是 `act` 线的 5 倍）。
   **排除三个限位接近区后 `action[t] ≈ state[t+3]`**，残差 0.13°（关节）/ 1.2 mm + 0.3°（末端），
   比策略自身误差低一个数量级——**非限位帧上标签不是瓶颈**。
   两个评测集一条触限 episode 都没有（0/53、0/100），四个训练集各有 3 %，
   所以**现行离线尺子在一个可测量的维度上比训练分布更容易**。
   （ACT-ACTEEF-results §2.3，2026-09-09 补测，`scripts_action_state_audit/`）
7. **两个方法论坑，每次新实验都得重复说：**
   `job.json` 看不出权重差别，只有 `config.json` 能（eef-state-head §1.1）；
   十几个 `config` 字段是死字段（只在别的 head 下被读）。
   七次 `act_dit` 训练全都 `dataset_episodes: null`，**从来没有留出验证集**
   （act_dit-matrix §4）。

---

## 1. 五条策略线，全部权重清单

### 1.1 `act` / `act_eef`（现役线）

| 短名 | job（`/mnt/robot_platform/jobs/`） | 动作空间 | 训练集 | 评过？ | 报告 |
|---|---|---|---|---|---|
| `act_baseline` | `act_..._batch_success_361_2026-08-17_12-42-42` | 16-D 关节 | 361 | ✓ | scripts_act_eval_test、act_dit-matrix §3 |
| `acteef_361` | `act_eef_..._361_eef_2026-08-26_06-10-46` | 14-D EEF | 361_eef | ✓ | 616-baseline |
| `acteef_505` | `act_eef_..._505_eef` | 14-D EEF | 505_eef | ✓ | eef-independent-eval、616-baseline |
| **`acteef_533`** | `act_eef_..._533_eef_2026-08-27_14-48-42` | 14-D EEF | 533_eef | ✓ | **`deploy_config_eef.yaml` 现役**；几乎每份 09 月报告的参照 |
| `acteef_616` | `act_eef_..._616_eef_2026-09-03_03-20-50` | 14-D EEF | 616_eef | ✓ | 616-baseline（真机最好那份） |
| `acteef_616as` | `act_eef_..._616_eef_actionsame` | 14-D EEF | 616_eef_actionsame | ✓ | 616-baseline（同配方孪生，用作噪声尺） |
| 失败数据三组 | `act_..._361{,_fail_36,_fail_72}` | 16-D 关节 | 361/+36/+72 | ✓（仅离线） | failure-data |
| quality 四组 | `act_quality_..._quality_labeled{,_ternary,_reentry,_reentry_finish}` | 16-D 关节 | 435 | **✗ 从没评过** | — |
| 单 batch 早期 | `act_..._batch_{2,3,4,5,6,7}` 等 8 个 | 16-D 关节 | 各 batch | ✗ | — |

五份 `act_eef` 的 `config.json` **逐字段相同**（resnet18 / d512 / ff3200 / enc 4 / dec 1 /
chunk 100 / VAE 开 / lr 1e-5 / 200 k / seed 1000 / bs 16）——**唯一变量是训练数据**
（616-baseline probes/provenance.txt §3）。

### 1.2 `act_delta`（相对动作，已停）

| 短名 | job | 结论 |
|---|---|---|
| ABS200k | `act_delta_..._batch_5_2026-08-13_03-52-51` | `use_relative_actions=false`——**自变量为空，等价基线 ACT**（batch5-failure §1） |
| REL100k | `act_delta_..._batch_5_rel100_keep-gripper_2026-08-14_06-34-30` | 相对动作**是对的**：留出集好 1.83×，但两者都远达不到任务精度（rel100 §结论） |
| — | `act_delta_..._batch_7_rel100_keep-gripper_2026-08-16_22-15-36` | **从没评过** |

### 1.3 `act_dit`（七个权重，详见 `act_dit/act_dit-matrix.md`）

共有：chunk 100 / n_action 100 / n_obs 1 / VAE 关 / cross-attn 开 / resnet18 / bs 16 / 361。
扫过的轴：`objective`（fm / diffusion）、`lr`（1e-4 / 1e-5）、`EMA`、积分器（euler / rk4）、
`state_in_adaln`。**当前最好：`fm_lowlr`（08-31_05-22-33），MAE 0.06730 / 2.48× null，
第一个超过 ACT baseline 的 act_dit**（fm-lowlr）。
**唯一没评过的：`08-20_02-32-38`（fm / 1e-4 / 无 EMA / 300 k，也是唯一的 300 k 权重）。**

### 1.4 `patch_policy`（七个权重，详见 `patch_policy/patch_policy-matrix.md`）

共有：`dino_patch` 冻结 / resize 224 / lr 5.5e-5 常数 / seed 1000 / 200 k。
扫过的轴：`action_head`（act / diffusion / vqbet）、`use_robot_state`、`n_obs_steps`（5 / 2）、
动作空间（关节 / EEF）。
另有 09-03 两个 EEF arm（`pp_eef_nostate`、`pp_eef_act5`）与 09-04 的 `pp_old`
（`2026-09-04_09-52-06`，DINOv3 骨干）。

### 1.5 `patch_new_policy`（最新线，六个权重）

| 短名 | job 后缀 | 相对 `base` 动的字段 | 参数量 | 评过？ |
|---|---|---|---:|---|
| `base` / `pp_new` | `2026-09-04_17-41-12` | —（flow head / dino_patch / 8L×d256） | 32 249 486 | ✓ new-arch、param-ablation |
| `head_diff` | `2026-09-07_02-55-46` | `action_head` flow→**diffusion** | 32 249 486 | ✓ param-ablation |
| `depth1` | `2026-09-07_02-57-48` | `n_decoder_layers` 8→**1** | 24 875 406 | ✓ param-ablation |
| `width512` | `2026-09-07_03-03-41` | `dim_model` 256→**512** | 53 313 166 | ✓ param-ablation |
| **未命名 A** | `2026-09-07_19-36-54` | `action_head` flow→**act** + `vision_encoder` dino_patch→**resnet18_imagenet**（2 字段） | — | **✗ 从没评过** |
| **未命名 B** | `2026-09-07_19-50-23` | `action_head` flow→**act** + `vision_encoder`→**resnet18_random** + `freeze_vision_encoder` True→**False**（3 字段） | — | **✗ 从没评过** |

两个未评权重都存了 050000–200000 四个中间步。**它们是当前最前沿的两格**，见 §5。

### 1.6 `vita`（三个权重，只做过部署诊断，无离线精度表）

`batch_3` / `batch_5` / `batch_success_361` 三个 job。三份报告全部是**部署链路与结构审查**，
不是精度评测。核心结论：**机器人从来没有执行过 VITA 输出的轨迹**——
100 % 的 chunk 被换成零速起止的 Hermite S 曲线（deploy-inference-code-audit §0）。
在 P0-1 修好之前，**真机不能用来区分任何 VITA 消融臂**（arch-ablation §0）。

### 1.7 不在本任务线上的（`package_head` / `luosi` / `express` / `sort_blocks`）

`jobs/` 里 86 个 job 中约 30 个属于别的任务（打包头、螺丝、快递、分拣），
与 `tidy_up_stationery_le` 无关，本总账不覆盖。

---

## 2. 数据集家谱

36 个目录，四类。`*_eef` 是同一批数据经 `tr_joint_to_eef.py` 转成 14-D 末端位姿的版本
（episode / frame 数与关节版逐位相同）。

### 2.1 成功集主线（唯一在用的一条）

```
361 (363ep/300k帧) ─┐
                    ├─→ 505 (505ep/419k) ─→ 533 (535ep/444k) ─→ 616 (616ep/507k) ─→ 749 (749ep/614k)
```

**注意这条线不是严格包含关系**（616-baseline §1.3 按 episode 指纹核对过）：
`505 ⊂ 616`，但 `533` 有 **67 集不在 616 里**、`361` 有 49 集不在。
616 = 505 + 111 集新数据 − 533 在 505 之上加的那 67 集。那 67 集单独评过，**放弃它们不是风险**。

| 数据集 | ep | 帧 | 训过谁 |
|---|---:|---:|---|
| `batch_success_361` / `_eef` | 363 | 300 689 | act_baseline、全部 act_dit、前五个 patch_policy、acteef_361 |
| `batch_success_505` / `_eef` | 505 | 419 330 | acteef_505、全部 EEF patch_policy、全部 patch_new_policy |
| `batch_success_533` / `_eef` | 535 | 443 885 | **acteef_533（现役）** |
| `batch_success_616` / `_eef` | 616 | 506 646 | acteef_616、acteef_616as |
| **`batch_success_749` / `_eef`** | **749** | **614 259** | **09-07 新落盘，还没有任何权重用过它** |

⚠️ **这条线上四个数据集全部含左腕触限工况**，比率一致（关节侧 3 %、EEF 侧 14–16 % 的 episode），
重灾 episode 在索引上完全打散（616 中分 53 段，最长 10），**不是某一次录制或某一批标定的问题**。
`*_eef` 版本因 FK 把单个腕滚关节的误差同时投到 `l_roll` / `l_yaw` / `l_x`（三者互相关 0.996），
**同一缺陷在 EEF 侧放大 5.3 倍**。逐数据集数字见 ACT-ACTEEF-results 表 3b。

### 2.2 评测集（唯一的一把尺子）

| 数据集 | ep | 帧 | anchor(stride 20) | 用途 |
|---|---:|---:|---:|---|
| `batch_success_53_eval_data` | 53 | 40 132 | 2007 | 16-D 关节侧全部评测 |
| `batch_success_53_eval_data_eef` | 53 | 40 132 | 2007 | 14-D EEF 侧全部评测 |
| **`batch_success_100_eval`** | **100** | **81 647** | **4083** | **09-08 新增，16-D 关节侧** |
| **`batch_success_100_eval_eef`** | **100** | **81 647** | **4083** | **09-08 新增，14-D EEF 侧** |

08-21 独立录制，与全部训练集**指纹去污染后丢弃 0 集**。

**09-09 更新：现在是两把尺子，不是一把。** 100 集那一对 09-08 落盘，
与 53 集那一对**互不相交**（0 集重叠）、与六个训练集**全部 0 污染**、
关节与 EEF 版指纹 100/100 孪生（dual-evalset §2.1）。合并留出 **153 集 / 121 779 帧 /
6090 anchor**。**两个集不等难**：`hold_state` 空基线 0.07266（53）vs 0.05213（100），
新集上"原地不动"好 28 %，所以新集上的低 MAE 是集的性质而非权重变好（dual-evalset §2.2）。

⚠️ **但它比训练分布容易**：`J7_L @ −π/2` 触限 episode 在 `53_eval_data` 是 **0/53**、
在 `100_eval` 是 **0/100**，而四个训练集各有 3 %。`J6_R` 撞完全相同的 +π/3 限位却测不出
train/eval 差异，正因为评测集也做那个动作——所以这是**动作覆盖不匹配，不是数据质量问题**。
零成本补法：从 `616` 划出 `|mean(a−s)| > 0.2 rad on J7_L` 的那 19 条作难例留出集，
与现有评测集并列报告（ACT-ACTEEF-results §2.3.7、建议 10）。

### 2.3 失败 / 纠错 / 质量标注（一条已结论的死路）

`batch_fail_36` / `batch_fail_72` / `batch_fail_aim_1` / `batch_fail_aim_test`，
以及 `361_fail_{36,72}`、`_trimmed` 与四个 `_quality_labeled_*` 变体。
**结论：以现在这种形式不该加**（failure-data §0）——加进去的不是"纠错信号"，
是一个背景明显不同的独立采集会话块，策略拿它当模式选择捷径。
四个 `quality_labeled` 变体训了权重但**一个都没评过**。

### 2.4 早期 batch 与 rel100（存档）

`batch_1`–`batch_8`（63/30/120/80/30/61/163/200 集），
`batch_5_rel100_keep-gripper` / `batch_7_rel100_keep-gripper` 是相对动作实验用的。
`batch_1`–`batch_4` 曾作 held-out，**已被 53 集评测集取代，两套数字不要混用**
（scripts_act_dit_lowlr/README）。

---

## 3. 三把尺子，混用就是错

跨报告读数最容易出错的地方。**同一份 MAE 数字，口径不同差 2 倍以上。**

| 尺子 | horizon | 口径 | null `hold_state` | 出现在 |
|---|---|---|---:|---|
| **A** 全 horizon 关节 | 100 | `policy_raw` | 0.1672 / 0.1506 | act_dit lowlr §4、scripts_act_eval_test |
| **B** 部署窗口关节 | 50 | raw / deployed | 0.09705 | patch_policy-matrix §3、flowmatching-deployed |
| **C** 部署窗口 EEF | 50 / 60 | raw / deployed | 0.07266（h50）/ 0.08216（h60） | 全部 09 月 EEF 报告 |
| **D** 逐帧执行动作 | **1** | replan_1 / ens / chunk | **0.02091** | 09-08 ensembling（**未成文**，§5） |

**尺子 D 的 null 已被分解**（ACT-ACTEEF-results §2.3、附录 B）：0.02091 里约 **17 %** 是
指令→实测的 100 ms 执行滞后（`k* = 3` 帧，全部 12 个数据集一致）、约 **60 %** 是三处关节限位饱和，
剩下才是测量噪声。**`hold_state` 不是"伺服滞后"的读数**，把它当滞后读会高估 6 倍。

**规则：**
- 尺子之间**只能读排序，不能相减**（new-arch §1、act_dit-matrix §3 都写了这条）。
- 每个精度数字**必须带同批 anchor 的 null**。EEF 的 null 有两个（h50 / h60），别串。
- **判读线**（采样噪声下界，实测）：flow **1.70 %**、diffusion **2.00 %**、
  act head **0**（确定性，两个种子逐位相同）、跨训练复现噪声 **1.93 %**（616 孪生）。
  **差距小于判读线的一律读作"没测出来"，不是"更好"。**
- **【09-09 补】还有第二条、而且更大的判读线：评测集判读线 中位 5.86 % / 最坏 15.21 %**
  （dual-evalset §4.2，两个互不相交留出集上 10 个权重实测；同轮采样线复测 ≤1.73 %）。
  **是采样线的 3.4 倍。** 正确读法应为：**差距必须在两个集上同号、且都超过 5.86 %**
  才算测出来。按这条复读本表以下各节，会有若干条降级为"没测出来"。
- `--filters none` 时 `policy_deployed` ≡ `policy_raw`。09-04 那轮全程 none，
  **它没有部署列**（acc-tau §开头）。

### 3.1 指标家族（`offline_chunk_eval.py`）

`mae` / `rmse` / `norm_*`（按训练集 action std 逐关节归一）/ `tail_ratio` /
`mae_at_horizon[1,10,25,50,H]` / `mae_per_joint` / **`acc@τ`**（τ ∈ 0.1/0.25/0.5/1.0 σ，永远算）/
**末端位姿误差**（位置 m + 姿态 rad，四通道，故意不给标量汇总，按适用性自动开）。

`acc@τ` 给过一个 MAE 给不出的结论：**5 mm 容差上三个 patch_policy 全部输给 null**
（0.042/0.035/0.049 vs hold_state 0.063），同时它们的平均位置误差只有 null 的一半。
读法是"**它们从不真正到位，只是处处差不多近**"（acc-tau §1.3）。

---

## 4. 扫过的参数轴 vs 空格子

### 4.1 已经有干净答案的（一次只动一个字段）

| 轴 | 结果 | 出处 |
|---|---|---|
| `use_robot_state` on→off（EEF/diffusion） | **纯亏 5.2 %**，@1 亏 15.9 %，位置 +8.7 % | eef-state-head §1.2 |
| `action_head` flow→diffusion（patch_new） | raw −3.0 %（读得出），deployed 持平 | param-ablation §1.2 |
| `n_decoder_layers` 8→1 | raw −2.5 %，参数 −22.9 %，快 26 %，**部署列最好** | param-ablation §1.2 |
| `dim_model` 256→512 | **纯亏**：+65 % 参数、慢 1.44×、MAE −0.03 %（读不出）、记忆 2.14× | param-ablation §1.3 |
| `lr` 1e-4→1e-5（act_dit） | 救回塌缩的 encoder（但与 EMA 绑在一起，见下） | lowlr、fm-lowlr |
| 动作空间 关节→EEF | patch_policy 专属收益：15.55 → 11.28 mm | eef-independent-eval §7 |
| 训练数据 363→616 集 | 离线跨度仅 7.4 %，**不是杠杆** | 616-baseline §1.5 |

### 4.2 归因不干净的（用结论时必须带这句）

| 对照 | 同时变了 | 出处 |
|---|---|---|
| `prev_act_head` ↔ `prev_diffusion` | head + chunk 50→64 + n_action 50→32 | patch_policy-matrix §2 |
| `new_state5` ↔ `pp_eef` | 动作空间 + n_obs 5→2 + 训练集 361→505_eef | patch_policy-matrix §2 |
| act_dit `08-22` ↔ `08-24` | lr **和** EMA | act_dit-matrix §4 |
| act_dit `08-31_06-17` | `state_in_adaln` **和** 积分器 euler→rk4 | act_dit-matrix §4 |
| `pp_new` ↔ `pp_old` | head + 视觉骨干 + trunk 形状（三样） | new-arch §7 |
| `pp_eef_act5` ↔ 基线 | head + n_obs + state（三样） | eef-state-head §1.1 |

### 4.3 空格子（按值得补的程度排）

| 缺口 | 为什么值得 / 不值得 |
|---|---|
| **09-07 晚上两个权重（act head + resnet18）** | 已训完，四个 checkpoint 齐全，**一次评测就能填**。见 §5.2 |
| **09-08 ensembling 那轮没有 README 也没有报告** | 数据已在磁盘上，是本月最有信息量的一轮。见 §5.1 |
| **`acteef_616` @100 k 上机** | 离线最优点在 100 k（0.03655）反超 533@200k（0.03689），五份权重全停在 200 k 而 200 k 已经过了最优点（616-baseline §1.4、§7.2） |
| **`batch_success_749`** | 749 集已落盘 09-07，还没有任何权重用过 |
| **含左腕触限工况的第二评测集** | 现行两个评测集 0/53、0/100 触限 episode，训练集各 3 %，**离线尺子量不到最难的那一段**。零成本做法：从 616 划出那 19 条作难例留出集（ACT-ACTEEF-results §2.3.7） |
| **剔除触限 episode 的敏感性重训** | 那 19 条影响 6.3 % 的帧，且标签在其上不可学。**但不要直接改配方**——它们记录的是真机上真实会发生的工况，须以敏感性实验判定（同上，建议 12） |
| `patch_new_policy` 的 `depth1` 上机 | 部署列最好且省 23 % 参数，离线已经说完话，剩下只有真机能判 |
| act_dit `08-20_02-32-38` | 唯一没评过的 act_dit，也是唯一 300 k |
| act_dit `vqbet` @200 k | "码本卡死"的结论建立在半程 100 k 权重上 |
| 固定 chunk 的 head 对照（patch_policy） | §4.2 第一条那个不可分离归因，至今没解开 |
| `act` head + `use_robot_state=true` | head-comparison 推荐的组合从没被训练出来 |
| 四个 `quality_labeled` 权重 | 训了没评；但 failure-data §0 已经给了负面先验，优先级低 |
| VITA 任何离线精度表 | 三份报告全是部署诊断；且 P0-1 不修，真机数据无意义 |
| cosine / EMA / warmup（patch_policy 侧） | optimization-proposals 提了，一条都没落到权重上 |
| 留出验证集 | 全部训练 `dataset_episodes: null`，每次都要事后跑 `offline_chunk_eval.py` |

---

## 5. 两块还没归档的（截至 09-08）

### 5.1 `runs/20260908_temporal_ensembling/` — 有数据，**没有 README，没有报告**

五个权重跑完了（`acteef_533` / `acteef_616` / `acteef_616as` / `base` / `depth1`），
`results/` 里 JSON + log 齐全。这是**尺子 D**（逐帧执行动作，horizon 1），
回答的问题是：真机上"temporal ensembling 成功率 90 %"这件事，功劳到底在
**换 replan 频率**、**平均**、还是**桥被关掉**——`deploy_config_coeff.yaml` 一次改了三样。

`ensemble_eval.py` 把三者拆开了（同一批前向，六种归约方式）。已经跑出来的数：

| run | `replan_1` | `ens_0` | `ens_-0.05` | `chunk_raw` | `chunk_deploy` | null `hold_state` |
|---|---:|---:|---:|---:|---:|---:|
| `acteef_533` | **0.01688** | 0.04041 | 0.02736 | 0.03593 | 0.03850 | 0.02091 |
| `acteef_616` | 0.01704 | 0.04250 | 0.02803 | 0.03689 | 0.03936 | 0.02091 |
| `acteef_616as` | 0.01722 | 0.04243 | 0.02817 | 0.03680 | 0.03935 | 0.02091 |
| `base` | 0.01647 | 0.03031 | 0.02470 | 0.03340 | 0.03755 | 0.02091 |
| `depth1` | **0.01644** | 0.02866 | 0.02362 | 0.03289 | 0.03733 | 0.02091 |

三条拆解（每条都是同一行里两个数相减）：
- **replan 频率**（`replan_1` − `chunk_raw`）= **−53 %~−51 %**，是三者里最大的一块。
- **平均**（`ens_*` − `replan_1`）= **+44 %~+152 %**，在 MAE 上是**纯亏**。
- **桥**（`chunk_deploy` − `chunk_raw`）= **+7 %~+13 %**，与尺子 C 上测到的量级一致。

**这张表和真机结论直接冲突，而冲突本身就是结论。** MAE 说 ensembling 更差，
真机说它成功率 90 %。分歧写在另外两列上：

| run | `plan_consistency` MAE | 二阶差分 `replan_1` | 二阶差分 `ens_0` |
|---|---:|---:|---:|
| `acteef_533` | **0.00339** | 0.00296 | 0.00021（**14× 更平**） |
| `acteef_616as` | 0.00353 | 0.00300 | 0.00021 |
| `base` | **0.01279**（3.8×） | 0.01159 | 0.00118 |
| `depth1` | **0.01589**（4.7×） | 0.01406 | 0.00165 |

**两条读数**：(a) ensembling 买的不是精度是**平滑**——二阶差分掉一个数量级；
(b) **两个 patch 权重的计划自洽性比 act 差 3.8–4.7 倍**，它们每一拍都在重写自己的计划。
这是 MAE 表看不见的、而且是 patch 线目前最像"真问题"的一个量。

**待办**：补 README（三段：评的哪个 checkpoint / 跑的命令 / 关键读数）+ 一份报告
`patch_policy/` 或新开 `deploy/`。另外 `results_pre_plan_consistency/` 与 `results/`
的 MAE **逐位相同**（已核对），只是前者没有 `plan_consistency`——
**是被取代的副本，可以删，或在 README 里注明它为什么留着。**

### 5.2 09-07 晚上的两个权重 — 训完了，**没进任何实验**

| job 后缀 | 相对 `base` | 存的步 |
|---|---|---|
| `2026-09-07_19-36-54` | `action_head` flow→**act**、`vision_encoder` dino_patch→**resnet18_imagenet** | 50k/100k/150k/200k |
| `2026-09-07_19-50-23` | `action_head` flow→**act**、`vision_encoder`→**resnet18_random**、`freeze_vision_encoder` True→**False** | 50k/100k/150k/200k |

这是 09-07 消融之后的自然下一步（把 patch_new 的视觉骨干换掉），
但 A 动了 2 个字段、B 动了 3 个——**按 §4.2 的标准，B 的归因不干净**。
A 可以和 `base` 直接比（虽然 head 和 backbone 绑在一起）。

---

## 6. 下一次模型迭代的建议

按"每单位工作量能读出的信息"排。前三条不需要训练任何新权重。

### P0 — 先把已经花掉的算力兑现（本周，0 次训练）

1. **给 09-08 那轮补 README + 报告。** 数据都在磁盘上，成本是写作不是 GPU。
   它握着当前唯一一条能解释"真机 90 % 但离线 MAE 更差"的证据链，
   而且 `plan_consistency` 这一列**第一次给出了 patch 线的一个真缺陷**（3.8–4.7× churn）。
   不写下来，下一轮又会从头问一遍。
2. **评 09-07 晚上那两个权重**（§5.2）。一次 `run_eval.sh`，模板照抄
   `runs/20260907_pp_new_param_ablation/`。它回答的是"视觉骨干值多少"——
   这是 patch 线上**唯一还没被扫过的大轴**（前面全部扫的是 head / 深度 / 宽度 / state / n_obs）。
3. **把 `acteef_616` @100 k 送上机。** 616-baseline §7.2 点名要的，一次真机 rollout。
   离线说 100 k（0.03655）反超现役 533@200k（0.03689），而"五份权重全停在 200 k"
   是一个跨全部 `act_eef` 的系统性错误——**如果 100 k 真的更好，这条一次性提升整条线。**

### P1 — 把工程力气放在回报最大的地方（1–2 周）

4. **修 K=40 桥，而不是继续调权重。** 这是本月被四份独立报告重复指出的同一件事：
   桥的代价 10–15.5 %，四个旋钮之间的全部差距 ≤3.0 %。**投入产出比差 4–12 倍。**
   09-08 那轮还给了一条更强的证据：**关掉桥 + 每拍 replan（`replan_1`）比现在的
   `chunk_deploy` 好 55–56 %**（0.0169 vs 0.0385）。方向不是"把桥调好"，
   可能是"证明它可以不要"——而 `deploy_config_coeff.yaml` 已经是这条路了。
   注意：桥同时是 VITA 那条线的 P0-1，修一次两条线都解锁。
5. **`depth1` 上机验证"越小越好"。** 离线已经三次说同一件事（部署列最好、
   记忆倍数最低 1.46×、快 26 %），离线不会再说出新东西了。
   如果真机确认，下一轮的默认配置就该是 1 层而不是 8 层。
6. **补 `plan_consistency` 到顶层 harness。** 现在它只活在 09-08 的 run 目录里。
   它不需要 ground truth，是同一批 chunk 的免费归约，而且是目前唯一能解释
   "离线 MAE 好但机器人抖"的指标。**加进 `offline_chunk_eval.py` 之后，
   历史权重全部可以免费重读一遍。**

### P2 — 训练侧（需要 GPU，按信息量排）

7. **下一批权重训 100 k 不训 200 k，并且留出验证集。**
   三条独立曲线（`acteef_616` / `acteef_616as` / `acteef_505`）一致地在 200 k 变差；
   `pp_new` 100 k 之后只降 1.6 %（噪声内）。**后 100 k 步买到的是记忆。**
   同时把 `--dataset.eval_split` 从 0.0 改掉——十几次训练全都没有验证集，
   每次都要事后跑离线评测才知道泛化，这是纯粹的返工。
8. **用 `batch_success_749` 训一个 `act_eef`，但把它当"数据不是杠杆"的检验，不是提升。**
   616-baseline §1.5 已经说了 363→616 只值 7.4 %。749 值得跑一次是因为它是
   **对那条结论的外推检验**（如果 749 也只值几个百分点，就可以正式停掉"加数据"这条线，
   把采集预算换到别处）。**不要指望它解决精度问题。**
9. **一次干净的 head 对照**（固定 chunk / 固定 backbone / 固定 trunk，只动 head）。
   `patch_policy-matrix §2` 那条不可分离归因挂了一个月，
   而 09-07 已经证明在 `patch_new_policy` 里做"一次动一个字段"是可行的。
10. **停掉的三条线**，明确写下来免得再花力气：
    `act_delta`（相对动作是对的，但瓶颈是数据不是表示，rel100 §结论②）、
    失败/纠错数据（failure-data §0，除非换采集方式重录）、
    `vqbet`（被码本卡死；补一个 @200 k 评测确认后即可关闭）。
    **VITA 不是停，是被 P0-1 阻塞**——桥修好之前它的任何消融都读不出来。

### 一句话

**这个月的全部离线证据指向同一个方向：模型侧已经进入噪声区（四个旋钮极差 1.2 %，
判读线 1.70 %），而部署侧还躺着一个 10–15 % 的效应和一个 55 % 的效应没人动。
下一轮该做的不是再调一个参数，是把 `replan_1` 那条路走通、把 `plan_consistency`
加进主 harness、并且从此不再训到 200 k。**

---

## 7. 报告 → 权重 → 数据 → run 目录 对照表

| 报告 | 日期 | 主角权重 | 评测集 | run 目录 |
|---|---|---|---|---|
| `act/ACT-layer-depth-analysis.md` | 08-10 | （代码分析，无权重） | — | — |
| `act/ACT-improvement-proposals-2026.md` | 08-10 | （文献提案） | — | — |
| `act/ACT-experiment-plan-2026-08.md` | 08-11 | （方案） | — | — |
| `vita/vita-2507.13231.md` | 08-12 | （论文笔记） | — | — |
| `vita/vita-deploy-vibration-2026-08.md` | 08-12~13 | vita batch_2 / 361 | 真机录制 | `scripts_vita_chunk` |
| `vita/vita-deploy-inference-code-audit-2026-08.md` | 08-13 | （链路审计） | — | `scripts_deploy_audit` |
| `vita/vita-arch-ablation-2026-08.md` | 08-13 | （结构审查） | — | — |
| `act/act_delta-batch5-failure-analysis-2026-08.md` | 08-14 | act_delta batch_5 | — | `scripts_act_delta_audit` |
| `act/act_delta-rel100-precision-analysis-2026-08.md` | 08-15 | act_delta rel100 | batch_6 留出 31 集 | `scripts_act_delta_audit` |
| `act/failure-data-in-imitation-2026-08.md` | 08-18 | act 361 / +36 / +72 | 离线 + 遮挡消融 | `scripts_fail_data` |
| （无报告） | — | act baseline 361 | 53 集 | `scripts_act_eval_test`、`_fix` |
| `act_dit/act_dit-encoder-collapse-2026-08.md` | 08-下 | act_dit 08-20_22-59 | 53 集 | `scripts_act_dit_probe` |
| `act_dit/act_dit-lowlr-diffusion-2026-08.md` | 08-下 | act_dit 08-24 | 53 集 | `scripts_act_dit_lowlr` |
| `act_dit/act_dit-flowmatching-deployed-eval-2026-08.md` | 08-下 | act_dit 08-27 | 53 集 | `scripts_act_dit_eval_fix` |
| `patch_policy/patch_policy-no-proprioception-2026-08.md` | 08-下 | `prev_diffusion` | 53 集 | `scripts_patch_policy_probe` |
| `patch_policy/patch_policy-head-comparison-2026-08.md` | 08-下 | `prev_act_head` ↔ `prev_diffusion` | 53 集 | `scripts_patch_policy_compare` |
| `patch_policy/patch_policy-state-and-window-2026-08.md` | 08-下 | `new_state5` / `new_obs2` | 53 集 | `scripts_patch_policy_eval_fix` |
| `patch_policy/patch_policy-vqbet-and-eef-2026-08.md` | 08-下 | `vqbet` / `pp_eef` | 53 集 | `scripts_patch_policy_eval_0831` |
| `patch_policy/patch_policy-optimization-proposals-2026-08.md` | 08-下 | （提案） | — | — |
| `patch_policy/patch_policy-eef-independent-eval-2026-09.md` | 09-02 | `pp_eef` + 三个 ACT-EEF | 53_eef | `scripts_patch_policy_eval_0902` |
| `act_dit/act_dit-fm-lowlr-2026-09.md` | 09-03 | `fm_lowlr` 五 arm 同尺 | 53 集 | `2026-09-03_act_dit_fm_lowlr` |
| `patch_policy/patch_policy-eef-state-head-2026-09.md` | 09-03 | `pp_eef_nostate` / `pp_eef_act5` | 53_eef | `20260903_pp_eef_state_head` |
| `patch_policy/patch_policy-acc-tau-eef-pose-metrics-2026-09.md` | 09-04 | （同 09-03 四权重，换尺子） | 53_eef | `20260904_acc_tau_eef_pose` |
| `patch_policy/patch_policy-new-arch-flow-head-2026-09.md` | 09-05 | `pp_new` / `pp_old` / `acteef_533` | 53_eef | `20260905_patch_new_policy_flow` |
| `act/act_eef-616-baseline-2026-09.md` | 09-06 | 五份 act_eef | 53_eef | `20260906_acteef_616_baseline` |
| `patch_policy/patch_policy-param-ablation-2026-09.md` | 09-07 | `head_diff`/`depth1`/`width512` | 53_eef | `20260907_pp_new_param_ablation` |
| **（缺）** | **09-08** | **五权重 × 六归约** | **53_eef** | **`20260908_temporal_ensembling`** |
| `patch_policy/patch_policy-dual-evalset-2026-09.md` | **09-09** | **全部 15 个 patch 权重 + `acteef_533`/`616`** | **53_eef + 100_eval_eef（及关节孪生）** | **`20260908_dual_evalset_sweep`** |
| **`act/ACT-ACTEEF-results-2026-09.md`**（+ `.html`） | **09-09** | act / act_eef 两线全部权重 | 53_eef + 100_eval_eef | **二次汇编**；§2.3 为一手补测 |
| └ 支撑脚本 | 09-09 | （无权重，直读数据集） | 12 个数据集 / ~260 万帧 | `experiment_report/scripts_action_state_audit/` |

另有 `LEROBOT-VERSION-COMPAT{,.zh-CN}.md`（训练 v0.6.0 / 推理 v0.5.2 的分支兼容性）
与 `runs/2026-09-03_metric_cdf/`（指标 CDF 探针，无报告）、
`runs/scripts_act_layer_bench/`（ACT 架构基准，无报告）。
