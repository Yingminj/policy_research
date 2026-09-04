# `patch_policy` 已测配置矩阵

**整理日期** 2026-09-03 · 权重根目录 `/mnt/robot_platform/jobs/`
**用途** 一张表看清"动作头 × 参数设置"跑过哪些、结论来自哪份报告、哪些还是空的。
明细与方法学以各报告为准，本文件只做索引，不重新解释结论。

---

## 1. 七个权重

短名后接 job 目录名的可辨识部分（前缀均为 `patch_policy_tidy_up_stationery_le_`）。
共有配置：`vision_encoder=dino_patch`、`freeze_vision_encoder=true`、`resize_shape=224`、
`optimizer_lr=5.5e-5` 常数、`seed=1000`、`steps=200000`。

| 短名 | job | head | chunk / n_action | n_obs | robot_state | 动作空间 | 训练集 | bs | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| `prev_act_head` | `..._2026-08-20_21-21-58` | `act` | 50 / 50 | 5 | ✗ | 16-D 关节 | 361 | 16 | done |
| `prev_diffusion` | `..._2026-08-20_21-31-19` | `diffusion` | **64 / 32** | 5 | ✗ | 16-D 关节 | 361 | 16 | done |
| — | `..._2026-08-27_05-23-22` | `vqbet` | 50 / 50 | 5 | ✗ | 16-D 关节 | 361 | **8** | **failed @116 k** |
| `vqbet` | `..._2026-08-30_11-19-40` | `vqbet` | 50 / 50 | 5 | ✗ | 16-D 关节 | 361 | 16 | done |
| `new_state5` | `..._2026-08-30_11-27-35` | `diffusion` | 50 / 50 | 5 | **✓** | 16-D 关节 | 361 | 16 | done |
| `new_obs2` | `..._2026-08-30_11-31-30` | `diffusion` | 50 / 50 | **2** | ✗ | 16-D 关节 | 361 | 16 | done |
| `pp_eef` | `..._batch_success_505_eef_2026-08-31_13-14-33` | `diffusion` | 50 / 50 | **2** | **✓** | **14-D 末端位姿** | 505_eef | 16 | done |

## 2. 实际被扫过的坐标

沿着"改一件事"的轴看，跑过的其实只有四条：

| 轴 | 取值 | 对照对 | 报告 |
|---|---|---|---|
| `action_head` | `act` / `diffusion` / `vqbet` | `prev_act_head` ↔ `prev_diffusion` ↔ `vqbet` | head-comparison、vqbet-and-eef |
| `use_robot_state` | false / true | `prev_diffusion` ↔ `new_state5` | state-and-window |
| `n_obs_steps` | 5 / 2 | `new_state5` ↔ `new_obs2` | state-and-window |
| 动作空间 | 关节 / EEF | `new_state5` ↔ `pp_eef` | vqbet-and-eef、eef-independent-eval |

**两条轴没有被干净地分离，用它们的结论时要带上这句：**

- `prev_act_head` ↔ `prev_diffusion` **同时**变了 head、`action_chunk_size`（50→64）、
  `n_action_steps`（50→32）——head-comparison §2 自己标了方法学警告，"换 head 值 7%"
  归因不可分离。想干净测 head，得固定 chunk 再跑一次。
- `new_state5` ↔ `pp_eef` **同时**变了动作空间、`n_obs_steps`（5→2）、训练集（361→505_eef）。
  eef-independent-eval §7 的跨空间对照带着这三重差异。

## 3. 结果（同一把尺子）

关节侧：`batch_success_53_eval_data`，53 ep / 2007 anchor，数字取自 vqbet-and-eef §5.1，
这是唯一一份把六个关节权重放在同一评测集上的表。MAE 越小越好。

| run | @1 | @50 | vs null | 单次推理 |
|---|---:|---:|---:|---:|
| `act_baseline`（非 patch_policy） | 0.02529 | **0.05112** | **1.90x** | — |
| `new_state5` | 0.03961 | 0.06121 | 1.59x | — |
| `new_obs2` | 0.04022 | 0.06190 | 1.57x | — |
| `prev_act_head` | 0.04107 | 0.06349 | 1.53x | 10.6 ms |
| `vqbet` @100 k | 0.05352 | 0.06723 | 1.44x | — |
| `prev_diffusion` | 0.04455 | 0.06773 | 1.43x | **614 ms** |
| *null* `hold_state` | 0.01556 | 0.09705 | — | — |

EEF 侧：`batch_success_53_eval_data_eef`，同一批录制的独立场次，数字取自 eef-independent-eval §4。

| run | @1 | @50 | vs null | 位置误差 |
|---|---:|---:|---:|---:|
| `acteef_533`（部署中，非 patch_policy） | 0.01694 | **0.03443** | **2.11x** | 9.78 mm |
| `acteef_505`（对齐基线） | 0.01814 | 0.03689 | 1.97x | 10.23 mm |
| `pp_eef` @200 k | 0.02254 | 0.03721 | 1.95x | 11.28 mm |
| *null* `hold_state` | 0.02098 | 0.07266 | — | 23.62 mm |

**跨表读数注意**：两张表的 null 基线不同（0.09705 vs 0.07266），关节与 EEF 的 @50
不可直接相减；`pp_eef` 的收益在 eef-independent-eval §7 里是与 `pp_joint_state5`
（15.55 mm → 11.28 mm）比出来的，不是与关节表比出来的。

## 4. 空的格子

| 缺口 | 说明 |
|---|---|
| **`vqbet` @200 k** | job 已 `done`，磁盘上有 150 k/200 k，但报告评的是训练中途的 100 k。"码本卡死"的结论建立在半程权重上——唯一值得补跑的一格。 |
| `act` head + `use_robot_state=true` | 打开本体感觉只在 `diffusion` 上试过。head-comparison §1 的取舍建议是"保留 act + n_obs 2 + state on"，这个组合从没被训练出来。 |
| `act` head + EEF 空间 | patch_policy 侧的 EEF 只有 `diffusion`。ACT 侧有三个 EEF 权重，patch_policy 没有对应项。 |
| 固定 chunk 的 head 对照 | §2 那条不可分离的归因，至今没有被一次受控实验解开。 |
| cosine 衰减 / EMA / 100 k 总步数 | optimization-proposals 提的三条，一条都还没落到权重上。 |
| 08-27 那个 failed job | `vqbet` bs=8，`FileExistsError` 崩在 116 k，被 08-30 的 bs=16 完整跑取代。**不必评测。** |

## 5. 报告索引

| 报告 | 覆盖的权重 |
|---|---|
| `patch_policy-no-proprioception-2026-08.md` | `prev_diffusion`（主）、`prev_act_head`（对照） |
| `patch_policy-head-comparison-2026-08.md` | `prev_act_head` ↔ `prev_diffusion`，含 B 的 50/100/150/200 k 收敛曲线 |
| `patch_policy-state-and-window-2026-08.md` | `new_state5`、`new_obs2` @200 k |
| `patch_policy-vqbet-and-eef-2026-08.md` | `vqbet` @50 k/@100 k、`pp_eef`；**EEF 部分已被 09-02 报告推翻** |
| `patch_policy-eef-independent-eval-2026-09.md` | `pp_eef` @100 k/@200 k + 三个 ACT-EEF 基线 |
| `patch_policy-optimization-proposals-2026-08.md` | 不评测权重，是基于上述结果的改进提案 |
