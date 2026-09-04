# `act_dit` 已测配置矩阵

**整理日期** 2026-09-03 · 权重根目录 `/mnt/robot_platform/jobs/` · 源码 `lerobot/src/lerobot/policies/act_dit/`
**用途** 一张表看清"目标 × LR × EMA × 通路"跑过哪些、结论来自哪份报告、哪些还是空的。
本文件只做索引，不重新解释结论，明细与方法学以各报告为准。

---

## 1. 七个权重

短名后接 job 目录名的可辨识部分（前缀均为 `act_dit_tidy_up_stationery_le_batch_success_361_`）。
共有配置：`chunk_size=100`、`n_action_steps=100`、`n_obs_steps=1`、`use_vae=false`、
`use_cross_attention=true`、ResNet18（ImageNet 预训练）、`optimizer_lr_backbone=1e-5`、
`num_inference_steps=10`、`bs=16`、`seed=1000`、训练集 `batch_success_361`、无 scheduler、
`dataset_episodes=null`（**七次训练全都没有留出验证集**）。

| job | objective | EMA | lr | 积分 | `state_in_adaln` | steps | 磁盘 checkpoint | 末步 loss | 状态 |
|---|---|---|---|---|---|---|---|---:|---|
| `08-20_02-32-38` | flow_matching | ✗ | 1e-4 | euler | 缺字段 → **on** | **300k** | 300000, last | 0.025 | done |
| `08-20_22-59-31` | flow_matching | ✓ | 1e-4 | euler | 缺字段 → **on** | 200k | 200000, last | 0.029 | done |
| `08-22_02-42-49` | **diffusion** | ✗ | 1e-4 | euler | 缺字段 → **on** | 200k | 050000–200000, last | 0.013 | done |
| `08-24_06-21-05` | **diffusion** | ✓ | **1e-5** | euler | 缺字段 → **on** | 200k | 200000, last | 0.015 | done |
| `08-27_04-32-02` | flow_matching | ✓ | 1e-4 | euler | 缺字段 → **on** | 200k | 200000, last | 0.029 | done |
| `08-31_05-22-33` | flow_matching | ✓ | **1e-5** | euler | 缺字段 → **on** | 200k | 050000–200000, last | 0.019 | done |
| `08-31_06-17-39` | flow_matching | ✓ | **1e-5** | **rk4** | **false** | 200k | 100000, 200000, last | 0.021 | done |

**`state_in_adaln` 那一列要当心。** 补丁把默认值定成 `false`，但**只有 08-31_06-17-39 的
`config.json` 里真的有这个字段**；其余六个权重写盘时字段还不存在。按
`configuration_act_dit.py:56` 自己的说明，缺字段 = 训练时 state 走了 adaLN。
所以**七个权重里有六个是"塌缩配方"，加载它们必须手工往 `config.json` 里补
`"state_in_adaln": true`，否则在 adaLN 的 `Linear` 形状上直接报错。**

**loss 不可跨目标比。** 08-22 的 0.013 是七个里最低的，但它在评测集上是最差的一个
（lowlr §6.4 已经把这条写成结论：diffusion loss 比 flow matching 低一半，动作精度差 42%）。

## 2. 报告覆盖了哪些

| job | 被哪份报告评过 | 角色 |
|---|---|---|
| `08-20_02-32-38` | **没有** | 见 §4，唯一仍未评的权重 |
| `08-20_22-59-31` | encoder-collapse（主）、flowmatching-deployed（对照）、lowlr §4（表内 "08-20" 行） | 塌缩的那一版 |
| `08-22_02-42-49` | lowlr §2 §4（表内 "08-22" 行） | 分离 LR 与 objective 的对照 |
| `08-24_06-21-05` | lowlr（主） | P0 处方首次执行 |
| `08-27_04-32-02` | flowmatching-deployed（主） | 08-20 的复跑，逐字段相同 |
| `08-31_05-22-33` | **fm-lowlr（09-03）**（主） | 处方本身，当前最好的 act_dit |
| `08-31_06-17-39` | **fm-lowlr（09-03）** §4.3 | P1 arm，输了；两变量 |

## 3. 结果（两把不同的尺子，别混着读）

> **2026-09-03 更新**：`act_dit-fm-lowlr-2026-09.md` §3 把五个 arm 放在**同一把尺子**上重跑了
> （含本节引用的三个参照，数字逐条对上）。**要横向比就用那张表**，本节保留是为了
> 标明每个数字的原始出处。

**尺子 A — 全 horizon（chunk 100）**，评测集 `batch_success_53_eval_data`，2007 anchor，
数字取自 lowlr §4。

| checkpoint | 目标 / LR / EMA | 训练集内 | 评测集 MAE | vs 空策略 | enc3 img-sens |
|---|---|---:|---:|---:|---:|
| ACT baseline `08-17` | — / 1e-5 / — | 0.0125 | **0.0685** | **2.44×** | 4.99e-2 |
| `08-20` | fm / 1e-4 / ✓ | 0.0370 | 0.0743 | 2.25× | **2.0e-6**（塌） |
| `08-24` | diff / 1e-5 / ✓ | 0.0954 | 0.1056 | 1.58× | 2.19e-2 |
| `08-22` | diff / 1e-4 / ✗ | 0.1275 | 0.1276 | 1.31× | **3.0e-6**（塌） |
| *null* `hold_state` | — | 0.1506 | 0.1672 | 1.00× | — |

**尺子 B — 部署忠实、executed horizon 50**，同一评测集，数字取自 flowmatching-deployed §1 §3。

| checkpoint | raw MAE | deployed MAE | vs 空策略 |
|---|---:|---:|---:|
| ACT baseline | **0.0511** | **0.0509** | — |
| `08-27` fm/1e-4 | 0.0567 | 0.0528 | 1.84× |
| `08-20` fm/1e-4 | 0.0571 | 0.0530 | 1.84× |
| *null* `hold_state` | 0.0970 | — | — |

**两张表不可相减**（horizon 100 vs 50，raw vs deployed）。跨表只能读排序。

**验收线**（lowlr §6.5）：评测集 `mae@10` 必须低于 `hold_state` 的 **0.0317**。
现状 ACT 0.0312（刚好打平）、fm 0.0485、diff-1e-5 0.0771 —— **三个都没真正过线。**

## 4. 空的格子

| 缺口 | 说明 |
|---|---|
| ~~`08-31_05-22-33` 从没评过~~ | **2026-09-03 已补**：MAE 0.06730 / 2.48×，第一个超过 ACT baseline 的 act_dit；首帧 1.04× 空策略，`mae@10` 0.0235 带余量过线。见 `act_dit-fm-lowlr-2026-09.md`。 |
| ~~`08-31_06-17-39` 从没评过~~ | **2026-09-03 已补**：全线更差（MAE +5.1%、首帧 +63%、图像敏感度 −64%），P1 不采纳。两变量的保留意见仍在（缺 `state_off` + euler 的第三个 arm）。 |
| **`08-20_02-32-38` 从没评过** | fm / 1e-4 / **无 EMA** / 300k。lowlr §6.1 把 `(fm, 1e-4, 无)` 算进"已覆盖的四格"，但三份报告里**没有它的任何数字**——那句"四种组合"实际是三次测量加一次假定。它同时还是唯一的 300k 权重。 |
| 08-22 ↔ 08-24 的归因 | lowlr §4 写成"LR 1e-4 → 1e-5"，两个 job 其实**同时**差了 EMA（无 → 有）。方向大概率没错，但 −25% 这个数不是纯 LR 的。 |
| **`state_in_adaln` 默认值** | 源码默认 `false`，但七个权重里六个是 state on，**加载即崩**。当前推荐的 `fm_lowlr` 正是 state on 那一类。需要改默认值或加按 adaLN 权重宽度自动推断的兼容层。见 fm-lowlr §7.4。 |
| `state_in_adaln=false` + euler | P1 的干净 arm。08-31_06-17 把它和 rk4 绑在一起了。 |
| `n_obs_steps` / `chunk_size` | 七次训练全是 1 / 100，从没动过。 |
| 1e-4 + warmup | encoder-collapse §8 的 P0 备选（保住收敛速度、给 post-norm encoder 挡住第一步）。至今没跑过任何 arm。 |
| ~~第一帧偏差~~ | **2026-09-03 已解**：`fm_lowlr` 首帧 0.01611 vs 空策略 0.01556（1.04×），第 2 帧即穿过空策略。它是 LR 的函数，不是独立于 LR 的缺陷。 |
| 验证集 | 七次训练全是 `dataset_episodes: null`，`--dataset.eval_split` 还是 0.0。每次都要事后跑 `offline_chunk_eval.py` 才知道泛化。 |

## 5. 报告索引

| 报告 | 覆盖 |
|---|---|
| `act_dit-encoder-collapse-2026-08.md` | `08-20_22-59` 的塌缩诊断 + 8000 步受控消融；给出 P0（lr 1e-5）/ P1（state 出 adaLN）两条处方 |
| `act_dit-lowlr-diffusion-2026-08.md` | `08-24` 主评 + `08-22`/`08-20` 对照；证明 P0 有效、objective 是回归来源 |
| `act_dit-flowmatching-deployed-eval-2026-08.md` | `08-27` 部署忠实评测；确认它与 `08-20` 是同一个模型（复跑） |
| **`act_dit-fm-lowlr-2026-09.md`** | **`08-31_05-22` + `08-31_06-17`，五个 arm 同尺重跑；处方超过 ACT baseline、首帧偏差消失** |
| `act_dit-state-in-adaln.patch` | P1 的 16 行补丁，**已合入源码**（`configuration_act_dit.py:91`），仅 `08-31_06-17-39` 用上 |
