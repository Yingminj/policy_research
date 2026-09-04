# ACT-DiT：处方跑完了 —— flow matching + lr 1e-5 + EMA 第一次超过 ACT baseline，chunk 首帧偏差消失

**被评 checkpoint**
- `act_dit_..._2026-08-31_05-22-33-507248/run/checkpoints/200000/pretrained_model`（下称 `fm_lowlr`）
- `act_dit_..._2026-08-31_06-17-39-059317/run/checkpoints/200000/pretrained_model`（下称 `fm_lowlr_noadaln_rk4`）

**Policy** `act_dit` — `objective: flow_matching`（Euler / `num_integration_steps` 10、
`timestep_sampling` beta 1.5/1.0），`chunk_size` 100，`n_obs_steps` 1，`use_vae` false，
`use_cross_attention` true，`use_ema` true（decay 0.9999），ResNet18，200 000 步，
**常数 LR 1e-5 / backbone 1e-5**，无 scheduler。第二个 arm 另外把 `state_in_adaln`
置 false、积分器换成 rk4。

**训练集** `tidy_up_stationery_le/batch_success_361`（363 episodes，`dataset_episodes: null`，仍无留出验证集）
**评测集** `tidy_up_stationery_le/batch_success_53_eval_data`（53 ep，40 132 帧，2026-08-21 录制；指纹去污染实测丢 **0** 个 episode）
**评测方法** 顶层 [`eval_policy/offline_chunk_eval.py`](../../../eval_policy/README.md)，未分叉未修改。stride 20 → 2 007 anchor，开环 teacher-forced
**对照** `fm_hilr`（08-27）、`diff_lowlr`（08-24）、ACT baseline（08-17）——**三个都在同一把尺子上重跑，不是从旧报告抄的数**
**前作** [`act_dit-encoder-collapse-2026-08.md`](act_dit-encoder-collapse-2026-08.md)（P0/P1 处方）、[`act_dit-lowlr-diffusion-2026-08.md`](act_dit-lowlr-diffusion-2026-08.md)（§6.1 点名要跑这一格）、[`act_dit-flowmatching-deployed-eval-2026-08.md`](act_dit-flowmatching-deployed-eval-2026-08.md)
**测量日期** 2026-09-03，mgmt01（RTX 4090），`/opt/robot-platform/train-venv`
**脚本与原始数据** [`../../../eval_policy/runs/2026-09-03_act_dit_fm_lowlr/`](../../../eval_policy/runs/2026-09-03_act_dit_fm_lowlr/)

---

## 1. 结论

1. **处方生效了，而且比预期更彻底。** `fm_lowlr` 在评测集上全 horizon MAE **0.06730**，
   比"原地不动"好 **2.48×**——**这是第一个在这个集上超过 ACT baseline（0.06848 / 2.44×）
   的 act_dit 权重**。前一版 flow matching（`fm_hilr`，同配方只差 LR）是 0.07373。
   把 LR 从 1e-4 降到 1e-5，**objective 与 EMA 都不动**：0.07373 → 0.06730（**−8.7%**）。
   这一次是干净的单变量对照（§3）。

2. **三篇报告追了一路的"chunk 首帧偏差"没了。** 首帧 MAE **0.01611 rad**，
   `hold_state` 是 0.01556——**比值 1.04×**。前作的同一个数是 3.0×（`fm_hilr` 0.0471）、
   4.4×（`diff_lowlr` 0.0690）。误差曲线**第 2 帧**（0.07 s）就穿到 `hold_state` 以下，
   ACT baseline 第 6 帧，`fm_hilr` 第 11 帧。
   **lowlr §6.5 立的验收线（评测集 `mae@10` < 0.0317）第一次被真正跨过**：
   `fm_lowlr` 0.0235，ACT baseline 0.0312 只是打平（§3.2）。

3. **因此部署侧的 Hermite 桥在这个权重上第一次变成净损失。** K=40 桥使
   MAE 从 0.04734 涨到 0.04961（**+4.8%**），方向与 ACT（−0.5%）一致、与所有旧
   act_dit 权重相反（`fm_hilr` −6.9%、`diff_lowlr` −17.9%）。
   桥是拿"从实测位姿出发的 S 曲线"换掉 50 帧里的前 40 帧——**首帧不再错之后，这笔交易就亏了**。
   前作"act_dit 在真机上还行是因为输出被替换掉 80%"的解释，对这个权重不再成立（§5）。

4. **编码器不只是活着，它比 ACT 更看图。** enc3 输出 LayerNorm 增益 **0.9670**，
   图像敏感度 **0.158**——ACT baseline 是 0.0499，塌掉的 `fm_hilr` 是 **1e-6**。
   **这是七个 act_dit 权重里图像敏感度最高的一个，高过 ACT 3.2 倍**（§4.1）。

5. **P1（把 state 从 adaLN 拿掉）这个 arm 输了，而且方向和假设相反。**
   `fm_lowlr_noadaln_rk4` 全线更差：MAE 0.07073（+5.1%）、首帧 0.02625（+63%）、
   穿过 null 第 6 帧、**图像敏感度反而从 0.158 掉到 0.0575**。
   拿掉那条 state 通路并没有"逼模型去看图"。
   **但这个 arm 同时换了积分器（euler→rk4），是两变量实验**——正是 encoder-collapse §8
   自己警告过的那种。归因不可分离，见 §6 的保留意见。

6. **encoder-collapse §8 P1 的判据，两条各中一条。** 判据原文是"enc3 signal 在 200k 内
   仍掉 20% 以上，**或** adaLN 份额仍高于图像份额，就把补丁作为第二个 arm 跑掉"。
   实测：signal 相对 enc0 只掉 **4.4%**（0.796 → 0.761），第一条不成立；
   但通路归因里 adaLN 仍占 **35.8%**、图像 **30.8%**，第二条成立。
   **判据判对了该跑这个 arm，跑完的结果是不要采纳它**（§4.2）。

**一句话**：lowlr §6.1 点名的那一格跑完了，结论是**处方本身就够了**——
flow matching + lr 1e-5 + EMA，不要叠任何补丁。这是目前最好的 act_dit 配置，
也是第一个在离线动作精度和首帧行为上都不输 ACT baseline 的版本。

---

## 2. 口径

| 项 | 取值 |
|---|---|
| 评测集 | `batch_success_53_eval_data`，53 ep / 40 132 帧，2026-08-21 录制 |
| 去污染 | `--train-root batch_success_361`，action SHA1 指纹，**丢弃 0 个 episode** |
| anchor | stride 20 → **2 007** 个，逐帧从演示状态开环起评（teacher forced，非闭环 rollout） |
| horizon 100 | 全 chunk，与 [lowlr §4](act_dit-lowlr-diffusion-2026-08.md) 同尺 |
| horizon 50 | 执行窗口 + 部署重写，与 [flowmatching-deployed §3](act_dit-flowmatching-deployed-eval-2026-08.md) 同尺 |
| `policy_raw` | 模型原始 chunk，截断到该 horizon |
| `policy_deployed` | 过完 `rollbacks → gripper_loops → smoothing → excursions → bridge(K=40) → gripper_clip` |
| `hold_state` | 空策略：整段保持当前实测关节角 |
| 解释器 | `/opt/robot-platform/train-venv/bin/python`（训练这些 checkpoint 的环境；`act_dit` 在 `.eval()` 时自行换入 EMA 影子权重） |

**噪声底**：`fm_lowlr` horizon 100 原地重跑一次，MAE 0.06730 → 0.06733（**0.04%**），
首帧 0.016109 → 0.016118（0.06%）。**本报告里小于 0.5% 的差一律不解读。**

**关于 `state_in_adaln`**：七个 act_dit 权重里六个的 `config.json` 早于这个字段，
直接加载会在 adaLN `Linear` 上崩。本次为每个权重建了影子目录（符号链接 + 补过字段的
`config.json`），**归档权重未改动一个字节**；`state_in_adaln` 的取值不取自 config，
而是从 adaLN 权重的输入宽度反推（256 = 只有 timestep，768 = state 也在），
脚本对两者做断言。细节见 [run README](../../../eval_policy/runs/2026-09-03_act_dit_fm_lowlr/README.md)。

**一致性核对**：同一把尺子重跑的三个已发表 arm，四个数全部对上（`act_baseline`
h100 0.06848 vs 0.0685、h50 0.05112 vs 0.0511、`diff_lowlr` h100 0.10546 vs 0.1056、
`diff_lowlr` 通路归因 0.271/0.366 vs 27.1%/36.6%）。第一条已写成 `summarise.py`
里的断言。**跨代 harness 拼表是不可信的，所以参照 arm 全部重跑。**

---

## 3. 主结果

### 3.1 全 horizon（100 帧，3.3 s）

`policy_raw` MAE，单位 rad，越小越好。

| checkpoint | 目标 / LR / EMA / adaLN | MAE | RMSE | norm. MAE | vs 空策略 |
|---|---|---:|---:|---:|---:|
| **`fm_lowlr`（08-31_05-22）** | fm / **1e-5** / ✓ / state on | **0.06730** | 0.12472 | **0.2061** | **2.48×** |
| ACT baseline（08-17） | — / 1e-5 / — / — | 0.06848 | 0.12396 | 0.2109 | 2.44× |
| `fm_lowlr_noadaln_rk4`（08-31_06-17） | fm / 1e-5 / ✓ / **state off, rk4** | 0.07073 | 0.12502 | 0.2150 | 2.36× |
| `fm_hilr`（08-27） | fm / 1e-4 / ✓ / state on | 0.07373 | 0.13813 | 0.2275 | 2.27× |
| `diff_lowlr`（08-24） | **diff** / 1e-5 / ✓ / state on | 0.10546 | 0.16620 | 0.3270 | 1.59× |
| `hold_state` | — | 0.16721 | 0.30704 | 0.5136 | 1.00× |
| `train_mean` | — | 0.31850 | 0.38435 | 0.8814 | 0.53× |

**唯一变量的对照**（其余字段逐一相同）：

- **LR 1e-4 → 1e-5**（`fm_hilr` → `fm_lowlr`）：0.07373 → 0.06730，**−8.7%**。
  这是三篇报告里第一次干净地量出 P0 处方在 flow matching 上值多少。
  前作在 diffusion 上量到的 −17% 是 LR 与 EMA 混在一起的（lowlr §4 的两行差了两个字段），
  这次没有这个问题。
- **objective diffusion → flow matching**（`diff_lowlr` → `fm_lowlr`，LR 都是 1e-5）：
  0.10546 → 0.06730，**−36%**。方向与 lowlr §4 一致，量级更大。

### 3.2 短 horizon：首帧偏差，以及验收线

| checkpoint | @1 | @10 | @25 | @50 | 穿过 `hold_state` |
|---|---:|---:|---:|---:|---:|
| **`fm_lowlr`** | **0.01611** | **0.02352** | **0.03405** | **0.04727** | **第 2 帧（0.07 s）** |
| ACT baseline | 0.02529 | 0.03115 | 0.03972 | 0.05112 | 第 6 帧（0.20 s） |
| `fm_lowlr_noadaln_rk4` | 0.02625 | 0.03139 | 0.04036 | 0.05244 | 第 6 帧（0.20 s） |
| `fm_hilr` | 0.04707 | 0.04719 | 0.04901 | 0.05596 | 第 11 帧（0.37 s） |
| `diff_lowlr` | 0.06895 | 0.07705 | 0.08563 | 0.09415 | 第 25 帧（0.83 s） |
| `hold_state` | 0.01556 | 0.03174 | 0.05721 | 0.09705 | — |

**这一格是本次最值得看的。** 前三篇报告都把"chunk 第一帧就是错的"记成一个
**与编码器塌缩相互独立、且没有被任何一次改动修好**的缺陷：
`fm_hilr` 首帧是空策略的 3.0×，`diff_lowlr` 是 4.4×，而且"编码器修好了，
首帧偏差没有跟着修好"（lowlr §5）。

`fm_lowlr` 的首帧是空策略的 **1.04×**。**这个缺陷在只降 LR 之后消失了。**

前作把它和编码器塌缩判成两个独立缺陷，是基于 `diff_lowlr` 那次观测——
但那次同时换了 objective。现在看，**首帧偏差在 flow matching 下是 LR 的函数**
（1e-4: 3.0× → 1e-5: 1.04×），在 diffusion 下不是（1e-5 仍有 4.4×）。
两个缺陷确实不同源，但"独立于 LR"这个更强的说法不成立。

**验收线**（lowlr §6.5：评测集 `mae@10` 必须低于 `hold_state` 的 0.0317）：

| checkpoint | mae@10 | 判定 |
|---|---:|---|
| **`fm_lowlr`** | **0.02352** | **过线（余量 26%）** |
| ACT baseline | 0.03115 | 打平（余量 1.9%，在噪声底之外但不多） |
| `fm_lowlr_noadaln_rk4` | 0.03139 | 打平（余量 1.1%） |
| `fm_hilr` | 0.04719 | 不过 |
| `diff_lowlr` | 0.07705 | 不过 |

**这是这条线立起来之后第一次有权重带着实质余量跨过去。**

---

## 4. 探针

### 4.1 编码器体检

`probe_encoder_collapse.py`，enc3（最后一层）是整条观测通路的总开关。

| checkpoint | enc3 mean&#124;γ&#124; | enc3 signal | enc3 img-sens | γ<0.05 占比 |
|---|---:|---:|---:|---:|
| **`fm_lowlr`** | 0.9670 | 0.76084 | **0.158464** | 0.000 |
| `fm_lowlr_noadaln_rk4` | 0.9678 | 0.70876 | 0.057519 | 0.000 |
| ACT baseline | 1.0037 | 0.81420 | 0.049903 | 0.000 |
| `fm_hilr` | **0.1321** | **0.00382** | **0.000001** | 0.346 |
| `diff_lowlr`（lowlr §2） | 0.9731 | 0.5162 | 0.0219 | — |

`fm_lowlr` 的图像敏感度 **0.158**，是 ACT baseline 的 **3.2 倍**、
`diff_lowlr` 的 **7.2 倍**，七个 act_dit 权重里最高。
enc3 signal 相对 enc0（0.79568）只掉 **4.4%**。

**塌缩这条线可以结案了**：同一个 adaLN 架构，只把 LR 降回 1e-5，
编码器不但不塌，还比 ACT 更依赖图像。encoder-collapse §1 判定的"主因是 LR、
adaLN 通路只是帮凶"，被这次单变量对照证实。

### 4.2 通路归因

`probe_conditioning.py`，`delta_frac_of_spread` = 换掉某条路的输入时，
发出的 chunk 移动了多少（占帧间自然差异的比例）。

| checkpoint | images | state→adaLN | state→token | state 置零 |
|---|---:|---:|---:|---:|
| `fm_lowlr` | **0.3083** | 0.3581 | 0.0053 | 0.4859 |
| `diff_lowlr`（存档复核） | 0.2706 | 0.3662 | 0.0281 | 0.5002 |
| `fm_hilr` | **0.0000** | 0.5606 | 0.0000 | 0.7864 |

`fm_hilr` 那一行的 0.0000 是字面意义的零——三路相机全换掉，输出一帧都不动。

`fm_lowlr` 的图像份额从 0.271 抬到 0.308，但 **adaLN 上的 state 仍然是最大的一条路
（0.358 > 0.308）**。按 encoder-collapse §8 P1 的判据，这一条成立，
所以那个 arm 该跑——而它已经跑了，结果在 §4.3。

> **`fm_lowlr_noadaln_rk4` 没有这一行。** `probe_conditioning.py` 按 road 3 存在
> 来构造条件向量，而这个权重根本没有 road 3，探针直接在 shape 上崩。
> 这不是失败，是那个 arm 的定义。没有为它分叉探针——下一节的 img-sens 已经把
> 要问的问题回答了。

### 4.3 P1 那个 arm：拿掉 adaLN 上的 state，编码器并没有更看图

P1 的假设是：state 走 adaLN 是一条比"看 900 个视觉 token"便宜得多的捷径，
堵死它，模型就只能去看图。encoder-collapse §4 的 8000 步消融支持这个方向
（相同 loss 下图像敏感度高 3–6×）。

200k 步之后不成立：

| | `fm_lowlr` | `fm_lowlr_noadaln_rk4` |
|---|---:|---:|
| enc3 img-sens | **0.158464** | 0.057519（**−64%**） |
| enc3 signal | 0.76084 | 0.70876 |
| MAE（h100） | **0.06730** | 0.07073（+5.1%） |
| 首帧 | **0.01611** | 0.02625（+63%） |
| mae@10 | **0.02352** | 0.03139（+33%） |

**把 state 从 adaLN 拿掉，图像敏感度降了 64%，动作精度全线变差。**
8000 步上成立的方向，到 200k 步反转了——这本身值得记一笔：
**短程消融的结论不能外推到完整训练预算**，encoder-collapse §4 的那组数字
（每 arm 8000 步）在这一点上误导过一次。

---

## 5. 部署重写：桥第一次变成净损失

horizon 50，`policy_raw` → `policy_deployed`，以及逐级滤波归因。

| checkpoint | raw | deployed | 桥的作用 | vs 空策略（deployed） |
|---|---:|---:|---:|---:|
| **`fm_lowlr`** | **0.04734** | 0.04961 | **+4.8%** | 1.96× |
| ACT baseline | 0.05112 | 0.05088 | −0.5% | 1.91× |
| `fm_lowlr_noadaln_rk4` | 0.05237 | 0.05160 | −1.5% | 1.88× |
| `fm_hilr` | 0.05630 | 0.05240 | −6.9% | 1.85× |
| `diff_lowlr` | 0.09418 | 0.07729 | −17.9% | 1.26× |
| `hold_state` | 0.09705 | — | — | — |

逐级归因显示，五级滤波里前四级（rollbacks / gripper_loops / smoothing / excursions）
在所有权重上都只值 ±0.5%，**全部作用都来自 K=40 的 Hermite 桥**
（`diff_lowlr`: `filt_5_bridge` −17.9%、`filt_bridge_only` −18.3%）。

**桥的收益与首帧误差严格同号。** 把上表按首帧误差排序，桥的作用单调翻转：

| 首帧 / `hold_state` | 桥的作用 |
|---|---:|
| `diff_lowlr` 4.43× | −17.9% |
| `fm_hilr` 3.02× | −6.9% |
| `fm_lowlr_noadaln_rk4` 1.69× | −1.5% |
| ACT baseline 1.63× | −0.5% |
| **`fm_lowlr` 1.04×** | **+4.8%** |

flowmatching-deployed §7 的判断——"桥对 chunk 开头本来就错的模型是修补，
对开头本来就对的模型是破坏"——在这张表上是一条单调曲线，不再是两个点的对比。

**部署侧的直接后果**：`fm_lowlr` 是第一个应该**关掉或调小** K=40 桥的 act_dit 权重。
它现在的 raw（0.04734）比任何权重的 deployed 都好，桥只是在把它拉回来。
本次没有扫 K，`--filters` 支持逐级选择，这是一次不需要重训的实验。

---

## 6. 保留意见

1. **P1 那个 arm 是两变量的。** `fm_lowlr_noadaln_rk4` 同时改了 `state_in_adaln`
   （true→false）和积分器（euler→rk4）。§4.3 把全部劣化记在 state 通路上，
   严格说不成立：rk4 在 `num_integration_steps` 10 下的实际步长与 euler 不同，
   单独也可能是负收益。**要下"P1 无效"的定论，需要 `state_in_adaln=false` + euler
   的第三个 arm。** 不过就决策而言这不重要——两个变量一起改的结果是全线更差，
   而单独跑 P0 已经过了验收线，没有理由再往上叠。
2. **仍然没有留出验证集。** 七次训练都是 `dataset_episodes: null`，本次也是事后评。
   `--dataset.eval_split` 还是 0.0。
3. **只评了 200k。** 两个 08-31 job 磁盘上都有中间 checkpoint
   （05-22 有 50k/100k/150k，06-17 有 100k），本次没评，所以**不知道 `fm_lowlr`
   是在哪一步超过 ACT 的，也不知道它是否已经收敛**。
4. **开环，非闭环。** 每个 anchor 都从演示状态起评。真机上 chunk N+1 从 chunk N
   实际停在哪里起步，这套评测复现不了。
5. **首帧偏差"消失"是在这个评测集上、这个 anchor 分布上的结论。**
   1.04× 已经贴着 `hold_state`，进一步的改善没有测量意义——这条指标到顶了，
   后续要换更严的判据。

---

## 7. 建议

1. **把 `fm_lowlr`（08-31_05-22，200k）定为当前的 act_dit 配置基线。**
   flow matching + lr 1e-5 + EMA + state 留在 adaLN，**不打 P1 补丁**。
   它是七个权重里唯一同时满足：MAE 优于 ACT baseline、首帧贴住 `hold_state`、
   `mae@10` 带余量过线、编码器图像敏感度最高。
2. **上真机之前先扫 K。** 桥在这个权重上是 +4.8% 的净损失，且五级滤波里只有桥有作用。
   `--filters` 可以逐级关，不需要重训。**在没扫 K 之前，不要拿真机表现判断这次改进**
   ——现在的部署链路会把它的优势削掉。
3. **评 50k / 100k / 150k。** 磁盘上就有。lowlr 系列尚未量过 flow matching + 1e-5 的
   收敛曲线；如果 100k 已经追平 200k，训练预算可以砍半（与 patch_policy 那边
   optimization-proposals 的结论一致）。
4. **P1 补丁降级为"不采纳"，但保留字段。** §4.3 的数字建议不要在 200k 预算下打它。
   `state_in_adaln: bool = False` 这个默认值需要改回 `True`，否则**六个已有权重
   全部加载即崩**，而当前唯一推荐的权重恰好是 state on 的那一类。
   要么改默认值，要么给 `ACTDiTConfig` 加一段按 adaLN 权重宽度自动推断的加载兼容
   （本次的 `make_ckpt_shims.py` 就是这个逻辑，16 行）。**这是一个会绊倒下一个人的坑。**
5. **下一个变量选首帧之外的。** 首帧指标到顶了。剩下没碰过的是
   `n_obs_steps`（七次训练全是 1）与 `chunk_size`（全是 100）。

---

## 8. 复现

```bash
cd ~/YING/paper/eval_policy/runs/2026-09-03_act_dit_fm_lowlr
./run.sh                                                    # 10 次评测，约 6 min
/opt/robot-platform/train-venv/bin/python summarise.py --selftest
```

探针命令、`state_in_adaln` 影子 checkpoint 的构造方式、以及与已发表数字的逐条核对，
见 [run README](../../../eval_policy/runs/2026-09-03_act_dit_fm_lowlr/README.md)。
