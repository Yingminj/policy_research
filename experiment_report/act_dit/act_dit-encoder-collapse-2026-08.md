# ACT-DiT 在真机上"只会背轨迹"的原因：观测编码器被训练关掉了

**Checkpoint** `/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-20_22-59-31-790981/run/checkpoints/200000/pretrained_model`
**Policy** `act_dit` — flow matching, `chunk_size` 100, `n_decoder_layers` 4, `use_vae` false, `use_ema` true (decay 0.9999), `use_cross_attention` true, ResNet18, 200 000 步，常数 LR 1e-4，无 scheduler
**训练集** `tidy_up_stationery_le/batch_success_361`（363 episodes，300 689 帧，`dataset_episodes: null` — 全部用于训练，没有留出验证集）
**对照** `act` baseline，同一训练集，`.../act_tidy_up_stationery_le_batch_success_361_2026-08-17_12-42-42-097328`
**测量日期** 2026-08-22，mgmt01（RTX 4090），`lerobot` conda env + `PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src`
**脚本** `../test_scripts/scripts_act_dit_probe/`（`probe_encoder_collapse.py` / `probe_conditioning.py` / `train_ablation.py` / `sweep_sampling.py`）
**补丁** `act_dit-state-in-adaln.patch`

> **后续（2026-08-27）**：§1 的处方（LR 降回 1e-5）已在 08-24 那次训练里执行并验证有效——
> 编码器没有塌，图像敏感度回到 2.2e-2。但那次同时把 objective 换成了 diffusion，离线动作精度
> 反而全线变差。见 [`act_dit-lowlr-diffusion-2026-08.md`](act_dit-lowlr-diffusion-2026-08.md)。

---

## 1. 结论

这个 checkpoint **在结构上不可能看图**。它的观测编码器在训练早期就被梯度下降关掉了：
最后一层 encoder 的输出 LayerNorm 增益从 1.0 降到 0.13，编码器输出幅值衰减到 ACT 的
1/200，而且**对图像内容完全不敏感**——换一张完全无关的图，编码器输出的相对变化是
`2e-6`（ACT 是 `5e-2`，差 25 000 倍）。cross-attention 拿到的是一个近似常数向量。

于是策略只剩一条路可走：`observation.state` → adaLN → 动作。这正是"背轨迹"的定义——
不管场景里有什么，它都复现名义轨迹。

**原因有两个，8000 步的受控实验（§4）把它们分开了**：

1. **`ACTDiTConfig` 把 `optimizer_lr` 从 ACT 的 1e-5 提到 1e-4，而 ACT preset 没有任何
   scheduler（无 warmup），encoder 又是 post-norm。** 这是主因：把学习率降回 1e-5，
   encoder 输出幅值在 8000 步内**完全不衰减**（0.796 → 0.800），代价只是多花 3.5× 步数
   到达同样的 loss。
2. **`act_dit` 相对 ACT 多出来的 adaLN 通路**，把 `observation.state` 乘性地送进每一层
   解码器。把它拿掉，在相同 loss 下图像敏感度高 3–6×，且几乎不牺牲收敛速度——但只是
   减缓衰减，没有阻止。

第二个独立缺陷（§5）：chunk 的**第一帧**就偏 0.042 rad（训练集内）/ 0.064 rad（held-out），
比"原地不动"还差 4 倍。实测排除了采样噪声这个解释——积分步数 10→50 收益为 0，8 次采样取
均值只降 3%——所以这是系统性偏差。部署侧那一整套滤波 + Hermite 桥接（§6）就是在盖这个洞，
代价是发给机器人的 50 帧里有 40 帧根本不是模型输出。

---

## 2. 离线指标：ACT-DiT 并没有比 ACT 更能泛化

> **2026-08-27 补注。** 本节的 held-out 数字用的是 `batch_1`–`batch_4`（263 ep，08-07…08-12
> 录制）。指定评测集改为 `batch_success_53_eval_data`（53 ep，08-21）之后，这两个
> checkpoint 重测为 ACT-DiT **0.0743**（2.25× 空策略）、ACT **0.0685**（2.44×）——
> 绝对水平好很多，**但两者互换了名次**：旧集上 ACT-DiT 略优（0.1289 vs 0.1355），
> 评测集上 ACT 略优（0.0685 vs 0.0743）。两处差距都在 5% 以内，本节
> "换掉解码器几乎没有改变泛化能力"的结论两边都成立。短 horizon 上评测集给的答案好一点：
> `mae@10` ACT 0.0312 vs `hold_state` 0.0317（打平），ACT-DiT 0.0485（仍差 1.5×）。
> 换集造成的落差见 [`act_dit-lowlr-diffusion-2026-08.md`](act_dit-lowlr-diffusion-2026-08.md) §4.3。
> 下面表里的数字保持原样。

用 `scripts_act_eval_test/offline_chunk_eval.py`（同一套指纹去污染 + `hold_state` 空策略基线）
在**训练时没见过**的 263 个 episode 上评测。`hold_state` = 整个 chunk 都输出当前实测关节角，
也就是"什么都不做"。

| | anchors | policy MAE | hold_state MAE | policy vs. null |
|---|---:|---:|---:|---:|
| ACT-DiT，训练集内 | 15 035 | 0.0370 | 0.1506 | 4.07× |
| ACT-DiT，held-out | 8 897 | **0.1289** | 0.1636 | **1.27×** |
| ACT，训练集内 | 15 035 | 0.0125 | 0.1506 | 12.0× |
| ACT，held-out | 8 897 | 0.1355 | 0.1636 | 1.21× |

held-out 上 ACT-DiT 比"什么都不做"好 27%，ACT 好 21%。**换掉解码器几乎没有改变泛化能力。**
（ACT-DiT 训练集内误差比 ACT 大 3×，那是 flow matching 的采样噪声 + EMA，不是它学得更少。）

按 horizon 拆开，问题更清楚——下面是 held-out 上 chunk 前 N 帧的 MAE：

| horizon | 1 | 10 | 25 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| ACT-DiT | 0.0635 | 0.0656 | 0.0741 | 0.0930 | 0.1289 |
| ACT | 0.0453 | 0.0544 | 0.0702 | 0.0944 | 0.1355 |
| hold_state（什么都不做） | **0.0156** | **0.0320** | **0.0579** | 0.0975 | 0.1636 |

**在真正会被执行的那一段（前 10–25 帧，0.3–0.8 s）上，两个策略都不如原地不动。**
ACT-DiT 在第 1 帧比 `hold_state` 差 4.1 倍。一个动作 chunk 如果连"下一时刻手臂在哪"
都答不对，后面所有的轨迹形状都是在错误起点上外推。

---

## 3. 机制：encoder 被关掉了

ACT 的 encoder 是 **post-norm**（`pre_norm: false`）：每层结尾是 `x = norm2(x + ffn(...))`。
所以**最后一层的 `norm2.weight`（gamma）就是整条观测通路的逐通道总增益**——它是一个
512 维的"总开关"。

普通 ACT 关不掉它：ACT 解码器的输入是全零张量，cross-attention 是观测到动作的唯一通路，
关掉它 loss 会爆。`act_dit` 能关掉：它额外给了解码器一条 adaLN 通路，把
`observation.state` **乘性地**送进每一层。如果那条路就能拟合训练集，缩小 encoder 增益
就是免费的 loss 下降；而一旦增益变小，回传到相机的梯度也随之消失，塌缩自锁。

`probe_encoder_collapse.py` 的输出（`mean|gamma|` 是 encoder 各层输出 LayerNorm 的平均绝对增益，
初始为 1.0；`img-sens` 是换一张图后该层输出的**相对**变化）：

| checkpoint | enc0 γ | enc1 γ | enc2 γ | **enc3 γ** | enc3 输出幅值 | **enc3 img-sens** |
|---|---:|---:|---:|---:|---:|---:|
| act_dit 50k | 0.998 | 0.997 | 0.959 | **0.488** | 0.0969 | **7e-6** |
| act_dit 100k | 0.998 | 0.997 | 0.951 | **0.250** | 0.0456 | **3e-6** |
| act_dit 150k | 0.998 | 0.998 | 0.945 | **0.180** | 0.0047 | **5e-6** |
| act_dit 200k | 0.998 | 0.998 | 0.943 | **0.131** | 0.0038 | **2e-6** |
| **act 200k** | 1.004 | 1.004 | 1.004 | **1.004** | 0.8140 | **5e-2** |

只有最后一层塌，前三层几乎停在初始值——这正是"总开关被关"的形状，不是训练不稳定
（不稳定会伤及全部层）。到 200k 时 enc3 有 35% 的通道 `|γ| < 0.05`。

`probe_conditioning.py` 从输出端确认同一件事。48 个训练帧，**固定同一个初始噪声**，
分别替换某一条通路的输入，看发出的 chunk 移动多少（单位 rad，除以帧间自然差异 0.374 rad）：

| 干预 | Δ chunk (rad) | 占帧间差异 |
|---|---:|---:|
| 三个相机全部换成另一帧的图 | **0.000000** | **0.0 %** |
| 只把 encoder token 那条路的 state 换掉 | **0.000000** | **0.0 %** |
| 只把 adaLN 那条路的 state 换掉 | 0.190663 | 51.0 % |
| 两条路的 state 都换掉 | 0.190663 | 51.0 % |
| state 置为数据集均值 | 0.294721 | 78.9 % |

图像的贡献是 **0**——原始数值是 1e-8 量级，即浮点噪声，不是"很小"。整个 encoder 输出
（图像 + state token）对结果没有可测量的影响，唯一起作用的是 adaLN。这不是"视觉占比低"，
是"视觉通路断了"。

> 早先的记录（memory `act-dit-ignores-cameras`）把这件事记成"视觉占比 8.9%"。两者并不矛盾：
> 那次每个条件都重新采一次初始噪声，测到的 8.9% 全部来自采样方差。固定噪声之后，
> 图像本身的贡献是 0。

### 3.1 一个必须说清楚的边界

encoder 塌缩解释的是"为什么 ACT-DiT 完全不看图"，不解释"为什么这个任务难"。同一数据集上的
ACT baseline 编码器是健康的（img-sens 0.05、γ=1.004），但它在 held-out 上也只比"什么都不做"
好 21%。所以：**修好视觉通路是必要条件，不是充分条件。** 数据本身（场景多样性、物体位置分布）
可能仍然限制上限——那需要另一次测量，不在本报告范围内。

---

## 4. 受控实验：adaLN 上的 state 就是原因

`train_ablation.py`，三个 arm 各在真实数据集上训满 **8000 步**（每个 ~106 min，单卡 4090），
每 250 步量一次 encoder 健康度。三个 arm 只差一件事：

- `shipped` — 就是产出上面 checkpoint 的配置（lr 1e-4，state 走 adaLN）
- `no_state_adaln` — adaLN 只带 flow timestep，state 只走 encoder token（和 ACT 一样）
- `lowlr` — `optimizer_lr` 降回 ACT 的 1e-5，其它不变

`image_sensitivity` = 换一张图之后 encoder 输出的**相对**变化（初始 0.42–0.57）。
`signal` = encoder 输出的绝对幅值（初始 ~0.79）。

### 4.1 按 step 比较

| step | `shipped` img-sens | `no_state_adaln` img-sens | `lowlr` img-sens | `shipped` signal | `no_state_adaln` signal | `lowlr` signal | `shipped` loss | `no_state_adaln` loss | `lowlr` loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.423 | 0.442 | 0.571 | 0.783 | 0.803 | 0.796 | 2.551 | 2.241 | 2.551 |
| 1000 | 0.165 | 0.547 | 1.111 | 0.659 | 0.715 | 0.802 | 0.182 | 0.211 | 0.352 |
| 2000 | 0.169 | 0.497 | 1.111 | 0.593 | 0.668 | 0.802 | 0.138 | 0.144 | 0.252 |
| 4000 | 0.175 | 0.412 | 1.080 | 0.494 | 0.596 | 0.802 | 0.104 | 0.112 | 0.175 |
| 6000 | 0.117 | 0.346 | 1.055 | 0.412 | 0.520 | 0.801 | 0.087 | 0.093 | 0.140 |
| 8000 | 0.210 | 0.351 | 1.034 | 0.363 | 0.473 | 0.800 | 0.083 | 0.086 | 0.124 |

### 4.2 按训练 loss 对齐比较（`lowlr` 学得慢，按 step 比对它不公平）

| 训练 loss | `shipped` step / img-sens / signal | `no_state_adaln` | `lowlr` |
|---|---|---|---|
| 0.300 | 750 / **0.084** / 0.672 | 750 / **0.560** / 0.728 | 1500 / **1.111** / 0.802 |
| 0.220 | 1000 / **0.165** / 0.659 | 1000 / **0.547** / 0.715 | 2750 / **1.091** / 0.802 |
| 0.175 | 1250 / **0.121** / 0.636 | 1500 / **0.485** / 0.693 | 4250 / **1.077** / 0.801 |
| 0.145 | 1750 / **0.147** / 0.608 | 2000 / **0.497** / 0.668 | 6000 / **1.055** / 0.801 |
| 0.125 | 2250 / **0.175** / 0.583 | 3000 / **0.439** / 0.630 | 8000 / **1.034** / 0.800 |

### 4.3 读法 — 与 8000 步之前的初步结论**不同**

在 500–2500 步的窗口里，`no_state_adaln` 和 `shipped` 的差距看起来是决定性的（12×），
当时据此写下"adaLN 上的 state 是直接原因，学习率至多是加速因素"。**跑满 8000 步之后这个
排序反了。** 在每一个 loss 档上：

- **`lowlr` 的 encoder 完全没有衰减。** signal 从 0.796 到 0.800，八千步纹丝不动；
  img-sens 稳定在 1.03–1.11。它确实要多花 3.5× 的步数才到同样的 loss（8000 vs 2250 到
  loss 0.125），但**它到得了**。这是三个 arm 里唯一一个 encoder 毫发无损的。
- **`no_state_adaln` 稳定地比 `shipped` 好 3–6×**（img-sens 0.44–0.56 vs 0.08–0.18），
  signal 衰减也更慢，而且 loss 下降速度基本齐平（到 loss 0.125 只多花 33% 的步数）。
  但它**没有阻止**衰减，只是减速。
- **`shipped` 在两个维度上都最差**，signal 在 8000 步内掉了 54%（0.783 → 0.363）。

所以正确的因果排序是：**10× 的学习率是主因，adaLN 上的 state 是显著但次要的加速因素。**
两者叠加才产生了 §3 里那个 200k 步之后彻底死掉的 encoder。

一个独立佐证：ACT baseline 用 lr 1e-5、也没有 adaLN 通路，训 200k 步后 encoder 完好
（γ=1.004、signal=0.814）——正是 `lowlr` arm 那条水平线延伸到 200k 的样子。

**本表的边界**：8000 步时三个 arm 的 `gamma_last` 分别是 0.962 / 0.973 / 0.996，都还接近初始值；
而真实 checkpoint 在 50k 步时已经是 0.488。也就是说 **γ 的塌缩发生在 8k–50k 之间，本次消融
只看到了衰减的起点，没有看到终点。** "`lowlr` 是避免还是仅仅推迟"这个问题，需要一次更长的
复现才能回答。

> **后续（2026-08-27）已回答：是避免，至少到 200k 步。** 08-24 那次 lr 1e-5 的全长训练在
> 200k 步时 `gamma_last` = 0.973、`frac<0.05` 全程为 0，img-sens 2.2e-2（塌掉那版是 3e-6）。
> 但 enc3 的输出幅值在同一段里仍降了 25%（0.691 → 0.516）——**衰减机制没有被消灭，只是被减速到
> 200k 步内无害**。见 [`act_dit-lowlr-diffusion-2026-08.md`](act_dit-lowlr-diffusion-2026-08.md) §2。

---

## 5. 第二个独立缺陷：chunk 第一帧就是错的，而且不是采样噪声

即使在**训练集内**，ACT-DiT 的 chunk 第一帧误差是 0.0418 rad，而且**几乎不随 horizon 下降**
（0.0418 → 0.0370）。ACT 在同样条件下是 0.0103。held-out 上第一帧是 0.0635 rad，
比"原地不动"的 0.0156 差 4.1 倍。

**最初的猜测是采样方差**：flow matching 每次从一个新的高斯样本出发，跑 10 步 Euler，
所以粗积分 + 随机起点会在输出上留一层噪声。这个猜测被实测否掉了。

`sweep_sampling.py`，`batch_3` held-out，300 个 anchor，28 055 个 anchor-step
（两个旋钮都只影响推理，不需要重训）：

| 配置 | mae@1 | mae@10 | mae@100 |
|---|---:|---:|---:|
| `steps=10, samples=1`（当前部署配置） | 0.07314 | 0.07350 | 0.14018 |
| `steps=50, samples=1` | 0.07336 | 0.07568 | 0.14123 |
| `steps=10, samples=8`（8 次采样取均值） | 0.07084 | 0.07181 | 0.13760 |
| `steps=50, samples=8` | 0.06842 | 0.06960 | 0.13645 |
| `hold_state`（什么都不做） | **0.02125** | **0.03876** | 0.17042 |

- 积分步数 10 → 50：**没有改善**（0.0731 → 0.0734）。Euler 已经收敛，误差不在积分上。
- 8 次采样取均值：只降 3.1%（0.0731 → 0.0708）。也就是说**采样方差最多只解释这个误差的 3%**。

剩下的 97% 是**系统性偏差**：模型对"下一时刻手臂应该在哪"的期望值本身就偏了 0.07 rad
（≈4°）。这和 §3 的结论自洽——策略只能从 `observation.state` 经 adaLN 出发，输出的是
一条粗粒度地按当前位姿索引出来的平均轨迹，而不是当前这一帧真正该接下去的动作。

**所以不要去调 `num_integration_steps` 或加采样平均**，那两个旋钮各值 0% 和 3%。
这个偏差要和编码器一起修。

> 附带一条与此相关但未被验证为主因的观察：`timestep_sampling_strategy="beta"`
> （α=1.5, β=1.0, s=0.999）把训练样本压向 t≈0 的高噪声端，`P(t > 0.9) ≈ 3.1%`——
> Euler 积分最后 10% 的精修路程，训练时只见过 3% 的样本。上面的 steps sweep 说明这
> 目前不是瓶颈，但修好视觉通路之后值得重新量。

---

## 6. 推理侧：发给机器人的 50 帧里，40 帧不是模型输出

`lerobot_vlahost/src/lerobot/rollout/strategies/core.py::send_next_action_chunk` 在把 chunk
交给机器人之前依次做了五件事：

1. `remove_small_rollbacks` — 删除 10 帧窗口内 ≤2 帧的反向毛刺
2. `remove_open_gripper_loops` — 删除张爪时的小环路
3. `smooth_action_chunk(passes=1)` — 前 14 个关节局部平滑
4. `smooth_large_excursions(wave_threshold=100°)` — 削大波峰/波谷
5. **`cubic_hermite_segment`，`bridge_steps = min(40, len(chunk))`** — 用一条三次 Hermite
   曲线**整段覆盖前 40 帧**，起点是当前实测关节角、`start_velocity=0.0`，终点是 `chunk[39]`

部署配置 `deploy_config_act_dit.yaml` 里 `n_action_steps: 50`。所以**每次推理真正送到机器人的
50 帧中，前 40 帧是一条合成的、零初速的 S 曲线**，模型只贡献了一个路标点 `chunk[39]` 和
末尾 10 帧（0.33 s）。策略实际上被当成一个 1.3 s 一次的路标发生器在用，chunk 的形状被丢掉了。

两个直接后果：

- 每个 chunk 都从**速度 0** 起步 → 机械臂每 1.67 s 减速到停一次，这就是观测到的顿挫。
- `record_chunk.txt` 里"36 个 chunk、抓着不动约 12 s、指令幅度 <0.07 rad"的现象，在
  `chunk[39]` 落在当前位姿附近时是必然的：整条桥接曲线退化成原地不动。

这套滤波是针对一个 chunk 本身就是噪声的策略调出来的（§5 的 0.042 rad 地板正是它的成因）。
**在编码器修好之前，它掩盖了问题；修好之后，它会掩盖改进。** 任何一次重训之后，先把
`bridge_steps` 降到 5–10 再评估，否则看不出差别。

其它推理侧核对结果（均无误）：

- BGR→RGB 转换存在（`marvain_m6_http.py::_decode_image`）
- quad 拼图切分与训练 profile `marvin-gripper-quadtile` 的 `camera_tiles` 在
  1280×1440 输入下逐像素一致（顶部 2/3 全宽 → resize 到 640×480；底部左右各 640×480）。
  但部署侧 resize 的目标尺寸是 `(w/2, h/3)` 这个**相对**尺寸，训练侧固定 640×480；
  一旦服务端下调 `image_max_width/height`，两者就会分叉。
- 夹爪标定 `gripper_state_calibration: [[0.0, 1.341], [0.0, 1.300]]` 已生效，把 /state 原始
  反馈映回训练用的 [0,1]
- EMA 权重确实被用于推理：`from_pretrained` 先 load_state_dict 再 `eval()`，
  影子权重与在线权重的相对差中位数 3.2%（最大 13.3%），`num_updates = 200000`

---

## 7. 其它代码审查发现

| 位置 | 问题 | 影响 |
|---|---|---|
| `job.json: dataset_episodes: null` | 363 个 episode 全部进训练，没有验证集 | 这次失败在训练日志里完全不可见；loss 一路降到 0.029 |
| `config.json: observation.velocity` | 16 维 STATE 特征进了 `input_features`，但 `robot_state_feature` 只匹配 `observation.state`，ACT 系没有任何代码读它 | 每个 batch 白解码、白归一化、白传 GPU |
| `/opt/robot-platform/train-venv`（mgmt01） | 里面的 `act_dit` 没有 EMA 相关代码，比 repo 落后；这个 checkpoint 在该 venv 上加载会直接失败 | 训练实际跑在 gpu03。两台机器的 venv 不同步，下一次跑的可能不是你以为的那份代码 |
| `SinusoidalPosEmb(t)`，t ∈ [0,1] | 最高频率是 1.0 rad，整个 256 维嵌入退化成 t 的近似线性函数；参考实现（SD3 / pi0）都先把 t 乘 1000 | 实测速度场对 t 仍有响应（跨 t 的输出散布 0.61，输出幅值 1.41），所以不致命，但嵌入被浪费了 |
| `ACTDiTConfig` 无 scheduler | lr 相对 ACT 提高 10× 到 1e-4，而 ACT preset 的 `get_scheduler_preset()` 返回 None，post-norm 结构没有 warmup | **§4 实测这是 encoder 塌缩的主因**，见 P0 |

---

## 8. 建议的改动，按优先级

**P0 — 把 `optimizer_lr` 降回 1e-5，或者给 1e-4 加 warmup。** 这是 §4 里效果最大的一项：
`lowlr` arm 的 encoder 在 8000 步内零衰减。ACT baseline 在这个集群上用 1e-5 训到 200k 步
encoder 依然完好，是已经验证过的配方。代价是收敛慢 3.5×（到同样的 loss），对 200k 步的
训练预算来说完全付得起。

```bash
# 直接改：不需要动代码
--policy.optimizer_lr 1e-5
```

如果想保留 1e-4 的收敛速度，就必须加 warmup——`ACTDiTConfig` 目前的
`get_scheduler_preset()` 继承自 ACT，返回 None，所以 post-norm 的 encoder 在第一步就吃满
1e-4。（warmup arm 本次未跑，属于后续。）

**P1 — 把 state 从 adaLN 拿掉。** 补丁见 `act_dit-state-in-adaln.patch`：给 `ACTDiTConfig`
加一个 `state_in_adaln: bool = False`，adaLN 只带 flow timestep，state 回到 encoder token
这条 ACT 原有的路。改动 16 行、2 个文件，相同 loss 下图像敏感度高 3–6×，收敛速度几乎不变。

> **2026-08-27 降级：不要和 P0 一起做，留作下一轮的第二个 arm。** 理由有二。其一，P0 单独就
> 够用了：lr 1e-5 训到 200k 步 encoder 完好（上面 §4.3 的后续），而本次消融里
> `no_state_adaln` 从来只是把衰减减速 3–6×、没有阻止过。其二，下一次重训是
> flow matching + 1e-5 + EMA，是四种组合里唯一缺的那格，再叠一个 `state_in_adaln=False`
> 就又成了两变量实验。
>
> 但这条问题**没有消失**：08-24 那次的归因表里 adaLN 那条 state 通路仍占 36.6%，压过图像的
> 27.1%，而 ACT 原有的 encoder token 通路只有 2.8%——解码器还是主要从 adaLN 拿 state；
> img-sens 2.2e-2 也仍比 ACT baseline 的 5.0e-2 低 2.3 倍。
>
> **判据**：fm + 1e-5 那次训完之后跑两个探针，若 enc3 signal 在 200k 内仍掉 20% 以上，或
> adaLN 份额仍高于图像份额，就把这个补丁作为 A/B 的第二个 arm 跑掉。训练预算一旦超过
> 200k 步，直接打。

```bash
cd /home/kewei/YING/robot_data_platform/lerobot
git apply /home/kewei/YING/paper/policy/experiment_report/act_dit/act_dit-state-in-adaln.patch
# 推理侧同一份代码也要打（train/deploy 必须一致）
cd /home/kewei/YING/lerobot_vlahost && git apply .../act_dit-state-in-adaln.patch
```

注意：旧 checkpoint 的 `config.json` 没有这个键，会按新默认值 False 构建，然后在 adaLN
`Linear` 的形状上**明确报错**（不是静默走错结构）。要加载旧 checkpoint，在它的 `config.json`
里补 `"state_in_adaln": true`。

**P2 — 每次训练都监控 encoder。** `mean|gamma|` 和图像敏感度在 loss 还在正常下降的时候
就已经塌了。把 `probe_encoder_collapse.py` 里的两行指标接进训练日志，比任何事后分析都便宜。

**P3 — 留出验证集。** `dataset_episodes` 留 10% 不训，用 `offline_chunk_eval.py` 出一个跨
policy family 可比的数字。验收线写死：**held-out `mae@10` 必须低于 `hold_state` 的
`mae@10`（0.0320）**。当前 ACT-DiT 是 0.0656，ACT 是 0.0544，两个都不及格。

**P4 — 推理侧先松开 Hermite 桥接再评估。** `bridge_steps` 40 → 5~10。否则重训的效果被
合成曲线吃掉，看不出来。

**P5 — 如果 P1 之后仍想给解码器一条低维快通路**，加 state dropout：训练时以 p≈0.2–0.5 把
`observation.state` 换成数据集均值。让策略必须能只凭视觉动作，而不是把这条路堵死。
（本次未测量，属于后续。）

**~~P6 — 采样噪声~~ — 已实测排除。** §5 的 sweep 显示 `num_integration_steps` 10→50 收益为
0，8 次采样取均值只有 3%。chunk 第一帧那 0.07 rad 是系统性偏差，不是采样方差，**不要花时间
在这两个旋钮上**。`timestep_sampling_strategy` 改 `uniform` 可以留到视觉通路修好之后再量。

---

## 9. 复现

```bash
cd /home/kewei/YING/paper/policy/experiment_report/test_scripts/scripts_act_dit_probe
export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src
CK=/mnt/robot_platform/jobs/act_dit_tidy_up_stationery_le_batch_success_361_2026-08-20_22-59-31-790981/run/checkpoints
D=/mnt/robot_platform/datasets/tidy_up_stationery_le

# encoder 健康度（秒级，只读权重 + 一次前向）
conda run -n lerobot python probe_encoder_collapse.py \
  --checkpoint $CK/050000/pretrained_model --checkpoint $CK/200000/pretrained_model

# 通路归因（固定噪声，48 帧）
conda run -n lerobot python probe_conditioning.py \
  --checkpoint $CK/200000/pretrained_model --dataset-root $D/batch_success_361

# 受控训练消融（每个 arm 约 106 min @ 8000 步，单卡独占）
conda run -n lerobot python train_ablation.py --arm lowlr \
  --dataset-root $D/batch_success_361 --steps 8000

# 推理旋钮 sweep（不需要重训）
PYTHONPATH=$PYTHONPATH:../scripts_act_eval_test conda run -n lerobot python sweep_sampling.py \
  --checkpoint $CK/200000/pretrained_model --dataset-root $D/batch_1 \
  --train-root $D/batch_success_361 --steps 10 --steps 100 --samples 1 --samples 8

# held-out 泛化（复用已有脚本）
conda run -n lerobot python ../scripts_act_eval_test/offline_chunk_eval.py \
  --checkpoint $CK/200000/pretrained_model \
  --dataset-root $D/batch_1 --dataset-root $D/batch_2 \
  --dataset-root $D/batch_3 --dataset-root $D/batch_4 \
  --train-root $D/batch_success_361
```

每个脚本都带 `--selftest`，主流程运行前会自动跑一遍。

**相关**：`../act/act_delta-batch5-failure-analysis-2026-08.md`（同一数据集上 ACT 的记忆化）、
`../../warmup_and_loss_curve_audit.md`、`../vita/vita-deploy-vibration-2026-08.md`（同一套
Hermite 桥接对 VITA 的影响）。
