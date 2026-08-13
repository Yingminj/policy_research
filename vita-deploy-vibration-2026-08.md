# VITA 真机部署实验记录与 chunk 抖动诊断

> **2026-08-13（下午）更新**：200k 训练完成，并拿到第一份**未经滤波**的真机录制。
> 见 [§9 一页结论](#9-一页结论本节修正-5-和-7-的两处判断) 与 [§10 修订后的建议](#10-修订后的建议取代-7)。
> **§9 修正了 §5.3 和 §7 的两处判断**：滤波的贡献不小而是几乎全部；训练量在 100k 已到平台。
> 另发现夹爪观测在真机上超出训练分布（§9.1）。
>
> **2026-08-13 更新**：新增 [§5 实验三：100k steps 复测](#5-实验三100k-steps-复测2026-08-13)、
> [§6 `n_obs_steps` / `horizon` / `n_action_steps` 对推理的影响](#6-n_obs_steps--horizon--n_action_steps-对推理的影响)、
> [§7 下一步建议（取代 §3）](#7-下一步建议取代-3)。
> **§1–§4 保持 08-12 原样存档，其结论已被 §5 部分修正**——
> 8/13 的录制经过了一层 8/12 当天新加的手写平滑滤波，与 8/12 的录制**不是同一条链路**，
> 不能直接比较。详见 §5.1。

> 撰写日期：2026-08-12。
> **代码基线**：`/home/kewei/YING/lerobot_vlahost` @ `dev-yw`，VITA 迁移见 commit `79cc122`（"add vita policy"，13 files / +2869）。
> **权重**：`/mnt/robot_platform/jobs/vita_tidy_up_stationery_le_batch_2_2026-08-12_02-57-02-431745/run/checkpoints/last/pretrained_model`
> **训练数据**：`/mnt/robot_platform/datasets/tidy_up_stationery_le/batch_2`（30 episodes / 21568 frames）
> **真机录制**：`/home/kewei/YING/lerobot_vlahost/record_chunk.txt`（93 chunks × 8 actions × 16 joints）
> 配套论文笔记：[`vita-2507.13231.md`](./vita-2507.13231.md)（VITA 算法本身，以 ICLR 2026 v4 为准）。

---

## 0. 一页结论

**工程侧**：VITA 已完整迁进 `lerobot_vlahost`，43 个 VITA 测试 + 114 个相关测试全绿，
`deploy.py` → `ChunkInferenceEngine` → `marvain_m6_http` 的端到端链路已用 mock server 跑通
（863 次 `POST /action_chunk`，字段名与 8 步 chunk 长度均正确）。**部署侧无阻塞问题。**

**模型侧**：真机 chunk 内部的抖动**不来自机器人、不来自数据、也不来自 flow 采样噪声，
而是模型本身尚未拟合出的残差**。关键证据：

| 指标 | 训练数据（30 eps） | VITA 输出（93 chunks） | 含义 |
|---|---|---|---|
| 单步 \|Δ\| 均值 | **0.00398** rad (0.23°) | **0.01323** rad (0.76°) | 3.3× |
| 步差 lag-1 自相关 | **+0.920** | **−0.439** | 平滑 → 近似交替锯齿 |
| 8 步 path/net（仅运动关节） | **1.00**（p90 1.87） | **4.68**（p90 30.3） | 单调 → 反复折返 |
| chunk 内方向反转率 | — | **60.2%** | — |

去掉每个 chunk 的线性趋势后，残差 **0.00762 rad (0.44°)**，趋势跨度 0.02347 rad (1.3°)，
**噪声/信号 = 0.32**。在 30 fps 下这就是一个 ~15 Hz、~0.9° 峰峰值的振荡叠加在真实运动上。

**主因排序**：① `SimpleActionDecoder` 的平铺 `nn.Linear(hidden, horizon×action_dim)` 读出头
在时间轴上无任何结构、L1 重建损失也无平滑项 → 拟合残差沿时间轴白化；
② 严重欠训练（10k steps × batch 8 ≈ **3.7 epochs**）；
③ `flow_matcher_type="exact"` 在 batch=8 上做 OT 重配对，而负责纠偏的 FLD 尚未收敛。

---

## 1. 实验一：VITA 迁移与部署链路验证

### 1.1 迁移内容

源：`/home/kewei/YING/robot_data_platform/lerobot/src/lerobot/policies/vita` → 目标：`src/lerobot/policies/vita/`
（`configuration_vita.py` / `flow_matching.py` / `modeling_vita.py` / `processor_vita.py` / `__init__.py` / `README.md`）。

注册与依赖（缺一不可）：

```python
# src/lerobot/policies/__init__.py
from .vita.configuration_vita import VitaConfig as VitaConfig   # 没有它，draccus 不认识 "vita"

# src/lerobot/policies/factory.py —— 四处显式分支
get_policy_class:        elif name == "vita": ... return VitaPolicy
make_policy_config:      elif policy_type == "vita": return VitaConfig(**kwargs)
make_pre_post_processors: elif isinstance(policy_cfg, VitaConfig): make_vita_pre_post_processors(...)
```

```toml
# pyproject.toml
# scipy 只被 flow_matcher_type="exact"（默认）用到；diffusers 只被训练期的 cosine LR preset 用到。
# 在已保存的 checkpoint 上做推理两者都不需要。
vita = ["lerobot[diffusers-dep]", "lerobot[scipy-dep]"]
```

> **踩坑记录**：`make_policy_config` 的通用回退路径把 `config_cls(**kwargs)` 包在 `try/except Exception` 里，
> 会把 `VitaConfig.__post_init__` 抛出的 `ValueError` 吞掉，对外报成
> `Policy type 'vita' is not available.`。6 个移植测试因此失败，加显式分支后修复——
> 这也是仓库里其他 policy 的既有写法。

### 1.2 唯一的行为性修改：`predict_action_chunk` 的观测步轴

`ChunkInferenceEngine` 从不调用 `select_action`，因此 policy 内部队列始终为空，
`predict_action_chunk` 会走"单帧"分支；而 `VitaModel.generate_actions` 是**从 state 上读观测步数**的：

```python
n_obs_steps = batch[OBS_STATE].shape[1]
assert n_obs_steps == self.config.n_obs_steps       # modeling_vita.py:246 —— 真机上这里炸了
```

单帧分支只给 images 补了轴，state 仍是 `[1, 16]`，于是断言读到的是 **state 维度 16** 而不是观测步数。修复：

```python
queues_populated = any(len(q) > 0 for q in self._queues.values())
if queues_populated:
    batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
else:
    batch = dict(batch)
    for key in self.config.image_features:
        if batch[key].ndim == 4:
            batch[key] = batch[key].unsqueeze(1)
    # generate_actions 从 state 上读观测步数，所以 images 长轴时 state 必须同步长轴，
    # 否则断言读到的是 state 维度而非 n_obs_steps。
    if batch[OBS_STATE].ndim == 2:
        batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
    batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
return self.vita.generate_actions(batch)
```

**为什么不在 `chunk.py` 里改**：`ChunkInferenceEngine` 的 `_obs_history` 只在 `n_obs_steps > 1` 时启用
（`chunk.py:105-108`），对 `n_obs_steps=1` 的 VITA 不生效；而如果改成"总是堆叠"，会打断 ACT——
`modeling_act.py:132` 期望 images 是一组 `[B, C, H, W]` 的普通 list。修复留在 policy 侧才是对的边界。

> 该修复尚未回灌到 `/home/kewei/YING/robot_data_platform/lerobot`，**上游仍有此 bug**。

### 1.3 `chunk.py:105-114` 对 VITA 是否够用

**够用**。本 checkpoint `n_obs_steps=1`，`_obs_history` 不启用，走单帧路径 + §1.2 的修复即可。
如果将来训练 `n_obs_steps=2` 的 VITA，`_stacked_observation()` 产出的 `[B, T, ...]` 布局
正好是 `generate_actions` 期望的布局，也能直接跑。已用参数化回归测试锁住两种情况：

```python
@pytest.mark.parametrize("n_obs_steps", [1, 2])
def test_chunk_engine_drives_vita(n_obs_steps):   # tests/policies/vita/test_vita_rollout.py
```

（临时撤掉 §1.2 的修复 → 3 个测试失败；装回 → 全绿。）

### 1.4 checkpoint 关键配置

`type=vita`，`n_obs_steps=1`，`horizon=16`，`n_action_steps=8`，state/action dim 16，3 相机，
`flow_matcher_type="exact"`，`num_sampling_steps=6`，`action_decoder_type="simple"`，
`recon_loss_type="l1"`，`decode_flow_latents=true`，归一化 STATE/ACTION=MIN_MAX、VISUAL=MEAN_STD。
`job.json`：`steps=10000`，`batch_size=8`，`current_loss=0.031`。

checkpoint 里多声明了一个 `observation.velocity` 输入特征，而 `marvain_m6_http` 不产出它——
**无害**：`NormalizerProcessorStep._normalize_observation` 有 `and key in new_observation` 保护
（`normalize_processor.py:281`），且 VITA 只读 `OBS_STATE` / `OBS_IMAGES`。已实测确认。

### 1.5 测试与端到端验证

- `tests/policies/vita/test_vita.py`（486 行，从源仓库逐字移植）：38 项，覆盖四种 flow matcher、
  形状、梯度、FLD、processor、配置校验。
- `tests/policies/vita/test_vita_rollout.py`（173 行，本仓库新增）：5 项，覆盖 chunk 引擎驱动、
  chunk 长度截断、到 Marvain HTTP payload 的落地、sync 引擎的 `select_action`。
- 汇总：**VITA 43 项通过**；`tests/policies/vita` + `test_rollout.py` + `test_yaml_policy_path.py` + `tests/robots/`
  合计 **114 passed / 2 skipped**。

端到端（`workflows/robot_interaction/deploy_config_vita.yaml` + `mock_echo_server.py`）：

```
Policy loaded: type=vita, device=cuda
ChunkInferenceEngine initialized (device=cuda, n_action_steps=8, chunk_interval_s=0.250, action_keys=16)
… 863 × POST /action_chunk，body 字段 jointcmd_left / jointcmd_right / gripper_left / gripper_right，每次 8 个 action
```

### 1.6 部署机器上能否直接 `pip install -e .`

可以，但注意两点（本机踩过）：

1. **PATH 污染**：`deploy.py` 的 `find_lerobot_rollout` 用 `shutil.which`，可能选到 base 环境里
   那份坏掉的 `lerobot-rollout`（本机报 `huggingface-hub==1.26.0` 版本冲突）。
   装完后确认 `which lerobot-rollout` 指向目标 env 的 bin。
2. **代理变量**：`http_proxy` / `all_proxy` 会让本机 HTTP 请求返回 403。
   跑本地 server 时加 `no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost`。

---

## 2. 实验二：真机 chunk 抖动诊断

### 2.1 数据与方法

`record_chunk.txt`：93 个 chunk，每个含 1 行 16 关节 `Robot State` + 8 行 `Action N`。
解析成 `A ∈ R^{93×8×16}`、`S ∈ R^{93×16}`（脚本见 §4）。对照组是训练集 30 个 episode 的 `action` 列，
**按 `episode_index` 切分后再做差分**（不切会跨 episode 边界，把自相关从 +0.920 稀释到 +0.738）。

### 2.2 测量结果

**幅度**

```
intra-chunk |step delta|   mean=0.01323  p95=0.04094  max=0.15918   (rad)
chunk-seam  |jump|         mean=0.02017  p95=…        max=0.36298   (rad)
seam / intra ratio         1.53×
|action0 - robot_state|    mean=0.032092  max=0.347707
拼接后的执行流 |step delta| mean=0.01409  p99=0.07116 → 30 fps 下均值 0.42 rad/s、峰值 10.9 rad/s
```

**结构**

```
path/net ratio            median=4.69（全关节）/ 4.68（仅运动关节，与训练集口径一致）
lag-1 autocorr of Δ       −0.439            （平滑→+1；白噪声→0；交替锯齿→−0.5）
方向反转率                 60.2%
线性趋势跨度               0.02347 rad
去趋势残差                 0.00762 rad  →  噪声/信号 0.32
```

**噪声在 chunk 内的分布**（无边缘效应，说明不是"chunk 首尾拼接"造成的）：

```
0->1: 0.01559   1->2: 0.01254   2->3: 0.01419   3->4: 0.01277
4->5: 0.01350   5->6: 0.01234   6->7: 0.01165
```

**逐关节**（rad）：抖动最大的是 `right_arm_joint_4`（intra_mean 0.02289、max 0.15918）、
`left_arm_joint_7`（0.01902）、`left_arm_joint_1`（0.01737）；
反转率最高的是 `right_gripper` 74.9%、`right_arm_joint_3` 71.0%、`left_arm_joint_6` 68.1%。
seam 处最大跳变出现在 `right_gripper`（0.36298）和 `left_gripper`（0.20791）——夹爪在 chunk 边界上会跳。

### 2.3 三个排除性诊断

**① 不是数据。** 训练集单步 Δ 均值 0.00398 rad、lag-1 自相关 **+0.920**（中位数 +0.981）、
8 步 path/net 中位数 **1.00**（p90 1.87）——演示轨迹在 8 步窗口内几乎完全单调。
抖动是策略**生成**出来的，不是学来的。

**② 不是采样噪声。** 用同一份观测连续调两次 `predict_action_chunk`，输出**逐比特相同**（max abs diff = 0.0）。
VITA 的 flow 从确定性的视觉隐向量出发（这正是它相对高斯源 FM 的卖点），推理是一个固定函数。
抖动被"烤"进了这个函数本身。

**③ 不是隐空间层面的共同扰动。** 去趋势残差的跨关节相关几乎为零：
`mean r = +0.029`、`mean|r| = 0.223`、仅 **8%** 的关节对 `|r| > 0.5`。
若抖动源自被扰动的隐向量，16 个关节应当同步晃动。事实是每个关节各抖各的。

### 2.4 机制：读出头没有时间结构

②+③ 合起来指向解码器。`SimpleActionDecoder`（`modeling_vita.py:667`）末端是

```python
self.output_proj = nn.Linear(dec_hidden_dim, horizon * action_dim)   # 16 × 16 = 256 个标量读出
```

即**用一个隐向量并联出 256 个互相独立的标量**，前面是若干层 `Mlp`——
沿时间轴既无卷积、也无位置编码、更无自回归。
对比：Diffusion Policy 用时间维 U-Net 卷积、ACT 用带位置编码的 per-step transformer query，
两者都自带平滑先验；VITA 的 simple decoder 一个都没有。
损失侧 `recon_loss_type="l1"` 是逐元素的，**没有任何平滑/时间差分惩罚**。

结果：剩余的拟合误差沿时间轴是白的 → 表现为 −0.44 的 lag-1 自相关，也就是振动。

### 2.5 为什么拟合误差还这么大

`10000 steps × batch 8 / 21568 frames ≈ **3.7 epochs**`，final loss 0.031。**这个模型几乎没训练。**
0.44° 的残差就是没收敛掉的那部分。

叠加因素：`flow_matcher_type="exact"` 在 **batch=8** 上做 minibatch-OT 重配对，训练的是
`z_img[i] → z_act[perm[i]]`，靠 FLD 把逐样本配对还原回来（VITA README 自己也提示了这一点）。
FLD 是开着的（`decode_flow_latents=true`），但 3.7 个 epoch 不足以让它完成这件事。

---

## 3. 建议（按优先级）

1. **加大 batch、延长训练。** 这是主项，单靠它就能解释大部分残差。
   建议至少 30–50 epochs（对应 batch 64 下 ~10–17k steps，或 batch 8 下 ~80–135k steps）。
   评估时**不要只看 loss**，直接把 §4 的三个指标（单步 Δ 均值、lag-1 自相关、path/net）
   算在验证集预测上，目标是自相关从 −0.44 拉向训练集的 +0.92。

2. **消融 `flow_matcher_type=conditional` vs `exact`。** VITA README 明确推荐做这个消融；
   在 batch=8 这种小批量下 OT 配对尤其噪声大。`conditional` 保留原始配对，可作为诊断对照。

3. **如果充分训练后抖动仍在，那就是结构性的**，修复应落在解码器：
   - 把 `SimpleActionDecoder` 的平铺读出换成时间卷积头（或加 per-step 位置编码的 transformer 头）；
   - 或在重建损失里加二阶差分平滑项：`L += λ · ‖Δ²a‖`，λ 用训练集自身的 `‖Δ²a‖` 量级标定。
   两者都是在把 DP/ACT 的时间先验补回来。

4. **次要问题：chunk 接缝。** seam 跳变是 chunk 内步长的 1.53×，夹爪最大 0.363 rad。
   等 §3.1 解决后再处理；可选做法是在 `n_action_steps` 上留重叠并做首尾 blend，
   或改用 RTC 引擎（`execution_horizon` < chunk 长度）让新 chunk 与在执行的 chunk 对齐。
   目前 `deploy_config_vita.yaml` 的 `n_action_steps=8` / `chunk_interval_s=0.25`
   （8 步 @30fps ≈ 0.27 s）时间上是匹配的，接缝问题不是配置错。

5. **不建议**在部署侧加低通滤波/插值来盖住抖动。策略层已有插值回退路径，但那是拿滞后换平滑，
   并不能修复一个还没学会轨迹的模型，而且会掩盖上面 1–3 的评估信号。

6. **工程收尾**：把 §1.2 的 `predict_action_chunk` 修复回灌 `robot_data_platform/lerobot`；
   在 `AGENTS.md` 的 policy 列表里补上 `vita`。

---

## 4. 复现脚本

已归档到 [`scripts_vita_chunk/`](./scripts_vita_chunk/)（含解析好的 `A.npy` / `S.npy`，可直接重跑）。
脚本里的路径常量仍指向原 scratchpad 目录，换机器时改 `SP` 即可。

| 脚本 | 作用 |
|---|---|
| `analyze_chunk.py` | 解析 `record_chunk.txt` → `A.npy`/`S.npy`；幅度、seam、反转率、逐关节 |
| `analyze2.py` | path/net、lag-1 自相关、趋势/残差分解、chunk 内位置分布 |
| `train_stats.py` | 训练集对照（**必须按 `episode_index` 切分再差分**） |
| `matched.py` | 同口径 path/net + 去趋势残差的跨关节相关 |
| `chunk_e2e.py` / `chunk_no_vel.py` | 端到端加载、`observation.velocity` 缺失验证、确定性验证 |
| `deploy_config_vita_smoke.yaml` | mock server 端到端跑通所用配置（跑出 863 次 `POST /action_chunk`） |

核心指标的口径（复用到后续训练评估）：

```python
d = np.diff(A, axis=1)                       # chunk 内步差
autocorr = corrcoef(x[:-1], x[1:])           # 平滑 +1 / 白噪声 0 / 锯齿 -0.5
path_net = |diff(w)|.sum(0) / |w[-1]-w[0]|   # 1.0 = 单调
resid    = y - polyfit_line(y)               # 噪声/信号 = mean|resid| / trend_span
```

---

# 2026-08-13 追加

> **代码基线**：`/home/kewei/YING/lerobot_vlahost` @ `dev-yw` = `cf98aca`（08-13 merge）。
> 关键区别见 §5.1：链路里多了 commit `971aa0b`（"Resolve the jitter problem"，08-12）引入的**手写后处理滤波栈**。
> **权重**：`/mnt/robot_platform/jobs/vita_tidy_up_stationery_le_batch_3_2026-08-12_12-06-10-620381/run/checkpoints/100000`
> **训练数据**：`.../datasets/tidy_up_stationery_le/batch_3`（**120** episodes / **72743** frames）
> **训练配置**：`steps=200000`、`batch_size=32`、`lr=1e-4` cosine、`resume` 自 `033000`。
> 100k step ≈ **44 epochs**（对比 08-12 的 3.7 epochs）；截至分析时训练已跑到 184k step、`loss=0.005`。
> **真机录制**：`/home/kewei/YING/robot_data_platform/record_chunk.txt`（65 chunks × 8 actions × 16 joints）

## 5. 实验三：100k steps 复测（2026-08-13）

### 5.0 一页结论

**抖动确实实质性好转，而且是真的模型进步——但今天的录制同时被一层新加的手写滤波修饰过，
两者必须拆开看。**

四个变量同时改变，不是单变量实验：

| | 08-12 | 08-13 |
|---|---|---|
| checkpoint | 10k steps | **100k steps** |
| batch_size | 8 | **32** |
| 数据集 | batch_2（30 eps / 21568 frames） | **batch_3（120 eps / 72743 frames）** |
| 发送前后处理 | **无** | **5 级手写滤波栈**（§5.1） |

拆解结论（方法见 §5.2、§5.3）：

- **模型本身的进步是真实且主导的**。用**完全没被滤波碰过的夹爪通道**做对照，
  lag-1 自相关从 **−0.495 → +0.410**，反转率 **70.9% → 24.9%**，path/net **8.62 → 2.97**。
  §2.4 说的"读出头无时间结构导致残差沿时间轴白化"——训练量上去之后这个残差被大幅压掉了，
  **不需要改结构**。
- **滤波栈的边际贡献在今天已经很小，但它掩盖了真实指标**。把今天的滤波原样回放到昨天的数据上，
  只能把自相关从 −0.439 抬到 **+0.125**（今天实测 +0.444）——所以今天的好转不是滤波造成的。
  但反过来，今天手臂关节的漂亮数字（反转率 4.7%）**不能当作模型指标**，因为它是滤过的。
- **剩下的"明显波动"已经换了性质**：不再是 §2 那种 chunk 内 ~15 Hz 的高频毛刺，
  而是**chunk 边界处每 0.27 s 一次的低频顿挫（~3.7 Hz）**。
  接缝速度是 chunk 内部速度的 **2.88×**，且 **83% 的接缝跳变方向与运动方向相反**（即回退）。
  机理见 §5.4：**滤波栈按设计保留首尾帧，把抖动能量从 chunk 内部挤到了 chunk 边界上**，
  再叠加"新 chunk 的第一帧被钉回实测位置"这一动作，形成"加速—减速—回弹—再加速"。

### 5.1 首要发现：今天的录制不是模型原始输出

`record_chunk.txt` 由 `src/lerobot/rollout/strategies/core.py` 写出，
而写文件之前 chunk 已经过 **5 级后处理**（`core.py:440-630`，实现在 `src/lerobot/rollout/trajectory.py`）：

| 顺序 | 函数 | 作用 | 作用范围 |
|---|---|---|---|
| ① | Hermite 降级插值 | 首帧偏离实测位置超阈值时，连续拒绝 2 次后用三次 Hermite 重建前 K=40 帧 | 前 14 关节 |
| ② | `remove_boundary_rollbacks` | 把"实测位置 → 第一帧"与后 10 步主方向相反的边界回退删掉 | 前 14 关节 |
| ③ | `remove_small_rollbacks` | 删除 10 步窗口内 ≤2 步的短暂反向毛刺 | 前 14 关节 |
| ④ | `remove_open_gripper_loops` | 删除张爪状态下的小环路 A→B→C | 前 14 关节 |
| ⑤ | **`smooth_action_chunk`** | **二项式 `[0.25, 0.5, 0.25]` 平滑 1 遍，首尾帧不动** | **前 14 关节** |
| ⑥ | `smooth_large_excursions` | 抹平偏离两端 >100° 的大波峰/波谷（阈值极大，实际几乎不触发） | 前 14 关节 |

三点必须记住：

1. **08-12 的基线 `79cc122` 里没有这些代码**（`git merge-base --is-ancestor 79cc122 971aa0b` → 否；
   `git show 79cc122:.../core.py | grep -c smooth_action_chunk` → 0）。
   §2 的所有数字是**原始模型输出**，§5 的手臂数字是**滤过的输出**。
2. **⑤ 是一个在 Nyquist 频率上增益为 0 的滤波器**：`|H(ω)| = 0.5 + 0.5·cos(ω)`，ω=π 时为 0。
   它**按定义**把"逐帧交替锯齿"完全消掉——也就是把 §2 里那个 −0.44 的 lag-1 自相关
   直接抹成正数。这正是 §3.5 当时不建议做的事（"不建议在部署侧加低通滤波盖住抖动"），
   现在它在链路里了。作为工程手段可以接受，但**评估必须绕开它**。
3. **所有 6 级都写死 `joint_count=14`，两个夹爪（index 14/15）从头到尾没被碰过。**
   这是个意外的礼物：**夹爪通道就是今天唯一的未污染对照组**。

证据本身也自洽：今天手臂关节反转率 1–2%、夹爪 19.5% / 30.3%，
断层恰好落在 index 14 这条滤波边界上——模型不可能产生这种分裂。

### 5.2 用夹爪通道做对照（未被滤波污染）

| 指标 | 08-12 夹爪（10k，原始） | 08-13 夹爪（100k，原始） | 训练集 batch_3 夹爪 |
|---|---|---|---|
| 单步 \|Δ\| 均值 | 0.00855 | **0.00546** | — |
| lag-1 自相关 | **−0.495** | **+0.410** | +0.731 |
| 8 步 path/net | 8.62 | **2.97** | — |
| 反转率 | 70.9% | **24.9%** | **0.8%** |
| 噪声/信号 | 0.38 | **0.12** | — |

**自相关从 −0.495 翻到 +0.410 是模型的真实进步**，与滤波无关。
§3.1 的建议（加大 batch、训到 30–50 epochs）被验证有效：44 epochs 把锯齿从"近似交替"拉到了"基本平滑"。

同时它也标出了**还差多远**：夹爪反转率 24.9% vs 训练集 0.8%，仍有 **30×** 差距。
夹爪是目前最脏的通道，而且它恰好是抓取成败的关键自由度。

### 5.3 滤波回放实验：把今天的滤波打到昨天的数据上

复现 `smooth_action_chunk`（二项式 1 遍、首尾帧固定、仅前 14 列）后逐条对齐：

| 全部 16 关节 | \|Δ\| 均值 | lag-1 自相关 | path/net | 反转率 | 噪声/信号 |
|---|---|---|---|---|---|
| 08-12 10k，原始（§2 报告值） | 0.01323 | −0.439 | 4.68 | 60.2% | 0.32 |
| 08-12 10k，**加今天的滤波** | 0.00750 | **+0.125** | 2.52 | 38.3% | 0.21 |
| 08-13 100k，实测（已滤波） | **0.00452** | **+0.444** | **1.00** | **7.2%** | **0.05** |

滤波单独只能把昨天的模型带到 +0.125，离今天的 +0.444 还差很远——
**所以今天的好转主要来自训练，不是来自滤波**。反过来，在今天这个质量水平上
滤波已经没多少 Nyquist 能量可削，边际贡献很小（夹爪 +0.410 → 手臂 +0.448）。

**结论：滤波栈现在是"锦上添花 + 指标污染"，不是"遮丑"。**
但既然它已经不解决主要问题了，建议按 §7.3 处理。

### 5.4 剩下的波动：接缝顿挫（现在的主要问题）

今天的 65 chunks，全 16 关节：

```
intra |step delta|   mean=0.00452  p95=0.02078  max=0.16933   (rad)
chunk-seam  |jump|   mean=0.00850  p95=0.03033  max=0.23016   (rad)
seam / intra         1.88×          （08-12 是 1.53×，相对更差了）
```

接缝的方向性是关键证据：

```
接缝跳变方向与"前一 chunk 末速度"一致：   16.7%      → 83% 是回退
接缝跳变方向与"后一 chunk 初速度"一致：   23.1%
前一 chunk 末速度 vs 后一 chunk 初速度同向：73.8%      → 运动本身是连贯的
|seam| / |v_prev| 中位数                  3.18×      → 回退幅度是一个正常步长的 3 倍
```

即：**运动方向本身没问题（73.8% 连贯），是位置在边界上被"拽回去"了。**

机理有两个叠加因素：

**(a) 机器人跟不上 chunk，新 chunk 又把目标钉回实测位置。**

```
|state(k+1) − last_action(k)|  mean=0.02511  max=0.23016   ← 请求下一个 chunk 时，手臂落后末帧 0.025 rad
|action0(k+1) − state(k+1)|    mean=0.01959  max=0.23742
手臂关节中 action0 与实测位置逐比特相等的比例：70.8%（保持 1 帧，不延续到第 2 帧）
```

0.025 rad 的落后相当于 **5.6 个 chunk 内步长**。`remove_boundary_rollbacks` 把第一帧钉回实测位置，
于是每 0.25 s 目标位置就朝后一跳。这就是那 83% 的回退接缝。

**(b) 滤波栈保留首尾帧，把抖动能量挤到边界。**
归一化的 chunk 内速度剖面（1.0 = 该 chunk 均值）：

```
今天 手臂（滤过）    0.634 0.903 1.187 1.311 1.286 0.997 0.684   峰/边 = 1.99
今天 夹爪（未滤）    1.289 0.709 0.731 1.133 1.103 1.280 0.755   峰/边 = 1.11
08-12 手臂（未滤）   1.174 1.008 1.046 1.019 1.013 0.910 0.830   峰/边 = 1.02
训练集 batch_3       1.033 0.997 0.973 0.969 0.978 1.004 1.045   峰/边 = 1.00
```

**训练数据是平的，昨天未滤波的输出也是平的，只有今天滤过的手臂是个 1.99 的钟形。**
这个钟形是滤波造成的，不是模型学出来的：②把开头的边界回退删平，⑤又固定首尾帧只压中间。

两者合起来，拼接执行流按 8 步周期的相位速度：

```
phase 0: 0.05053   phase 1: 0.06692   phase 2: 0.08563   phase 3: 0.09566
phase 4: 0.08958   phase 5: 0.07097   phase 6: 0.04730   phase 7: 0.13606  ← 接缝
调制深度 max/min = 2.88×
```

**每 8 帧（0.27 s，≈3.7 Hz）一次"慢—快—慢—猛跳"**。这就是现在肉眼看到的波动，
性质与 §2 那个 ~15 Hz 的高频毛刺完全不同。

**(c) 时序上还有一个 6% 的错配。** `deploy_config_vita.yaml`：`n_action_steps=8`、
`chunk_interval_s=0.25`，而 8 帧 @30 fps = **0.2667 s**。每个 chunk 的最后一帧
系统性地来不及执行就被下一个 chunk 抢占，等于长期只执行 7.5 步——而第 8 帧恰好是
滤波器**不平滑**的那一帧。改成 `chunk_interval_s = 0.2667` 是零成本的一步。

### 5.5 其余指标（供后续对比）

08-13 / 100k / 全 16 关节，以及训练集 batch_3 对照：

| 指标 | 08-12（10k，原始） | 08-13（100k，已滤波） | 训练集 batch_3 |
|---|---|---|---|
| 单步 \|Δ\| 均值 | 0.01323 | 0.00452 | **0.00437** |
| lag-1 自相关 | −0.439 | +0.444 | **+0.943**（中位 +0.983）|
| 8 步 path/net（仅运动关节）| 4.68（p90 30.3）| 1.00（p90 2.39）| **1.00**（p90 1.82）|
| 反转率 | 60.2% | 7.2% | 手臂 4.4% / 夹爪 0.8% |
| 去趋势噪声/信号 | 0.32 | 0.05 | — |

单步幅度已经和训练集**一致**（0.00452 vs 0.00437），path/net 也已**完全一致**（1.00 / p90 2.39 vs 1.82）。
自相关 +0.444 vs +0.943 的剩余差距，主要由夹爪（+0.410）和接缝拖着。

去趋势残差的跨关节相关：`mean r = −0.020`、`mean|r| = 0.321`、25% 的关节对 `|r| > 0.5`
（08-12 是 0.223 / 8%）。相关性上升是滤波把各关节往同一个低通形状上压的副作用，
不构成"隐向量被扰动"的新证据。

---

## 6. `n_obs_steps` / `horizon` / `n_action_steps` 对推理的影响

本 checkpoint：`n_obs_steps=1`、`horizon=16`、`n_action_steps=8`
（VITA 论文里叫 `obs_horizon` / `pred_horizon` / `action_horizon`，
对应关系写在 `configuration_vita.py:38-40`）。约束是
`n_action_steps <= horizon − n_obs_steps + 1`（`configuration_vita.py:211`）。

### 6.1 `n_obs_steps = 1`：模型只看当前这一帧

每次推理只喂 1 帧观测（3 路相机 + 16 维 state），**没有任何历史**。
观测编码维度直接由它决定：`obs_dim = n_obs_steps × (num_cameras × feature_dim + state_dim)`
（`modeling_vita.py:385`）。

对推理的影响：

- **好处**：无状态、无预热。部署侧 `ChunkInferenceEngine` 的 `_obs_history` 只在 `n_obs_steps > 1`
  时启用（`chunk.py:105-108`），所以 `=1` 时走单帧路径，第一个 chunk 就能发，
  也不会因为丢帧而污染历史队列。§1.2 那个修复正是为了让这条单帧路径走通。
- **代价**：**模型无法观测速度**。位置相同、速度不同的两个时刻，对模型是同一个输入，
  但正确的动作不同——这是一种部分可观测。它必须从图像里猜运动趋势，猜不准的部分
  就变成沿时间轴的残差，也就是 §2.4 那个抖动的一部分来源。
  这也解释了为什么 checkpoint 里声明了 `observation.velocity` 输入特征（虽然真机不产出、§1.4 已确认无害）——
  **如果真机能提供速度，把 `observation.velocity` 接上比把 `n_obs_steps` 提到 2 更省算力**
  （后者会让视觉 backbone 的前向翻倍）。

改成 2 的代价：视觉编码要跑 2 帧，`obs_dim` 翻倍，推理延迟接近翻倍——
在 `chunk_interval_s=0.25` 的预算下要先量一下延迟。

### 6.2 `horizon = 16`：一次生成 16 步，但只有 8 步会被执行

`horizon` 是**动作自编码器和 flow 的作用长度**，不是执行长度：

- `SimpleActionDecoder.output_proj = nn.Linear(dec_hidden_dim, horizon × action_dim)`
  = 16×16 = 256 个标量读出（`modeling_vita.py:695`）。
- 训练时 loss 覆盖全部 16 步（`modeling_vita.py:272` 断言 `action` 形状为 `(B, horizon, A)`）。
- 推理时 `generate_actions` 生成 16 步，然后**只切出 `[start : start + n_action_steps]`**，
  `start = n_obs_steps − 1 = 0`（`modeling_vita.py:262-264`）→ **后 8 步直接丢弃**。

所以 `horizon > n_action_steps` 的那部分是"垫料"：它让被执行的 8 步位于预测窗口的**中前段**，
而不是贴着边界。边界处的预测通常最差（缺少后续上下文约束），把它丢掉是划算的。
这也是为什么 §5.4 观察到的钟形速度剖面**不是** horizon 末端的衰减——
我们执行的是 0–7 步，正处在模型最有把握的前半段。

另外 `horizon` 有两个硬约束（`configuration_vita.py:196-207`）：
`latent_dim` 必须被 `horizon` 整除（simple 编码器）；`horizon >= 2**action_ae_num_layers`
（cnn 编码器每层折半，本 checkpoint `action_encoder_type="cnn"`、`num_layers=4` → `horizon >= 16`，**刚好卡在下限**）。
想把 `horizon` 降到 8 就必须同时把 `action_ae_num_layers` 降到 3。

### 6.3 `n_action_steps = 8`：重规划周期，直接决定接缝频率

执行 8 步再重新推理一次。它同时定义了三件事：

1. **闭环频率** = `fps / n_action_steps` = 30/8 = **3.75 Hz**。这就是 §5.4 那个顿挫的频率——
   **接缝频率就是重规划频率**，不是巧合。
2. **开环时长** = 8/30 = **0.267 s**。这段时间里模型看不到任何新观测，
   靠 0.267 s 前的一帧图像外推。手臂在这期间累积了 0.025 rad 的跟踪误差（§5.4a）。
3. **训练侧的 `drop_n_last_frames = 8`**，注释给的公式是
   `horizon − n_action_steps − n_obs_steps + 1`（`configuration_vita.py:82-83`）
   = 16−8−1+1 = 8。改 `n_action_steps` 时这个值要跟着改，否则 episode 末尾的采样窗口会错。

**调参方向**：

- **调小**（例如 4）：闭环更快，跟踪误差和接缝幅度都会下降，但推理频率翻倍
  （0.133 s 一次），要先确认 GPU 端跟得上；`num_sampling_steps=6` 的 flow 采样是主要开销。
- **调大**：开环更久，误差累积更多，接缝更大。不建议。
- **注意**：`n_action_steps` 是**训练期就固化进 checkpoint** 的语义
  （通过 `drop_n_last_frames` 影响采样窗口），推理时在 `deploy_config_vita.yaml` 里
  只能**往小调**（配置里写的值会被 clamp 到 `config.n_action_steps`，见 yaml:134 注释）。
  往小调是安全的，等于"只执行前几步就重规划"，不需要重训。

### 6.4 三者与本次抖动的关系（一句话版）

| 参数 | 对抖动的作用 |
|---|---|
| `n_obs_steps=1` | 看不到速度 → 部分可观测 → 贡献沿时间轴的拟合残差（§2.4 的次要来源） |
| `horizon=16` | 只是垫料，**不是抖动来源**；且它已卡在 cnn 编码器的下限，不建议动 |
| `n_action_steps=8` | **直接决定接缝频率 3.75 Hz 和开环误差 0.025 rad**——现在的主要问题就在这 |

---

## 7. 下一步建议（取代 §3）

按投入产出排序。

1. ~~**先把评估口径从滤波里摘出来（零成本，最重要）。**~~ **已于 08-13 实施。**
   `core.py:384` 在任何后处理之前 `raw_chunk = chunk.clone()`，
   `record_chunk.txt` 改为记录这份原始输出（仍过 `robot_action_processor`，
   所以单位/键名与下发的一致，也**与 08-12 的基线录制口径一致、可直接比较**）；
   发给机器人的仍是滤波后的 chunk，真机行为不变。
   每个 chunk 头部多写一行 `Source: raw policy output (pre-filter)` 作为来源标记
   （现有解析脚本按前缀匹配，会忽略它）。
   > 注意：写文件是**追加**模式。下次跑之前先把机器人上旧的
   > `/home/snorlax/Documents/test_chunk/record_chunk.txt` 移走，
   > 否则滤波后的旧数据会和原始的新数据混在同一个文件里。

2. **`chunk_interval_s: 0.25 → 0.2667`（零成本）。**
   消掉 §5.4c 那个 6% 的时序错配，让第 8 帧真正被执行完。

3. **接缝是现在的主要矛盾，按这个顺序处理：**
   - **(a) 优先试 `n_action_steps: 8 → 4`**（推理侧改配置即可，会被 clamp，不用重训）。
     闭环频率翻倍到 7.5 Hz，开环误差和接缝幅度大致减半。先量一次推理延迟确认能跑到 0.133 s。
   - **(b) 改边界策略：不要把第一帧钉回"实测位置"，而应接续"上一条已发出的指令轨迹"。**
     现在 `remove_boundary_rollbacks` 用实测位置做锚（70.8% 的手臂关节被逐比特钉住），
     而手臂天然落后指令 0.025 rad，于是每个 chunk 都朝后弹一次。
     以上一 chunk 的对应帧为锚做 blend，能直接消掉那 83% 的回退接缝。
   - **(c) 或者直接换 RTC 引擎**（`deploy_config_vita.yaml` 已有 `execution_horizon` 字段）。
     它按设计就是让新 chunk 与在执行的 chunk 对齐，是 (b) 的正规做法。

4. **夹爪是现在最脏的通道**（反转率 24.9% vs 训练集 0.8%，chunk 内最大跳变 0.169 rad、
   接缝最大 0.177 rad），而且被所有滤波排除在外。抓取成败就靠它。
   建议：等 200k 训练跑完先复测——夹爪从 70.9% 掉到 24.9% 说明它对训练量很敏感，
   可能还在继续收敛。若 200k 后仍 >10%，再考虑单独处理。

5. **继续训练是划算的，但要盯住收益曲线。** 100k（44 epochs）已经拿到了主要收益；
   现在 184k / loss 0.005。等 200k 跑完，**用同一份观测在 100k 和 200k 两个 checkpoint 上
   离线跑 §4 的三个指标**（不用上真机），确认还有没有继续训的价值。
   评估目标：夹爪 lag-1 自相关 +0.410 → 训练集的 +0.731。

6. **§3.3 的结构性修改（时间卷积读出头 / 二阶差分平滑损失）现在可以降级为"暂不需要"。**
   §2.4 的诊断没错，但 44 epochs 已经把那个残差压到接近训练集水平
   （单步幅度 0.00452 vs 0.00437，path/net 1.00 vs 1.00）。
   结构改动应该等 §7.1 的干净评估口径建立之后再谈。

7. **仍未完成的工程收尾（§3.6 遗留）**：§1.2 的 `predict_action_chunk` 修复还没回灌到
   `robot_data_platform/lerobot`；`AGENTS.md` 的 policy 列表仍缺 `vita`。

## 8. 08-13 复现脚本

已归档到 [`scripts_vita_chunk_0813/`](./scripts_vita_chunk_0813/)（含解析好的 `A100k.npy` / `S100k.npy`
和当天的 `record_chunk_0813.txt`，可直接重跑）。`tr3.py` / `trg.py` 需要 `conda run -n lerobot`
（base 环境的 pandas 与 numpy 2.x 不兼容）。

| 脚本 | 作用 |
|---|---|
| `an.py` | 解析今天的 `record_chunk.txt` → `A100k.npy`/`S100k.npy`，输出 §5.5 全部指标 |
| `seam.py` | 接缝方向性、跟踪落后量、chunk 内速度剖面、相位调制（§5.4） |
| `a0.py` | `action0` 与实测位置逐比特相等的比例（§5.4a 的 70.8%） |
| `filt.py` | **滤波回放实验**：复现二项式滤波并打到昨天数据上（§5.3），按手臂/夹爪分通道 |
| `tr3.py` / `trg.py` | 训练集 batch_3 对照，含逐关节反转率与自相关 |

口径与 §4 完全一致，新增两条：

```python
# 接缝是否为回退：与前一 chunk 末速度同向的比例（<50% 即为回退主导）
agree = mean(sign(A[1:,0]-A[:-1,-1]) == sign(A[:-1,-1]-A[:-1,-2]))
# chunk 内归一化速度剖面（训练集应为平的 ~1.0；钟形 = 滤波痕迹）
sp = |diff(A,axis=1)|.sum(axis=2); prof = (sp / sp.mean(axis=1,keepdims=True)).mean(axis=0)
```

---

# 2026-08-13（下午）追加：200k 原始输出复测

> **权重**：`.../vita_tidy_up_stationery_le_batch_3_.../run/checkpoints/200000`（训练已 `done`，`exit_code=0`，`loss=0.004`）
> 200k × batch 32 / 72743 frames ≈ **88 epochs**。
> **真机录制**：`record_chunk.txt`，52 chunks，**全部带 `Source: raw policy output (pre-filter)` 标记**——
> 这是第一份未经滤波的手臂数据，§5 里靠夹爪通道代理的推断现在可以直接验证。
> 归档：[`scripts_vita_chunk_0813pm/`](./scripts_vita_chunk_0813pm/)

## 9. 一页结论（本节修正 §5 和 §7 的两处判断）

**三件事，两件是坏消息。**

1. **100k → 200k 没有再带来收益。** 在唯一可跨录制比较的夹爪通道上：
   lag-1 自相关 **+0.410 → +0.267**，单步幅度 **0.00546 → 0.00831**，path/net **2.97 → 3.38**，
   反转率 24.9% → 24.0%。**训练量这条杠杆已经用尽**，再往上堆 steps 不解决问题。
   （样本量小、两次录制轨迹不同，所以判定为"已平台化"而非"变差"。）

2. **§5.3 关于"滤波边际贡献很小"的估计是错的，现在可以直接测量。**
   把同一个二项式滤波打到 200k 的原始手臂数据上：

   | 200k 手臂 | \|Δ\| | lag-1 自相关 | path/net | 反转率 |
   |---|---|---|---|---|
   | 原始输出 | 0.00797 | **−0.080** | 1.12 | **27.1%** |
   | + 二项式滤波 | 0.00715 | **+0.558** | 1.00 | **8.0%** |

   **滤波贡献了几乎全部的"平滑"外观。** §5 里 100k 那个 +0.448 / 4.7% 的漂亮数字
   基本上是滤波器的读数，不是模型的。**模型在 88 epochs 之后，手臂仍然在逐帧锯齿**
   （自相关 −0.080，训练集是 **+0.973**；反转率 27.1%，训练集 **4.4%**）。

3. **接缝是现在的头号缺陷，而且它不是跟踪滞后造成的。**

   ```
   intra |step delta|   mean=0.00801   seam |jump| mean=0.03522   seam/intra = 4.40×
   相位速度 0.143 0.140 0.136 0.118 0.137 0.100 0.124 | 0.564(接缝)   调制 5.65×
   ```

   分解接缝（见 `gap200.py`）：

   ```
   总接缝跳变                0.03522
     可由手臂滞后解释的部分   0.02915   与接缝的相关系数仅 +0.057
     残差（模型自身偏移）      0.05721   ← 比总接缝还大
   ```

   相关系数 +0.057 说明**接缝与"手臂落后多少"几乎无关**——不是"追赶实测位置"造成的，
   而是模型每次拿到新观测后给出的首帧本身就偏得远：
   `|action0 − state|` 手臂 **0.0378**，而训练集里 `|action[t] − state[t]|` 只有 **0.0203**，**1.86×**。

   顺带证实 §5.4b 的判断：原始数据的 chunk 内速度剖面是**平的**
   （`1.110 1.069 1.036 0.937 1.058 0.804 0.986`，峰/边 **0.89**），
   §5 里那个 1.99 的钟形**确认是滤波伪影**。

## 9.1 新发现：夹爪观测在部署时超出训练分布

`|action0 − state|` 在夹爪上是 **0.185 rad**，而训练集里这个量是 **0.00012**——**1544×**。
追下去发现是**量纲/量程不匹配**：

| | action 范围 | state 范围 |
|---|---|---|
| 训练集 batch_3 | `[0.0000, 1.0000]` | `[0.0000, 1.0000]`，且 **action ≡ state**（差 0.00012）|
| 真机部署 | `[−0.007, 1.029]` / `[−0.020, 1.058]` | **`[0.007, 1.2095]` / `[0.018, 1.2995]`** |

- 模型**输出**是对的，老老实实待在训练范围 `[0, 1]` 附近。
- 模型**输入**的夹爪 state 在真机上跑到了 **1.21 / 1.30**，**超出训练集上界 1.0**。
  `STATE` 用的是 `MIN_MAX` 归一化，所以这些值归一化后越界，属于**分布外输入**。
- 回归拟合：`action0 ≈ 0.798 × state`（两只夹爪都是 0.797±0.001）。
  `1/0.798 ≈ 1.25`——真机上报的夹爪 state 大约被放大了 1.25×。

**这是一个部署侧的 bug，不是模型问题**，而且很便宜就能修：
把 `marvain_m6_http` 的夹爪 state 换算回训练用的 `[0,1]` 量程即可。
它同时解释了为什么夹爪一直是最脏的通道。

> 另一条需要记住的背景：训练集里夹爪的 `action` 列**就是 observation 的恒等复制**
> （差 0.00012）。按 `CLAUDE.md` 的说法，`marvin-gripper-quadtile` 这类 profile
> "Float32 grippers, no feedback"，末端执行器观测退化成 `command_echo`。
> 所以夹爪的训练目标本身噪声就更大（lag-1 自相关 +0.731，手臂是 +0.973）——
> 拿夹爪当对照通道衡量**输出平滑度**是有效的，但它的绝对目标值不能和手臂比。

## 9.2 三个 checkpoint 的完整对照

全部 16 关节：

| | 10k 原始 | 100k **滤波后** | 200k 原始 | 训练集 |
|---|---|---|---|---|
| 单步 \|Δ\| | 0.01323 | 0.00452 | 0.00801 | 0.00437 |
| lag-1 自相关 | −0.439 | +0.444 | **−0.037** | **+0.943** |
| path/net | 4.68 | 1.00 | 1.19 | 1.00 |
| 反转率 | 60.2% | 7.2% | **26.7%** | 手臂 4.4% |
| 噪声/信号 | 0.32 | 0.05 | 0.07 | — |

夹爪通道（三份都是原始，唯一可比的模型指标）：

| | 10k | 100k | 200k |
|---|---|---|---|
| lag-1 自相关 | −0.495 | **+0.410** | +0.267 |
| 单步 \|Δ\| | 0.00855 | **0.00546** | 0.00831 |
| path/net | 8.62 | **2.97** | 3.38 |
| 反转率 | 70.9% | 24.9% | **24.0%** |

**10k → 100k 是真实的大幅进步；100k → 200k 平台化。**

## 10. 修订后的建议（取代 §7）

1. **修夹爪 state 量程（§9.1）。** 最便宜、最确定的一项：真机夹爪 state 除以 ~1.25
   回到训练的 `[0,1]`，消除分布外输入。改完直接复测，不需要重训。

2. **停止靠加 steps 解决问题。** 88 epochs 已到平台（§9.2）。
   200k 这个 run 可以收了，`loss=0.004` 继续降也不再转化为轨迹质量。

3. **§7.6 作废——结构性修改现在是主线，不是备选。**
   §2.4 的诊断（`SimpleActionDecoder` 平铺读出头在时间轴上无结构 + L1 逐元素损失无平滑项）
   在 88 epochs 之后依然成立：手臂原始自相关 −0.080 vs 训练集 +0.973。
   **训练量已经证明治不好这个残差**，剩下的就是结构和损失：
   - 在重建损失里加二阶差分平滑项 `L += λ·‖Δ²a‖`，λ 用训练集自身的 `‖Δ²a‖` 量级标定
     （改动最小，先试这个）；
   - 或把平铺 `nn.Linear(hidden, horizon×action_dim)` 换成时间卷积头 / 带位置编码的
     per-step transformer 头，把 DP、ACT 自带的时间先验补回来。

4. **接缝（§9 第 3 条）需要单独处理，它不会被 3 顺带解决。**
   接缝与跟踪滞后无关（相关 +0.057），是模型首帧偏移 1.86× 造成的。
   - 先试 `n_action_steps: 8 → 4`（配置即可，会被 clamp，不用重训）：重规划更频繁，
     每次偏移的绝对量更小；
   - 再考虑 RTC 引擎让新旧 chunk 对齐。
   - `chunk_interval_s: 0.25 → 0.2667` 仍然该改（§5.4c）。

5. **滤波栈怎么办**：既然它贡献了几乎全部的表观平滑（§9 第 2 条），
   在 3 落地之前**不要摘掉**——它现在是真机能跑的原因。
   但评估必须一律走 `record_chunk.txt` 的原始输出（08-13 已改，见 §7.1）。

