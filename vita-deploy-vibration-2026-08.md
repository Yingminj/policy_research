# VITA 真机部署实验记录与 chunk 抖动诊断

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
