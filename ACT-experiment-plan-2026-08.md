# ACT 三方向实验方案：#2 相对动作 / #4 相机身份 / #9 视觉 backbone

> **代码基线：`/home/kewei/YING/robot_data_platform/lerobot` @ `22bd7a2f`（lerobot 0.6.2）**，
> 工作树干净、无本地改动。ACT 实现在 `src/lerobot/policies/act/{configuration_act,modeling_act,processor_act}.py`。
> 配套：[`ACT-improvement-proposals-2026.md`](./ACT-improvement-proposals-2026.md)（提案）、
> [`act-camera-embedding.patch`](./act-camera-embedding.patch)（已验证可干净应用，见 §4.1）。
> 撰写日期：2026-08-11。

**前一版方案针对 `ACT_rosbag2`（tonyzhaozh/act 的 DETR 式实现）写成，与本代码库出入很大，本文已整体重写。**
两者的关键差异：

| | ACT_rosbag2（DETR 式） | **lerobot（本方案目标）** |
|---|---|---|
| decoder 层数 | `dec_layers=7`，`transformer(...)[-1]` | **`n_decoder_layers=1`**，循环全部层 + norm，无 `[0]/[-1]` bug |
| 多相机 token | 沿 W 轴 `cat` feature map | 逐相机 flatten 后 `extend` 进 token 列表（`modeling_act.py:476-488`） |
| 归一化 | `utils.py` 手写 `norm_stats` | **processor pipeline**（`NormalizerProcessorStep` + `dataset_stats`） |
| 相对动作 | 无 | **已实现**：`RelativeActionsProcessorStep` / `AbsoluteActionsProcessorStep` + `compute_relative_action_stats` |
| 训练循环 | 无 scheduler / 无 clip / 无 AMP / val 每 1000 epoch | `grad_clip_norm`、`accelerator.autocast`、`lr_scheduler`、**留出 eval split + `eval_steps`**、EMA、FSDP2 |
| 执行步数 | 部署脚本硬编码 3 | 配置项 `n_action_steps`（默认 100），`select_action` 走 action queue |
| temporal ensembling | 手写、YAML 里关着 | `ACTTemporalEnsembler`，要求 `n_action_steps=1` |

结论：**提案 #1（decoder 深度/checkpoint 守卫）在本代码库上不成立**（lerobot 默认 `n_decoder_layers=1`
且没有那个 bug）；**提案 #3 的训练配方大部分已具备**。真正要做的工作量比原提案估计的小得多，
但 **#2 的难点整个转移到了推理端**——见 §2.3。

---

## 0. 三方向在本代码库中的实际状态

| 方向 | 上游状态 | 本方案要做的事 | 工作量 |
|---|---|---|---|
| **#2 相对动作** | **算法已实现且经过测试**（`processor/relative_action_processor.py`、`datasets/compute_stats.py:686`、`tests/policies/test_relative_actions.py`），但**只接进了 pi0 / pi0_fast / pi05 / groot，没有接 ACT** | ① 给 `ACTConfig` 加开关；② 按 OpenPI 顺序重写 `make_act_pre_post_processors`；③ 用 `chunk_size=100` 重算相对空间 stats；④ **解决推理端的 chunk 锚点漂移**（上游已知未解，见 §2.3） | 1–2 天，其中 ④ 占大头 |
| **#4 相机身份** | 无。`ACTSinusoidalPositionEmbedding2d` 输出只依赖 feature map 形状 → 逐相机逐字节相同 | patch 已就绪且可干净应用；再补两个对照 arm | 2 小时 |
| **#9 视觉 backbone** | 只支持 resnet：`configuration_act.py:134` 显式 `raise`；`modeling_act.py:355` 用 `backbone_model.fc.in_features`；单个共享 `self.backbone` | 放开校验、per-camera backbone、DINOv3 接入、token 预算控制 | 2–4 天 |

### 0.1 命名陷阱（必须先说清楚）

lerobot 里 **`delta_timestamps` / `action_delta_indices`（`configuration_act.py:171`）指的是时间偏移**
（「取未来第 0..99 帧的 action」），**不是**相对动作表示。相对动作表示在 lerobot 里一律叫
**`relative`**（`use_relative_actions`）。本方案全程用「相对动作 / relative」，
提案里的「delta」一词在本文只在指代文献时出现。写代码和写论文时都别混。

---

## Phase 0：评估口径（约半天）

lerobot 已经提供了 ACT_rosbag2 缺的绝大部分基础设施，Phase 0 因此从「重建管线」缩小成「补一个指标 + 冻住变量」。

### 0.1 已有的、直接用

| 能力 | 位置 | 用法 |
|---|---|---|
| 确定性留出集 | `datasets/factory.py:151` `make_train_eval_datasets`——**按 task 取最后 `ceil(n·eval_split)` 条 episode**，无随机性 | `--dataset.eval_split=0.2` |
| 周期性 eval | `scripts/lerobot_train.py:748, 774-793` | `--eval_steps=500` |
| 梯度裁剪 | `lerobot_train.py:212-213` | `--optim.grad_clip_norm=1.0` |
| 混合精度 | `lerobot_train.py:180` `accelerator.autocast()` | accelerate 配置 |
| 固定 step 预算 | `cfg.steps` | 所有 arm 用同一个 `--steps` |

### 0.2 仍然缺的一件事：eval 指标不可用于本方案

`lerobot_train.py:777-785` 的 eval 循环调的是 `policy.forward(eval_batch)`，即
`modeling_act.py:139-166` 的**训练路径**。由此产生三个问题：

1. **用的是后验 latent**。`modeling_act.py:401,409` 的条件是 `self.config.use_vae and ACTION in batch and self.training`。
   eval 时 `policy.eval()` 使 `self.training=False` → 走 `latent_sample=zeros` 分支，
   这一点其实是对的（与部署一致）。**但 `mu`/`log_sigma_x2` 因此为 `None`**，
   `forward` 里 `if self.config.use_vae and log_sigma_x2_hat is not None` 为假 →
   `eval_loss` 退化成纯 L1，而 `train_loss` 含 KL。**两条曲线不同量纲，不要放一起看。**
2. **`eval_loss` 在归一化空间**。R0（绝对）与 R1/R2（相对）用**不同的 `dataset_stats`**，
   相对动作的 std 显著更小 → **相对 arm 的 `eval_loss` 天然更小**。
   直接比是错的，这是本方案最容易犯的错误。
3. **是一个标量**，看不到 `MAE(k)` 的形状。ACT 的改进几乎总表现为曲线尾部下压。

### 0.3 补一个部署口径的评估

新增 `tools/eval_act_offline.py`，独立于训练循环运行（对已保存的 checkpoint 求值），
**在物理量、绝对动作空间上**计算：

```python
@torch.no_grad()
def eval_mae(policy, preprocessor, postprocessor, eval_loader, chunk):
    """全部指标在 *反归一化后的绝对动作空间* 上计算，因此跨 R0/R1/R2 可比。"""
    err = torch.zeros(chunk, ACT_DIM); cnt = torch.zeros(chunk, 1)
    for batch in eval_loader:
        gt_abs = batch[ACTION].clone()                    # 数据集里就是绝对量
        proc   = preprocessor(batch)                      # rename→batch→device→relative→normalize
        pred   = policy.predict_action_chunk(proc)        # latent=0，与部署一致
        pred_abs = postprocessor(pred)                    # unnormalize→absolute→cpu
        m = (~batch["action_is_pad"]).float().unsqueeze(-1)
        err += ((pred_abs - gt_abs).abs() * m).sum(0);  cnt += m.sum(0)
    return err / cnt.clamp(min=1)                         # (chunk, action_dim)
```

关键点：**必须走完整的 postprocessor**（`unnormalize → AbsoluteActionsProcessorStep → to_cpu`），
让相对 arm 的输出被还原成绝对量。用 `predict_action_chunk`（`modeling_act.py:127`）而不是
`select_action`，绕开 action queue 和 temporal ensembler——这两者的正确性是 §2.3 单独要查的事。

**指标定义**（关节维与夹爪维分开报）：

| 指标 | 定义 | 用途 |
|---|---|---|
| **`MAE_exec`** | `mean_{k<n_action_steps}` | **主指标**。实际会被执行的那一段 |
| `MAE_head` | `k∈[0,10)` | 短程拟合 |
| `MAE_tail` | `k∈[50,100)` | 长程外推；#2 的机理指标 |
| `JERK` | `mean_k \|â_{k+1} − 2â_k + â_{k−1}\|` | 平滑度 |
| `D_bnd` | `\|â^{(t+E)}_{0} − â^{(t)}_{E}\|`，`E=n_action_steps` | chunk 边界跳变 |
| `LAT` | batch=1 单帧 ms（中位数 / 100 次） | 部署闸门 |

### 0.4 冻住的变量（所有 arm 一致）

| 项 | 取值 | 理由 |
|---|---|---|
| `--steps` | 同一个值（见 §5.2） | 按 step 而非 epoch 计预算 |
| `--dataset.eval_split` | `0.2` | 确定性划分，无需自己冻 split 文件 |
| `--dataset.image_transforms.enable` | **`false`** | 图像增广会与 #9 的扰动鲁棒性指标混淆；也是 #9 特征缓存的前提 |
| `--ema.enable` | **`false`** | 新特性（commit `266be2bd`），会掩盖 arm 间差异 |
| scheduler | 无（`get_scheduler_preset()` 返回 `None`） | 保持 ACT 预设；改它属提案 #3，另开一条线 |
| `--optim.grad_clip_norm` | `1.0` | 相对动作 arm 的 loss 尺度会变，不裁剪引入不可控方差 |
| `--policy.n_action_steps` | 固定（见 §2.4） | 它同时进入 `MAE_exec` 的定义，跨 arm 必须一致 |
| seed | `0/1/2` | 每 arm 3 个 |

**统计**：所有 arm 在同一留出集上评估 → 逐样本误差做配对 bootstrap（10000 次）出 95% CI。
**预注册门槛**：`MAE_exec` 相对改善 **≥ 5% 且 CI 不跨 0**。

---

## Phase 2：相对动作表示（约 9 次训练 + 1–2 天工程）

### 2.0 前置闸门（不训练，15 分钟）

上游的 `compute_relative_action_stats`（`datasets/compute_stats.py:686`）本身就会打印
相对维数与统计量。直接跑一次，拿到逐维 `std(relative)/std(absolute)`：

```bash
python -c "
from lerobot.datasets.compute_stats import compute_relative_action_stats
# chunk_size 必须 = policy.chunk_size = 100（函数默认是 50，见 §2.2 陷阱 ②）
stats = compute_relative_action_stats(hf_dataset=ds.hf_dataset, features=ds.meta.features,
                                      chunk_size=100, exclude_joints=['gripper'], num_workers=8)
"
```

**判据**：关节维 `std` 比值的中位数 **< 0.5** → 机理成立（ICML 2026 报 31–37%）；
**> 0.8** → 机理不成立，只跑 R2，不跑 R1。

### 2.1 实验 arm

| ID | 配置 | 角色 |
|---|---|---|
| **R0** | `use_relative_actions=false` | 基线 |
| **R1** | `use_relative_actions=true, relative_exclude_joints=[]` | 全维相对 |
| **R2** | `use_relative_actions=true, relative_exclude_joints=["gripper"]` | **上游 pi0/pi05 的默认值**（`configuration_pi0.py:55`） |

3 arm × 3 seed = **9 次训练**。

注意 R2 正是我在前一版方案里作为「关键修改」提出的「夹爪不做相对」——
**上游已经把它设成了 pi0 系列的默认值**，这是对该判断的独立支持。
`_build_mask`（`relative_action_processor.py:106-123`）按 `action_names` 做子串匹配，
所以要确认你的数据集 action 维度命名里夹爪维确实含 `gripper` 字样，否则 mask 全 True、
R2 会静默退化成 R1。**这一条必须 assert，不能靠肉眼。**

提案里的「逐步差分」负对照（`a_k − a_{k−1}`）上游**不支持**——
`to_relative_actions` 只做 chunk 锚定（`state` 广播到整个时间维，
`relative_action_processor.py:56`）。要跑它得自己写 ProcessorStep，
**本方案列为可选**，优先级低于 §2.3 的部署问题。

### 2.2 接线改动

**(1) `configuration_act.py`** —— 加三个字段（与 pi0 同名，便于复用工具链）：

```python
use_relative_actions: bool = False
relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
action_feature_names: list[str] | None = None      # 由 dataset meta 填入，供 _build_mask 使用
```

**(2) `processor_act.py`** —— 现在只有一行 `return make_default_pre_post_processors(...)`
（`processor_act.py:50`），要换成 pi0 的显式组合（`processor_pi0.py:127-156`）：

```python
relative_step = RelativeActionsProcessorStep(
    enabled=config.use_relative_actions,
    exclude_joints=getattr(config, "relative_exclude_joints", []),
    action_names=getattr(config, "action_feature_names", None),
)
s = make_default_policy_processor_steps(config, dataset_stats, normalizer_device=config.device)

# OpenPI 顺序：raw → relative → normalize → model → unnormalize → absolute
input_steps  = [s.rename_observations, s.add_batch_dim, s.to_device, relative_step, s.normalize]
output_steps = [s.unnormalize,
                AbsoluteActionsProcessorStep(enabled=config.use_relative_actions,
                                             relative_step=relative_step),
                s.to_cpu]
return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)
```

**顺序不能错**：`relative` 必须在 `normalize` **之前**、`absolute` 必须在 `unnormalize` **之后**。
放反了就是拿归一化后的 action 减去未归一化的 state，量纲直接错。

**(3) 重算 stats。** `NormalizerProcessorStep` 用的是 `dataset_stats`，而它默认是在**绝对**
action 上算的。必须用 `dataset_tools`（`datasets/dataset_tools.py:1600` 起）重算：

```python
recompute_stats(dataset, relative_action=True,
                relative_exclude_joints=["gripper"],   # 必须与 policy config 一致
                chunk_size=100,                        # 必须 = ACTConfig.chunk_size
                num_workers=8)
```

**三个静默失效的陷阱：**

① **`relative_exclude_joints` 在 stats 与 policy config 之间不一致** → 归一化用的是
   「夹爪相对」的统计量、模型学的是「夹爪绝对」的目标（或反之）。两边都不报错。
② **`chunk_size` 默认是 50**（`dataset_tools.py:1592`），而 ACT 默认 `chunk_size=100`。
   不显式传就会用 50 帧窗口的相对分布去归一化 100 帧的目标——尾部被系统性低估。
③ **忘了重算** → 用绝对 stats 归一化相对 action，std 大 3 倍，loss 尺度塌掉。

三条都必须在训练启动时 assert，不能靠流程纪律：

```python
assert stats_meta["relative_action"] == config.use_relative_actions
assert stats_meta["relative_exclude_joints"] == config.relative_exclude_joints
assert stats_meta["chunk_size"] == config.chunk_size
```

### 2.3 推理端阻塞项（本 Phase 的真正难点）

上游**明确知道**相对动作在逐帧推理下是坏的，并且选择了直接拒绝而不是修复。
`rollout/context.py:477-483`：

```python
if isinstance(cfg.inference, SyncInferenceConfig) and any(
    isinstance(step, RelativeActionsProcessorStep) and step.enabled for step in preprocessor.steps):
    raise NotImplementedError(
        "SyncInferenceEngine does not support policies with relative actions for now."
        "Use --inference.type=rtc or remove relative action processor steps ...")
```

原因写在 `rollout/inference/sync.py:20-30`：

> 每一帧都会刷新 `RelativeActionsProcessorStep._last_state`，于是**后续帧从队列里弹出的
> 缓存动作，会被重新锚定到当前机器人状态上，绝对目标在整个 chunk 内漂移。**

这对 ACT 是致命的：`select_action`（`modeling_act.py:102-125`）正是「一次预测、
分 `n_action_steps` 帧弹出」的 queue 结构。**所以 `use_relative_actions=true` +
`n_action_steps>1` 在当前上游代码上直接不可部署。**

**同一个根因还污染 temporal ensembling**：`ACTTemporalEnsembler.update`
（`modeling_act.py:223-257`）在 `select_action` 内部、对**模型原始输出**（相对空间）做加权平均，
而这些 chunk 锚定在不同的 `state` 上——正是提案 §2 第 4 点警告的偏置。

**三个可选方案：**

| 方案 | 做法 | 代价 | 评价 |
|---|---|---|---|
| **P-a（推荐）** | 采用 `sync.py:26-30` 里写的候选修法：用 `predict_action_chunk` 取整个 chunk，**一次性过 postprocessor**，再把还原成绝对量的动作放进本地 FIFO 逐帧弹出 | 需要绕开 `select_action`；ACT 的 temporal ensembler 得跟着搬到绝对空间 | 从构造上消除漂移，还省掉每帧的 pre/post 开销 |
| **P-b** | `n_action_steps=1` | 闭环频率 = 推理频率，`T_replan` 压力最大 | 只适合当对照，不适合部署 |
| **P-c** | 训练用相对、**部署前把模型转成绝对**（把 state 偏置折进 action head 的 bias） | 只在 mask 全 True 且 state/action 维度对齐时成立 | 数学上可行但脆弱，不推荐 |

**本方案采用 P-a**，并把它作为 Phase 2 的**交付物之一**：

```python
class ChunkFIFOEngine:
    """相对动作安全的推理引擎：整块 postprocess，再逐帧弹出。"""
    def get_action(self, obs):
        if not self._fifo:
            proc  = self._preprocessor(obs)              # 刷新 _last_state = 本次观测的 state
            chunk = self._policy.predict_action_chunk(proc)[:, : self._n_action_steps]
            abs_chunk = self._postprocessor(chunk)       # 用*同一个* _last_state 还原整块
            self._fifo.extend(abs_chunk.transpose(0, 1))
        return self._fifo.popleft()
```

关键不变式：**`_last_state` 在一个 chunk 的生命周期内只能被写一次**。
写一个回归测试断言它：构造两帧 state 差异很大的观测，断言
`ChunkFIFOEngine` 弹出的绝对动作与「一次性 postprocess 整块」逐元素相等。
`tests/policies/test_relative_actions.py` 里已有的用例可以作模板。

**temporal ensembling**：本方案 Phase 2 全程 `temporal_ensemble_coeff=None`（上游默认），
把「相对空间下的 ensembling 偏置」列为已知问题、不在本 Phase 解决——
它属于提案 #5（BID/TAS）那条线。

### 2.4 `n_action_steps` 的处理

`MAE_exec` 的定义依赖 `n_action_steps`，而它同时是 P-a 里 FIFO 的长度。
**Phase 2 固定 `n_action_steps=100`（上游默认，全开环）**，先把动作表示这一个变量隔离干净。
`n_action_steps` 的扫描（100 → 50 → 25，提案 #7.1，纯推理端、零训练成本）
在 Phase 2 结束后对**胜出的 checkpoint** 单独做——它不需要重训，
不该占用训练预算，也不该和动作表示混在一个矩阵里。

### 2.5 判据（预注册）

| 观察 | 结论 |
|---|---|
| `MAE_exec(R1 或 R2)` 比 R0 改善 ≥ 5%，CI 不跨 0 | **通过** |
| `MAE_exec` 无差异但 `MAE_tail` 改善 ≥ 15% | **部分通过**：收益全在长程 → 记录，并作为 `n_action_steps` 扫描的论据 |
| `JERK` / `D_bnd` 改善 ≥ 10% | 独立通过（平滑度收益，文献一致） |
| `R2 > R1` | 采纳 R2，并印证 pi0 系列把 `["gripper"]` 设为默认的选择 |
| 前置闸门比值 > 0.8 且各指标无差异 | **判负**，止损 |

---

## Phase 4：相机身份嵌入（约 9 次训练）

### 4.0 前置验证（不训练，30 分钟）

`ACTSinusoidalPositionEmbedding2d.forward`（`modeling_act.py:707-739`）的输出
**只由 `x` 的形状决定**（`not_mask = torch.ones_like(x[0, :1])`），且 `y_range`/`x_range`
各自 rescale 到 `[0, 2π]`。ACT 要求所有相机图像同形状 → **三份位置编码逐字节相同**。

```python
# tools/probe_campos.py
pe = ACTSinusoidalPositionEmbedding2d(256)
f0 = model.backbone(img0)["feature_map"]; f1 = model.backbone(img1)["feature_map"]
assert torch.equal(pe(f0), pe(f1))          # 预期 True —— 缺陷本身
```

再用现有 checkpoint 做**视角交换测试**：交换两个 wrist 相机的图像，测输出变化量。
若 `S_swap` 比 `MAE_exec` 小一个量级 → 模型确实视角盲，#4 有明确空间。

### 4.1 patch 状态

`policy/act-camera-embedding.patch` **在 `22bd7a2f` 上可干净应用**（`git apply --check` 通过，无 fuzz）。
它做的是：`ACTConfig.use_camera_embedding`（默认 `False`，保证旧 checkpoint 照常加载）＋
`encoder_cam_id_embed = nn.Embedding(n_cams, dim_model)`，在
`for cam_index, img in enumerate(batch[OBS_IMAGES])` 里加到 **`cam_pos_embed`** 上，
初始化 `std=0.02`。

这份 patch 就是下面的 **C1**。

### 4.2 实验 arm

固定 `use_relative_actions=false`（与 Phase 2 解耦，两个 Phase 可并行跑）。

| ID | 做法 | 参数量 | 机理 |
|---|---|---|---|
| **C0** | 无（= R0） | — | 基线，复用 Phase 2 权重，不重跑 |
| **C1** | 现有 patch：embedding 加到 **`cam_pos_embed`** | +1.5K | 只进 q/k → **只改变注意力路由** |
| **C2** | embedding 加到 **`cam_features`**（`encoder_img_feat_input_proj` 之后） | +1.5K | 进 value 与残差流 → **改变 token 内容** |
| **C3** | **per-camera `encoder_img_feat_input_proj`**（`nn.ModuleList`） | +0.79M | 身份隐含在投影权重里 |

3 个新 arm × 3 seed = **9 次训练**。

**C1/C2 必须分开的依据**（`modeling_act.py:558-559`）：

```python
q = k = x if pos_embed is None else x + pos_embed
x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
```

`pos_embed` **不进 `value`**。decoder 的 cross-attention 同理
（`modeling_act.py:651-654`：`key=...+encoder_pos_embed, value=encoder_out`）。
所以 C1 注入的视角身份只能改变「谁看谁」，改变不了「看到的内容里带不带视角标签」。
提案 §4 写的是「加到对应 token 块上」，而 patch 加的是 `pos`——**这两件事不同**，
用 C1 vs C2 把歧义消掉。

C2 的实现（在 `modeling_act.py:479` 之后）：

```python
cam_features = self.encoder_img_feat_input_proj(cam_features)
if self.config.use_camera_embedding and self.config.camera_embedding_target == "src":
    cam_id = torch.tensor([cam_index], device=cam_features.device)
    cam_features = cam_features + self.encoder_cam_id_embed(cam_id)[:, :, None, None].to(cam_features.dtype)
```

**初始化诊断**：patch 用 `std=0.02`（理由正确——sine PE 元素量级 O(1)，
`nn.Embedding` 默认 `N(0,1)` 会一上来压过空间信号）。若 C1/C2 都无效，
**先看训练结束时 `encoder_cam_id_embed.weight.norm()`**：
若基本没离开初值，说明梯度不推它（= 模型不需要视角身份，真判负）；
若离开了却没收益，才是「用了但没用处」。必要时补一次 `std=0.5` 的单 seed 诊断。

### 4.3 T2 机理指标（#4 的**主判据**）

MAE 上的差异大概率落在噪声内，用它当主判据必然得到假阴性。主判据用机理指标：

| 指标 | 计算 | 预期方向 |
|---|---|---|
| **`ΔMAE_swap`** | 交换两个 wrist 图像后 `MAE_exec` 的**恶化幅度** | C0 几乎不恶化（视角盲）→ 修复后**大幅恶化** |
| `S_swap` | 交换前后输出差 `mean\|â−â_swap\|`（rad） | C0 小 → C1/C2/C3 变大 |
| `P_view` | 从 encoder 输出的视觉 token 训 n 类 logistic 回归（视角分类），held-out acc | C0 > 1/n（内容本身有区分度）→ 修复后趋近 1 |
| `ΔMAE_drop(c)` | 把相机 c 置零后的 `MAE_exec` 恶化 | 给出逐视角重要性排序 |

`ΔMAE_swap` 是这里最锋利的判据：**符号明确、不依赖效应量大小**。
一个视角盲的模型，交换左右腕图像后性能不该变；它变了，就证明视角身份进了决策。

### 4.4 判据（预注册）

| 观察 | 结论 |
|---|---|
| `ΔMAE_swap` 从 C0 的 < 5% 涨到 ≥ 30% | **机理修复成功**（无论 MAE 是否改善） |
| 同时 `MAE_exec` 改善 ≥ 5% | **完全通过**，采用 |
| 机理修复成功但 MAE 无差异 | **有条件通过**：代价仅 1.5K 参数，保留；但不作性能卖点。它解锁了相机 dropout 增广 |
| C2 > C1 | 「视角身份需要进 value 通路」——对提案 #4 与现有 patch 的实质修正 |
| C3 ≫ C1/C2 | 瓶颈是**共享投影**而非位置编码 → 直接指向 Phase 9 的 D1 |
| 全部无差异且 `weight.norm()` 未离开初值 | **判负** |

---

## Phase 9：视觉 backbone（约 12 次训练）

### 9.0 先算 token 预算

当前：resnet18 stride 32。设图像 `H×W`，则每相机 `(H/32)×(W/32)` 个 token。
以 480×640 为例 = **300 token/相机**，3 相机 900，加 1D token 共 903
（`modeling_act.py:463-492`：latent + state + 图像 token）。

直接换 **ViT/16** 则每相机 `(H/16)×(W/16)` = **1200**，3 相机 3600 token。
encoder 单层代价（`dim_model=512`、`dim_feedforward=3200`）：

| 项 | 900 token | 3600 token | 倍数 |
|---|---|---|---|
| 投影 | 1.89 GFLOP | 7.5 | 4× |
| 注意力 | 1.66 | 26.5 | **16×** |
| FFN | 5.9 | 23.6 | 4× |
| **单层** | **9.47** | **57.6** | **6.1×** |
| ×`n_encoder_layers=4` | 37.9 | 230 | |

再叠上 ViT backbone 本身（ViT-S/16 @1200 token × 3 相机 ≈ 232 GFLOP，
而 resnet18 三相机合计仅 33 GFLOP），总量从 ~71 涨到 ~460 GFLOP。
**所以 token 数必须锁死在 300/相机**，这是 #9 的硬约束，不是优化项。

| 路线 | 做法 | token/相机 | backbone FLOP 相对 resnet18 | 形状兼容 |
|---|---|---|---|---|
| **① DINOv3-ConvNeXt** | 原生 stride 32（stem 4 + 3×2 下采样） | **300** | ~2.5×（Tiny） | **精确 drop-in**，只需把投影改成 `Conv2d(768, 512, 1)` |
| ② DINOv3-ViT-S/16 @ 半分辨率 | 输入降采样一半 | **300** | ~1.3× | 兼容，损失一半分辨率 |
| ③ DINOv3-ViT-S/16 @ 全分辨率 + 2×2 池化 | 30×40 → 15×20 | **300** | ~7× | 兼容、保分辨率，backbone 最贵 |

**路线① 是本次最实用的结论**：DINOv3 的 ConvNeXt 变体输出 stride 与 resnet18 `layer4` 完全一致，
下游 token 结构和全部已实测的 encoder/decoder 成本**保持不变**，
实验变量被干净地隔离成「特征提取器」这一个。

本地权重齐全（`/home/kewei/YING/dinov3/dino_ckpt/`）：
`dinov3_convnext_{tiny,small,base,large}_pretrain_lvd1689m-*.pth`、
`dinov3_vit{s16,s16plus,b16,l16}_pretrain_lvd1689m-*.pth`。
（`vit7b16` / `vith16plus` 量级完全不匹配，不考虑。）
DINOv3 用 ImageNet mean/std，与 `NormalizationMode.MEAN_STD` 的 VISUAL 归一化路径兼容。

### 9.1 前置闸门：冻结特征岭回归（不训 transformer，约 1 小时）

在训任何 arm 之前回答：**DINOv3 特征真的比 ImageNet-resnet18 携带更多任务相关信息吗？**
对留出集抽三种冻结特征（全局平均池化）+ state，闭式岭回归到未来 100×`action_dim`，比 val MAE。

**判据**：DINOv3 不优于 resnet18 → 只跑 D1，跳过 D2–D4，省 9 次训练。
这是**单向判据**——全局池化会低估 ViT 的稠密 patch 优势
（正是 `patch-policy-2607.18236.md` 批评的对象），输了只降优先级，不完全否定。

### 9.2 实验 arm

所有 arm 固定携带 Phase 4 胜出配置 `C*`（去混淆的必要条件）与 Phase 2 胜出配置 `R*`。

| ID | backbone | 可训练？ | 角色 |
|---|---|---|---|
| **D0** | resnet18 ×1 共享 | 是 | 基线（= Phase 3 结果，不重跑） |
| **D1** | resnet18 ×n **独立** | 是 | **必需对照**：隔离「不再共享权重」 |
| **D2** | DINOv3-ConvNeXt-T 共享，**冻结** | 否 | 冻结大模型特征，全视角 |
| **D3** | 外部视角 = DINOv3 冻结；wrist = resnet18 可训练 | 混合 | **StereoPolicy 式分视角决策** |
| **D4** | DINOv3-ViT-S/16 冻结，全分辨率 + 2×2 池化 | 否 | ViT vs ConvNeXt |

4 个新 arm × 3 seed = **12 次训练**。

**D1 为什么必需**：D3 用了多套不同 backbone，这本身就赋予模型区分视角的能力。
没有 D1，D3 的收益无法归因到「DINOv3 特征」还是「backbone 不共享」。

### 9.3 代码改动

**(1) 放开 backbone 校验**（`configuration_act.py:134-137`）：

```python
if not self.vision_backbone.startswith("resnet"):
    raise ValueError(...)
```
改成允许 `dinov3_*` 前缀，并对每种前缀走不同的构建分支。

**(2) 投影层的输入通道数**（`modeling_act.py:354-356`）当前是
`backbone_model.fc.in_features` —— resnet 专有属性。改为由 backbone 包装类导出
`num_channels`（ConvNeXt-T = 768、ViT-S = 384、resnet18 = 512）。

**(3) 单 backbone → `nn.ModuleList`**（`modeling_act.py:336`）。
**关键约束**：`get_optim_params`（`modeling_act.py:74-93`）按名字前缀
`n.startswith("model.backbone")` 分配 `optimizer_lr_backbone`，
所以 per-camera backbone 的属性名**必须仍以 `backbone` 开头**（如 `self.backbones`
会失配——用 `self.backbone = nn.ModuleList([...])` 保持前缀）。**这一条错了不会报错，
只会让 backbone 用错学习率。**

**(4) 冻结的三个细节**：
- `requires_grad_(False)` 后，`get_optim_params` 的 `p.requires_grad` 过滤会自动排除它们——
  但要 assert 一次，否则 AdamW 会给冻结参数建 state 白吃显存。
- resnet 路径靠 `FrozenBatchNorm2d`（`modeling_act.py:331`）保证 BN 统计量不漂移；
  **换 backbone 后这层保护就没了**。ConvNeXt 用 LayerNorm（无 running stats）相对安全，
  但仍建议覆写 `train()` 强制冻结子模块保持 `eval()`（dropout 同理）。
- 冻结省下反传 → 显存和步时长下降。**但本方案统一保持 batch size 不变**以维持可比性；
  「用省下的预算换 batch」记为后续独立实验。

**(5) 特征缓存**：Phase 0 已强制 `image_transforms.enable=false`，所以冻结特征对固定
`(episode, frame)` 是确定的，理论上可缓存。但算盘：`15×20×768` fp16 = 460 KB/相机 →
多相机每帧 >1 MB → 十万帧量级即上百 GB。**全量缓存不划算，不做**；
只对留出集缓存（约 1 GB),加速 Phase 0 的周期性评估。

### 9.4 T2 泛化指标（#9 的**主判据**）

backbone 的文献主张全是**泛化**主张，不是拟合主张。同分布留出集上大概率测不出差异
（冻结 backbone 甚至可能略差），主判据必须是扰动/迁移：

| 扰动 | 幅度 |
|---|---|
| 亮度 | ±20% / ±40% |
| 对比度 | 0.8× / 1.25× |
| 高斯噪声 | σ=5 / 15（8-bit 尺度） |
| 平移 | ±8 / ±16 px（模拟相机轻微位移） |
| 模糊 | 高斯 r=1 / 2 |
| **不同光照/时段/摆放的 session** | 若数据支持则**务必做**，价值最高 |

统一报**退化斜率** `(MAE_perturbed − MAE_clean) / MAE_clean`。
DINOv3 的主张若成立，应表现为**斜率显著更平**，即使 clean MAE 相当甚至略差。

注意：这些扰动必须在评估脚本里手工施加，**不要**打开
`cfg.dataset.image_transforms`——那是训练侧增广，打开会同时改变训练分布。

同时**逐视角**报 `ΔMAE_drop(c)`：StereoPolicy 的核心结论是「外部视角受益、wrist 视角反而变差」，
这个结论只有逐视角报才检验得了。D2（全视角 DINOv3）vs D3（仅外部视角用 DINOv3）
的差就是对它的直接复现检验。

### 9.5 判据（预注册）

| 观察 | 结论 |
|---|---|
| `LAT` 超出部署预算 | **直接淘汰**，无论 MAE 多好 |
| D1 ≈ D0 | 共享 backbone 不是瓶颈 → D3 的收益可归因到 DINOv3 特征 |
| D1 > D0 显著 | **共享 backbone 本身就是瓶颈** → 比 DINOv3 便宜得多的收益，优先采用 |
| D3 clean MAE ≈ D0 但扰动斜率低 30%+ | **通过**（泛化收益，符合预期形态）→ 进 T3 |
| D2 < D3 | **复现了 StereoPolicy 的分视角结论**，采用 D3 |
| D2 ≈ D3 | wrist 域不匹配在本任务不存在 → 用更简单的 D2 |
| 全部扰动斜率无差异 | **判负**：瓶颈不是视觉泛化 → 预算转向提案 #6/#7 |

---

## 5. 矩阵、顺序与预算

### 5.1 阶段

```
Phase 0  评估口径 + 变量冻结                              0 runs（纯工程）
   │
   ├── Phase 2  动作表示  R0/R1/R2      ×3 seed   9 runs  ┐
   │            + P-a 推理引擎（工程）                     ├─ 可并行
   └── Phase 4  相机身份  C1/C2/C3      ×3 seed   9 runs  ┘（C0 复用 R0）
                     │
                     ▼
       Phase 3  交互检验  R* × C*       ×3 seed   3 runs
                     │
                     ▼
       Phase 9  backbone D1/D2/D3/D4    ×3 seed  12 runs  (D0 = Phase 3 结果)
                     │
                     ▼
       推理端扫描  n_action_steps ∈ {100,50,25}  —— 零训练成本，只对胜出 ckpt
                     │
                     ▼
       T3 实机     top-2 arm × 20 rollout
```

**合计 33 次训练。** Phase 2 与 Phase 4 故意解耦（各自在对方的基线配置下筛选），
好处是可以并行、且各自的结论直接可比现有基线；代价是需要 Phase 3 的交互检验：
跑 `R*+C*`，与**可加性预测值**比较。若实测显著差于预测，说明两者收益重叠——
这很可能发生，因为相对表示降低了目标方差，也就降低了模型对视角消歧的依赖。

### 5.2 预算

按 **step** 而非 epoch 计：所有 arm 用同一个 `--steps`。ACT 在 lerobot 上的常用量级是
`--steps=100000`、`--batch_size=8`。实测参考（RTX 4090、3 相机 480×640、
`n_encoder_layers=4`、`n_decoder_layers=1`）：resnet18 arm 约 100 ms/step
（比 ACT_rosbag2 实测的 135 ms/step 快，因为 decoder 从 7 层降到 1 层）。

| Phase | runs | 相对步时长 | 小计（`steps=100k`） |
|---|---|---|---|
| 2 | 9 | 1.0× | ~25 GPU·h |
| 4 | 9 | 1.0× | ~25 GPU·h |
| 3 | 3 | 1.0× | ~8 GPU·h |
| 9（D1） | 3 | ~1.6× | ~13 GPU·h |
| 9（D2/D3） | 6 | ~1.3× | ~22 GPU·h |
| 9（D4） | 3 | ~2.5× | ~21 GPU·h |
| **合计** | **33** | | **≈ 114 GPU·h** |

开 `accelerator.autocast`（预期 1.5–1.8×）后 **≈ 65–75 GPU·h**。

**本平台是 Slurm 集群**（`robot_data_platform`：mgmt01 + gpu01），
所以这些 run 应该作为 **job array** 提交，而不是串行跑。
`--steps` 与 seed 作为数组参数，`--output_dir` 按 `arm_id` 分目录。
注意 `lerobot_train.py` 支持 FSDP2/HSDP 与梯度累积——**本方案一律用单卡单进程**
（`gradient_accumulation.steps=1`、不启用 FSDP），因为并行训练会改变有效 batch size
与数值路径，破坏跨 arm 可比性。**计时类指标（`LAT`、步时长）必须在独占卡上单独测。**

`--steps` 的具体取值取决于数据集规模（episode 数），启动前用一条基线 run 确认
`eval_loss` 已经平台化，再把该值固定给所有 arm。

### 5.3 日志 schema（`results/arms.jsonl`，每 arm 一行）

```json
{"arm_id":"R2_C2_D3_s0", "phase":9, "seed":0, "lerobot_commit":"22bd7a2f",
 "use_relative_actions":true, "relative_exclude_joints":["gripper"],
 "stats_relative":true, "stats_chunk_size":100,
 "use_camera_embedding":true, "camera_embedding_target":"src", "per_cam_input_proj":true,
 "vision_backbone":"per_cam:dinov3_convnext_tiny:frozen,resnet18,resnet18",
 "steps":100000, "n_action_steps":100,
 "MAE_exec":0.0123, "MAE_head":0.0141, "MAE_tail":0.0388, "MAE_grip":0.021,
 "JERK":1.9e-4, "D_bnd":0.0031,
 "S_swap":0.0142, "dMAE_swap":0.41, "P_view":0.98,
 "dMAE_drop":{"cam0":0.31,"cam1":0.22,"cam2":0.20},
 "slope":{"bright40":0.18,"noise15":0.27,"shift16":0.33,"blur2":0.14},
 "LAT_ms":16.4, "TRAIN_ms":128.0, "VRAM_GB":9.1, "params_M":41.2}
```

---

## 6. 风险与已知陷阱

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| **相对动作的 chunk 锚点漂移** | `use_relative_actions=true` + `n_action_steps>1`，上游 `SyncInferenceEngine` 直接 `raise`（`rollout/context.py:482`） | §2.3 的 P-a 引擎 + 「`_last_state` 每 chunk 只写一次」的回归测试 |
| **stats 与 config 不一致** | `relative_exclude_joints` / `chunk_size` / `relative_action` 三者任一失配 | §2.2 的三条 assert，写进训练启动路径 |
| **`_build_mask` 静默全 True** | 数据集 action 维度命名里没有 `gripper` 字样 | 启动时打印 mask 并 assert `sum(mask) == expected` |
| **归一化空间的 eval_loss 跨 arm 比较** | 相对 arm 的 std 更小 → 数字天然更好看 | 只用 §0.3 的物理量绝对空间指标；不看 `eval_loss` 的跨 arm 对比 |
| **train_loss 含 KL、eval_loss 不含** | `modeling_act.py:153` 的 `log_sigma_x2_hat is not None` 在 eval 下为假 | 两条曲线不同量纲，别放一张图 |
| **per-camera backbone 用错学习率** | 属性名不再以 `backbone` 开头 → `get_optim_params:82,89` 失配 | 保持 `self.backbone` 名称；assert 两个 param group 的参数量 |
| **冻结 backbone 的 norm/dropout 在 `train()` 下漂移** | 换掉 resnet 就失去 `FrozenBatchNorm2d` 保护 | 覆写 `train()`，强制冻结子模块 `eval()` |
| **EMA / image_transforms / FSDP 引入的组间差异** | 新特性默认值随版本变动 | Phase 0 §0.4 全部显式冻死，并把实际生效值写进 `arms.jsonl` |
| **`delta_timestamps` 与「delta 动作」混淆** | lerobot 里 `action_delta_indices` 指时间偏移 | 全程用 `relative` 一词；见 §0.1 |
| **离线 MAE 好但实机差** | MAE 与成功率不是一回事 | T1/T2 只用于筛候选，T3（实机 rollout）是最终判据 |

---

## 7. 交付物清单

**代码（全部相对 `22bd7a2f` 的 patch 形式维护，与现有 `act-camera-embedding.patch` 同一套路）**
- `configuration_act.py`：`use_relative_actions` / `relative_exclude_joints` / `action_feature_names` / `use_camera_embedding` / `camera_embedding_target` / `per_cam_input_proj` / 放开 backbone 校验
- `processor_act.py`：按 OpenPI 顺序显式组合 pre/post pipeline
- `modeling_act.py`：相机身份 embedding（pos / src 两路）、per-camera 投影、per-camera backbone、DINOv3 包装类
- `rollout/inference/chunk_fifo.py`：§2.3 的 P-a 引擎；`rollout/context.py` 放行相对动作策略
- `tools/eval_act_offline.py`：§0.3 的全部 T1/T2 指标
- `tools/probe_campos.py`、`tools/gate_backbone_probe.py`（前置闸门）
- `tests/`：`_last_state` 每 chunk 只写一次；stats/config 一致性；mask 非全 True

**数据**
- `results/arms.jsonl`（33 行）+ 每 arm 的训练日志
- `results/mae_curves.png`：`MAE(k)`，k=0..99，按 arm 分组，标注 `n_action_steps` 处的执行边界

**结论文档**
- 每个 Phase 一节：前置闸门结果 → arm 表 → 判据命中 → 采纳/否决
- 最终配置表：三个开关各取什么值，及其代价（参数量、`LAT`、步时长）
