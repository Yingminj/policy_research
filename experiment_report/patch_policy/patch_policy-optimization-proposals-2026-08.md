# `patch_policy` 训练参数与模型结构的 10 条优化建议（文献对照版）

**写作日期** 2026-08-31
**对象** `lerobot/policies/patch_policy/`（`configuration_patch_policy.py` 390 行 / `modeling_patch_policy.py` 621 行 / `processor_patch_policy.py` 87 行）
**证据来源（本地实测）**
`patch_policy-head-comparison-2026-08.md`、`patch_policy-state-and-window-2026-08.md`、`patch_policy-no-proprioception-2026-08.md`
**证据来源（文献）** 附录 A，15 篇，全部核对过 arXiv 条目（核对日期 2026-08-31）
**这份文档不是什么** 不是新的实测报告。所有"实测"数字都引自上面三份前置报告；所有"预期"都是估计，标注了依据强度。

---

## 0. 先说清楚这 10 条要解决的是哪一个问题

三份前置报告收敛到同一句话：

> **误差的主体不是"轨迹外推不准"，而是"起点锚错了"。**
> chunk 首帧与实测位姿的偏差 **0.040–0.043 rad**，而示教动作与实测位姿只差 **0.0140 rad**——偏了 **2.8–3.0 倍**。
> 部署侧的 Hermite 桥接（把起点强行拉回实测位姿）能一口气救回 **26–30%**，这就是锚定误差的直接测量。
> 而一个 8 帧 waypoint 需要表达的全部运动量只有 **0.0386 rad**——**误差比信号还大**。

并且这个缺陷**已经被证明不是"没接本体感觉"造成的**：`use_robot_state=true` 之后，state token 拿到 8.65 倍于均分份额的 cross-attention 质量、范数 0.76（正常量级），但换掉它输出只动 6.7%，锚定偏差纹丝不动（2.97×）。

**所以下面 10 条按"是否直接打这个靶心"排序，而不是按新颖度排序。**

| # | 建议 | 类别 | 打靶心？ | 需重训 | 预期收益 | 证据强度 |
|---|---|---|---|---|---|---|
| 1 | 训练目标换成**相对动作** | 目标 | ✅ 直接 | 是 | **15–25%**（有实测代理） | 强（本地 + 文献） |
| 2 | 本体感觉改走 **adaLN/FiLM 全局条件** | 结构 | ✅ 直接 | 是 | 大（见 §2） | 强（文献有对照表） |
| 3 | `n_obs_steps` 5 → **1 或 2** | 参数 | ❌ 省算力 | 是 | 精度 ±0，**延迟 −50%** | 强（本地实测） |
| 4 | **LR warmup + cosine 衰减** | 参数 | ⚠️ 间接 | 是 | 3–8% | 中（惯例 + 同仓库对照） |
| 5 | 加 **EMA 权重** | 参数 | ⚠️ 间接 | 是 | 3–8%，方差显著下降 | 中（文献 + 代码注释自陈） |
| 6 | 打开**图像增强** | 参数 | ⚠️ 间接 | 是 | 抗过拟合 | 中 |
| 7 | ACT head **decoder 加深** 1 → 4 | 结构 | ❌ 补脆性 | 是 | 鲁棒性大幅改善 | 中（本地实测 + 上游注释） |
| 8 | 视觉侧：**提分辨率 / 换 DINOv3 / 末层解冻** | 结构 | ⚠️ 间接 | 是 | 5–15% | 中 |
| 9 | 扩散头**去噪步数 100 → 10**（DDIM/flow/蒸馏） | 参数 | ❌ 省延迟 | 否 | 延迟 **−90%** | 强（文献） |
| 10 | 部署侧：**放大执行窗口 + 实时分块** | 部署 | ✅ 直接 | 否 | 见 §10 | 强（本地实测 + 文献） |

> **第 1 和第 2 条是同一件事的两半**：一个改输出空间，一个改输入通路。**建议一起上，但分两次训**（原因见 §11）。

---

## 1. 把训练目标换成相对动作（`action − state`）

### 现状
`normalization_mapping` 里 `ACTION = MIN_MAX`，模型回归的是**绝对关节角**。在示教数据里 `action_t ≈ state_t`（差 0.0140 rad），也就是说模型要学的主要是一个**恒等映射**，而这个恒等映射必须穿过
`state_projector(MLP) → 加性进 memory → cross-attention → 去噪器/decoder` 这条长链路才能实现。梯度信号太弱，从来没被学出来。

模型实际学到的是"看画面像任务的哪个阶段，就输出那个阶段的平均位姿"——在 loss 上是个好解（363 episode、单场景，平均轨迹信息量本来就大），作为控制指令则封顶在 0.066 rad。

### 实测代理（这条建议的收益下界）
| 测量 | 数值 | 出处 |
|---|---:|---|
| 显式 re-anchor（`state_t + (pred − pred[0])`），`act` 头 | **−18.8%** | head-comparison §5.2 |
| 部署侧 Hermite 桥接（同一件事的部署实现），horizon 8 | **−26 ~ −30%** | state-and-window §6 |
| `act_delta` 相对化的 pre-gate 中位数（`batch_success_361`, chunk 50） | **0.529**（逐维 0.376–0.722，J7 收缩最多） | state-and-window §7.3 |

注意最后一行：**§4.3 里误差最大的 J7 腕转，正是相对化收缩最多的维度**。这不是巧合——J7 的绝对角在不同 episode 间分布最宽，相对化把这块方差直接删掉。

### 文献依据
- 动作空间的选择本身就是一个一等超参数，而不是实现细节：[arXiv:2312.03673] 系统比较了关节空间 / 任务空间 / 绝对 vs 增量表示对操作学习与 sim-to-real 迁移的影响。
- π₀ / π₀.₅ 把动作按**每维 1%/99% 分位数**映射到 [−1,1]，而不是 min-max，理由是对离群值鲁棒（[arXiv:2410.24164]、[arXiv:2504.16054]）。你们现在用的 `MIN_MAX` 会被单个异常 episode 的极值把整个尺度压扁。
- 相对表示让 chunk 与机器人绝对起始位姿解耦，这正是当前误差的来源。

### 具体改动
树里已有成熟实现，**不需要新写算法**：
```
lerobot/policies/act_delta/
  ├── configuration_act_delta.py   use_relative_actions / relative_exclude_joints
  ├── prepare_relative_stats.py    先算相对动作的统计量
  └── processor_act_delta.py       "raw → relative → normalize" 的正确顺序
```
移植到 `patch_policy` 需要动三处，都很小：
1. `configuration_patch_policy.py`：加 `use_relative_actions: bool = False`、`relative_exclude_joints: list[str] = ["gripper"]`（夹爪维的 pre-gate ≈0.99，相对化无意义）。
2. `processor_patch_policy.py`：在 `NormalizerProcessorStep` **之前**插入相对化步骤，`UnnormalizerProcessorStep` **之后**插入反相对化。顺序错了会静默出错。
3. `modeling_patch_policy.py:480 predict()`：chunk 的锚点是 `n_obs_steps - 1` 那一帧的 state，不是第 0 帧。这一点 `act_delta` 的 `allow_unsafe_relative_select_action` 注释里有踩过的坑，照抄。

**顺带一条（同一主题，成本 0）**：把 `ACTION` 的归一化从 `MIN_MAX` 换成分位数归一化或 `MEAN_STD`。

### 验证
用 `scripts_patch_policy_eval_fix/offline_chunk_eval.py` 直接复测，重点看两个数：
- `chunk 首帧 vs 实测位姿`（现在 0.0417）应当掉到 **0.015 量级**——这是这条建议成不成的判据；
- `policy_deployed @8` 应当从 0.031 掉到 **0.028 以下**（即赢过 `hold_state` null）。

### 风险
相对化会把误差从"位置"搬到"速度"上：如果 state 的读数本身有偏（memory `gripper-state-scale-mismatch` 记录了夹爪训练侧是命令回显、部署侧是过标定反馈），相对动作会把这个偏差直接注入每一步。**保留 `--exclude-joints gripper` 就是为了这个。**

---

## 2. 本体感觉改走 adaLN / FiLM，而不是拼一个 token 进 memory

### 现状
`modeling_patch_policy.py:429-434`：
```python
if self.config.use_robot_state:
    state_token = self.state_projector(batch[OBS_STATE]).unsqueeze(2)  # (b, s, 1, e)
    patch_tokens = torch.cat([patch_tokens, state_token], dim=2)       # 每帧 768 -> 769
```
即 **in-context token concatenation**：把 state 变成 memory 里的第 3846 个槽位，让 cross-attention 自己去找。

### 文献依据（这条是本文最硬的一条）
[arXiv:2410.10088]《The Ingredients for Robotic Diffusion Transformers》(Dasari, Mees, Zhao, Srirama, Levine, 2024) 就是专门做这个消融的。其 Table III 在双臂 ALOHA 上比较了同一个扩散 transformer 的三种条件化方式：

| 条件化方式 | Pick Place | Pen Uncap |
|---|---:|---:|
| **adaLN-Zero**（他们的选择） | **50%** | **100%** |
| adaLN 但不做 zero-init | 38% | 80% |
| Cross-attention（100 步去噪） | 38% | 70% |
| Cross-attention（10 步 DDIM） | **0%** | **0%** |
| **In-context 拼接** | **0%** | **0%** |

作者的原话是标准 joint-attention 机制"simply not able to learn policies as stably"。

**`patch_policy` 现在用的正是那张表里得 0 分的那一种。** 而且注意倒数第二行：**cross-attention 在 10 步去噪下会彻底崩掉，adaLN 不会**——这条直接约束了 §9（减去噪步数）能不能做。

同一篇的 Table IV 还给出：把视觉 tokenizer 从独立 ResNet-26 换成纯 transformer，成功率从 50%/100% 掉到 0%/0%（小模型）或 13%/20%（放大后），结论是"CNNs should still be considered as encoders for robotics tasks, particularly for low-data regimes"。你们是 363 episode 单场景，正是 low-data regime。

### 具体改动
在 `TransformerForDiffusion` 里，把 state 与 diffusion timestep embedding **合并**成一个全局条件向量，走 adaLN-Zero：
```python
# 现在：time_emb 单独进，state 混在 cond memory 里
# 改成：
global_cond = self.time_emb(timestep) + self.state_projector(state)   # (B, n_emb)
# 每个 decoder block 前：
shift, scale, gate = self.adaLN_modulation(global_cond).chunk(3, dim=-1)  # 最后一层 zero-init
x = x + gate * block(modulate(self.ln(x), shift, scale), memory)
```
`ACT` 头（`PatchACTHead`）没有 timestep，可以走更简单的 FiLM：用 state 生成每层的 `(γ, β)` 去调制 decoder 的输入。

⚠️ **zero-init 不能省**：上表里 adaLN 少了 zero-init，Pick Place 从 50% 掉到 38%。

### 为什么这条和 §1 都要做
它们修的是同一个洞的两侧：§1 让"用 state"变成**输出上的恒等式**（不用学），§2 让 state**在结构上无法被忽略**（不能不看）。前置报告已经证明单独翻 `use_robot_state` 这个布尔开关无效——因为它两侧都没动。

---

## 3. `n_obs_steps` 从 5 降到 1 或 2

### 实测（本地，最干净的一组）
| 干预 | `new_state5` | `new_obs2` | `prev_act_head` |
|---|---:|---:|---:|
| 历史帧时间顺序**倒转** | **0.5%** | **0.2%** | 1.0% |
| 历史帧全部替换为最新帧 | **2.8%** | **1.5%** | 3.5% |
| （参照）换掉图像 | 135.7% | 139.6% | 139.7% |
| （参照）换扩散采样种子 = 噪声地板 | **3.3%** | 1.3% | — |

**把 5 帧观测的时间顺序完全翻转，模型输出只动 0.2–0.5%。** 模型把历史当成无序集合，没有任何速度信息。而 `n_obs_steps` 5→2 的延迟收益是实测的 **0.595 s → 0.296 s**，精度差 1.1%（噪声内）。

按上表，`n_obs_steps=2 → 1` 的代价上界约 **1.5%**，也在 3.3% 的采样噪声内。

### 文献依据
1. **参考实现里 5 不是调出来的。** [arXiv:2607.18236] Patch Policy 原文的 window size 是按环境给的预设（Push-T 5、LIBERO Goal 10、BlockPush 3），**论文里没有任何 window size 的消融**。你们继承的 5 是 Push-T 的值，不是针对这套真机的结论。
2. **更长的历史在 BC 里可能主动有害。** [arXiv:1905.11979]《Causal Confusion in Imitation Learning》(de Haan, Jayaraman, Levine, NeurIPS 2019) 指出 BC 会抓住与专家动作相关的 nuisance 特征；[arXiv:2010.14876]《Fighting Copycat Agents in Behavioral Cloning from Observation Histories》(Wen, Lin, Darrell, Jayaraman, Gao, NeurIPS 2020) 把它具体化为 **copycat 问题**：当专家动作时序高度相关时，模仿者会去预测"上一个动作"而不是"下一个动作"，导致**从观测历史学习反而比只用最新一帧更差**。你们的 `action_t ≈ state_t`（差 0.0140 rad）正是"专家动作时序高度相关"的教科书情形。
3. **同仓库的对照**：`lerobot` 自己的 `diffusion` 策略默认 `n_obs_steps = 2`，`act` 默认 `n_obs_steps = 1`（且 `n_obs_steps != 1` 直接报错）。而 ACT 基线在同一评测集上比最好的 patch_policy 好 17%。

### 具体改动
```bash
--policy.n_obs_steps=2      # 保守；或 =1 更激进
```
连带效果：`cond_pos_emb` 从 984 576 参数降到 393 472（或 196 736），memory 从 3 846 槽位降到 1 538（或 769），**state token 的占比从 0.13% 升到 0.33%（或 0.65%）**——这本身对 §2 有利。

### ⚠️ 一个必须先确认的边界
`n_obs_steps=1` 时 `horizon = n_obs_steps + action_chunk_size - 1 = action_chunk_size`，`unpack_actions` 只产出 1 个 chunk，块因果掩码退化成全可见。逻辑上没问题，但 `generate_mask_matrix(npatch, nwindow=1)` 这条路径**没有被测过**。降到 1 之前先跑一遍 `tests/policies/patch_policy`。降到 2 没有这个顾虑。

---

## 4. 加 LR warmup + cosine 衰减（现在是常数学习率，无 scheduler）

### 现状
```python
def get_scheduler_preset(self) -> None:
    # The reference trains at a constant learning rate; there is no scheduler in train_policy.py.
    return None
```
`optimizer_lr = 5.5e-5` 全程恒定，200 000 步。

### 为什么这是个问题（用你们自己的曲线说）
| step | 1 k | 10 k | 50 k | 100 k | 150 k | 200 k |
|---|---:|---:|---:|---:|---:|---:|
| 训练 loss (`new_state5`) | 0.193 | 0.057 | 0.032 | 0.022 | 0.019 | **0.016** |
| held-out MAE (`prev_diffusion`) | — | — | 0.0840 | 0.0773 | 0.0760 | 0.0712 |
| held-out @1 (`prev_diffusion`) | — | — | 0.0668 | 0.0621 | **0.0678** ⚠️ | 0.0456 |

**训练 loss 从 50 k 到 200 k 降 52%，held-out 只降 15%，而且 @1 在 150 k 处非单调地变差。** 常数学习率在后期就是在最优点附近来回弹——这正是 cosine 衰减要消除的东西。

### 文献依据
- 同仓库对照：`lerobot` 的 `diffusion` 策略默认 `scheduler_name = "cosine"`、`scheduler_warmup_steps = 500`。**同一个仓库里，扩散策略有 scheduler，patch_policy 没有。**
- [arXiv:2405.12213] Octo：AdamW + 线性 warmup + cosine 衰减是通用机器人策略训练的标准配方。
- [arXiv:2410.10088] DiT-Policy：AdamW + cosine schedule，250 K iterations。

### 具体改动
```python
def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
    return DiffuserSchedulerConfig(
        name="cosine",
        num_warmup_steps=self.scheduler_warmup_steps,   # 新字段，默认 2000（= 1% of 200k）
        num_training_steps=...,                          # 训练循环注入
    )
```
warmup 取总步数的 1–3%（200 k → 2 000–6 000 步）。

**顺带**：既然 held-out 在 100 k 就基本停滞，**把总步数从 200 k 降到 100 k 并把 cosine 压缩到 100 k**，很可能得到同等或更好的 held-out，同时省一半训练时间。这是一个可以顺手做的对照。

---

## 5. 给扩散头加 EMA 权重

### 现状（代码里自陈的偏离）
`configuration_patch_policy.py` 的 docstring：
> `- No EMA on the diffusion head. The reference samples from an EMA copy of the denoiser; lerobot's training loop has no post-optimizer-step hook to update one.`

即：**参考实现有 EMA，你们这份移植没有，理由是工程限制而不是实验结论。**

### 文献依据
[arXiv:2303.04137]《Diffusion Policy: Visuomotor Policy Learning via Action Diffusion》(Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song) 在 4 个 benchmark、12 个任务上平均提升 46.9%，其配方里 EMA 是标准组成部分。后续的机器人扩散策略工作普遍沿用（典型衰减 0.75–0.9999）。

### 与你们实测的对应
- 采样噪声地板 **3.3%**（同一 checkpoint 换种子）；
- held-out @1 在 150 k 处**非单调**变差；
- 逐 episode 极差 1.47–1.81×。

这三条都是"权重在最优点附近抖"的症状，EMA 直接压这一类抖动。

### 具体改动
`lerobot` 的训练循环里没有 post-optimizer hook，但加一个不难——在 `train.py` 的 `optimizer.step()` 之后：
```python
if ema is not None:
    ema.step(policy.model.head)   # 只对 head，不对冻结的 encoder
```
保存 checkpoint 时同时存 EMA 副本，评测和部署用 EMA 副本。

**成本**：显存 +（head 参数量）≈ +40 MB，可忽略（现在 1.91–4.28 GB）。

---

## 6. 打开图像增强（现在完全没有）

### 现状
`processor_patch_policy.py` 的 pipeline 只有四步：`Rename → AddBatchDim → Device → Normalize`。**没有任何数据增强。** 而 `VISUAL` 归一化是 `IDENTITY`（由冻结 ViT 自己做 ImageNet 归一化），所以连隐式的扰动都没有。

数据面：363 episodes / 单场景 / 200 k 步 × batch 16 = **10.6 epoch**。训练 loss 还在降而 held-out 在 100 k 停滞——这是过拟合的形状。

### 文献依据
- [arXiv:2303.04137] Diffusion Policy 的标准配方包含 **random crop** 图像增强。
- `lerobot` 自己的 `diffusion` 策略保留了 `crop_shape` / `crop_is_random=True` 的通路（默认 `crop_ratio=1.0` 即关闭，但机制在）。

### 具体改动
`lerobot` 已有现成的 `ImageTransformsConfig`（`transforms/transforms.py:166`），默认包含 brightness / contrast / saturation / hue / sharpness / RandomAffine(±5°, translate 0.1)，只需要在训练命令上加：
```bash
--dataset.image_transforms.enable=true --dataset.image_transforms.max_num_transforms=3
```

### ⚠️ 三条重要的限定，不要盲目开
1. **编码器是冻结的，增强只能正则化 head。** 增强的主要价值（让 backbone 学出不变性）在冻结设置下拿不到。这条建议的收益因此比在端到端策略上小，**除非同时做 §8 的解冻**。
2. **几何增强会破坏"像素位置 ↔ 关节角"的对应。** RandomAffine 的平移/旋转会让同一张图对应不同的正确动作。腕部相机尤其敏感。**建议先只开光度增强（brightness/contrast/saturation/hue/sharpness），把 `affine` 的 weight 设 0。**
3. **增强与"预计算冻结特征"互斥。** 见 §12。

---

## 7. `act` 头的 decoder 从 1 层加到 4 层

### 现状与实测
```python
n_decoder_layers: int = 1
```
head-comparison §7.2 的实测（`act` 头 vs `diffusion` 头，屏蔽单路相机）：

| | `act`（1 层 decoder） | `diffusion`（8 层） |
|---|---:|---:|
| 屏蔽任一路相机后 MAE 恶化 | **3.7–4.2 倍** | 1.4–2.0 倍 |
| 屏蔽后预测标准差 / 全观测 | 38–48% | 74–92% |
| **全黑输入下的预测标准差** | **0.0000**（对每个样本输出完全相同的动作） | 0.1332 |

**`act` 头在全黑输入下彻底塌回一个无条件常数。** 机制上说得通：1 层 decoder 要用一次 cross-attention 从 3 840 个 memory token 里读完全部信息，学到的是一个依赖三路 token 联合分布的、脆的读出方式；8 层的扩散去噪器被迫学出冗余表征。

### 文献依据
`lerobot` 自己 `act/configuration_act.py:108-111` 的注释：
> `# Note: Although the original ACT implementation has 7 for n_decoder_layers, there is a bug in the code ...`

即上游 ACT ([arXiv:2304.13705], Zhao, Kumar, Levine, Finn) **原本是 7 层**，`lerobot` 的 1 是为了复现一个上游 bug，不是设计结论。`patch_policy` 的 `PatchACTHead` 直接抄了这个 1。

### 具体改动
```bash
--policy.n_decoder_layers=4
```
**延迟预算完全够**：`act` 头现在是 10.5 ms，部署窗口 267 ms，占空比 4%。加到 4 层大约 25–30 ms，占空比 ~11%，仍然远比扩散头的 111–223% 宽裕。参数量 +约 6 M（`dim_model=512`, `dim_feedforward=3200`，每层约 2 M）。

### 为什么这条只排第 7
它修的是**鲁棒性**（掉相机时不塌），不是**精度**（锚定偏差）。在三路相机总是同时到达的当前部署场景下，脆性不是活跃约束。但它便宜，而且是 §1/§2 之后 `act` 头能不能吃下更强条件信号的前提。

---

## 8. 视觉侧：先提分辨率，再考虑换 encoder / 末层解冻

### 实测：空间细节是被用上的，而且不够用
| 干预 | `new_state5` | `new_obs2` |
|---|---:|---:|
| patch 网格 → 每帧 1 个 token | **55.6%** | **66.2%** |
| **1/4 分辨率** | **16.3%** | **18.8%** |

对照文献：[arXiv:2607.18236] Patch Policy 原文 Table 4（Push-T）:

| patch 数 | 256 | 64 | 1 |
|---|---:|---:|---:|
| 成功率 | **0.69** | 0.52 | 0.48 |

原文结论：`"spatially downsampling the features results in a significant decrease in task success."`

**两边一致：空间分辨率是这个架构的主要信息通路。** 而你们现在把 **480×640 双线性下采样到 224×224** 才送进 ViT-S/14 —— 宽度压掉 65%。J7（腕转）的 MAE 是 J2 的 3.2–3.8 倍，而腕转恰恰是在低分辨率下最难看清的自由度。

### 三个梯度的改动，按成本排序

**(a) 提分辨率（最便宜，一个字段）**
```bash
--policy.resize_shape="(308, 308)"     # 308/14 = 22 -> 484 patch/相机（现在 256）
```
代价：token 数 ×1.9，编码器 FLOPs ×~1.9。配合 §3 把 `n_obs_steps` 降到 2，**净 token 数仍然比现在少**（2×3×484 = 2 904 vs 现在 5×3×256 = 3 840）。这是本报告里"省下的算力投回去"的最佳去处。

**(b) 换 encoder preset（零代码，已在 `PATCH_ENCODER_PRESETS` 里）**
`dinov3_patch`（`facebook/dinov3-vits16plus-pretrain-lvd1689m`）已经配好。DINOv3 的密集特征质量优于 DINOv2 同级模型。注意它是 16px patch、224² 下只有 196 token，比现在**少**——所以 (b) 应当和 (a) 一起做。

**(c) 末层解冻或 LoRA（最贵，最后做）**
[arXiv:2304.06600]《Lossless Adaptation of Pretrained Vision Models For Robotic Manipulation》(Sharma et al., ICLR 2023) 指出：机器人任务上**完全冻结的预训练特征达不到最优**，而全量微调会破坏预训练表征；插入轻量 adapter 可以在接近全量微调的性能下保住原表征。[arXiv:2410.10088] Table IV 也指出低数据场景下独立 CNN 编码器显著优于纯 transformer tokenizer。

反方证据要一并记住：[arXiv:2607.18236] 原文 `"For all experiments, we freeze the image encoder and exclusively train the policy"`，理由是可以预计算视觉嵌入、大幅加速训练；而且他们的真机实验是**双腕相机的单一 cable insertion 任务**，末端执行器在画面里一目了然。你们的顶视相机几乎看不见手臂——**同一个默认值在两个场景里的合理性不一样**。

具体做法：`freeze_vision_encoder=False` 是全解冻，风险大（7.58 M 可训练参数 → 29 M，且冻结时被覆写的 `train()` 逻辑要重新审视）。**推荐先做最后 2 个 block 的解冻 + 独立的低学习率**（backbone lr = head lr / 10，参考 ACT 的 `optimizer_lr_backbone`）。

### ⚠️ 关于 register token，一条否定建议
[arXiv:2309.16588]《Vision Transformers Need Registers》(Darcet, Oquab, Mairal, Bojanowski, ICLR 2024) 说明 DINOv2 的 patch 特征里有高范数 artifact token 污染密集任务。**但该文同时指出这些 outlier 主要出现在 ViT-Large 及以上的模型里**，而你们用的是 **ViT-S/14**。所以"换成带 register 的权重"**不是**当前的高优先级项——列在这里是为了让下一个人不用再查一遍。

---

## 9. 扩散头：去噪步数 100 → 10（纯推理侧，不必重训）

### 实测：这是延迟的全部
| run | 中位延迟 | 编码器图像数 | 去噪步数 | 部署窗口 0.267 s 的 duty |
|---|---:|---:|---:|---:|
| `new_state5` | 0.595 s | 15 | 100 | **2.23×** ❌ |
| `new_obs2` | 0.296 s | 6 | 100 | **1.11×** ❌ |
| `prev_act_head`（`act` 头） | **0.0105 s** | 15 | — | 0.04× ✅ |
| `act_baseline`（ACT） | **0.0065 s** | 3 | — | 0.02× ✅ |

同样 15 张图，`act` 头 10.5 ms，扩散头 595 ms——**57 倍差距全部来自 100 步 DDPM**。`num_train_timesteps: 100` 且 `num_inference_steps: null`（回落到 100）。

### 三个梯度
**(a) DDIM 10 步（零成本，改一个字段，不重训）**
```bash
--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
```
预期 0.296 s → **~0.035 s**，duty 从 1.11× 降到 0.13×。用 `offline_chunk_eval.py` 直接复测精度。
⚠️ **必须和 §2 一起看**：[arXiv:2410.10088] Table III 显示 cross-attention 条件化在 10 步 DDIM 下成功率掉到 **0%**，而 adaLN 保持 50%/100%。**你们现在正是 cross-attention 条件化**，所以单独做 (a) 很可能精度崩掉。**顺序应当是先 §2 后 §9(a)。**

**(b) flow matching（要重训）**
π₀ ([arXiv:2410.24164]) 用 flow matching，**K=10 步**即可。你们树里 `act_dit` 已经跑过 flow matching（`act_dit-flowmatching-deployed-eval-2026-08.md`），有现成经验可以复用。

**(c) 一致性蒸馏（要额外一轮训练）**
[arXiv:2405.07503]《Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation》(Prasad, Lin, Wu, Zhou, Bohg, 2024) 从预训练的 Diffusion Policy 蒸馏出单步/少步采样器，宣称推理速度比最快的替代方法再快一个数量级，成功率保持竞争力。

### 但先问一个更根本的问题
head-comparison §4 已经测过：**在匹配 horizon 上 `act` 头在每一个 horizon 都比 `diffusion` 好 6–8%，且推理快 58 倍、确定性无采样抖动。** 扩散头唯一的优势是单相机缺失时的鲁棒性，而这不是当前部署的约束。

**所以 §9 真正的建议是：先确认还要不要扩散头。** 如果只是想要多模态表达能力，先做 §7（把 `act` 头 decoder 加深）——成本低得多。

---

## 10. 部署侧：放大执行窗口，或上实时分块

### 现状（这是最被低估的一条）
`deploy_config_patch_policy.yaml`：`n_action_steps: 8`，`chunk_interval_s: 0.26667`。
而 `send_next_action_chunk` 里 `bridge_steps = min(40, chunk.shape[0])`——**chunk 只有 8 步时，Hermite 桥接覆盖它的全部 8 步**。

后果（state-and-window §5、§6）：
- 执行窗口内**每一步都是从实测位姿出发的三次 S 曲线**，策略的唯一贡献是这条曲线的终点；
- 桥接一项贡献 **−26 ~ −30%** 的误差改善，其余四个滤波器合计 < 3%；
- **四个 patch_policy 权重在 horizon 8 上全部劣于 `hold_state` null**（0.0309–0.0336 vs 0.0283），唯一赢过 null 的是 ACT（0.0243）。

**花 0.3–0.6 s 推理去决定一条 S 曲线的一个端点，这个投入产出比是失衡的。**

### 两条路
**(a) 把 `n_action_steps` 从 8 放到 50（零成本）**
此时桥接（`min(40, len)`）不再覆盖 chunk 的全部，策略的 chunk 形状真正参与执行。代价是反应性下降。`new_obs2` 在 1.667 s 窗口下 duty = 0.18，**跑得动**。

**(b) 实时分块（RTC / BID），保留反应性**
- [arXiv:2506.07339]《Real-Time Execution of Action Chunking Flow Policies》(Black et al., Physical Intelligence, NeurIPS 2025)：在执行当前 chunk 的同时生成下一个，把保证会执行的动作"冻结"、其余部分"inpaint"。**对任意 diffusion/flow VLA 开箱即用，不需要重训**——这正好对上你们"推理 0.3–0.6 s、窗口 0.267 s"的失配。
- [arXiv:2408.17355]《Bidirectional Decoding》(BID)：测试时采样多个 chunk，按 backward coherence（与上一次决策一致）+ forward contrast（未来计划的高似然）挑一个。纯推理时算法。
- 最便宜的版本：ACT 的 **temporal ensembling**（[arXiv:2304.13705]），对重叠 chunk 做指数加权平均。`lerobot` 的 `ACTTemporalEnsembler` 已经实现了，但注意 `act_delta` 的注释说明**它与相对动作不兼容**（ensembler 平均的是锚定在不同 state 上的相对偏移）——如果做了 §1，就不能用它，要用 RTC/BID。

---

## 11. 怎么排这 10 条（含一条流程铁律）

### 🔴 先说流程铁律
head-comparison §2 和 state-and-window §2 记录了同一件事：**两次实验的差异完全来自节点上 `train-venv` 的默认值漂移，不是设计。** 同一份 sbatch 落在 gpu04 / gpu05 得到两个不同的模型；两次对比里都有 2–3 个字段同时变化，导致归因不可分离。

**在跑下面任何一组实验之前，先在 sbatch 里显式写死每一个关心的字段：**
```bash
--policy.action_head=act \
--policy.n_obs_steps=2 \
--policy.action_chunk_size=50 \
--policy.n_action_steps=50 \
--policy.use_robot_state=true \
--policy.n_decoder_layers=1 \
--policy.resize_shape="(224,224)" \
--policy.optimizer_lr=5.5e-5
```
**并保留中间 checkpoint**（A 的 50 k/100 k/150 k 已被清理，导致收敛分析只能对 B 做）。

### 建议的实验顺序

| 轮次 | 改动 | 变量数 | 成本 | 判据 |
|---|---|---:|---|---|
| **R0** | 基线：`act` 头 + `n_obs_steps=2` + 全部字段显式写死 | — | 4.5 h | 建立可比基线；确认与 `prev_act_head` 一致 |
| **R1** | **§1 相对动作**（单独一项） | 1 | 4.5 h | **chunk 首帧 vs 实测位姿 < 0.020 rad**；`deployed @8` < 0.0283（赢过 null） |
| **R2** | R1 + **§2 adaLN/FiLM** | 1 | 5 h | 换 state 的输出位移应从 6.7% 升到 > 30% |
| **R3** | R2 + **§4 cosine + §5 EMA**（这两条互不干扰，可同轮） | 2 | 5 h | held-out @1 单调；逐 episode 极差收窄 |
| **R4** | R3 + **§8(a) resize 308 + §7 decoder 4 层** | 2 | 7 h | J7 的 MAE 下降幅度应大于其它关节 |
| **R5** | R4 + **§6 光度增强**，步数 100 k | 1 | 3 h | held-out 不再在 100 k 停滞 |
| **推理侧**（任何时候，不必重训） | **§9(a) DDIM 10 步**（仅扩散头，且必须在 R2 之后）、**§10(a) 窗口 50**、**§10(b) RTC** | — | 分钟级 | 用 `offline_chunk_eval.py` 直接复测 |

R1 是**唯一的必答题**。如果 R1 之后 chunk 首帧偏差没有掉到 0.02 rad 以下，说明诊断错了，后面的都不用做——先回来重新找原因。

### ❌ 不要再调的旋钮（前置报告已实测无效）
| 旋钮 | 实测依据 |
|---|---|
| 继续加训练步数（200 k → 400 k） | 100 k → 200 k 只买到 8%，且 @1 非单调 |
| **再翻一次 `use_robot_state` 布尔开关** | 已翻过，锚定偏差纹丝不动（2.97× vs 2.81×） |
| **把 state token 复制成 K 份** | 已证伪：token 拿到 8.65× 均分份额的注意力、范数 0.76，没有被淹没；复制只会拆分同一份 softmax 质量 |
| 提高 `n_obs_steps` | 现有 5 帧里时间顺序只值 0.2–0.5% |
| 单纯换 action head（act ↔ diffusion） | 值 6–8%，比锚定偏差小一个数量级 |
| 换 DINOv2-with-registers 权重 | artifact 主要出现在 ViT-L 及以上，你们是 ViT-S/14（§8 末） |

---

## 12. 一条附带的工程建议（不占 10 条名额）

**冻结编码器 + 无增强 = 视觉特征可以预计算并缓存。**
[arXiv:2607.18236] 原文明说这么做 `"significantly accelerating training"`。你们每一步都在把同样的图片重新过一遍 DINOv2（`new_state5` 每步 15 次编码器前向，`updt_s = 0.187 s`）。预计算之后训练步长会掉到只剩 head 的成本。

**但它与 §6（图像增强）和 §8(c)（解冻）互斥。** 三选一：
- 要**最快训练** → 预计算特征，放弃增强与解冻；
- 要**最好精度** → 增强 + 末层解冻，接受慢训练；
- **折中** → 预计算 N 份不同增强的特征（例如 4 份），训练时随机取一份。磁盘换算力。

按当前证据（held-out 在 100 k 停滞 = 过拟合），我倾向**第二条**。

---

## 附录 A：文献清单与核对状态

核对日期 **2026-08-31**。"全文核对"= 抓取了 arXiv 页面并核对了标题/作者/摘要与引用的具体数字；"条目核对"= 核对了 arXiv ID 与标题/作者匹配，具体数字引自检索摘要，使用前建议再读原文。

| # | 引用 | arXiv | 用在 | 核对 |
|---|---|---|---|---|
| 1 | Zhou, Cui, Langford, Tan, LeCun, Pinto. *Patch Policy: Efficient Embodied Control via Dense Visual Representations* (2026-07-20) | [2607.18236](https://arxiv.org/abs/2607.18236) | §3, §8, §12 | **全文核对**（Table 4 的 0.69/0.52/0.48、冻结编码器原句、window size 预设、无 window 消融） |
| 2 | Dasari, Mees, Zhao, Srirama, Levine. *The Ingredients for Robotic Diffusion Transformers* (2024-10-14) | [2410.10088](https://arxiv.org/abs/2410.10088) | **§2**, §8, §9 | **全文核对**（Table III adaLN vs cross-attn vs in-context；Table IV ResNet-26） |
| 3 | Wen, Lin, Darrell, Jayaraman, Gao. *Fighting Copycat Agents in Behavioral Cloning from Observation Histories* (NeurIPS 2020) | [2010.14876](https://arxiv.org/abs/2010.14876) | §3 | **全文核对**（摘要 + 作者 + venue） |
| 4 | Prasad, Lin, Wu, Zhou, Bohg. *Consistency Policy: Accelerated Visuomotor Policies via Consistency Distillation* (2024-05-13) | [2405.07503](https://arxiv.org/abs/2405.07503) | §9(c) | **全文核对**（"order of magnitude" 原句；具体成功率数字未取） |
| 5 | de Haan, Jayaraman, Levine. *Causal Confusion in Imitation Learning* (NeurIPS 2019) | [1905.11979](https://arxiv.org/abs/1905.11979) | §3 | 条目核对 |
| 6 | Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023) | [2303.04137](https://arxiv.org/abs/2303.04137) | §5, §6 | 条目核对（EMA / random crop 属其标准配方，具体消融表未逐项核对） |
| 7 | Black et al. *Real-Time Execution of Action Chunking Flow Policies* (NeurIPS 2025) | [2506.07339](https://arxiv.org/abs/2506.07339) | §10(b) | 条目核对（另见 pi.website 与 NeurIPS proceedings） |
| 8 | *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling* (2024) | [2408.17355](https://arxiv.org/abs/2408.17355) | §10(b) | 条目核对 |
| 9 | Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA, 2023) | [2304.13705](https://arxiv.org/abs/2304.13705) | §7, §10(b) | 条目核对 |
| 10 | Black et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control* (2024) | [2410.24164](https://arxiv.org/abs/2410.24164) | §1, §9(b) | 条目核对（1%/99% 分位数归一化、K=10 flow 步） |
| 11 | *π₀.₅: a VLA Model with Open-World Generalization* (2025) | [2504.16054](https://arxiv.org/abs/2504.16054) | §1 | 条目核对 |
| 12 | Sharma et al. *Lossless Adaptation of Pretrained Vision Models For Robotic Manipulation* (ICLR 2023) | [2304.06600](https://arxiv.org/abs/2304.06600) | §8(c) | 条目核对 |
| 13 | Darcet, Oquab, Mairal, Bojanowski. *Vision Transformers Need Registers* (ICLR 2024) | [2309.16588](https://arxiv.org/abs/2309.16588) | §8（**否定建议**） | 条目核对（"outlier 出现在 ViT-L 及以上"来自检索摘要，采纳前请读原文） |
| 14 | Octo Model Team. *Octo: An Open-Source Generalist Robot Policy* (2024) | [2405.12213](https://arxiv.org/abs/2405.12213) | §4 | 条目核对 |
| 15 | *On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer* (2023) | [2312.03673](https://arxiv.org/abs/2312.03673) | §1 | 条目核对 |

---

## 附录 B：这份文档的边界

- **没有新的实测。** 所有本地数字都引自三份前置报告，那三份报告自己的边界（全部是离线开环、近分布 held-out、无真机成功率、评测集偏乐观）**全部继承到这里**。
- **预期收益全部是估计。** 只有 §1 有实测代理（re-anchor −18.8%、桥接 −26~30%），其余各条的"预期"来自文献在**别的机器人、别的任务**上的数字，不能直接搬。
- **§2 是本文最强的一条建议，但它的证据来自 ALOHA 上的双臂任务**（[2410.10088] Table III），不是你们这套硬件。对照关系成立是因为架构形状一致（扩散 transformer + 多模态条件），但迁移不是自动的。
- **§8 的两个方向互相矛盾**（Patch Policy 原文坚持冻结，Lossless Adaptation 说冻结达不到最优）。文中已把两边都列出来；差别在于**你们的顶视相机看不见手臂**，而原文的真机实验是双腕相机的单一插拔任务。这是我的判断，不是文献结论。
- **没有查的方向**：多任务 / 多数据集协同训练（你们是单任务 363 episode）、语言条件、3D / 点云输入、RL 微调（HIL-SERL 就在 `~/YING/HIL-SERL`）、以及 DAgger 数据配比（`~/YING/paper/PA-DAgger` 和 `dagger-strategy-and-data-allocation-2026-08.md` 是独立的一条线）。
