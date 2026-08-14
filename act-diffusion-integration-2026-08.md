# ACT × Diffusion Policy 融合可行性研究：架构差异、DiT 剖析与十个落地方案

- **日期**：2026-08-14
- **代码基线**：`/home/kewei/YING/robot_data_platform/lerobot` @ `8db548d4`（lerobot 0.6.2 系）
- **审查对象**：`src/lerobot/policies/{act, act_delta, diffusion, multi_task_dit, vita, rtc}/`
- **文献范围**：2023-03 ~ 2026-08
- **配套阅读**：[`experiment_report/act/ACT-improvement-proposals-2026.md`](./experiment_report/act/ACT-improvement-proposals-2026.md)、
  [`experiment_report/act/ACT-experiment-plan-2026-08.md`](./experiment_report/act/ACT-experiment-plan-2026-08.md)、
  [`flowmatching-image-to-action-plan-2026-08.md`](./flowmatching-image-to-action-plan-2026-08.md)、
  [`experiment_report/vita/vita-arch-ablation-2026-08.md`](./experiment_report/vita/vita-arch-ablation-2026-08.md)

> **证据分级说明。** 本文所有 `file:line` 均为本次实际阅读所得，参数量为按配置解析式计算（见 §2.3，计算脚本内联）。
> 文献引用分两级：**[✓]** = 本次直取 arXiv abs 页核对过标题/作者/日期/摘要；**[○]** = 仅从检索摘要获得，未逐篇核实。
> 未标记者为领域内奠基性工作（ACT / Diffusion Policy / DiT）。

---

## 0. 一页结论

**三件事。**

**一、"ACT vs Diffusion Policy" 这个对立命题在本仓库里是错的——ACT 在推理期根本不是生成模型。**
`modeling_act.py:452-458`：非训练态时 `latent_sample` 被强制置零。也就是说 CVAE 的随机性只在训练期存在，
部署时 ACT 是一个**完全确定性**的 chunk 回归器。`use_vae=True` 的作用是训练期正则化，不是推理期多模态。
因此"融合"的真实问题不是"两个生成模型怎么合"，而是**"如何在不放弃单步前向的前提下，把分布建模能力接回推理路径"**。
这个重新表述直接决定了后面十个方案里哪些是真的、哪些是伪需求。

**二、ACT decoder 与 DiT block 的结构距离，比文献给人的印象小得多——只差 timestep 与调制方式。**
ACT decoder（`modeling_act.py:596-665`）= self-attn(chunk 内) + cross-attn(观测 token) + FFN，query 是
`chunk_size` 个可学 positional embedding、输入恒为零张量（`:495-499`）。
DiT block（`multi_task_dit/modeling_multi_task_dit.py:496-554`）= self-attn(chunk 内) + FFN + adaLN-Zero 调制，
输入是含噪动作。**把 ACT 的 `decoder_in` 从 `zeros` 换成 noisy action、加一路 timestep embedding、把 L1 换成 flow-matching 回归，
就地得到一个保留 token 级 cross-attention 的 DiT**——这是方案 S1，也是本文认为收益/风险比最高的一条。

**三、仓库里已有的 `multi_task_dit` 存在参数预算错配，按你的 3 相机配置，85% 的 DiT 参数花在一个"把观测压成 6 个调制向量"的投影上。**
按 `apex` 式配置（3 相机 / `state_dim=16` / `n_obs_steps=2` / CLIP ViT-B/16）解析计算：
`conditioning_dim = 5664`，每个 block 的 `adaLN_modulation` 是 `Linear(5920, 3072)` = **18.19M** 参数，
6 层共 **109.1M**；而 attention+FFN 主干 6 层合计只有 **18.9M**。比值 **85.2%**。
更严重的是这 109M 参数的输出只有 `6×512=3072` 维、且**逐 chunk 恒定**——空间结构在 `CLIPVisionEncoder`
取 CLS token 那一步（`:214-219`）就已经被丢光了。ACT 同配置下 encoder 有 **902 个** token 供 decoder 逐层 cross-attend。
**在这个仓库里，DiT 现在的实现比 ACT 看得少。** 直接拿它当 ACT 的替代品去比，比出来的不是"扩散 vs 回归"，是"CLS vs 空间 token"。

**如果只做三件事：S1（ACT-DiT 同构改造）+ S5（latent 上做扩散）+ S3（ACT 作为 warm-start）。**
三者共用一份 ACT encoder 资产，互不冲突，且分别覆盖"重训一次拿上限"、"零延迟拿多模态"、"不重训拿增量"三种预算。

---

## 1. 两条路线的本质差异

### 1.1 差异不在"回归 vs 生成"，在四个正交维度上

社区常见的二分（ACT=确定性快、DP=多模态慢）掩盖了真正的设计分歧。逐条拆开：

#### (a) 目标函数与它隐含的分布假设

| | ACT | Diffusion Policy |
|---|---|---|
| 损失 | `L1(a, â)` + `KL·kl_weight`（`modeling_act.py:145-160`，`kl_weight=10.0`） | `MSE(ε̂, ε)`（`modeling_diffusion.py:378-385`，`prediction_type="epsilon"`） |
| 隐含假设 | 条件分布是**单峰**的，L1 收敛到条件中位数 | 不假设峰数，学的是 score / 噪声场 |
| 多模态承载体 | CVAE latent（32 维，`latent_dim=32`） | 采样噪声 `x_T`（`horizon × action_dim` 维） |
| 推理期是否用上 | **否**（latent 置零，`:456-458`） | 是（`torch.randn`，`:247-253`） |

L1 在多模态数据上的行为是取中位数而非均值，比 L2 稍好（中位数落在某个模态上的概率更高），但仍然会在
"左绕/右绕"这类良分离双峰上给出穿过障碍物的中间轨迹。2026 年有一篇专门做这件事的理论工作
[✓ Mazza et al., 2026, arXiv:2605.22493] 给出了两侧的机制：latent-variable 策略受
**posterior-prior 正则的两难**支配——正则强则 latent 里的 action-conditioned 信息被抹掉（正是 ACT 推理期置零的等价后果），
正则弱则依赖 prior 是否覆盖到相关 latent 区域；而 action-space 生成式策略（扩散/流）的多模态能力受
**base→action 传输映射的 Lipschitz 常数**上界约束，覆盖多个良分离模态必然要求 base 空间的突变或 action 空间的
off-support 桥接区。**两条路线各有各的天花板，不是一方免费碾压另一方。**

#### (b) 条件注入路径——本文认为这是最被低估的一维

| 策略 | 观测→动作的通路 | 每层可见的观测信息量 |
|---|---|---|
| **ACT** | ResNet18 feature map 逐相机 flatten 成 token（`:474-486`），decoder 每层 cross-attend | 480×640 输入下 **902 token**；240×320 下 212 token |
| **Diffusion Policy (UNet1d)** | 所有观测 concat + flatten 成一个 `global_cond` 向量（`:269-305`），经 FiLM 注入每个 conv block | **1 个向量**（`global_cond_dim × n_obs_steps`），空间结构由 SpatialSoftmax 压成 keypoints |
| **multi_task_dit (本仓库)** | CLIP CLS token concat → flatten → adaLN-Zero（`:322-335`, `:536-539`） | **1 个向量**，空间结构完全丢失 |
| **Dita** [✓ arXiv:2503.19757] | in-context conditioning：raw visual token 与 action token 同序列 | 全部视觉 token |
| **DiT-X (ManiFlow)** [✓ arXiv:2509.01819] | adaptive cross-attention + adaLN-Zero 并用，低维 state 走 adaLN、视觉走 cross-attn | 全部视觉 token + 调制 |

结论：**ACT 在条件通路上其实站在"高信息量"的一侧，Diffusion Policy 的经典实现站在"低信息量"的一侧。**
社区常把 ACT 的成功归给 action chunking，但至少一部分要归给它是唯一默认保留逐相机空间 token 的那个。
这也解释了你在 `ACT-improvement-proposals-2026.md` 里实测到的"相机位置编码逐相机逐字节相同"为什么是个真缺陷——
它损害的正是 ACT 最强的那条通路。

#### (c) 时间轴的三套不同参数

| | ACT | Diffusion Policy | multi_task_dit |
|---|---|---|---|
| `n_obs_steps` | **1**（`configuration_act.py:84`，且 `:151-154` 硬性 raise） | 2 | 2 |
| 预测长度 | `chunk_size=100` | `horizon=64` | `horizon=32` |
| 执行长度 | `n_action_steps=100`（**全开环**） | 32 | 24 |
| 聚合 | `ACTTemporalEnsembler`（要求 `n_action_steps=1`，`:138-141` 互斥） | 无（receding horizon） | 无 |

`n_action_steps=100` 在 30 Hz 下是 **3.3 秒开环**。这不是"ACT 反应性差"的架构宿命，是默认配置选的。
而 temporal ensembling 与 chunk 执行在 lerobot 里是**互斥**的（`configuration_act.py:138-141` 显式 `NotImplementedError`），
所以现状是二选一：要么 3.3 s 开环，要么每步都推一次全量前向。这一条是后面 S9 的直接动机。

#### (d) 推理成本

- ACT：**1 次**前向。ResNet18×3 + 4 层 encoder(902 token) + 1 层 decoder(100 query)。
- Diffusion Policy：`num_inference_steps` 默认等于 `num_train_timesteps=100`（`modeling_diffusion.py:227-230`）。
  UNet 前向 ×100，但观测编码只做一次（`global_cond` 在循环外算好，`:322-325`）。这是好设计。
- multi_task_dit：同上，`num_train_timesteps=100`；FM 分支 `num_integration_steps=100`、`integration_method="euler"`。
  **默认 100 步是 2024 年的配置**，2026 年的 FM 实践普遍在 4–10 步（见 `flowmatching-image-to-action-plan-2026-08.md` §1.2）。

关键观察：**扩散的迭代成本只落在动作分支上，视觉编码不重复。** 按 480×640 输入解析核算（MACs×2）：

| 组件 | GFLOPs |
|---|---:|
| ResNet18 × 3 相机 @480×640 | 33.4 |
| ACT encoder 4 层 × 902 token（`dim_ff=3200`） | 37.9 |
| ACT decoder 1 层 × 100 query | 2.1 |
| **ACT 单次前向合计** | **≈ 73** |
| DiT 每个去噪步（6 层 / 512 宽 / 100 token / 含 adaLN） | 4.1 |
| DiT 4 步 / 10 步 / 100 步 | 16.5 / 41.2 / **411.6** |

**准确的说法是：100 步默认值确实不可接受（是 ACT 单次前向的 5.6 倍），但 4–10 步只是 ACT 的 0.23–0.56 倍，
属于同一量级、可接受。** 所以"扩散太慢"不是范式问题，是把 `num_inference_steps` 留在 100 的配置问题。
注意这里没算 ACT encoder 的 902-token 自注意力其实比 DiT 的 100-token 昂贵得多（37.9 vs 4.1 per step）——
**ACT 把预算花在了看观测上，DiT 把预算花在了迭代上。** 实测确认见 §5 的 M0。

### 1.2 文献上的胜负关系

不存在单向碾压，分任务类型：

- **确定性、高精度、单一策略的任务**：ACT 赢。磁控微机器人双臂操作上 ACT 的 RMSE 79.56 ticks，
  Diffusion Policy 153.52、Flow Matching 140.50 [○ Mag-VLA, arXiv:2605.28486]。
- **多模态演示、大数据量**：Diffusion Policy 赢，这是 [Chi et al., 2023, arXiv:2303.04137] 的原始主张。
- **小数据（<100 demos）、边缘部署**：ACT 赢，主要是数据效率与延迟。
- **开放手术缝合跟随的四策略评测** [✓ Wang et al., 2026, arXiv:2605.28736]：ACT / DP / SmolVLA / π₀ 在理想条件下
  都只有 50–75% 成功率，**主导失效模式是深度误差，跨架构一致**。π₀ 凭预训练 VLM backbone 最强。
  这条证据很重要：**在真实任务上，架构差异被感知瓶颈压过去了。** 它是对"换生成范式就能涨点"的一个直接反例。
- **2026 年的产业实践**：主流栈不再二选一，而是把其中一个当作 VLA 之下的低层 action head [○ 多处综述]。

**对本项目的推论：** 如果你的 `express` / `pack_airpods` / `tea_2` 数据是单一策略演示（大概率是，遥操作数据通常如此），
换成扩散**不会**自动涨点，甚至可能掉点。融合的正当理由应当是下面三条之一，而不是"扩散更先进"：
1. 你确实观察到 chunk 边界抖动 / 多模态歧义（VITA 的振动报告指向这里）；
2. 你需要在同一模型上支持多任务/多策略（`multi_task_dit` 的 text 条件动机）；
3. 你想要一个可以被 RL 后训练的分布式策略（扩散/FM 有现成的 policy-gradient 配方）。

---

## 2. DiT 架构剖析

### 2.1 三种条件注入范式

DiT [Peebles & Xie, arXiv:2212.09748] 原文比较了三种把条件塞进 transformer 的方式，机器人领域三种都有人用：

| 范式 | 机制 | 参数代价 | 信息带宽 | 代表工作 |
|---|---|---|---|---|
| **in-context** | 条件当成额外 token 拼进序列 | 近乎零 | 高（全 token） | Dita [✓ 2503.19757] |
| **cross-attention** | 动作 token 作 query，观测作 key/value | 中（每层一组 KV 投影） | 高 | ACT decoder、MDT |
| **adaLN-Zero** | 条件 → 6 组 (shift, scale, gate) 逐层调制 | **高**（与 cond_dim 线性） | 低（每层 6×d 维，chunk 内恒定） | DiT-Block Policy [○]、本仓库 `multi_task_dit` |

DiT-Block Policy（"The Ingredients for Robotic Diffusion Transformers"）[○ dit-policy.github.io] 的主张是
**用 adaLN-zero 替换 cross-attention 反而更好**，理由是默认 attention 实现导致的训练动力学问题，报告平均 +20%，
并在 ALOHA 1500+ 步长时序任务上大幅领先。**但要注意它的前提**：它同时给每个相机配了独立 ResNet-18，
即它并没有把视觉压成 CLS——它是在"每相机独立编码 + 池化"的条件下比较的。

2025–2026 的走向是**不再二选一**：
- **DiT-X / ManiFlow** [✓ 2509.01819]：低维 robot state 走 adaLN-Zero，多模态观测走 adaptive cross-attention，
  cross-attn 的输入输出再被学出来的 scale/shift 调制。消融确认"非对称设计 + adaLN 优于纯 cross-attn +
  双向注意力优于因果注意力"三条，尤其在多阶段任务上。
- **Tenma** [○ arXiv:2509.11865]：每个 block 既 cross-attend 观测序列，又用 adaLN-zero 注入 diffusion timestep。
- **AC-DiT** [○ arXiv:2507.01961]：移动操作场景下的自适应协调 DiT。

**共识（本文的读法）：timestep 这种"每 chunk 一个标量"的条件天然适合 adaLN；视觉这种"高维、有空间结构"的条件天然适合 attention。
把两者混在一个 adaLN 里是范式误用。** 这正是本仓库 `multi_task_dit` 的问题。

### 2.2 本仓库 `multi_task_dit` 的实现审查

结构（`modeling_multi_task_dit.py`）：

```
ObservationEncoder (:258-383)
  ├─ CLIPVisionEncoder ×N_cam  → 只取 CLS token，reshape 成 (B, 768, 1, 1)   :214-219
  ├─ robot_state (原样)                                                       :342
  └─ CLIPTextEncoder (冻结) → Linear(768→512)                                 :225-255
  → concat → flatten(start_dim=1) → conditioning_vec                          :382-383

DiffusionTransformer (:557-629)
  ├─ time_mlp: SinusoidalPosEmb(256) → MLP → 256                              :574-580
  ├─ cond_features = cat([timestep_emb, conditioning_vec])   ← cond_dim       :619
  ├─ input_proj: Linear(action_dim → 512)                                     :583
  ├─ 6 × TransformerBlock:
  │     adaLN_modulation = SiLU + Linear(cond_dim → 6×512)                    :534
  │     RoPEAttention(自注意力，仅 chunk 内)                                    :451-493
  │     MLP(512→2048→512)                                                     :528-532
  └─ output_proj: Linear(512 → action_dim)                                    :607
```

**注意 `RoPEAttention` 是纯 self-attention（`:475-493`，q/k/v 全来自 `x`），没有任何 cross-attention。
整个网络看到观测的唯一途径就是 adaLN 的 6 组调制系数。**

### 2.3 参数预算错配：85% 的参数在一个低带宽通路上

按你的实际配置（3 相机、`state_dim=16`、`n_obs_steps=2`、CLIP ViT-B/16 hidden=768、`hidden_dim=512`、
`num_layers=6`、`timestep_embed_dim=256`、`text projection_dim = hidden_dim = 512`）解析计算：

```python
h, L, ff, t = 512, 6, 4*512, 256
cond = (768*3 + 16 + 512) * 2                 # = 5664   conditioning_dim
cd   = t + cond                                # = 5920   adaLN 输入维
adaln_per_block = cd*(6*h) + 6*h               # = 18,189,312
trunk_per_block = (h*3*h+3*h) + (h*h+h) + (h*ff+ff + ff*h+h)   # = 3,150,336
# → adaLN 6 层 109,135,872 ; 主干 6 层 18,902,016 ; 占比 85.24%
```

| 组件 | 参数量 | 占 DiT 主体 |
|---|---:|---:|
| `adaLN_modulation` × 6 | **109.14 M** | **85.2%** |
| attention + FFN × 6 | 18.90 M | 14.8% |
| `time_mlp` | 0.26 M | — |
| `input_proj` + `output_proj` | 0.017 M | — |

**三个后果：**

1. **信息瓶颈。** 5664 维观测被压到每层 3072 个调制标量，且这些标量在 chunk 的 100 个时间步上**完全相同**
   （`:541`, `:550`，`shift.unsqueeze(1)` 广播）。也就是说 DiT 无法让第 3 步和第 97 步看到不同的观测细节。
   ACT 的 decoder 每个 query 可以独立地 attend 到不同的图像区域。这是**表达力上的硬差距**，不是调参能补的。
2. **优化困难。** 109M 个参数只通过 3072 个输出接受梯度，且 `_initialize_weights` 把最后一层 zero-init（`:610-613`，
   这是 adaLN-Zero 的正确做法）。训练早期梯度信号极稀薄。
3. **空间信息在编码器就没了。** `CLIPVisionEncoder.forward` 只返回 `last_hidden_state[:, 0]`（CLS），
   `get_output_shape()` 返回 `(768, 1, 1)`。ViT-B/16 @224 有 196 个 patch token，全部丢弃。

**修法很明确：让 adaLN 只吃 `timestep + robot_state`（16+256=272 维 → 每层 0.84M，6 层 5.0M），
视觉/语言走 cross-attention。** 这正是 DiT-X 的配方，也是方案 S10。

### 2.4 两个必须先修的实现问题

**(P0) 无语言数据时维度不匹配。** `_setup_vector_output` 无条件地 `total_dim += self.text_dim`（`:333`），
但 `encode()` 只在 `OBS_LANGUAGE_TOKENS in batch` 时才 append text features（`:373-380`）。
`text_encoder` 也是无条件构造的（`:290`）。因此在**没有 task 字段的数据集上**，`conditioning_vec` 的实际宽度比
`conditioning_dim` 少 `512×n_obs_steps = 1024`，会在第一个 `adaLN_modulation` 的 `Linear` 上抛 shape 错误。
你现有的 `express` / `pack_airpods` / `tea_2` / `dex_stack_box` 数据集是否都带 language token，需要先确认；
不带的话 `multi_task_dit` 现在跑不起来。修法：把 `total_dim += self.text_dim` 挪进条件分支，或引入
`use_language: bool` 配置项并在两处共用。

**(P1) 默认推理步数是 100。** `num_inference_steps: int | None = None` → 回落到 `num_train_timesteps=100`；
FM 分支 `num_integration_steps=100`。任何与 ACT 的延迟对比在这个默认值下都不公平。
对比实验前必须先扫 `{100, 20, 10, 4}`（DDIM / Euler），否则测的是配置不是架构。

---

## 3. 融合的三个正交轴

把文献里所有"AR/回归 × 扩散"的组合归一化后，只有三个正交的注入点。十个方案就是这三个轴上的取值组合。

```
                 观测 o
                   │
   ┌───────────────┼───────────────┐
   │        轴C：谁产生条件         │   ← 表征层：token 级 vs 全局向量 vs 共享 backbone
   └───────────────┬───────────────┘
                   ▼
   ┌───────────────────────────────┐
   │        轴A：谁定义分布          │   ← 损失层：L1 / 扩散 / 两者加权 / 分层(离散模式+连续细化)
   └───────────────┬───────────────┘
                   ▼
   ┌───────────────────────────────┐
   │        轴B：谁提供初值/迭代      │   ← 采样层：从噪声起 vs 从回归解起 vs 从上一 chunk 起
   └───────────────┬───────────────┘
                   ▼
              动作 chunk a_{1:H}
```

- **轴 A（损失）**：决定训练成本与上限。改这一轴必然重训。
- **轴 B（采样）**：决定延迟。可以完全不动训练（S3 冻结版、S9）。
- **轴 C（条件）**：决定天花板。本仓库最大的 headroom 在这里（§2.3）。

文献映射：HybridVLA [✓ 2503.10631] 动 A+B；STEP [✓ 2602.08245] 只动 B；Primary-Fine [✓ 2602.21684] 动 A；
Dita / DiT-X 只动 C；RTC [○ 2506.07339] 与 Soft RTC [✓ 2605.25537] 只动 B。

---

## 4. 十个方案

每个方案给出：**动机 / 改动点 / 是否重训 / 工作量 / 预期收益 / 风险 / 判决性消融 / 置信度**。
置信度指"文献+本仓库证据支持该方案能带来所述收益"的强度，不是"实现难度"。

---

### S1 ★ ACT-DiT 同构改造：给 ACT decoder 装上 timestep 与生成目标

**轴：A + C。** 保留 ACT 的全部条件通路（902 个空间 token 的 cross-attention），只把 decoder 从
"零输入 + L1 回归"改成"含噪动作输入 + flow-matching / 扩散回归"。

**动机。** §2.3 已证明本仓库 DiT 的瓶颈在条件通路而非生成目标。反过来做——保留 ACT 的强条件通路、
只换生成目标——可以在**不损失任何观测信息**的前提下拿到分布建模能力。文献侧的对照是 DiT-X [✓ 2509.01819]
（cross-attn + adaLN 并用）与 Tenma [○ 2509.11865]，两者都证明这个组合可训、且优于单一范式。

**改动点。**
- `act/modeling_act.py:495-499`：`decoder_in = torch.zeros(...)` → `self.action_in_proj(noisy_actions)`，
  形状不变 `(chunk, B, dim_model)`。
- `act/modeling_act.py:596-665` `ACTDecoderLayer`：加 `adaLN_modulation = Sequential(SiLU, Linear(t_dim, 6*dim_model))`，
  **只吃 timestep embedding + robot_state**（≈272 维 → 每层 0.84M，1 层 decoder 时仅 0.84M，可忽略）。
  zero-init 最后一层。cross-attn 分支保持原样。
- `act/modeling_act.py:145-164`：L1 → `MSE(v̂, a₁-a₀)`（conditional FM）或保留 `ε` 参数化。
- `act/modeling_act.py:126-135` `predict_action_chunk`：加 4–10 步 Euler 循环，**encoder 只跑一次**
  （照搬 `modeling_diffusion.py:322-325` 的做法，把 `encoder_out` 提到循环外）。
- CVAE 部分（`:299-322`, `:407-458`）可整块删掉（`use_vae=False`），因为分布建模已由 FM 承担。

**是否重训：是。** 工作量 **3–5 天**（含 encoder 输出缓存的重构）。

**预期收益。** 多模态保真 + 保留 ACT 的空间条件优势。这是本文认为**上限最高**的一条。
**风险。** decoder 只有 1 层（`n_decoder_layers=1`），作为 denoiser 可能太浅——扩散网络通常需要更多深度来
表达不同噪声水平下的不同函数。**这条风险是可测的**：先在 `n_decoder_layers ∈ {1, 2, 4}` 上扫。
另注意你在 `ACT-experiment-plan-2026-08.md` 已确认 lerobot 版本没有 `[0]/[-1]` 死层 bug，加深是安全的。

**判决性消融。** A: 原 ACT(L1)；B: S1 但 `n_dec=1`；C: S1 但 `n_dec=4`；D: S1 去掉 cross-attn 只留 adaLN
（= 退化成 DiT-Block Policy 式）。**B/C vs D 的差值直接量化"空间 token 值多少"**，这是本文最想看到的一个数。

**置信度：★★★★☆**

---

### S2 保留 L1 主干，加扩散头作为辅助损失（MAR 式 per-token diffusion loss）

**轴：A。** decoder 输出的每个 `h_t ∈ ℝ^512` 既送进原 `action_head`（L1），又作为条件送进一个小 MLP denoiser
（3 层、宽 256、输入 `action_dim + t_emb`），训练时 `loss = L1 + λ·diffusion_loss`。

**动机。** 这是把 [Li et al., NeurIPS 2024, "Autoregressive Image Generation without Vector Quantization", arXiv:2406.11838]
的 Diffusion Loss 搬到动作空间：**AR/回归骨干负责"这一步大概去哪"，逐 token 的小扩散头负责"该分布长什么样"**。
机器人侧已有等价做法——多个 diffusion-VLA 都是"VLM 出特征 + 小扩散头出连续动作" [○ 综述多处]。

**改动点。** `act/modeling_act.py:370` 旁边加 `self.diffusion_head`；`:137-164` 加权重 `λ`。
推理期**两条路可选**：快路径直接用 `action_head`（延迟与今天完全一致），慢路径用扩散头采 4 步。

**是否重训：是。** 工作量 **2 天**。
**预期收益。** 低风险的"可回退多模态"：`λ=0` 时退化成今天的 ACT，部署时可按任务阶段切换快慢路径。
**风险。** 两个头可能互相拖累（L1 拉向中位数、扩散拉向全分布），需要扫 `λ ∈ {0.1, 0.3, 1.0}`。
逐 token 独立扩散**不保证 chunk 内时序一致**——这是 MAR 式做法在动作上的固有弱点，可能加剧你在 VITA 报告里
记录的振动问题。建议配合 S9 的时序平滑一起上。

**判决性消融。** λ 扫描 × {只用 L1 头推理, 只用扩散头推理, 两者平均}。
**置信度：★★★☆☆**

---

### S3 ★ Warm-start：把现有 ACT checkpoint 当作扩散/FM 的初值

**轴：B。** 冻结你已经训好的 ACT（100k / 300k 步资产直接复用），在其输出上训练一个 2 步 refiner：
`a₀ = ACT(o) + σ·ε`，refiner 学从 `a₀` 到真值的速度场。

**动机。** STEP [✓ Li et al., ICML 2026, arXiv:2602.08245] 正是这个配方：用确定性预测替换随机噪声初值，
再加一个 velocity-aware 扰动注入，**2 步扩散**即达到有竞争力的结果，在 RoboMimic 与真机上分别比
BRIDGER / DDIM 高 21.6% / 27.5% 成功率。Soft RTC [✓ arXiv:2605.25537] 用的是同一思想的另一版本
（用上一 chunk 的部分去噪状态当先验，而非纯噪声），在 12 个 Kinetix 关卡上把 action delta 降 9.1%、jerk 降 9.6%。

**改动点。** 新建 `policies/act_refine/`，不动 `act/`。refiner 可以直接复用 `vita/flow_matching.py:1-599`
里已有的 FM 实现（`flow_matcher_type` 已支持多种概率路径）。

**是否重训：否（ACT 不动），只训 refiner。** 工作量 **2–3 天**。
**预期收益。** 这是十个方案里**唯一不作废现有 checkpoint** 的。若你已经有若干训到 100k+ 的 ACT，
这条的边际成本最低。
**风险。** 上限受 ACT 教师约束——若 ACT 已经把两个模态平均掉了，refiner 从平均点出发很可能停在附近，
**它修不了模式坍塌，只能修平滑度与精度**。要诚实地把它定位成"精修"而非"多模态修复"。

**判决性消融。** refiner 步数 {1, 2, 4} × σ {0.05, 0.1, 0.3}；关键对照是"从 ACT 输出起"vs"从纯噪声起"
在相同步数下的成功率差。
**置信度：★★★★☆**

---

### S4 Primary-Fine 解耦：离散模式选择 + 模式条件扩散

**轴：A。** 两阶段：(1) 轻量策略把 chunk 压成离散模式（VQ 码本或 K-way 分类），选出一个 coarse 模式；
(2) 模式条件的 MeanFlow / FM 策略生成连续细节。

**动机。** [✓ Lei et al., ICLR 2026, arXiv:2602.21684] 证明了两阶段设计有**严格更低的 MSE 上界**，
在 Adroit / DexArt / MetaWorld 共 56 个任务上超过单阶段生成式基线，并在真机触觉灵巧操作上验证。
好处是**模式在 chunk 内保持一致**（不会执行到一半跳模态），这直接对症 chunk 边界抖动。
本仓库已有 `vqbet/`，码本机制不用从零写。

**改动点。** 把 ACT 的 CVAE latent（`:299-322`）替换成 VQ 码本；推理期从码本先验采样（而不是置零）；
decoder 以码字为条件。

**是否重训：是。** 工作量 **1–2 周**（码本训练不稳定，需要 EMA/dead-code 重启等工程）。
**预期收益。** 在真有多模态演示时收益最大，且天然给出"这条 chunk 属于哪个模态"的可解释信号。
**风险。** 若你的数据其实是单模态，码本会退化（所有样本落到少数码字），白付工程成本。
**上这条之前必须先做诊断**：在训练集上统计"同一观测邻域下动作 chunk 的方差/双峰性"。

**判决性消融。** 码本大小 {8, 32, 128} × {硬选择, Gumbel 软选择}；以及"模式一致性率"这个诊断指标本身。
**置信度：★★★☆☆**（收益强依赖数据是否真多模态）

---

### S5 ★ 在 CVAE latent 上做扩散：零延迟代价的多模态

**轴：A + B，但只作用在 32 维 latent 上。** 保留 ACT 的整个网络与 L1 损失，
只把 latent 的先验从 `N(0, I)` 换成一个学出来的扩散/FM 先验；推理时对 **32 维**跑 4–10 步扩散得到
`latent_sample`，再走一次 ACT 前向。

**动机。** §0 结论一指出，ACT 推理期置零 latent 是多模态丢失的**直接机制**。而 [✓ arXiv:2605.22493]
的分析说 latent-variable 策略保住模式的关键是"prior 是否覆盖到相关 latent 区域"——
**学一个表达力足够的 prior 恰好是这个诊断给出的处方**。这是 Latent Diffusion Planning [○ arXiv:2504.16925]
在 planning 侧的等价思路。

**代价核算：32 维、10 步、一个 3 层 MLP denoiser（宽 256）= 1.64 MFLOPs，
相对 ACT 单次前向的 73 GFLOPs（§1.1d）是 1/44,800。** 也就是说**延迟增量在测量误差内**。
这是十个方案里唯一一个"多模态几乎白送"的结构位置——因为随机性被放在 32 维而不是 1600 维的动作空间上。

**改动点。**
- 训练：第一阶段照常训 ACT（拿到 `mu, log_sigma_x2`）；第二阶段冻结 ACT，在收集到的 `latent_sample` 上训扩散先验。
  两阶段可解耦，也可联合。
- 推理：`act/modeling_act.py:452-458` 的 `latent_sample = zeros` → `self.latent_prior.sample(cond=obs_embed)`。
  条件可以取 encoder 输出的池化，或直接用 robot_state。

**是否重训：ACT 可不重训（第二阶段独立）。** 工作量 **2–3 天**。
**预期收益。** 本文认为这是**收益/代价比最高**的一条：几乎零延迟、几乎零风险、可完全回退（把 prior 换回零向量
就是今天的行为），却直击 §0 结论一指出的那个真实机制缺陷。
**风险。** `kl_weight=10.0` 相当高，可能已经把 latent 压得接近各向同性——那样学出来的 prior 就是 `N(0,I)`，
方案自然失效但也无害。**先做一个 5 分钟的诊断**：在训练集上跑一遍 encoder，统计 `mu` 的经验分布，
看它是单峰高斯还是有结构。**这个诊断本身应该在任何方案之前做**，因为它同时告诉你数据到底有没有多模态。

**判决性消融。** `kl_weight ∈ {10, 1, 0.1}` × `{零 latent, N(0,I) 采样, 学出的扩散 prior}` 的 3×3。
`kl_weight` 降低 + 学 prior 是理论上最有道理的一格。
**置信度：★★★★☆**

---

### S6 残差扩散：ACT 出基线轨迹，DiT 出残差

**轴：A + B。** `a = ACT(o) + Δ`，扩散只建模残差 `Δ` 的分布。因为残差方差远小于动作本身，
所需去噪步数更少、信噪比调度更容易。

**动机。** AnchorRefine [○ arXiv:2604.17787] 的"轨迹锚点 + 残差细化"、
以及 VLA 出 coarse、DM 精修的一系列工作 [○]。与 S3 的区别：S3 是采样初值层面的（同一个动作空间的迭代），
S6 是**参数化层面**的（网络直接输出残差），后者训练信号更干净。

**改动点。** 独立的 `DiffusionTransformer`（可直接复用 `multi_task_dit` 的 block，但按 S10 修好条件通路），
target 为 `a_gt - ACT(o)`。
**是否重训：ACT 可冻结。** 工作量 **3–5 天**。
**预期收益。** 与 S3 类似但上限更高（残差网络可以有自己的条件通路，看到 ACT 看不到的东西）。
**风险。** 与 S3 共享同一个根本局限：**基线若已模式平均，残差分布会变成双峰且远离零，反而更难学**。
残差的尺度需要单独 normalize，否则 `MIN_MAX` 归一化下残差会挤在很窄的区间。

**判决性消融。** 残差 vs 全量参数化，在相同步数下比；以及残差 stats 用 `MEAN_STD` vs `MIN_MAX`。
**置信度：★★★☆☆**

---

### S7 反向蒸馏：扩散/FM 教师 → ACT 学生（单步多模态）

**轴：A。** 先训一个好的扩散或 FM 策略（或直接用你已有的 `vita`），再把它蒸馏进 ACT 的结构，
学生一次前向出 chunk。教师提供的是**分布信息**，而不只是均值。

**动机。** OneDP [○] 报告把动作预测频率从 1.5 Hz 提到 62 Hz（约一个数量级）。
Consistency Policy 类方法同样可 10× 加速，但共同的天花板是"受教师能力约束" [○]。
MP1 [○ arXiv:2507.10543] 用 MeanFlow 做到**免蒸馏**的单步生成，是这条路线的替代品
（你在 `flowmatching-image-to-action-plan-2026-08.md` §1.2 已记录）。

**改动点。** 新的训练脚本；学生沿用 `act/` 结构但可加 noise 输入以保留随机性
（否则单步确定性学生仍然只能表达一个模态——**这是关键细节**：要蒸馏成 `a = G(o, z)` 的形式，`z` 是噪声）。
**是否重训：两次（教师 + 学生）。** 工作量 **1–2 周**。
**预期收益。** 部署形态与今天完全一致（单次前向），但拿到多模态。
**风险。** 双倍训练成本；教师本身若不好则全盘皆输；蒸馏的模式覆盖损失需要专门度量。
**置信度：★★★☆☆**

---

### S8 共享 backbone 双头 + 推理期协同集成

**轴：A + C。** 一个 ResNet + 一个 ACT encoder，两个头：L1 回归头 + FM 头。
推理时按一致性/置信度仲裁或加权。

**动机。** HybridVLA [✓ Liu et al., 2025, arXiv:2503.10631] 把扩散去噪与 next-token 预测统一进一个 LLM，
并用 **collaborative action ensemble** 自适应融合两路预测，报告仿真 +14%、真机 +19% 平均成功率。
在没有 LLM 的小模型上，等价做法就是共享 encoder 的双头。

**改动点。** `act/modeling_act.py:370` 处并列两个 head；新增仲裁逻辑（两头预测的 L2 距离小则取平均，
大则取 FM 头——因为分歧大通常意味着多模态）。
**是否重训：是。** 工作量 **4–6 天**。
**预期收益。** 除了成功率，**"两头分歧度"本身是一个免费的在线不确定性信号**，可以用来触发降速/求助/
切换执行 horizon（与你 `ACT-improvement-proposals-2026.md` 的提案 #7 自适应 horizon 天然耦合）。
**风险。** 仲裁规则是启发式的，容易过拟合到某几个任务。
**置信度：★★★☆☆**（成功率增益 ★★★，不确定性信号 ★★★★）

---

### S9 执行层融合：用扩散/一致性准则替换 temporal ensemble

**轴：B。完全不动训练。** 把 `ACTTemporalEnsembler`（`:167-255`）的指数加权平均，换成
BID 式的多样本搜索或 RTC 式的 chunk inpainting 拼接。

**动机。** 三条相互印证的证据：
- BID [○ Bhattacharyya et al., arXiv:2408.17355]：每步采多个预测，按 **backward coherence**（与既往决策一致）
  与 **forward contrast**（未来计划高似然）择优。明确指出 receding horizon 在多模态演示下会产生抖动轨迹，
  而 EMA 平均正是**跨模态平均**——这在数学上和 L1 的模式平均是同一个病。
- RTC [○ Black et al., arXiv:2506.07339]：异步生成下一 chunk，对已执行前缀做 inpainting 引导，
  **对任何扩散/流策略免重训即插即用**，报告成功率至 94.1%。**本仓库已有 `policies/rtc/` 实现**
  （`action_interpolator.py` / `action_queue.py` / `latency_tracker.py` 俱全）。
- Soft RTC [✓ arXiv:2605.25537]：软化 RTC 的二值 mask，jerk −9.6%。
- TAS [○ arXiv:2511.04421] 与你 `ACT-improvement-proposals-2026.md` 提案 #5 已记录的路线一致。

**改动点。** 部署侧脚本；`configuration_act.py:138-141` 那条 `n_action_steps=1` 的互斥限制需要放开
（BID 需要每步采样但不需要执行长度为 1）。
**是否重训：否。** 工作量 **2–5 天**。
**预期收益。** 直接对症你在 VITA 部署报告里记录的振动问题，且**零训练成本、可当天验证**。
**风险。** BID 需要每步多次前向（K 个样本），延迟 ×K；对 ACT 这种确定性策略，多次采样需要先有随机源
（所以 S9 与 S5 是天然搭档：S5 提供 latent 随机性，S9 提供选择准则）。
**置信度：★★★★☆**

---

### S10 修 `multi_task_dit` 的条件通路：adaLN 只吃低维，视觉走 cross-attention

**轴：C。** 落实 §2.3 的诊断。

**改动点。**
- `modeling_multi_task_dit.py:214-219`：`CLIPVisionEncoder.forward` 返回 `last_hidden_state[:, 1:]`
  （196 个 patch token）而非 CLS；或换成 ResNet18 feature map 以对齐 ACT 的成本结构。
- `:322-335` `_setup_vector_output`：`conditioning_dim` 只保留 `robot_state`（并修 P0 的 text_dim 无条件累加）。
- `:496-554` `TransformerBlock`：加一路 `cross_attn(query=action_tokens, key=value=obs_tokens)`，
  `adaLN_modulation` 的输入维从 5920 降到 272。**参数量 109.1M → 5.0M，省 104M**；
  省下的预算全部可以还给主干（`num_layers` 6→12、`hidden_dim` 512→768 只需 ~60M）。
- 可选：按 DiT-X [✓ 2509.01819] 用学出的 scale/shift 调制 cross-attn 的输入输出。

**是否重训：是。** 工作量 **3–5 天**。
**预期收益。** 让 `multi_task_dit` 与 ACT 的对比变成一个**公平**的对比。在此之前，任何
"ACT vs DiT" 的实验数字都不可解释——这是本文对实验设计最强的一条建议。
**风险。** 100 个 action token × ~600 个视觉 token 的 cross-attn，显存与 FLOPs 都会明显上升，
需要控制 token 预算（与你 `ACT-experiment-plan-2026-08.md` 方向 #9 的 token 预算问题同源）。
**置信度：★★★★★**（这条是修 bug 级别的确定性改进，不是赌收益）

---

## 5. 可行性评估与优先级

### 5.1 总表

| # | 方案 | 轴 | 重训 | 工作量 | 延迟增量 | 复用现有 ckpt | 上限 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| S1 | ACT-DiT 同构改造 | A+C | 是 | 3–5 d | 中（×4–10 decoder） | 否 | **高** | ★★★★☆ |
| S2 | L1 + 扩散辅助头 | A | 是 | 2 d | 可选（双路径） | 否 | 中 | ★★★☆☆ |
| S3 | ACT warm-start refiner | B | 否 | 2–3 d | 低（2 步） | **是** | 中 | ★★★★☆ |
| S4 | Primary-Fine 离散+连续 | A | 是 | 1–2 w | 中 | 否 | 高 | ★★★☆☆ |
| S5 | latent 扩散先验 | A+B | 否* | 2–3 d | **≈0** | **是** | 中高 | ★★★★☆ |
| S6 | 残差扩散 | A+B | 否* | 3–5 d | 中 | **是** | 中高 | ★★★☆☆ |
| S7 | 扩散→ACT 反向蒸馏 | A | 两次 | 1–2 w | **0** | 否 | 中高 | ★★★☆☆ |
| S8 | 双头 + 协同集成 | A+C | 是 | 4–6 d | 中 | 否 | 中高 | ★★★☆☆ |
| S9 | BID/RTC 执行层聚合 | B | **否** | 2–5 d | ×K 或 ≈0 | **是** | 中 | ★★★★☆ |
| S10 | 修 DiT 条件通路 | C | 是 | 3–5 d | 降低 | 否 | — | ★★★★★ |

\* ACT 主干冻结，只训新增模块。

### 5.2 组合关系

- **天然搭档**：S5 + S9（前者提供随机源，后者提供选择准则）；S3 + S6（初值 vs 参数化，二选一即可，别同时上）；
  S10 + S1（修好 DiT 通路后，S1 的对照组 D 才有意义）。
- **互斥**：S1 与 S2（都改 decoder 出口，语义冲突）；S4 与 S5（都改 latent 语义）。
- **前置依赖**：所有涉及 `multi_task_dit` 的比较实验依赖 S10 与 §2.4 的 P0 修复。

### 5.3 三条最重要的可行性判断

**(1) "扩散太慢"是配置问题，不是范式问题——但必须先测。**
§1.1(d) 的解析核算给出：DiT 每步 4.1 GFLOPs，ACT 单次前向 73 GFLOPs。
100 步 = 5.6× ACT（确实不可接受），10 步 = 0.56× ACT，4 步 = 0.23× ACT（完全可接受）。
观测编码在两种策略里都只做一次，迭代只落在动作分支。
**前置实验 M0：**在实机上测 `{ACT 单次, DiT 4/10/100 步}` 的端到端 p50/p95 延迟，
并核对解析值与实测的偏差（kernel launch 开销在小 batch 多次迭代下会显著放大，解析 FLOPs 会低估 DiT）。
若 10 步的实测增量 < 10 ms，"为了速度而放弃生成范式"这个论证就失效了，方案选择应完全由分布建模需求驱动。

**(2) 你的数据是否真的多模态，决定了一半方案的价值。**
S4/S5/S7/S8 的收益全部押在"演示里有多个有效策略"上。遥操作采集的数据往往是单一操作员的单一策略，
此时 L1 的模式平均**根本不发生**，换生成范式只会引入采样方差。
**前置实验 M1（成本 <1 天）：**跑一遍 ACT 的 VAE encoder，统计 `mu` 的经验分布结构；
并在训练集上对相近观测（DINO 特征 kNN）分组，量化组内动作 chunk 的方差与双峰性。
这个诊断同时是 S5 的可行性门槛。

**(3) 最大的头顶空间可能不在"生成范式"上。**
开放手术四策略评测 [✓ arXiv:2605.28736] 显示 ACT / DP / SmolVLA / π₀ 的主导失效模式**跨架构一致**（深度误差），
而唯一明显胜出的 π₀ 靠的是预训练 VLM backbone 而非动作头形式。
结合你在 `ACT-improvement-proposals-2026.md` 里已实测的相机位置编码缺陷，
**"修条件通路"（S10、相机身份嵌入、DINOv3 backbone）的期望收益很可能高于"换生成范式"。**
这是本文对整个课题最诚实的一条评估：**融合值得做，但不应被当作首要收益来源。**

### 5.4 推荐路线

```
第 0 周   M0 延迟基准 + M1 多模态诊断 + multi_task_dit P0/P1 修复        (2–3 天)
          └─ 决策门：M1 显示单模态 → 跳过 S4/S7，直接走 S10 + S3
第 1 周   S5（latent 扩散先验，零延迟、可回退）+ S9（执行层，不重训）      (5 天)
          └─ 这一周结束就应该能看到抖动/成功率的第一个信号
第 2–3 周 S10（修 DiT 条件通路）→ 得到公平的 DiT 基线                    (5 天)
第 3–5 周 S1（ACT-DiT 同构改造）+ 判决性消融 A/B/C/D                     (8–10 天)
          └─ 这一步产出的 "空间 token 值多少" 是本课题最可发表的一个数
备选      S3（若已有高步数 ACT ckpt 且不想重训）随时可插入，与上述并行
```

---

## 6. 局限与未解问题

1. **本文没有跑任何训练，也没有跑任何延迟实测。** 参数量与 GFLOPs 均为按配置解析计算（MACs×2，
   ResNet18 按 1.82 GFLOPs@224 线性外推到 480×640），不含 kernel launch、内存带宽、数据搬运开销——
   **多步迭代的实际延迟一定高于 FLOPs 比例给出的估计**。所有收益预期来自文献外推 + 本仓库静态分析。
   §5.3 的三条判断都以前置实验（M0/M1）为条件。
2. **参数量计算依赖假设的配置。** §2.3 用的是 3 相机 / `state_dim=16` / `n_obs_steps=2` / CLIP ViT-B/16。
   若你实际用 `use_separate_rgb_encoder_per_camera=True` 或不同相机数，`conditioning_dim` 与 85% 这个比例会变
   （相机数越多，adaLN 占比越高，结论方向不变且更强）。
3. **`multi_task_dit` 的 P0（无语言时维度不匹配）是静态阅读推断，未实跑验证。** 建议用一个不带 task 字段的
   batch 跑一次 `forward` 确认。
4. **未覆盖的组合。** 本文没有讨论 RL 后训练（PAC-ACT [○ arXiv:2607.09590]、Q-chunking 类方法）——
   扩散/FM 策略在 RL 微调上的配方比 L1 策略成熟得多，这可能是"融合"的第四个动机，但超出本次范围。
   也没有讨论 3D/点云条件（FlowPolicy、DP3 系）。
5. **文献覆盖偏向 2025–2026 的 arXiv 预印本**，其中相当一部分尚未经同行评审；
   标 [○] 的条目仅凭检索摘要，未逐篇核对全文，引用前应按 `feedback_citation_verification` 的规程复核。
6. **S1 的核心风险（1 层 decoder 作为 denoiser 是否够深）在文献里没有直接答案**，
   因为没人从 ACT 侧改起——大家都是从 DiT 侧加 cross-attn。这既是风险也是这条路线的原创性所在。

---

## 7. 参考文献

**已直取 arXiv abs 核实 [✓]**

- Mazza, L., Datres, M., Rodriguez, A., Bodenstedt, S., Kutyniok, G., & Speidel, S. (2026). *Understanding multimodal failure in action-chunking behavioral cloning*. arXiv:2605.22493.
- Lei, X., Wang, M., Zhou, W., Lu, X., & Li, H. (2026). *Primary-fine decoupling for action generation in robotic imitation*. ICLR 2026. arXiv:2602.21684.
- Yan, G., Zhu, J., Deng, Y., Yang, S., Qiu, R.-Z., Cheng, X., Memmel, M., Krishna, R., Goyal, A., Wang, X., & Fox, D. (2025). *ManiFlow: A general robot manipulation policy via consistency flow training*. arXiv:2509.01819.
- Hou, Z., Zhang, T., Xiong, Y., Duan, H., Pu, H., Tong, R., Zhao, C., Zhu, X., Qiao, Y., Dai, J., & Chen, Y. (2025). *Dita: Scaling diffusion transformer for generalist vision-language-action policy*. arXiv:2503.19757 (earlier: arXiv:2410.15959).
- Li, J., Cong, Y., Wang, Y., Xia, H., Huang, S., Zhang, Y., Xu, N., & Dai, G. (2026). *STEP: Warm-started visuomotor policies with spatiotemporal consistency prediction*. ICML 2026. arXiv:2602.08245.
- Liu, D., Zheng, Z., Sun, Y., Zhang, L., Liu, Y., & Wan, H. (2026). *Action-prior denoising for smooth real-time chunking*. arXiv:2605.25537.
- Liu, J., Chen, H., An, P., Liu, Z., Zhang, R., Gu, C., … Zhang, S. (2025). *HybridVLA: Collaborative diffusion and autoregression in a unified vision-language-action model*. arXiv:2503.10631.
- Wang, X., Yang, Z., Zhang, X., Kim, S. E., Hardy, R., & Rajpurkar, P. (2026). *Imitation learning for robot assistance in open surgery: A multi-policy evaluation on suture following*. arXiv:2605.28736.

**奠基性工作**

- Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). *Learning fine-grained bimanual manipulation with low-cost hardware*. arXiv:2304.13705. （ACT）
- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). *Diffusion policy: Visuomotor policy learning via action diffusion*. arXiv:2303.04137.
- Peebles, W., & Xie, S. (2023). *Scalable diffusion models with transformers*. arXiv:2212.09748. （DiT / adaLN-Zero）
- Li, T., Tian, Y., Li, H., Deng, M., & He, K. (2024). *Autoregressive image generation without vector quantization*. NeurIPS 2024. arXiv:2406.11838. （Diffusion Loss，S2 的来源）

**仅检索摘要，未逐篇核实 [○]**

- Bhattacharyya, … (2024). *Bidirectional decoding: Improving action chunking via guided test-time sampling*. arXiv:2408.17355.
- Black, K., et al. (2025). *Real-time execution of action chunking flow policies*. arXiv:2506.07339. （RTC）
- *Training-time action conditioning for efficient real-time chunking*. arXiv:2512.05964.
- *Temporal action selection for action chunking*. arXiv:2511.04421. （TAS）
- Dasari, S., Mees, O., et al. *The ingredients for robotic diffusion transformers*. dit-policy.github.io / IEEE. （DiT-Block Policy）
- *Tenma: Robust cross-embodiment robot manipulation with diffusion transformer*. arXiv:2509.11865.
- *AC-DiT: Adaptive coordination diffusion transformer for mobile manipulation*. arXiv:2507.01961.
- *AnchorRefine: Synergy-manipulation based on trajectory anchor and residual refinement for VLA models*. arXiv:2604.17787.
- *Mag-VLA: Vision-language-action model for bimanual magnetically actuated microrobot manipulation*. arXiv:2605.28486. （ACT/DP/FM 的 RMSE 对比数字来源）
- *Mixture-of-experts action chunking transformers for high-precision robot imitation learning*. Robotics and Autonomous Systems. （MEAT）
- *PAC-ACT: Post-training actor-critic for action chunking transformers*. arXiv:2607.09590.
- *Latent diffusion planning for imitation learning*. arXiv:2504.16925.
- Wang, Z., et al. *One-step diffusion policy*. ICML 2025. （OneDP）
- Sheng, et al. (2026). *MP1: MeanFlow tames policy learning in 1-step*. arXiv:2507.10543.
- Gao, et al. *VITA: Vision-to-action flow matching policy*. arXiv:2507.13231.
- 代码内引用：arXiv:2507.05331（`multi_task_dit` 的参考实现来源）。

---

## 8. AI 使用声明

本报告由 AI 辅助研究工具（Claude Opus 5，通过 academic-research-skills deep-research 流程）生成。
仓库审查（`file:line`）与参数量计算为本次实际执行所得；文献部分按 §"证据分级说明" 标注了核实层级。
所有实验预期与收益估计均为基于文献的外推，未经本地实验验证。使用前请按 §6 的局限说明与
`feedback_citation_verification` 的引用规程复核。
