# VITA 网络结构审查与消融设计（配置层面）

> 撰写日期：2026-08-13（夜）
> 审查对象：`/home/kewei/YING/robot_data_platform/lerobot/src/lerobot/policies/vita/`
> （`configuration_vita.py` 273 行 / `modeling_vita.py` 738 行 / `flow_matching.py` 599 行）
> 依据实验：[`vita-deploy-vibration-2026-08.md`](./vita-deploy-vibration-2026-08.md)（§2/§5/§9/§11）、
> [`vita-deploy-inference-code-audit-2026-08.md`](./vita-deploy-inference-code-audit-2026-08.md)（P0–P2）
> 实测环境：conda `lerobot`，torch 2.11.0+cu130 / torchvision 0.26.0+cu130
> 本文只做**静态代码审查 + 小规模实测**，不含新的训练运行。

---

## 0. 一页结论

三件事。

**一、`vision_backbone` 的可替换范围比注释说的窄，也比校验器允许的宽。**
五个 policy（act / act_delta / diffusion / vqbet / vita）共用同一个 `startswith("resnet")` 校验器。
真正可直接替换、无需改代码的是 **5 个**；另有 **5 个** ResNeXt/WideResNet 在两条构造路径上都实测可用，
仅被那一行校验器挡住。`"resnet"` 这个字符串能通过校验但必然崩溃。详见 §1。

**二、消融必需参数只有 6 个轴，而其中最关键的两个在当前配置文件里不存在。**
§10.3 点名的两条主线——时间平滑损失项、时间结构读出头——分别是"配置里没有这个字段"和
"字段存在但被校验器锁死成单一取值"。也就是说**按现在的代码，主线假设根本无法做成对照实验**。
其余 4 个轴（`flow_matcher_type` / `decode_flow_latents` / `recon_loss_type` / `n_obs_steps`）
今天就能跑。同样重要的是**明确不该消融的 9 类参数**及其证据依据。详见 §2。

**三、结构问题的核心是参数预算错配。**
约 **12.9M** 参数是一个**无条件的、推理期逐比特确定的** 512→512 映射，
而真正写出被执行动作的读出头只有 **2.5M**，其最终一层是 `Linear(512, 256)`。
按投入产出给出 6 条建议，排序见 §3.7。

> **贯穿全文的约束**：审计 §1 已测出**每一块 chunk（52/52）**在下发前都被替换成
> 零速起止的三次 Hermite S 曲线。**在 P0-1 / P0-2 修好之前，任何结构变体在真机上产生的执行轨迹都是同一条**，
> 真机不能用来区分消融臂。本文所有评估建议一律指离线评估。

---

## 1. `vision_backbone` 的实际可替换范围

### 1.1 哪些 policy 有这个旋钮

`vision_backbone` 字段只存在于 5 个 policy：**act**、**act_delta**（复用 ACT 的 model）、
**diffusion**、**vqbet**、**vita**。其余都没有可换的视觉主干——
`pi0` / `pi05` / `pi0_fast` / `smolvla` / `groot` / `eo1` / `xvla` / `vla_jepa` 融的是 VLM 视觉塔，
`tdmpc` 是手写的 `nn.Conv2d` 堆叠（`tdmpc/modeling_tdmpc.py:701`）。

五个 policy 的校验器逐字相同（`act/configuration_act.py:134`、`vita/configuration_vita.py:151` 等）：

```python
if not self.vision_backbone.startswith("resnet"):
    raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {...}")
```

### 1.2 两条不同的构造路径

| policy | 构造方式 | 位置 |
|---|---|---|
| act / act_delta | `IntermediateLayerGetter(m, {"layer4": "feature_map"})`，通道数取 `m.fc.in_features`，且传 `replace_stride_with_dilation` | `act/modeling_act.py:326,334,353` |
| diffusion / vqbet / vita | `nn.Sequential(*list(m.children())[:-2])`，通道数由 `get_output_shape` 干跑得到 | `diffusion/modeling_diffusion.py:500`、`vqbet/modeling_vqbet.py:672`、`vita/modeling_vita.py:358` |

两条路径都**自动适配通道数**，512→2048 不需要手改任何维度常量。

### 1.3 实测结果

对每个候选同时跑两条路径（`torch.zeros(1,3,224,224)`）：

| 名称 | ACT 路径 | `Sequential[:-2]` 路径 |
|---|---|---|
| `resnet18` / `resnet34` | OK ch=512 map=(512,7,7) | OK (512,7,7) |
| `resnet50` / `resnet101` / `resnet152` | OK ch=2048 | OK (2048,7,7) |
| `resnext50_32x4d` / `resnext101_32x8d` / `resnext101_64x4d` | OK ch=2048 | OK (2048,7,7) |
| `wide_resnet50_2` / `wide_resnet101_2` | OK ch=2048 | OK (2048,7,7) |
| `resnet` | **FAIL** `'module' object is not callable` | 同左 |
| `efficientnet_b0` | FAIL（不接受 `replace_stride_with_dilation`） | "OK" (1280,7,7) |
| `convnext_tiny` / `mobilenet_v3_small` | FAIL（无 `layer4`） | "OK" (768/576,7,7) |
| `vit_b_16` | FAIL（不接受该 kwarg） | **假 OK** (768,14,14) |
| `regnet_y_400mf` | FAIL（不接受该 kwarg） | "OK" (440,7,7) |

**可直接替换（零改动）**：`resnet18` `resnet34` `resnet50` `resnet101` `resnet152`。

**结构上兼容、仅被校验器挡住**：`resnext50_32x4d` `resnext101_32x8d` `resnext101_64x4d`
`wide_resnet50_2` `wide_resnet101_2`。把那行 `startswith` 换成元组判断即可，无需其它改动。

### 1.4 三个坑

1. **`pretrained_backbone_weights` 必须同批改**。它是以裸字符串传给 torchvision 的，配错是硬报错：
   ```
   resnet50 + ResNet18_Weights.IMAGENET1K_V1  →  KeyError
   resnet50 + ResNet50_Weights.IMAGENET1K_V2  →  OK
   resnext50_32x4d + ResNeXt50_32X4D_Weights.IMAGENET1K_V1  →  OK
   ```
2. **`vision_backbone="resnet"` 能过校验器再崩溃**：`torchvision.models.resnet` 是子模块不是函数。
3. **非 ResNet 进 `Sequential[:-2]` 路径会"跑通但是错的"**。最危险的是 `vit_b_16`：
   `children()` 是 `[conv_proj, encoder, heads]`，`[:-2]` 只剩 **patch embedding 卷积**，
   整个 transformer 被丢掉——它能训练，只是没在做你以为的事。
   源码自己标了 `# TODO(alexander-soare): Use a safer alternative.`（`diffusion/modeling_diffusion.py:504`）。

另注：diffusion / vqbet 的 `use_group_norm=True` 与预训练权重互斥（显式 raise），
且 GroupNorm 用 `num_features // 16` 分组，对 ResNet 家族安全，对任意通道数不保证。

### 1.5 对本项目的结论

**VITA 不需要动 `vision_backbone`**，理由见 §2.4 第 5 条：报告 §2.3③ 已经用跨关节相关性
把视觉通路从抖动嫌疑里排除了。这一节的价值是为将来"换主干"类实验划出安全边界。

---

## 2. 消融必需参数

### 2.0 取舍准则

报告已经把**模型侧**缺陷收敛成两条可测量的断言，一个轴只有能推动其中之一才值得一次训练：

- **断言 A —— 读出残差沿时间轴是白的。**
  200k 原始手臂输出 lag-1 自相关 **−0.080**，训练集 **+0.973**；反转率 27.1% vs 4.4%。
  §9.2 已证 88 epochs 平台化，**训练量这条杠杆治不好它**。
- **断言 B —— 首帧偏移。** `|action0 − state|` = **0.0378**，训练集 **0.0203**（**1.86×**）。

真机上看到的其余现象全部归属部署侧（审计 P0-1…P1-5），不属于本消融的射程。

### 2.1 Tier 0：必需，但当前配置文件里做不了

**这是本节最重要的发现。**

**(1) 时间平滑损失权重——字段不存在。**
`compute_loss` 只在两处调用 `self.recon_loss_fn`（`modeling_vita.py:313` 的 FLD 重建、
`:319` 的编码器重建），二者都是逐元素的，全图没有任何东西作用于时间轴。
需要新增 `action_smoothness_weight: float = 0.0`（可再加阶数开关）。
**λ=0 恰好就是现基线**，所以对照臂免费。这是唯一直指断言 A 的轴。

**(2) `action_decoder_type`——字段存在但被校验器锁死。**
字段在 `configuration_vita.py:121`，而 `:191-194` 拒绝除 `"simple"` 外的一切取值，
即该字段当前只有一个取值，**无从对照**。

> **必须同时控制的混淆项**：加卷积/transformer 读出头时，
> `action_dec_hidden_dim` / `action_ae_num_layers` 要调到与 MLP 头**参数量对齐**，
> 否则"时间结构有用"与"容量更大有用"分不开——这是审稿人第一句会问的。

### 2.2 Tier 1：必需且今天就能跑（4 轴）

| # | 参数 | 对照臂 | 理由 |
|---|---|---|---|
| 3 | `flow_matcher_type` | `exact` → `conditional` | VITA README 明确推荐，§3.2 列过但一直没跑。`exact` 在 minibatch 上做 OT 重配对，是"逐样本不相关残差"最后一个站得住的假设，也是论文自己的招牌设计。 |
| 4 | `decode_flow_latents` | `True` → `False` | FLD 是把采样隐向量拴回**本样本**动作的机制，是 VITA 的核心贡献，不消融交不出去。 |
| 5 | `recon_loss_type` | `l1` → `l2` | 改一个字符。L1 是求中位数的，对小残差的符号模式近乎无所谓；L2 二次惩罚。平滑损失臂的廉价对照。 |
| 6 | `n_obs_steps` | `1` → `2` | 唯一针对 §6.1 部分可观测性的配置旋钮——模型看不到速度，位置相同/速度不同是同一个输入。是断言 A 的竞争解释。代价：视觉前向 ×2。 |

**3 和 4 必须做成联合网格，不能各跑各的**。且 `configuration_vita.py:51` 的 docstring 写明
`exact` + `decode_flow_latents=False` 是非法组合，所以 2×2 实际只有 **3 个合法格**：

```
(exact, FLD)        = 现基线
(conditional, FLD)
(conditional, noFLD)
```

这三格才能把"OT 重配对注入了残差"和"FLD 修复了 OT 打乱的配对"分开。任取两格都做不到。

### 2.3 Tier 2：有预算就各跑一次

- **`action_encoder_type`：`cnn` → `simple`。** 干净的对称对照：`SimpleActionEncoder` 是逐时间步 MLP，
  **跨时间零混合**；`CNNActionEncoder` 是 Conv1d k=5/s=2。编码器只在训练期使用
  （不在推理通路上），所以它若能推动指标，说的是隐空间几何而非读出头。
  现配置直接可跑：512 % 16 == 0 满足 `latent_dim % horizon` 约束。
- **`enc_contrastive_weight` / `flow_contrastive_weight`**（都是 0）：只为损失表补完整性行。

### 2.4 明确**不要**消融的参数（作为固定量报告，附证据）

1. **`n_action_steps`** —— 答案已在手上。§11：4 vs 8，所有尺度无关指标持平或略差。
   且推理期会被 clamp，本来就不是训练变量。再花一次训练是浪费。
2. **`horizon`** —— 被 `horizon >= 2**action_ae_num_layers`（`:202-209`）钉死在 16
   （cnn + 4 层）。动它就必须同时动 `action_ae_num_layers` 和 `drop_n_last_frames`，三重混淆。
   §6.2 已证它只是垫料。
3. **`num_sampling_steps`** —— §2.3② 已证同一观测下逐比特确定，**采样不是抖动来源**。
   只有在主张"推理成本"时才消融，那属于 MeanFlow 那条线。
4. **全部 `meanflow_*` 与 `flow_net_type`** —— 另一个论点（1-NFE 计算量），
   且被校验器绑定到 `simple_mean`。不要混进抖动表。
5. **`vision_backbone` / `pretrained_backbone_weights` / `freeze_backbone_batchnorm`**
   —— **你自己的数据已经排除了它们**。§2.3③：去趋势残差的跨关节相关近乎为零
   （mean r = +0.029，仅 8% 的关节对 |r|>0.5）。若是共享视觉隐向量被扰动，16 个关节应当同步晃。事实不是。
6. **`flow_hidden_dim` / `flow_num_layers` / `flow_mlp_ratio` / `flow_dropout` / `flow_time_embed_dim`**
   —— 同一条排除：flow net 完全活在隐空间里。属于容量轴，不是机理轴。（作为**结构建议**另说，见 §3.5。）
7. **`latent_dim`** —— 容量轴，且在 simple 编码器下与 `horizon` 耦合，两头混淆。
8. **`resize_shape` / `crop_shape` / `crop_is_random`** —— 审计 §6.1 查明真问题是训练/部署的
   **插值核不一致**（`INTER_LINEAR` vs `INTER_AREA`），那是部署侧一行修复，不是消融。
9. **全部 optimizer / scheduler 字段** —— 把它们固定住正是各臂可比的前提。

### 2.5 最小可辩护矩阵与协议

**9 个臂**：基线 + 3 个平滑 λ + 1 个卷积头 + 2 个 flow 格 + 1 个 `l2` + 1 个 `n_obs_steps=2`，
其中第一个就是现有 100k checkpoint。

三条协议，否则表格没有意义：

1. **离线评估，不上真机。** 审计 §1 测出 **100% 的 chunk** 在执行前被换成零速 Hermite S 曲线；
   P0-1/P0-2 修好之前真机对任何两臂给出的是同一条执行轨迹。
2. **每臂固定 ~100k steps × batch 32。** §9.2 证明 100k→200k 没有收益。
   这一条把每臂成本减半，是 9 个臂能负担得起的原因。
3. **手臂与夹爪通道分开报。** 夹爪的训练目标是 `command_echo`（action ≡ state，差 0.00012），
   它自己的天花板是 lag-1 **+0.731** 而非手臂的 +0.973；合并统计会让每个臂都显得更差。
   且真机数字入表前必须先修 P1-5（夹爪 state 跑到 1.21/1.30，训练量程是 [0,1]）。

每臂指标：Δ 的 lag-1 自相关、path/net、反转率、平均 |Δ|、`|action0 − state|`。
最后一项是唯一对应断言 B 的。

---

## 3. 网络结构建议

### 3.0 参数预算：问题的中心

| 模块 | 参数量 | 推理期作用 |
|---|---|---|
| ResNet-18 主干（3 相机共享） | ~11.2M | 图像 → 512/相机 |
| `obs_encoder.projection` `Linear(1552, 512)` | 0.80M | **全部**的视觉+状态融合 |
| `SimpleFlowNet`（4 × `FlowNetLayer` + time embed） | **~12.9M** | 确定性 512 → 512 |
| `CNNActionEncoder` | ~4.2M | 仅训练期，推理不用 |
| `SimpleActionDecoder` | 2.50M | 隐向量 → 真正被执行的 256 个数 |

约 13M 参数夹在**两个隐向量之间**，而写出动作的只有 2.5M，其最后一层是单个 `Linear(512, 256)`。
再叠加 §2.3② 证明的全程逐比特确定性：**部署时 VITA 是一个很贵的固定函数 `z_img → z_act`，
外挂一个又薄又没有时间结构的读出头**。这个失衡与两条断言都对得上，是结构问题的中心。

### 3.1 S1 —— 让读出头带限，而不只是被惩罚（针对断言 A）

`output_proj` 从一个向量并联出 256 个互不相关的标量，计算图和损失都不知道其中两个在时间上相邻。
三种形式，由便宜到贵：

**(a) 固定平滑基读出头。** 输出 `K × action_dim` 个系数，乘以冻结基 `B ∈ R^{K×horizon}`（三次 B 样条或 DCT）：

```python
self.output_proj = nn.Linear(dec_hidden_dim, K * action_dim)   # K ≈ 6，不是 horizon=16
coef = self.output_proj(x).view(-1, K, self.action_dim)
return torch.einsum("bka,kt->bta", coef, self.basis)           # basis: (K, horizon)，register_buffer
```

约 10 行，一个新超参，而且让逐帧锯齿**在结构上不可能出现**，而不是被软性劝阻——
当实测病理是 lag-1 −0.080 而目标 +0.973 时，硬约束才是对的工具。
K 从训练集 chunk 的 DCT 能量谱里选（§4 的脚本已经把数据加载好了）：K 必须覆盖示教的真实频谱内容，
而训练集 path/net = 1.00、p90 1.82 说明示教在 16 步窗口内几乎单调，**K=4–6 大概率够**。
这条同时**取代**了 §2.1(1) 的 Δ² 平滑损失，且比它强。

**(b) 时间卷积解码器** —— 把隐向量 reshape 成 `(B, horizon, latent/horizon)` 再走转置 Conv1d，
与已有的 `CNNActionEncoder` 对称。可留作"学习基 vs 固定基"的对照。

**(c) query token + cross-attention 解码器**（ACT 的做法）。最标准、最贵、最难做参数对齐。

建议先跑 (a)，(b) 作为第二格。

### 3.2 S2 —— 把动作锚在当前状态上（针对断言 B）

**解码器在写动作时根本看不到 `observation.state`。** 状态是 1552 维输入里的 16 维——
**约 1% 的输入宽度**——过一个 Linear，再要熬过 6 步 ODE。
而数据里 `action[t] − state[t]` 只有 0.0203 rad，部署实测首帧偏移是它的 1.86×。
改成预测 `a_t = state + Δ_t`，锚点由构造保证精确，整个误差模式消失。

**本仓库已经为 ACT 实现过这件事**：`policies/act_delta/` 有 `use_relative_actions`、
`relative_exclude_joints=["gripper"]`，以及预闸门脚本 `prepare_relative_stats.py`。
**先跑闸门再花 GPU**——在 `express` 上中位 `std(relative)/std(absolute)` 是 **0.661**，
落在"不确定"带内，效应量是随数据集变的。两条现成教训直接适用：

- **夹爪保持绝对。** VITA 的夹爪 action **就是** state（差 0.00012），
  相对化会让该目标恒等于零——可学，但那是掩盖而不是修复，与 profile 的 `command_echo` 退化同源。
- `act_delta` 禁止相对化 + temporal ensembling，因为不同偏移锚在不同 state 上。
  VITA 每块重新取锚，本身安全；但这意味着**锚必须取自产生该隐向量的同一帧观测**，
  不能用之后再读一次的 `/state`——这一点与审计 P1-4（客户端锚实测位置、服务端锚指令位置）是同一类错误。

### 3.3 S3 —— 给状态单独的编码通路并在融合前归一化

现在是 `cat([state(16), img(1536)]) → Linear(1552, 512)`，无 LayerNorm；
而两路的归一化模式还不同（STATE=MIN_MAX，VISUAL=MEAN_STD），到达时量级就不一致。
改成 `state_proj: MLP(16→128)` + `img_proj: Linear(1536→384)`，各自 LayerNorm 后再 concat。
本身很便宜，而且它是让 S2 条件良好的前提。

### 3.4 S4 —— `AdaptiveAvgPool2d((1,1))` 是最大的信息瓶颈

7×7×512 → 512/相机，把**"在哪里"**整个丢掉。
DiffusionPolicy 用 SpatialSoftmax 关键点（`policies/diffusion` 里就有实现），
ACT 保留整张特征图当 token。对操作任务，空间精度就是信号。

**诚实的限定**：§2.3③（跨关节残差相关 ≈ 0）**已把视觉通路从抖动嫌疑里排除**，
所以这条推不动断言 A 或 B。它有望推动的是**任务成功率**——一个报告至今没有测量过的轴。
按独立贡献处理，不要算作抖动修复。

### 3.5 S5 —— 把容量从 flow net 挪出来

12.9M 无条件参数做确定性 512→512，对 2.5M 写动作。推理只要 2.8 ms，
所以**这不是速度问题，是容量放在哪里有用的问题**。
把 `flow_num_layers` 从 4 减到 2，省下的给解码器；实验很便宜，
若指标不掉，本身就是论文里一个有意思的结论：**生成式机制相对于部署真正提取到的东西是超配的**。

### 3.6 S6 —— adaLN-zero 初始化被静默禁用（免费修复）

`modeling_vita.py:500-510` 自己写了注释：`self.apply(_basic_init)` 在构造完块之后执行，
覆盖掉每个 `FlowNetLayer.time_modulator` 输出层的零初始化，**adaLN-zero 实际没有生效**。
在 `apply` 之后重新置零是一行。零成本，且应当作为免费的一行进消融表——
"与参考实现意图的偏离"正是审稿人会挖的地方。

### 3.7 建议顺序

**S6（免费）→ S1a → S2（先过 `prepare_relative_stats` 闸门）→ S3 → S5 → S4（另立项）**

其中 S1a 与 §2.1(1) 是同一件事的强化版本：若采用 S1a，
消融表里的"平滑损失 λ"三臂可以缩成"基线 / 带限头 K=4 / 带限头 K=8"。

**同一条约束覆盖全部六条**：在审计 P0-1 / P0-2 修好之前，
没有任何一条能在真机上被验证——100% 的 chunk 在执行前被换成零速 Hermite S 曲线，
所有结构变体产出的执行轨迹是同一条。一律离线评估，手臂与夹爪分开报。

---

## 4. 复现

本文所有实测都是秒级的，不需要 GPU 训练。

```bash
# §1.3 主干替换矩阵：对每个候选同时跑 ACT 路径与 Sequential[:-2] 路径
conda run -n lerobot python - <<'PY'
import torch, torchvision, torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
x = torch.zeros(1,3,224,224)
for name in ["resnet18","resnet50","resnext50_32x4d","wide_resnet50_2","resnet","vit_b_16"]:
    try:
        m = getattr(torchvision.models, name)(replace_stride_with_dilation=[False,False,False],
              weights=None, norm_layer=torchvision.ops.misc.FrozenBatchNorm2d)
        print(name, "ACT ok", m.fc.in_features,
              tuple(IntermediateLayerGetter(m, {"layer4":"feature_map"})(x)["feature_map"].shape[1:]))
    except Exception as e: print(name, "ACT FAIL", type(e).__name__)
    try:
        m2 = getattr(torchvision.models, name)(weights=None)
        print("   Seq[:-2]", tuple(nn.Sequential(*(list(m2.children())[:-2]))(x).shape[1:]))
    except Exception as e: print("   Seq[:-2] FAIL", type(e).__name__)
PY

# §3.0 参数预算：直接对着实际配置构造模型统计
#   （需要一个 dataset meta 来填 image_features / robot_state_feature，
#    最省事的做法是加载 200k checkpoint 后 sum(p.numel() for p in module.parameters())）

# §3.2 相对动作预闸门（决定 S2 值不值得跑）
conda run -n lerobot python -m lerobot.policies.act_delta.prepare_relative_stats \
  --root /mnt/robot_platform/datasets/tidy_up_stationery_le/batch_3 \
  --chunk-size 16 --dry-run
```

判据（沿用 ACT 实验计划 §2.0）：中位 `std(relative)/std(absolute)` **< 0.5** 则 S2 的机制成立；
**> 0.8** 则不成立；**0.5–0.8** 为不确定带（`express` 上测得 0.661）。
注意 `relative_exclude_joints` 依赖动作维度名里含 `gripper`，
在没有该命名的数据集上会静默退化成全维相对。

---

## 5. 与既有文档的关系

| 本文 | 修正/补充了 |
|---|---|
| §2.1 | §10.3 把"读出头 / 二阶差分平滑项"列为主线——本文指出这两项**在当前配置文件里都做不成对照实验**，需先加字段、先解锁校验器 |
| §2.4 第 1 条 | 确认 §11.5 的结论：`n_action_steps` 到此为止，不必再花训练预算 |
| §2.4 第 5 条 | 把 §2.3③ 的跨关节相关证据正式用作"视觉主干不参与消融"的依据 |
| §3.1 | 提出比 §10.3 的 Δ² 平滑损失**更强**的替代（带限读出头：结构约束 vs 软惩罚） |
| §3.2 | 为 §9 的"首帧偏移 1.86×"给出结构侧解法，并接上 `act_delta` 的现成实现与闸门脚本 |
| §3.4 | 指出一条报告从未测量的轴（任务成功率），并明确它**不**解决抖动 |
| §3.6 | 新增：flow net 的 adaLN-zero 初始化实际未生效（源码注释已记录，未纳入实验） |
