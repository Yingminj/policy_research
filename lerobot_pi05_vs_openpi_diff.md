# LeRobot π₀.₅ 复现 vs openpi 官方实现：代码差异详细对比

> 对比对象：
> - **lerobot 复现版**：`/home/kewei/YING/robot_data_platform/lerobot/src/lerobot/policies/pi05/`
>   （`modeling_pi05.py` 1304 行、`configuration_pi05.py`、`processor_pi05.py`）+ `src/lerobot/policies/pi_gemma.py`
> - **openpi 官方**：<https://github.com/Physical-Intelligence/openpi>，clone 于 `/tmp/openpi_repo`，
>   HEAD = `15a9616 update output objects to support batching (#975)`。
>   JAX 主实现 `src/openpi/models/{pi0,gemma,model,tokenizer,pi0_config}.py`，
>   官方 PyTorch 移植 `src/openpi/models_pytorch/{pi0_pytorch,gemma_pytorch,preprocessing_pytorch}.py`。
>
> 对比日期：2026-08-21（本版**修订**了同日早些时候一版的一个错误结论，见 §0.1）。
>
> 运行环境事实（已核对，影响结论）：
> - `pyproject.toml:152` 固定 `transformers>=5.4.0,<5.6.0`；实际训练环境 `/opt/robot-platform/train-venv` 装的是 **5.5.4**。
> - `/opt/robot-platform/train-venv/.../lerobot/policies/pi05/*` 与本仓库 `src/lerobot/policies/pi05/*` **逐字节相同**，无版本漂移。
> - lerobot 仓库自带一份 openpi 参考实现用于数值对齐测试：`tests/policies/pi0_pi05/openpi_pytorch/`，
>   并有 parity 测试 `tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py`（forward loss rtol/atol=1e-4，
>   sample_actions rtol=1e-2/atol=5e-3；CI 中跳过，需手动在 GPU 上跑）。

---

## 0. 结论速览（TL;DR）

模型主体（双流 Gemma、adaRMS、flow matching、注意力掩码、prompt 格式、状态离散化、归一化公式、
去噪循环、KV cache）移植得**高度忠实**——把 lerobot 侧和 lerobot 自带的 openpi 参考实现逐行 diff，
差异只有导入路径、格式化和 transformers 版本适配。真正的差异集中在**模型之外**：损失归约、
数据增强、优化器/EMA、以及若干配置面。

| 优先级 | 差异 | 影响 |
|---|---|---|
| 🔴 高 | **loss 截断到真实动作维**（openpi 对 32 维全部求均值） | 真实动作维上的梯度被放大 **32/D** 倍（D=7 时 4.6×），等价于偷偷把学习率提高 4.6 倍；照抄 openpi 的 lr 2.5e-5 就不是同一个训练配置 |
| 🔴 高 | **训练时完全没有图像增强** | openpi 两个实现在 `train=True` 都做 RandomCrop95%+Rotate±5°+ColorJitter(0.3/0.4/0.5)；lerobot 一个都没有，且 lerobot 数据集级 `ImageTransformsConfig` 默认 `enable=False` 且内容不等价 |
| 🟠 中高 | **无 EMA** | openpi `TrainConfig.ema_decay` 默认 0.99（`pi05_libero` 用 0.999），发布的 checkpoint 是 EMA 权重 |
| 🟠 中高 | **`n_action_steps` 默认 = `chunk_size` = 50** | openpi 参考 runtime 只执行 8（droid）/10（aloha_sim）/25（aloha_real）步就重新推理；50 步开环在 30Hz 下是 1.7 秒盲走 |
| 🟠 中 | **分位数归一化的 eps 语义不同** | openpi 恒加 `1e-6` 到分母；lerobot 只在 `q99==q01` 时才替换。**近似恒定但不完全相等**的关节（锁死轴、几乎不动的夹爪）在 lerobot 侧会被放大到几百甚至上万，离散化状态直接饱和、动作目标爆炸 |
| 🟠 中 | weight decay 0.01 vs 1e-10（≈0） | 相差 8 个数量级 |
| 🟡 低 | 相机顺序/数量、缺相机策略、`discrete_state_input` 无开关、bf16 fp32 保留范围、adaRMS bias 初始化、TF32 默认、scheduler 余弦分母 | 见 §3 |

### 0.1 修订说明：`√2048` 语言 embedding 缩放**不是**差异

同日早些时候的一版文档把「lerobot 的 `embed_language_tokens` 少乘 `√hidden_size`」列为头号问题。
**这个结论是错的**，现予撤回。事实如下：

- openpi JAX `gemma.py:150` 里 `Embedder.encode` 显式 `x *= sqrt(embed_dim)`；
  openpi PyTorch `pi0_pytorch.py:214-217` 也显式乘 `math.sqrt(lang_emb_dim)`，
  并把 patch 版 transformers 里 `GemmaModel.forward` 的 `hidden_states * normalizer` 注释掉
  （`transformers_replace/models/gemma/modeling_gemma.py:515-516`）——因为 openpi 钉的是 **transformers 4.53.2**，
  那个版本 `embed_tokens` 是裸 `nn.Embedding`，缩放在 model.forward 里。
- lerobot 钉的是 **transformers ≥5.4**。在 5.5.4 里（`transformers/models/gemma/modeling_gemma.py:55-61, 381-383`）：

  ```python
  class GemmaTextScaledWordEmbedding(nn.Embedding):
      def forward(self, input_ids):
          return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
  ...
  self.embed_tokens = GemmaTextScaledWordEmbedding(
      vocab_size, hidden_size, padding_idx, embed_scale=self.config.hidden_size**0.5)
  ```

  即 `get_input_embeddings()(tokens)` **本身已经乘了 √2048**。lerobot 的
  `modeling_pi05.py:457-458` 直接返回它、并在 `pi_gemma.py:273-283` 删掉 normalizer 乘法，
  是**恰好缩放一次**的正确写法。lerobot 自带的 openpi 参考实现里也写明了这一点
  （`tests/.../openpi_pytorch/pi0_pytorch.py:207-213` 的注释 + HF PR #44432 链接），
  正是靠这一改动 parity 测试才能对齐到 1e-4。
- 图像侧同理：transformers 5.4+ 的 `PaliGemmaModel.get_image_features`
  （`modeling_paligemma.py:249-257`）不再除以 `√hidden_size`，与 openpi patch 后的语义一致。

**但这里留了一个真实的脆弱点（🟡）**：这份代码的正确性**依赖 transformers ≥5.4**。
本机 `~/anaconda3` base 环境是 **4.53.1**，`envs/auto_annotation`、`envs/locany` 是 4.57.1。
一旦在 <5.4 的环境里 import 这份 pi05，`embed_tokens` 退回裸 `nn.Embedding`，
而 lerobot 的 `PiGemmaModel.forward` 又绕过了 HF 的 normalizer——**√2048 会静默消失**，
语言条件被衰减 45 倍，且不会报任何错。跑之前请务必确认用的是 `train-venv`（5.5.4），
或者加一行断言。

---

## 1. 已确认一致的部分（移植忠实的核心）

逐项对过代码，以下**语义等价**：

| 模块 | openpi | lerobot | 说明 |
|---|---|---|---|
| Flow matching 目标 | `pi0.py:199-200` | `modeling_pi05.py:744-746` | `x_t = t·noise+(1−t)·a`，`u_t = noise−a`，MSE |
| 时间采样 | Beta(1.5,1.0)×0.999+0.001（`pi0.py:197`） | `configuration_pi05.py:45-48` 同参数 | 一致 |
| 时间嵌入 | `posemb_sincos`，min 4e-3 / max 4.0 | `create_sinusoidal_pos_embedding` | 一致 |
| π₀.₅ time MLP | `Linear→swish→Linear→swish → adarms_cond`（`pi0.py:162-168`） | `modeling_pi05.py:717-723`（SiLU≡swish） | 一致 |
| adaRMS | `gemma.py:112-129`：zero-init Dense→3·dim，`normed·(1+scale)+shift`，gated residual | `pi_gemma.py:84-128` + `_gated_residual` | 结构一致（仅 bias 初始化，见 §3.6） |
| 双 expert 注意力 | 每层拼 Q/K/V 联合注意力，o_proj 分开 | `compute_layer_complete`（`modeling_pi05.py:471-...`） | 一致 |
| 注意力掩码 | `make_attn_mask` cumsum 语义 + big_neg −2.3819763e38 | `make_att_2d_masks` + `OPENPI_ATTENTION_MASK_VALUE` | 一致；前缀双向、后缀因果 |
| position_ids | `cumsum(pad_mask)−1`；推理 `prefix_offsets + cumsum(suffix)−1` | 同 | 一致 |
| **π₀.₅ 无 state_proj** | `pi0.py:151-157` 仅 `if not pi05` 才加 state token | `embed_suffix(self, noisy_actions, timestep)` 无 state 入参 | 一致 |
| prompt 格式 | `Task: {text}, State: {256 bins};\nAction: `，BOS，pad id 0，max_len 200（`tokenizer.py:22-48`） | `processor_pi05.py:83` + HF paligemma tokenizer | 一致 |
| 状态离散化 | `np.digitize(state, linspace(-1,1,257)[:-1]) − 1`，**在 pad 到 32 维之前**（`config.py:127-135` TokenizePrompt 在 PadStatesAndActions 之前） | `processor_pi05.py:77`，同样用原始维度 | 一致 |
| 分位数归一化公式 | `(x−q01)/(q99−q01+1e-6)·2−1`（`transforms.py:139-144`） | `normalize_processor.py:394-401` | 公式一致，**eps 语义不同，见 §2.4** |
| 归一化次序 | `data_transforms → Normalize → model_transforms(tokenize)`（`data_loader.py:187-189`） | `AddBatchDim → relative → Normalizer → Pi05PrepareStateTokenizer → Tokenizer` | 一致（都是先归一化再离散化） |
| `resize_with_pad` | `image_tools.py:55-125`，`[-1,1]` 输入、clamp(−1,1)、pad 值 −1.0 | `modeling_pi05.py:162-230`，`[0,1]` 输入、clamp(0,1)、pad 值 0.0，之后 `*2−1` | **等价**（两边补边都是纯黑）；注释里 "exact copy" 说法不准，但语义对 |
| 动作 padding | `PadStatesAndActions(32)` | `pad_vector(..., max_action_dim=32)` | 一致 |
| LR schedule | warmup 1000 / peak 2.5e-5 / decay 30k / end 2.5e-6（`optimizer.py:16-31`） | `configuration_pi05.py:99-102` 同值 | 值一致（余弦分母细节见 §3.7） |
| Adam 超参 | b1 0.9 / b2 0.95 / eps 1e-8 / clip 1.0 | `configuration_pi05.py:92-95` 同值 | 一致 |
| chunk / 推理步数 | `action_horizon=50`，`num_steps=10` | `chunk_size=50`，`num_inference_steps=10` | 一致 |
| 去噪循环 | `while time >= -dt/2`，`time += dt` 累加 | `for step in range(num_steps): time = 1.0 + step·dt` | 数学等价（lerobot 数值更干净） |
| KV cache 推理 | prefix prefill 一次，suffix 每步复用 | 同 + 每步 `clone_past_key_values` 防原地写污染（openpi 参考实现用 `copy.deepcopy`） | 功能等价 |
| 语言/图像 embedding 缩放 | 显式 ×√2048 / 图像不缩放 | 由 transformers ≥5.4 的 `GemmaTextScaledWordEmbedding` 承担 | **等价**，见 §0.1 |

---

## 2. 关键差异详解

### 2.1 🔴 loss 对 padding 动作维的处理：梯度尺度差 32/D 倍

openpi（两个实现都是）对 **pad 到 32 维之后的全部 32 维**求均值：

```python
# openpi JAX —— pi0.py:214
return jnp.mean(jnp.square(v_t - u_t), axis=-1)          # 对 32 维求均值

# openpi PyTorch —— pi0_pytorch.py:374 + scripts/train_pytorch.py:536
return F.mse_loss(u_t, v_t, reduction="none")            # (B, H, 32)
loss = losses.mean()                                     # 除以 B·H·32
```

lerobot 先算 32 维，再**截断到真实动作维**再求均值：

```python
# lerobot —— modeling_pi05.py:1277-1278, 1285
original_action_dim = self.config.output_features[ACTION].shape[0]
losses = losses[:, :, :original_action_dim]              # (B, H, D)
...
loss = losses.mean()                                     # 除以 B·H·D
```

**这不只是"loss 数值不可比"。** 两边真实动作维上的梯度差一个常数因子：

```
∂L_lerobot/∂v_d  =  (32/D) · ∂L_openpi/∂v_d      (d ≤ D)
```

D=7（6 关节 + 夹爪）时是 **4.57×**，D=14（双臂）时是 **2.29×**。
损失是网络唯一的梯度源，所以这等价于把**全网学习率乘以 32/D**。
照抄 openpi 的 `peak_lr=2.5e-5`，lerobot 上的有效学习率其实是 ≈1.14e-4。
对 3B 参数的 VLA 微调，这个量级足以让训练从"收敛"滑向"抖动/塌到均值动作"。

顺带说明 openpi 那 25 个 padding 维在训练什么：`actions` 补零后 `u_t = noise − 0 = noise`，
`x_t = t·noise`，所以模型要在 padding 维上学 `v_t = x_t / t`——一个确定可学的恒等映射。
浪费一点容量，但不是噪声。lerobot 的做法本身**更合理**，问题只在于**没有同步调整 lr**。

**建议**：要么把 lr 除以 `32/D`（D=7 → 5.5e-6），要么改成
`losses[:, :, :D].sum(-1) / 32` 以完全对齐 openpi 的尺度，二选一，别两个都不做。

### 2.2 🔴 训练时完全没有图像增强

openpi 在 `train=True` 时对图像做增强，JAX 和 PyTorch 两条路都实现了：

```python
# openpi JAX —— models/model.py:168-187
if train:
    image = image / 2.0 + 0.5
    transforms = []
    if "wrist" not in key:                       # 非腕部相机才做几何增强
        transforms += [augmax.RandomCrop(int(w*0.95), int(h*0.95)),
                       augmax.Resize(w, h),
                       augmax.Rotate((-5, 5))]
    transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
    image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
    image = image * 2.0 - 1.0
```

PyTorch 版 `preprocessing_pytorch.py:52-142` 是等价手写实现
（brightness ×[0.7,1.3]、contrast ×[0.6,1.4]、saturation ×[0.5,1.5]、crop 95%、rotate ±5°）。

lerobot 的 `_preprocess_images`（`modeling_pi05.py:1149-1213`）**训练和推理走同一条路**，
只有 `resize_with_pad` + `[0,1]→[-1,1]`，**没有任何增强**，连 `train` 参数都没有。

lerobot 确实有数据集级增强 `ImageTransformsConfig`（`src/lerobot/transforms/transforms.py:166-...`），
但：(a) **默认 `enable=False`**；(b) 就算打开也**不等价**——只有 ColorJitter/Sharpness 之类，
**没有 RandomCrop、没有 Rotate**，且 jitter 幅度更窄（brightness `(0.8,1.2)` vs openpi 的 0.3 即 `[0.7,1.3]`），
还是"从若干变换里随机抽最多 3 个"而不是全部串联。

**影响**：π₀.₅ 公布的所有结果都是带增强训练的。少了 RandomCrop+Rotate，
策略对相机位姿的微小变化会非常敏感——这正好解释"实验室里录的数据训出来，
机器人一开机相机被碰歪 2 度就不行了"这类现象；也与本项目已记录的
[ACT 依赖背景像素] 观察同源（模型抓的是场景快捷方式而不是物体）。

### 2.3 🟠 无 EMA

openpi `TrainConfig.ema_decay` 默认 **0.99**（`training/config.py:490`），
`pi05_libero` 甚至用 **0.999**（`config.py:759`），且**官方发布的 checkpoint 存的是 EMA 权重**。
（只有 LoRA 低显存配置显式设 `ema_decay=None`。）

lerobot pi05 没有任何 EMA 逻辑，`get_optim_params()` 直接返回 `self.parameters()`
（`modeling_pi05.py:1123-1124`），训练脚本里也搜不到 ema。

**影响**：flow matching 末期 loss 仍在抖，直接评最后一版在线权重通常比 EMA 权重差。
和 openpi 公布数字对比时这不是对等比较。

### 2.4 🟠 分位数归一化 eps 语义不同——近似恒定的关节会爆炸

```python
# openpi —— transforms.py:141-144
q01, q99 = stats.q01[..., :x.shape[-1]], stats.q99[..., :x.shape[-1]]
return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0        # 恒加 1e-6

# lerobot —— normalize_processor.py:394-401
denom = q99 - q01
denom = torch.where(denom == 0, torch.tensor(self.eps), denom)   # 仅在恰好为 0 时替换
return 2.0 * (tensor - q01) / denom - 1.0
```

差别只在"几乎恒定"的维度上暴露，但那正是机器人数据里最常见的情形：
某个被锁死的关节、整个数据集里几乎不动的夹爪、或某个恒定的冗余通道。

设某维 `q99 − q01 = 1e-5`（不是 0，所以 lerobot 不触发替换）：
- openpi：分母 `1e-5 + 1e-6 ≈ 1.1e-5`，缩放 ≈ 1.8e5，但至少被 eps 兜住了上界（最坏 2e6）。
- lerobot：分母 `1e-5`，缩放 2e5；若 `q99−q01 = 1e-9`，lerobot 直接放大 2e9 倍，openpi 仍被钳在 2e6。

后果有两处，**都发生在 π₀.₅ 特有的路径上**：
1. 归一化后的 state 远超 `[-1,1]` → `np.digitize` 饱和成 `-1` 或 `255`
   （注意 `digitize−1` 在下溢时会给出 **−1**，prompt 里就是字符串 `"-1"`），
   语言 prompt 里那一维永远是同一个极值 token；
2. 归一化后的 action target 变成几百上千 → `u_t = noise − a` 被这一维主导 → MSE 被单维吞掉，
   其他维梯度相对消失。

**建议**：转数据集时检查 `q99 − q01` 的最小值；对 `< 1e-3` 的维度要么加 openpi 那个 `+1e-6`，
要么直接从 `action`/`state` 里剔除。

### 2.5 🟠 `n_action_steps` 默认 = 50，全 chunk 开环

`configuration_pi05.py:38-39`：`chunk_size = 50`，`n_action_steps = 50`。
`select_action` 一次推理后把 50 步全塞进队列逐个吐出（`modeling_pi05.py:1223-1232`）。

openpi 的参考 runtime 从不这么用：

| runtime | 文件 | 每次执行步数 |
|---|---|---|
| droid | `examples/droid/main.py:42` | `open_loop_horizon = 8` |
| aloha_sim | `examples/aloha_sim/main.py:21` | `action_horizon = 10` |
| aloha_real | `examples/aloha_real/main.py:18` | `action_horizon = 25` |

而且 `pi05_droid` 训练时 `action_horizon=15`、`pi05_libero` 是 `10`
（`training/config.py:630, 745`），根本没训 50 步的 chunk。

**影响**：30Hz 下 50 步 = 1.67 秒完全不看新观测。任何接触、滑动、抓空都无法纠正。
这是真机上"看着像会做但总在最后一下失手"的经典成因。
lerobot 里有 RTC（Real-Time Chunking）可以缓解，但 `rtc_config` 默认是 `None`（关闭）。

**建议**：先把 `n_action_steps` 降到 `8~15` 试一轮，这是零代码成本的一个变量。

### 2.6 🟠 weight decay：0.01 vs 1e-10

- openpi `AdamW.weight_decay = 1e-10`（`training/optimizer.py:73`），注释明说
  「设成 0 会莫名 OOM，所以取一个可忽略值」，并且通过 `weight_decay_mask` 只作用于部分参数。
- lerobot `optimizer_weight_decay = 0.01`（`configuration_pi05.py:94`），
  且 `get_optim_params()` 返回全部参数 → decay 会作用到 RMSNorm 的 `weight`、所有 bias、embedding 上。

相差 8 个数量级。对 3B 模型微调，0.01 的 decoupled decay 会持续把预训练权重往 0 拉。
配合 §2.1 那 4.6× 的有效 lr，两个偏差叠在一起离 openpi 的训练轨迹相当远。

**建议**：对齐成 0（或 1e-10）。

---

## 3. 其余差异（中低危）

### 3.1 🟡 相机顺序与数量

openpi 固定三路、**固定顺序**（`preprocessing_pytorch.py:11-15` / `models/model.py:149`）：

```python
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
```

lerobot 按 `config.image_features` 的字典顺序拼接图像 token（`modeling_pi05.py:1161`）。
π₀.₅ 预训练时相机槽位语义是固定的（第 1 组 256 个 token = 外部相机，第 2/3 组 = 左/右腕）。
**若数据集里 base 和 wrist 顺序颠倒，注意力会对错相机**，效果明显下降。
这是配置问题不是代码 bug，但极其常见——本项目的四宫格 mosaic 转换尤其要当心 tile 顺序。

### 3.2 🟡 缺相机的处理策略

- openpi：三路必须齐全，缺就 `raise ValueError`（`preprocessing_pytorch.py:31-32`）。
- lerobot：缺的相机补 `-1` 图像 + `mask=0`（`modeling_pi05.py:1206-1211`）。

`mask=0` 会让那 256 个 token 被 pad mask 剔出注意力，语义上是干净的。
但它把 openpi 的一个**硬报错**变成了**静默降级**——只用一路相机去微调三相机预训练模型，
不会有任何警告。用之前确认一下相机确实是有意省略的。

### 3.3 🟡 `discrete_state_input` 没有开关

openpi 的 `Pi0Config.discrete_state_input` 默认跟随 `pi05`（`pi0_config.py:40-41`），
但**官方自己的 `pi05_libero` 显式设成 `False`**（`training/config.py:745`）。
`pi05=True` + `discrete_state_input=False` 时（`pi0.py:151` 的 `if not self.pi05` 分支不走，
prompt 里也不带 state）→ **模型完全看不到本体状态**，纯视觉+语言。

lerobot 的 `Pi05PrepareStateTokenizerProcessorStep` 无条件把离散状态写进 prompt，没有开关。
这不算 bug（那是 π₀.₅ 的标准形态），但意味着**无法复现 openpi 的 libero 数字**，
也无法做"去掉 state 看是否更泛化"的消融。考虑到本项目已记录的
[夹爪状态尺度不匹配]（部署时状态比训练分布宽约 1.25×），这条消融其实很有价值：
状态一旦 OOD，把它写进 prompt 反而是负担。

### 3.4 🟡 默认精度与 fp32 保留范围

- openpi 默认 `dtype="bfloat16"`（`pi0_config.py:20`）；lerobot 默认 `"float32"`（`configuration_pi05.py:33`）。
- bf16 模式下保留 fp32 的参数集不同：
  - openpi（`gemma_pytorch.py:71-78`）：只留 patch_embedding 的 weight/bias、position_embedding，加所有 layernorm。
  - lerobot（`modeling_pi05.py:417-423`）：留**整个 `vision_tower` + `multi_modal_projector`** 加 layernorm，
    注释说明是为了避免 dtype 来回切换触发 optimizer 的 "same dtype" 报错。
    lerobot 自带的 openpi 参考实现也做了同样的放宽，所以 parity 测试对得上。

数值上 lerobot 更精确，本身不会变差，但显存/算力开销更大，训练数值轨迹也和 openpi 不同。

### 3.5 🟡 TF32 默认不同

openpi `pi0_pytorch.py:111` 在 `__init__` 里**无条件** `torch.set_float32_matmul_precision("high")`；
lerobot 只在 `compile_model=True` 时才设（`modeling_pi05.py:600`）。
默认 `float32` + 默认 `compile_model=False` 时，lerobot 走的是完整 fp32 matmul，
在 Ampere+ 上比 openpi 慢一大截。纯速度差异，不影响精度。

### 3.6 🟡 adaRMS Dense 的 bias 初始化

- openpi：flax `nn.Dense(kernel_init=zeros)`，flax 的 `bias_init` 默认也是 `zeros` → 起点严格等于普通 RMSNorm，
  `gate=0` 使每层残差分支在第 0 步完全静默（标准 DiT zero-init）。
- lerobot：`pi_gemma.py:98-99` 只 `nn.init.zeros_(self.dense.weight)`，
  **bias 是 `nn.Linear` 默认的 U(−1/√cond_dim, 1/√cond_dim)**（cond_dim=1024 时约 ±0.031）。

加载预训练权重时 bias 会被覆盖 → 无影响。**从头训练**时第 0 步 scale/shift/gate 非零，引入轻微扰动。

### 3.7 🟡 余弦衰减的分母

optax 的 `warmup_cosine_decay_schedule` 把 `decay_steps` 当**总步数**，余弦段实际跨
`decay_steps − warmup_steps = 29000` 步且从 warmup 结束处起算。
lerobot（`optim/schedulers.py:170-175`）用 `cos(π · current_step / actual_decay_steps)`，
分母是 30000 且分子是**绝对步数**（含 warmup）。差异很小（末端 lr 略高一点点），
但两条曲线不完全重合。另外 lerobot 会在 `steps < decay_steps` 时按比例自动缩放 warmup/decay
（`schedulers.py:149-153`），openpi 不会——openpi 的配置本来就是照 30k 设死的。

### 3.8 🟡 默认 batch size

- openpi：`TrainConfig.batch_size = 32`（`config.py:506`），`pi05_libero` 用 **256**。
- lerobot：`TrainPipelineConfig.batch_size = 8`。

batch 8 + lr 2.5e-5 和 batch 256 + lr 5e-5 不是同一个训练配置。
再叠上 §2.1 的 32/D 梯度放大，"照抄默认参"离 openpi 已经很远了。

### 3.9 ⚪ lerobot 额外增加、openpi 没有的功能（不影响对齐）

- **RTC**（Real-Time Chunking）推理，`rtc_config` 默认 `None`（关闭）。
- **`use_relative_actions`**（默认 `False`）：对应 openpi 的 `DeltaActions`，
  openpi 的 `LeRobotLiberoDataConfig.extra_delta_transform` 在不同配置里取值不同（`pi05_libero` 是 `False`）。
- `freeze_vision_encoder` / `train_expert_only` 布尔开关：openpi 用 `get_freeze_filter()` 的
  PathRegex + LoRA 表达（`pi0_config.py:88-117`），两者覆盖的场景不完全重合，lerobot 没有 LoRA 变体的 gemma。
- RA-BC 用的 per-sample loss（`reduction="none"`，`modeling_pi05.py:1281-1287`）。

---

## 4. 「效果不佳」的可能原因排序与验证建议

按预期影响排序，并给出**能单独验证**的动作：

| # | 怀疑项 | 验证/修复动作 | 成本 |
|---|---|---|---|
| 1 | **有效 lr 被放大 32/D 倍**（§2.1） | 把 `optimizer_lr` 从 2.5e-5 改成 `2.5e-5 · D/32`（D=7 → 5.5e-6）重训一轮对比；或改 loss 归约 | 一次训练 |
| 2 | **50 步全开环**（§2.5） | `--policy.n_action_steps=10`，**不用重训**，直接重评 | 零成本，先做这个 |
| 3 | **无图像增强**（§2.2） | 打开 `--dataset.image_transforms.enable=true`，或在 `_preprocess_images` 里按 openpi 加 crop95%+rotate±5°+jitter(0.3/0.4/0.5)（仅 `self.training` 时） | 一次训练 |
| 4 | **近恒定维被放大**（§2.4） | 直接查数据集 stats：打印每维 `q99−q01`，看有没有 `<1e-3` 的 | 五分钟，先做这个 |
| 5 | **相机顺序错配**（§3.1） | 打印 `config.image_features` 的顺序，确认是 base → left_wrist → right_wrist | 五分钟，先做这个 |
| 6 | **transformers 版本**（§0.1） | `python -c "import transformers;print(transformers.__version__)"`，必须 ≥5.4 | 一分钟，先做这个 |
| 7 | 无 EMA（§2.3） | 训练循环外挂 EMA(0.99)，评 EMA 权重 | 少量代码 |
| 8 | weight decay 0.01（§2.6） | 改成 0 | 配置 |
| 9 | batch size 8 vs 32/256（§3.8） | 对齐 batch 或按线性缩放调 lr | 显存决定 |
| 10 | 精度路径 / TF32（§3.4, §3.5） | 影响最小，最后再考虑 | — |

**#2 / #4 / #5 / #6 都是零成本的检查，建议先跑完这四项再动训练。**

## 5. 最小修复清单（对齐 openpi）

| # | 位置 | 改动 |
|---|---|---|
| 1 | `configuration_pi05.py:92` 或 `modeling_pi05.py:1277-1285` | 补上 32/D 的尺度：lr 乘 `D/32`，或把 loss 改成 `losses[:,:,:D].sum(-1).mean()/32` |
| 2 | 运行配置 | `n_action_steps` 50 → 8~15（或启用 RTC） |
| 3 | `_preprocess_images`（仅 `self.training` 分支） | 加 openpi 等价增强：非 wrist 相机 RandomCrop 95% + Resize + Rotate(±5°)；全相机 ColorJitter(0.3/0.4/0.5) |
| 4 | 数据转换 | 检查并处理 `q99−q01 < 1e-3` 的维度（剔除或加 `+1e-6`） |
| 5 | `configuration_pi05.py:94` | `optimizer_weight_decay` 0.01 → 0 |
| 6 | 训练循环 | 加 EMA(decay 0.99)，评测/导出用 EMA 权重 |
| 7 | 环境 | 断言 `transformers >= 5.4`，否则 √2048 静默丢失 |
| 8 | 数据/运行配置 | 确认相机顺序 = base → left_wrist → right_wrist |

---

### 附：对比所用的关键文件

- **lerobot**：`src/lerobot/policies/pi05/{modeling_pi05,configuration_pi05,processor_pi05}.py`、
  `src/lerobot/policies/pi_gemma.py`、`src/lerobot/processor/normalize_processor.py`、
  `src/lerobot/optim/schedulers.py`、`src/lerobot/transforms/transforms.py`、
  `tests/policies/pi0_pi05/{test_pi05_original_vs_lerobot.py,openpi_pytorch/*}`
- **openpi**（`/tmp/openpi_repo` @ `15a9616`）：`src/openpi/models/{pi0,pi0_config,gemma,tokenizer,model}.py`、
  `src/openpi/models_pytorch/{pi0_pytorch,gemma_pytorch,preprocessing_pytorch}.py`、
  `src/openpi/shared/image_tools.py`、`src/openpi/transforms.py`、
  `src/openpi/training/{config,optimizer,data_loader}.py`、`scripts/train_pytorch.py`、`examples/*/main.py`
- **transformers 5.5.4**（`/opt/robot-platform/train-venv`）：
  `models/gemma/modeling_gemma.py:55-61,381-383`、`models/paligemma/modeling_paligemma.py:249-257`
