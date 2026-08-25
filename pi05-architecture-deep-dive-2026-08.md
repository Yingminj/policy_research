# π₀.₅ 架构深度拆解：模块、张量形状、公式推导与代码对照

> 代码基准：本仓库 `/home/kewei/YING/openpi_yw`（openpi fork，分支 `hhw-dev`，HEAD `14e1e47`）
> 主实现 `src/openpi/models/{pi0,pi0_config,gemma,siglip,model,tokenizer}.py`
> PyTorch 移植 `src/openpi/models_pytorch/{pi0_pytorch,gemma_pytorch,preprocessing_pytorch}.py`
>
> 论文基准：
> - **π₀.₅**：*π₀.₅: a Vision-Language-Action Model with Open-World Generalization*，arXiv:2504.16054
> - **π₀**：*π₀: A Vision-Language-Action Flow Model for Robot Control*，arXiv:2410.24164（架构母本、flow matching 公式来源）
> - **KI**：*Knowledge Insulation for Vision-Language-Action Models*，arXiv:2505.23705（openpi 发布的 `pi05_base` 实际训练配方）
>
> 姊妹文档：`lerobot_pi05_vs_openpi_diff.md`（lerobot 复现 vs openpi 的逐行差异）。本文不重复那份，
> 本文回答的是「π₀.₅ 到底是怎么算的」，那份回答的是「两份代码哪里不一样」。
>
> 撰写日期：2026-08-24

---

## 0. 先把三个「π₀.₅」分清楚

这是读这套代码最容易踩的坑：**「π₀.₅」在三个语境下指三个不同的东西**。

| 语境 | 指什么 | 动作头 | 训练目标 |
|---|---|---|---|
| **π₀.₅ 论文**（2504.16054） | 完整系统 = 预训练 + 后训练 + 分层推理 | 预训练阶段是 **FAST 离散 token**（自回归）；后训练阶段**加挂** flow matching action expert | 预训练：纯交叉熵；后训练：交叉熵 + α·flow L2，α=10 |
| **KI 论文**（2505.23705） | 一种把 action expert 和 VLM 主干**梯度隔离**的联合训练配方 | flow matching（推理用）+ FAST token（只在训练时作表征学习信号） | `L_CO-VLA = E[−Σ M^ℓ_j log p_θ + α·M^act‖ω−a−f^a‖²]`，且 **stop-gradient** 切断 expert→backbone |
| **openpi 代码**（本仓库） | **只有 flow matching 头**。README 明说：*"we currently only support the flow matching head for both π₀.₅ training and inference"* | flow matching | 只有 flow L2，**没有任何交叉熵项、没有 FAST token、没有 stop-gradient、没有 high-level 子任务预测** |

也就是说：**你在本仓库里 finetune 的 `pi05_base`，是一个已经被 KI 配方训练好的 checkpoint，
但你的 finetune 过程本身退化成了纯 flow matching 回归。** 论文里的「co-training / 知识隔离 /
分层推理」这三件事，代码里一件都没有。这不是 bug，是 openpi 的发布范围。

后文 §1–§8 讲代码里**真实存在**的那条计算路径；§9 讲论文里有、代码里没有的部分；§10 讲这个 fork 自己的坑；
§11–§12 是训练实务：一个真实训练脚本的逐项拆解，以及「反向传播时参数到底动多少」的定量计算。

---

## 1. 模块清单与参数量

`Pi0.__init__`（`src/openpi/models/pi0.py:66-104`）构造的全部模块：

```
Pi0
├── PaliGemma
│   ├── img  : siglip.Module(variant="So400m/14", pool_type="none", num_classes=2048)
│   └── llm  : gemma.Module(configs=[gemma_2b, gemma_300m], adarms=pi05)
│        ├── embedder      (257152 × 2048)              # 只有 expert 0 有
│        ├── layers        (scan over depth=18)
│        │    ├── pre_attention_norm / pre_attention_norm_1
│        │    ├── attn     (qkv for expert0 / expert1，共享 softmax)
│        │    ├── pre_ffw_norm / pre_ffw_norm_1
│        │    └── mlp / mlp_1
│        └── final_norm / final_norm_1
├── action_in_proj   : Linear(action_dim=32 → 1024)
├── action_out_proj  : Linear(1024 → action_dim=32)
└── [pi05 分支] time_mlp_in / time_mlp_out : Linear(1024 → 1024) ×2
    [pi0  分支] state_proj : Linear(32 → 1024)
                action_time_mlp_in : Linear(2048 → 1024)
                action_time_mlp_out: Linear(1024 → 1024)
```

### 1.1 尺寸表（`gemma.get_config` + `siglip.decode_variant`）

| 组件 | width | depth | mlp_dim | num_heads | num_kv_heads | head_dim |
|---|---|---|---|---|---|---|
| SigLIP So400m/14 | 1152 | 27 | 4304 | 16 | — (MHA) | 72 |
| Gemma 2B（expert 0，VLM 主干） | 2048 | 18 | 16384 | 8 | **1** | 256 |
| Gemma 300M（expert 1，action expert） | 1024 | 18 | 4096 | 8 | **1** | 256 |

两个 expert 的 `depth` / `num_heads` / `num_kv_heads` / `head_dim` **必须相同**
（`gemma.py:158-161` 与 `Module.setup` 里有 assert），因为它们要在**同一个 softmax 里**做注意力。
只有 `width` 和 `mlp_dim` 不同 —— 这就是「MoE by token」的全部含义。

`num_kv_heads=1` 意味着这其实是 **MQA（Multi-Query Attention）**，不是常见的 GQA。
后果：KV cache 极小。968 个 prefix token 的 cache 只有
`2(k,v) × 968 × 1 × 256 × 18层 × 2字节(bf16) ≈ 17.8 MB / 样本`，
这也是 π₀.₅ 能在机器人上跑 10 步去噪还不爆显存的原因之一。

### 1.2 参数量（按 config 手算，可与论文口径对照）

单层 Gemma 2B：
- attn: `q_einsum (8,2048,256)=4.19M` + `kv_einsum (2,1,2048,256)=1.05M` + `attn_vec (8,256,2048)=4.19M` = **9.44M**
- mlp（GeGLU）: `gating (2,2048,16384)=67.1M` + `linear (16384,2048)=33.6M` = **100.7M**
- 合计 ≈ 110M/层 × 18 = **1.98B**，加 embedder 526M ⇒ **≈ 2.51B**

单层 Gemma 300M：
- attn: `2.10M + 0.52M + 2.10M = 4.72M`
- mlp: `8.39M + 4.19M = 12.58M`
- 合计 17.3M/层 × 18 = **311M**（代码注释 `# 311M params` 对得上）

SigLIP So400m/14：`(4×1152² + 2×1152×4304) × 27 + patch_conv 0.68M + posemb 0.30M + head 2.36M ≈ 414M`

**π₀.₅ 独有的 adaRMS 开销**（容易被忽略）：每个 adaRMS 的调制层是
`nn.Dense(width×3)`，对 action expert 是 `1024 → 3072`，即 `3.146M + 3072` 参数。
每层有两个（pre_attention_norm_1、pre_ffw_norm_1），加上 final_norm_1：

```
18 层 × 2 × 3.15M + 3.15M ≈ 116M
```

即 **adaRMS 让 action expert 从 311M 涨到 ≈ 427M（+37%）**。论文口径的「2B backbone + 300M expert = 2.3B」
没有把这块算进去，实际 checkpoint 里它是实打实存在的。总参数 ≈ 414M + 2.51B + 427M ≈ **3.35B**。

---

## 2. 输入侧：从 LeRobot 原始帧到模型 token

三组 transform 依次作用（`DataConfig`，`src/openpi/training/config.py:69-104`）：

```
raw LeRobot dict
   │
   ├─(1) repack_transforms      重命名 key → image / *_wrist_image / state / actions / prompt
   │
   ├─(2) data_transforms        <Robot>Inputs（维度布局）
   │        └─ DeltaActions（可选，把绝对动作变成相对 state 的增量）
   │
   ├─── Normalize               ← 归一化在这里，夹在 (2) 和 (3) 中间
   │
   └─(3) model_transforms       InjectDefaultPrompt → ResizeImages
                                → TokenizePrompt(discrete_state_input=?)
                                → PadStatesAndActions(32)
```

### 2.1 归一化：π₀.₅ 走分位数，π₀ 走 z-score

`config.py:197`：

```python
use_quantile_norm = (model_config.model_type != ModelType.PI0)
```

所以 **PI05 和 PI0_FAST 用分位数归一化，只有 PI0 用 z-score**。这不是随手选的，见 §2.2。

z-score（`transforms.py:137-139`）：

$$\tilde{x} = \frac{x - \mu}{\sigma + 10^{-6}}$$

分位数（`transforms.py:141-145`）：

$$\tilde{x} = 2\cdot\frac{x - q_{01}}{q_{99} - q_{01} + 10^{-6}} - 1$$

反归一化（`transforms.py:175-182`）是严格的逆运算，且对**超出 norm stats 维度的尾部维度原样透传**：

$$x = \frac{\tilde{x}+1}{2}\,(q_{99}-q_{01}+10^{-6}) + q_{01}$$

分位数归一化把训练数据的 1%–99% 分位映射到 **[-1, 1]**，尾部允许超出。
`1e-6` 是**恒加**到分母上的（不是「仅当 q99==q01 时才替换」），这一点在
`lerobot_pi05_vs_openpi_diff.md §3` 里已经指出过 lerobot 侧的差异。

### 2.2 状态离散化：π₀.₅ 与 π₀ 的第一处结构性分歧

π₀ 把 state 作为**连续向量**经 `state_proj` 变成 suffix 里的一个 token。
π₀.₅ 把 state **写进语言 prompt**，作为离散文本 token。论文原话：
*"The robot proprioceptive state is discretized and input to the model as text tokens."*

代码 `tokenizer.py:26-32`：

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
tokens = self._tokenizer.encode(full_prompt, add_bos=True)
```

**分箱公式展开**：`np.linspace(-1, 1, 257)` 步长 $2/256 = 0.0078125$，`[:-1]` 丢掉最后一个点，
得到 256 个边界 $b_i = -1 + i\cdot\frac{2}{256},\ i=0..255$（即 $-1 \dots 0.9921875$）。
`np.digitize(x, b)` 返回满足 $b_{k-1}\le x < b_k$ 的 $k$，再减 1：

$$\text{bin}(x)=\begin{cases}
-1 & x < -1 \quad\textbf{（注意：负一，不是 0）}\\
\left\lfloor 128\,(x+1)\right\rfloor & -1 \le x < 0.9921875\\
255 & x \ge 0.9921875
\end{cases}$$

**两个必须知道的边界行为**：

1. **下溢产生字符串 `"-1"`**。低于 q01 的状态值会被编码成文本 `-1`，SentencePiece 会把它切成
   `-` 和 `1` 两个 token。所以状态串的 token 数**不是定长的**，取决于有多少维溢出、有多少个三位数。
   这就是 `max_token_len` 在 pi05 下默认拉到 **200**（π₀ 是 48）的原因（`pi0_config.py:38`）。
2. **上溢饱和到 255**，且上边界是 0.9921875 不是 1.0 —— 恰好等于 q99 的值会落在 254 还是 255
   取决于浮点舍入。这在「几乎不动的关节 / 锁死轴」上会表现为整条轨迹的状态 token 常数化。

**这也解释了为什么 π₀.₅ 必须配分位数归一化**：分箱假设输入已经在 $[-1,1]$。
如果误用 z-score，状态量普遍落在 $[-3,3]$，超过 ±1 的部分全部饱和到 −1/255，
proprioception 直接废掉，而且**不会报任何错**。

### 2.3 Prompt 模板对照

| 模型 | 模板 | 典型长度 |
|---|---|---|
| π₀ | `{prompt}` + `"\n"`（`"\n"` 单独 encode 作为 start-of-answer） | ≤ 48 |
| π₀.₅ | `Task: {prompt}, State: {b1} {b2} ... {bD};\nAction: ` | ≤ 200 |
| π₀-FAST | `Task: {prompt}, State: {...};\n` + `Action: ` + FAST tokens + `\|` + EOS | ≤ 180~250 |

π₀.₅ 的 prompt 以 `"Action: "` 结尾但**后面什么都不接** —— 因为 openpi 版本里动作不是自回归生成的，
这个后缀纯粹是为了和预训练时（FAST 阶段）的格式对齐，让 VLM 的表征处在「准备输出动作」的状态。

### 2.4 图像预处理

`resize_with_pad`（保持长宽比 + padding，不是直接 resize），然后 `PadStatesAndActions` 把 state / action
零填充到 `action_dim=32`。训练时的增强在 `model.preprocess_observation`（`model.py:167-186`）：

```python
if train:
    image = image / 2 + 0.5                      # [-1,1] → [0,1]
    if "wrist" not in key:                       # 只对外部相机
        RandomCrop(0.95W, 0.95H) → Resize(W,H) → Rotate(-5°, +5°)
    ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)   # 所有相机
    image = image * 2 - 1
```

腕部相机**不做几何增强**（裁剪/旋转），因为腕部视角与末端位姿强耦合，几何扰动等于伪造了标定关系。
这是个物理正确的设计选择，照抄时别顺手删掉这个 `if`。

---

## 3. SigLIP 视觉塔

`siglip._Module.__call__`（`siglip.py:208-290`），`pool_type="none"`，所以**不做池化，输出全部 patch token**。

```
image [B, 224, 224, 3]  (float32, ∈[-1,1])
  │ Conv(width=1152, kernel=14, stride=14, padding=VALID)     # 在 float32 下做，注释说"更安全"
  ↓ [B, 16, 16, 1152]
  │ reshape
  ↓ [B, 256, 1152]
  │ + pos_embedding  (learned, shape [1, 256, 1152])          # ← 学习式，非 sincos
  │ cast → bfloat16
  ↓
  │ Encoder × 27:  x += MHSA(LayerNorm(x));  x += MLP(LayerNorm(x))    # Pre-LN
  ↓ [B, 256, 1152]
  │ head = Dense(num_classes=2048)                            # 投到 Gemma 2B 的 width
  ↓ image_tokens [B, 256, 2048]
```

几个要点：

- **patch 数 = (H/14) × (W/14)**。224×224 → 256 token/相机。三个相机 ⇒ **768 个图像 token**。
- **posemb 是 learned 参数，形状写死为 [1, h·w, width]**。改 `image_resolution` 会直接改变这个参数的形状 —— 见 §10.1。
- 视觉塔在 openpi 里是**全量参与梯度**的（没有默认冻结它的 freeze filter），LoRA 变体也只对 llm 生效
  （`get_freeze_filter` 只匹配 `.*llm.*`）。
- `scan=True`：27 层用 `nn.scan` 折叠，参数存储带一个 leading depth 轴。

---

## 4. 双专家 Gemma：token 布局与注意力掩码

### 4.1 序列布局

π₀.₅（`action_dim=32, action_horizon=50, max_token_len=200`，三相机 224²）：

| 段 | 来源 | 走哪个 expert | 长度 | embed dim |
|---|---|---|---|---|
| base 图像 | SigLIP + head | expert 0 (2048) | 256 | 2048 |
| left wrist | SigLIP + head | expert 0 | 256 | 2048 |
| right wrist | SigLIP + head | expert 0 | 256 | 2048 |
| 语言（含离散 state） | `llm.embed` | expert 0 | 200（padding 到定长） | 2048 |
| **prefix 小计** | | | **968** | |
| 动作 token | `action_in_proj(x_t)` | **expert 1 (1024)** | 50 | 1024 |
| **总计** | | | **1018** | |

π₀ 对比：prefix = 768 + 48 = 816；suffix = 1 (state) + 50 (action) = 51。

### 4.2 注意力掩码：`make_attn_mask` 的推导

`pi0.py:19-44`：

```python
cumsum = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]     # attn_mask[i,j] = (c[j] <= c[i])
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
return attn_mask & valid_mask
```

记 $c_i = \sum_{k\le i} \text{ar}_k$，则

$$M_{ij} = \mathbb{1}[c_j \le c_i]\cdot \mathbb{1}[\text{valid}_i]\cdot\mathbb{1}[\text{valid}_j]$$

即：**query $i$ 能看到 key $j$，当且仅当 $j$ 所在的「块」编号不大于 $i$ 的块编号**。
`ar_k = 1` 表示「在这里开一个新块」。

**π₀.₅ 的 ar 向量**（`embed_prefix` + `embed_suffix`）：

```
ar = [0]*768 (图像)  [0]*200 (语言)  |  [1] (第一个动作token)  [0]*49
c  = [0]*968                        |  [1]*50
```

代入：
- prefix query（$c=0$）→ 只能看到 $c_j\le 0$ ⇒ **只看得到 prefix**（图像+语言之间全双向）
- action query（$c=1$）→ 能看到 $c_j\le 1$ ⇒ **prefix 全部 + 全部 action token（双向）**

这正是 KI 论文说的
*"Information flows unidirectionally from the VLM to the action expert; no VLM embedding attends to the action expert."*
—— **VLM 的表征完全不被动作 token 污染**，这是 prefix KV cache 可以复用的前提（§6.5），
也是 KI「知识隔离」在**注意力层面**的一半（另一半 stop-gradient 在 openpi 里没实现，见 §9.3）。

**π₀ 的 ar 向量**（多一个 state token）：

```
ar = [0]*816  |  [1] (state)  |  [1] (第一个action)  [0]*49
c  = [0]*816  |  [1]          |  [2]*50
```

⇒ 三个块：prefix / state / actions，state 看得到 prefix+自己，action 看得到全部。
和 π₀ 论文的 *"blockwise causal attention mask with 3 blocks: [I¹…Iⁿ, ℓ], [q], [a^τ…]"* 完全对应。

**π₀.₅ 把 3 块压回 2 块**，因为 state 搬进了 prefix 的语言部分。

### 4.3 RoPE positions

```python
positions = jnp.cumsum(input_mask, axis=1) - 1
```

位置**只数有效 token**，padding 不占位置。推理阶段 suffix 的位置要接在 prefix 后面：

```python
positions = jnp.sum(prefix_mask, -1)[:, None] + jnp.cumsum(suffix_mask, -1) - 1
```

RoPE 本身是标准的（`gemma.py:_apply_rope`，`max_wavelength=10000`），但注意实现是
**split-half** 而不是 interleave：

$$x_1,x_2 = \text{split}(x,2),\quad \text{out} = [\,x_1\cos\theta - x_2\sin\theta,\ \ x_2\cos\theta + x_1\sin\theta\,]$$

$$\theta_d = \frac{p}{10000^{2d/H}},\quad d = 0..H/2-1,\ H=256$$

RoPE 在 float32 下算完再 cast 回 bf16（注释里明确说这是有意为之，和 DeepMind/HF 实现一致）。

### 4.4 一层 attention 的完整数据流

`gemma.Attention.__call__`（`gemma.py:155-236`）：

```
xs = [x⁰ (B,968,2048),  x¹ (B,50,1024)]        # 两个 expert 的 token，可以有一个是 None
  │
  │ 各自投影（权重不同，输出形状相同）
  │   expert0: q_einsum "BTD,NDH→BTNH"  (8,2048,256)     → q⁰ (B,968,8,256)
  │            kv_einsum "BSD,2KDH→2BSKH" (2,1,2048,256) → k⁰,v⁰ (B,968,1,256)
  │   expert1: 同构，但 D=1024
  │
  ↓ concat along seq axis
  q (B,1018,8,256)   k (B,1018,1,256)   v (B,1018,1,256)
  │
  │ q = RoPE(q); q *= head_dim^(-0.5) = 1/16      # 注意：缩放用 head_dim，不是 width
  │ k = RoPE(k)
  │ [推理时] k,v = concat(kv_cache, [k,v])
  │
  │ q → (B,T,K=1,G=8,H=256)
  │ logits = einsum("BTKGH,BSKH→BKGTS")   (float32 累加)
  │ masked = where(mask, logits, -2.3819763e38)   # 不是 dtype.min，对齐 gemma 官方实现
  │ probs  = softmax(masked, -1).astype(bf16)
  │ enc    = einsum("BKGTS,BSKH→BTKGH") → (B,T,8,256)
  │
  │ 按段切回去，各自用自己的 attn_vec_einsum "BTNH,NHD→BTD"
  ↓
out = [out⁰ (B,968,2048), out¹ (B,50,1024)]
```

**这是理解 π₀/π₀.₅ 的关键一步**：注意力是**联合**的、一个 softmax，
只有 QKV 投影和输出投影按 token 归属分给不同的 expert。
所以它不是「两个模型 + cross-attention」，而是**一个 transformer，两套逐 token 的权重**。

### 4.5 RMSNorm 与 adaRMSNorm

`gemma.RMSNorm.__call__`（`gemma.py:107-126`）。方差始终在 **float32** 下算：

$$\hat{x} = \frac{x}{\sqrt{\frac{1}{D}\sum_d x_d^2 + 10^{-6}}}$$

**普通 RMSNorm**（expert 0 恒用；π₀ 时 expert 1 也用）：

$$y = \hat{x}\odot(1+\gamma),\qquad \gamma\ \text{零初始化}$$

**adaRMSNorm**（π₀.₅ 的 expert 1）：给定条件向量 $c\in\mathbb{R}^{1024}$（= 时间 embedding），

$$[\,s\,\|\,b\,\|\,g\,] = W_{\text{mod}}\,c + \beta,\qquad W_{\text{mod}}\in\mathbb{R}^{1024\times 3072},\ \ \textbf{零初始化}$$

$$y = \hat{x}\odot(1+s) + b$$

而 gate $g$ 被送去做**门控残差**（`gemma.py:_gated_residual`）：

$$x \leftarrow x + g\odot \text{Block}(y)$$

注意 $s,b,g$ 的形状是 $(B,1,D)$ —— **对同一个样本的全部 50 个动作 token 共享**，
因为 flow matching 的时间步 $t$ 是逐样本的标量。

**零初始化的含义**（DiT 的 adaLN-Zero 同款）：初始化时 $s=b=g=0$，于是
- $y = \hat{x}$（纯 RMSNorm）
- $x \leftarrow x + 0\cdot\text{Block}(y) = x$

**整个 action expert 的 18 个 block 在初始化时是恒等映射**。新初始化的 π₀.₅ 的第一步前向等价于

$$v_t = W_{\text{out}}\cdot\text{RMSNorm}\big(W_{\text{in}}\,x_t\big)$$

即一个（近似）线性函数。这让新加的 action expert 在训练早期**不会扰乱预训练 VLM 的残差流** ——
这是 π₀.₅ 相对 π₀ 换掉时间注入方式的直接动机。
（从 `pi05_base` finetune 时这些参数已非零，此性质只对 from-scratch 有效。）

### 4.6 FFN：GeGLU

`gemma.FeedForward`（`gemma.py:239-263`）：

$$\text{FFN}(x) = \big(\text{GELU}(x W_g^{(0)}) \odot (x W_g^{(1)})\big) W_\ell$$

$W_g$ 形状 `(2, width, mlp_dim)`，$W_\ell$ 形状 `(mlp_dim, width)`。
标准 Gemma GeGLU，两个 expert 结构相同、维度不同。

### 4.7 一个 Block 的完整公式

$$
\begin{aligned}
(y^i, g^i_{\text{attn}}) &= \text{Norm}^i_{\text{pre-attn}}(x^i,\ c^i) \\
[o^0, o^1] &= \text{Attention}([y^0, y^1],\ \text{pos},\ M) \\
x^i &\leftarrow x^i + g^i_{\text{attn}}\odot o^i \\[4pt]
(z^i, g^i_{\text{ffn}}) &= \text{Norm}^i_{\text{pre-ffw}}(x^i,\ c^i) \\
x^i &\leftarrow x^i + g^i_{\text{ffn}}\odot \text{FFN}^i(z^i)
\end{aligned}
$$

其中 $c^0 = \text{None}$（VLM 主干永远用普通 RMSNorm），$c^1 = $ 时间 embedding（π₀.₅）或 None（π₀）。
使用普通 RMSNorm 时 $g \equiv 1$。

18 层结束后各自过 `final_norm` / `final_norm_1`（后者也是 adaRMS，同样吃 $c^1$）。

---

## 5. 时间条件注入

### 5.1 正弦时间嵌入

`pi0.posemb_sincos`（`pi0.py:47-63`），$d = 1024$（action expert width）：

$$\text{fraction}_i = \frac{i}{d/2-1},\quad i=0..511$$
$$T_i = T_{\min}\left(\frac{T_{\max}}{T_{\min}}\right)^{\text{fraction}_i},\qquad T_{\min}=4\times10^{-3},\ T_{\max}=4.0$$
$$\text{emb}(t) = \Big[\sin\!\big(\tfrac{2\pi t}{T_i}\big)\ \Big\|\ \cos\!\big(\tfrac{2\pi t}{T_i}\big)\Big]\in\mathbb{R}^{1024}$$

周期从 0.004 到 4.0 几何递增，即角频率从 $2\pi/0.004 \approx 1571$ 到 $2\pi/4\approx 1.57$。
选这个范围是因为 $t\in[0,1]$：最长周期 4.0 保证在整个区间内 cos 分量单调可分（不绕圈），
最短周期 0.004 提供 ~0.002 的时间分辨率，远细于 10 步积分的 0.1 步长。
`einsum` 用 `Precision.HIGHEST` 强制 float32，避免 bf16 下高频项相位坍缩。

### 5.2 两种注入方式

**π₀（拼接 + MLP）**：

$$\tau_h = \text{emb}(t)\ \text{（广播到全部 }H\text{ 个动作 token）}$$
$$\tilde{a}_h = W_2\,\text{swish}\big(W_1\,[\,W_a a_h^t \ \|\ \tau_h\,]\big),\quad W_1\in\mathbb{R}^{2048\times1024}$$

时间信息**混进 token 内容**，只在输入层注入一次，之后靠 18 层去传播。

**π₀.₅（adaRMS）**：

$$c = \text{swish}\big(W_{o}\,\text{swish}(W_{i}\,\text{emb}(t))\big),\qquad W_i, W_o\in\mathbb{R}^{1024\times1024}$$
$$\text{token}_h = W_a\,a_h^t\quad\text{（不含时间）}$$

时间信息**变成条件向量**，在**每一层的两个 norm 上（共 37 处）反复注入**。

注意 π₀.₅ 的 time MLP **末尾也有一个 swish**（`pi0.py:161`：`time_emb = nnx.swish(time_emb)`），
这不太常见 —— 通常条件向量不会过最终激活。PyTorch 侧 `pi0_pytorch.py:292-298` 一致地复制了这个行为。

**为什么换**：π₀ 的 concat-MLP 让时间和动作在输入层就纠缠，18 层里没有再提醒模型「现在噪声多大」；
adaRMS 把时间提升为每层的调制信号（DiT 的做法），对不同噪声水平的函数族拟合能力更强，
且零初始化门控让新模块在 finetune 初期无害。代价是 +116M 参数。

---

## 6. Flow Matching：完整推导

### 6.1 条件流与目标向量场

设干净动作块 $a \in \mathbb{R}^{H\times D}$（已归一化、已 pad 到 32 维），噪声 $\epsilon\sim\mathcal{N}(0,I)$。

**代码约定**（`pi0.py:203-206`，注释明确说「和 π₀ 论文相反，抱歉」）：

$$x_t = t\,\epsilon + (1-t)\,a,\qquad t\in[0,1],\quad t=1\ \text{是纯噪声},\ t=0\ \text{是数据}$$

**论文约定**（π₀ / π₀.₅ 都是）：$A_t^\tau = \tau A_t + (1-\tau)\epsilon$，$\tau=1$ 是数据。

两者的换元关系就是 $\boxed{t = 1-\tau}$。

沿这条直线路径求导：

$$u_t := \frac{dx_t}{dt} = \epsilon - a$$

**这个向量场与 $t$ 无关**（直线插值的性质），所以训练目标就是一个常量场：

```python
u_t = noise - actions          # pi0.py:207
```

论文写作 *"the model is trained to predict the flow vector field ω − a_t"*，完全一致
（论文用 $\omega$ 记噪声）。

### 6.2 时间分布：代码与论文的换元验证

**论文**（π₀，2410.24164）：$p(\tau) = \text{Beta}\!\left(\frac{s-\tau}{s};\ 1.5,\ 1\right)$，$s=0.999$

**代码**（`pi0.py:205`）：

```python
time = jax.random.beta(rng, 1.5, 1, batch_shape) * 0.999 + 0.001
```

验证：令 $u\sim\text{Beta}(1.5,1)$。
- 论文侧：$\frac{s-\tau}{s}=u \Rightarrow \tau = s(1-u) = 0.999 - 0.999u$
- 代码侧：$t = 0.999u + 0.001$
- 于是 $1-\tau = 1 - 0.999 + 0.999u = 0.001 + 0.999u = t$ ✓

**代码与论文严格等价**，只是换了 $t=1-\tau$ 的记号。

$\text{Beta}(1.5,1)$ 的密度 $\propto u^{0.5}$，单调递增，质量偏向 $u\to1$，
即代码里的 $t\to1$（**高噪声**）、论文里的 $\tau\to0$（同样是高噪声）。
**采样偏向高噪声区间**，因为那里向量场最难学、也最决定生成轨迹的全局结构。
$0.999/0.001$ 的裁剪保证 $t\in[0.001, 1.0]$，永不取到 $t=0$（那里 $x_t=a$，无信息）。

### 6.3 损失

`pi0.py:220`：

```python
return jnp.mean(jnp.square(v_t - u_t), axis=-1)      # → [B, H]
```

`scripts/train.py:150-151` 再 `jnp.mean` 一次 → 标量。展开：

$$\mathcal{L} = \mathbb{E}_{a,\epsilon,t}\left[\frac{1}{B\,H\,D_{\text{model}}}\sum_{b,h,d}\big(v_\theta(x_t,t,o)_{bhd} - (\epsilon-a)_{bhd}\big)^2\right]$$

**注意 $D_{\text{model}}=32$ 是 padding 后的维度，不是真实动作维**。
真实动作维 $D$（比如双臂 rot6d 是 20，单臂 joint 是 7）之外的维度目标恒为
$\epsilon - 0 = \epsilon$（因为 pad 值是 0），模型要去拟合纯噪声 —— 这部分是不可约的常数损失
$\approx \frac{32-D}{32}$（归一化后 $\epsilon$ 的方差为 1）。

两个后果：
1. **loss 曲线的绝对值没有跨配置可比性**，且有一个非零地板。$D=20$ 时地板是 $12/32=0.375$。
   看 loss 别指望它趋近 0。
2. **真实动作维上的梯度被稀释了 $D/32$ 倍**。这一点在 `lerobot_pi05_vs_openpi_diff.md`
   里作为「lerobot 截断到真实维」的差异被记过 —— 从这边看，openpi 的 lr 2.5e-5/5e-5
   是**在这个稀释系数下调出来的**，换成截断版必须相应下调 lr。

### 6.4 采样：Euler 积分的推导

`pi0.sample_actions`（`pi0.py:222-284`）：

```python
dt = -1.0 / num_steps                # num_steps=10 → dt = -0.1
x = noise;  t = 1.0
while t >= -dt/2:                    # 恰好循环 10 次
    v = model(x, t, obs)
    x = x + dt * v
    t = t + dt
return x
```

**为什么这能还原动作**：ODE 是 $\frac{dx}{dt} = u_t$，从 $t=1$（$x_1=\epsilon$）积到 $t=0$：

$$x_0 = x_1 + \int_1^0 u_t\,dt = \epsilon - (\epsilon - a) = a\ \checkmark$$

显式 Euler 离散：$x_{t+\Delta} = x_t + \Delta\cdot v_\theta(x_t,t)$，$\Delta = -0.1$。
由于真实向量场 $u = \epsilon - a$ **与 $t$ 无关**，若模型完美，
**Euler 一步就精确**；10 步是为了容忍 $v_\theta$ 的估计误差随 $x_t$ 变化。
这也解释了为什么 flow matching 用 10 步就够，而 DDPM 要几十上百步。

循环条件 `t >= -dt/2` 即 $t \ge 0.05$，是对浮点累积误差的防御（第 10 步后 $t\approx 0$ 退出）。

### 6.5 KV cache：10 步去噪只算一次 VLM

这是 π₀/π₀.₅ 能实时跑的核心工程点。

```python
# 1) prefix 前向一次，拿到 KV cache
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=...)

# 2) 10 步循环里只跑 suffix（50 个 token），prefix 的 K/V 直接复用
(_, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens], mask=full_attn_mask, positions=..., kv_cache=kv_cache, adarms_cond=[None, c])
```

**这在数学上成立，当且仅当 prefix 的表征不依赖 suffix** —— 正是 §4.2 证明的
「prefix query 的 $c=0$，看不到 $c=1$ 的动作 token」。掩码设计和 KV cache 复用是一回事的两面。

计算量对比（单次推理，224²×3 相机）：
- SigLIP：3 × 27 层 × 256 token —— **算 1 次**
- Gemma 2B：18 层 × 968 token —— **算 1 次**
- Gemma 300M：18 层 × 50 token（但 attention 的 key 长度是 1018）—— **算 10 次**

即 10 步去噪的边际成本只是一个 300M 模型上 50 个 token 的前向 ×10，
相对 2B 主干 ×968 token 几乎可以忽略。

推理时 `full_attn_mask` 的拼装（`pi0.py:257-267`）：

```
prefix_attn_mask: (B, 50, 968)   受 prefix padding 约束 —— action 看得到全部有效 prefix
suffix_attn_mask: (B, 50,  50)   全 1 —— action 之间双向
full_attn_mask  : (B, 50, 1018)
```

---

## 7. 训练流程

### 7.1 一步训练的完整数据流

```
batch (Observation, Actions)
  │
  ├─ preprocess_observation(train=True)            # resize + 图像增强
  │
  ├─ t ~ 0.999·Beta(1.5,1) + 0.001                 # [B]
  ├─ ε ~ N(0, I)                                   # [B,50,32]
  ├─ x_t = t·ε + (1-t)·a                           # [B,50,32]
  ├─ u_t = ε - a
  │
  ├─ embed_prefix  → tokens [B,968,2048], mask, ar=[0]*968
  ├─ embed_suffix  → tokens [B,50,1024],  mask, ar=[1]+[0]*49, adarms_cond [B,1024]
  │
  ├─ attn_mask = make_attn_mask(concat(masks), concat(ars))    # [B,1018,1018]
  ├─ positions = cumsum(mask) - 1
  │
  ├─ llm([prefix, suffix], mask, positions, adarms_cond=[None, c])   # 一次大前向
  │      → (prefix_out [B,968,2048], suffix_out [B,50,1024])
  │
  ├─ v_t = action_out_proj(suffix_out[:, -50:])    # [B,50,32]
  └─ loss = mean((v_t - u_t)²)
```

**训练时 prefix 和 suffix 一起前向（不用 KV cache）**，因为要对 prefix 求梯度。
`prefix_out` 被完全丢弃 —— 它没有任何 loss（这正是 §9 里 openpi 缺失交叉熵项的表现）。

### 7.2 优化器与调度

`src/openpi/training/optimizer.py`：

```python
AdamW(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_gradient_norm=1.0)
CosineDecaySchedule(warmup_steps, peak_lr, decay_steps, decay_lr)
    init_value = peak_lr / (warmup_steps + 1)      # 注意分母是 warmup_steps+1，不是 warmup_steps
```

- `b2=0.95`（不是 Adam 默认 0.999）—— 大 batch 视觉/语言训练的惯例。
- `weight_decay=1e-10` 实质是 **0**，注释说改成真 0 会 OOM（optax 的实现细节）。
- **全局梯度裁剪 1.0 在 AdamW 之前**（`optax.chain(clip, adamw)`）。

`pi05_libero` 的官方超参（`config.py:1013-1027`）可作为 π₀.₅ finetune 的参考基线：

```
batch_size=256, warmup=10_000, peak_lr=5e-5, decay_steps=1_000_000, decay_lr=5e-5,
ema_decay=0.999, num_train_steps=30_000
```

注意 `peak_lr == decay_lr == 5e-5` 且 `decay_steps=1M ≫ num_train_steps=30k`：
**这实际上是「warmup 10k 后恒定 lr」**，余弦段根本没走到。照抄时别以为自己在做余弦退火。

### 7.3 EMA

`scripts/train.py:169-174`：

$$\theta_{\text{EMA}} \leftarrow \lambda\,\theta_{\text{EMA}} + (1-\lambda)\,\theta$$

`TrainConfig.ema_decay` 默认 0.99，π₀.₅ 系配置用 0.999。
**发布的 checkpoint 是 EMA 权重**。$\lambda=0.999$ 的有效平均窗口 $\approx 1/(1-\lambda) = 1000$ 步，
在 30k 步的 finetune 里是合理的；但如果只跑 2k 步，EMA 权重会严重滞后于当前权重 —— 短训练要下调到 0.99 或关掉。

### 7.4 LoRA / freeze filter

`Pi0Config.get_freeze_filter`（`pi0_config.py:96-118`）用正则匹配 NNX 路径：

- `.*llm.*` → 整个 Gemma（含两个 expert）
- `.*llm.*_1.*` → 只有 action expert（靠 `_name(name, i)` 给 expert 1 加 `_1` 后缀）
- `.*lora.*` → LoRA 增量，**永远不冻结**

规则：
- `paligemma_variant` 含 lora ⇒ 冻结 `.*llm.*`；若 action expert 不用 lora，则 `Not(.*llm.*_1.*)` 把它排除出冻结集（即 action expert 全量训练）
- **SigLIP 视觉塔从不在冻结集里** —— LoRA 模式下它仍然全量训练。这是显存和过拟合上都值得注意的点。

`_name` 的后缀约定（`gemma.py:445-453`）是整套双专家设计的落地方式：
expert 0 的权重**没有后缀**，因此能从原始 PaliGemma checkpoint 无缝加载；
expert 1 的权重带 `_1`，从零初始化。

---

## 8. 推理链路

```
robot obs (dict)
  │ repack_transforms.inputs
  │ data_transforms.inputs          <Robot>Inputs：拼 state/actions 布局、选相机
  │ [DeltaActions]                  推理时不生效（只作用于 actions）
  │ Normalize(use_quantiles=True)
  │ model_transforms.inputs         ResizeImages → TokenizePrompt → PadStatesAndActions
  ↓
Observation.from_dict  →  Pi0.sample_actions(num_steps=10)
  ↓ [B, H, 32]
  │ model_transforms.outputs        （π₀.₅ 为空；π₀-FAST 在这里做 token→action）
  │ Unnormalize(use_quantiles=True)
  │ [AbsoluteActions]               如果训练时用了 DeltaActions，这里加回 state
  │ data_transforms.outputs         <Robot>Outputs：截回真实动作维、rot6d→矩阵 等
  ↓
robot actions
```

`Outputs` 必须是 `Inputs` 的**严格逆**，改一边必须改另一边 —— 这是本仓库 `*_policy.py` 的核心约定
（每个都有配套 `*_test.py`，改布局后先跑那个）。

`n_action_steps`（实际下发多少步再重新推理）**不在模型里**，在 runtime client 侧。
openpi 的参考 runtime 只执行 8（droid）/ 10（aloha_sim）/ 25（aloha_real）步。
`action_horizon=50` 全开环执行在 30Hz 下是 1.7 秒盲走。

---

## 9. 论文有、openpi 代码没有的部分

这一节是「读论文时会以为代码里有，其实没有」的清单。

### 9.1 离散 token 预训练阶段（缺失）

论文 π₀.₅ 的预训练（280k 步）是**纯自回归**的：
*"trained as a standard auto-regressive transformer, performing next-token prediction of text,
object locations, and FAST encoded action tokens."*

openpi 里 `Pi0.compute_loss` **对 `prefix_out` 不施加任何 loss**。`FASTTokenizer` 存在于
`tokenizer.py`，但只被 `ModelType.PI0_FAST` 使用，π₀.₅ 路径不碰它。

### 9.2 后训练的复合损失（缺失交叉熵项）

论文式（1）：

$$\mathbb{E}_{\mathcal{D},\tau,\omega}\Big[\underbrace{H\big(x_{1:M},\ f^\ell_\theta(o_t,\ell)\big)}_{\text{文本/离散 token 交叉熵}} + \alpha\underbrace{\big\|\omega - a_{t:t+H} - f^a_\theta(a^{\tau,\omega}_{t:t+H}, o_t,\ell)\big\|^2}_{\text{flow matching}}\Big],\quad \alpha=10.0$$

openpi 只实现了第二项，且没有 $\alpha$（等价于 $\alpha=1$ 且第一项系数为 0）。

**实践含义**：finetune `pi05_base` 时，VLM 主干上**没有任何语言监督在拉着它**。
主干会被 flow 回归的梯度慢慢拖离预训练分布，语言跟随能力随步数衰减。
这是「finetune 久了模型不听指令、只会做一个动作」的结构性原因之一。
缓解手段在本仓库里只有：冻结主干（LoRA）、或者少训。

### 9.3 KI 的 stop-gradient（缺失）

KI 论文的核心是
*"stop the gradient flow between the action expert and the backbone weights"*。
grep 全仓库，`compute_loss` 路径上没有任何 `jax.lax.stop_gradient` / `.detach()`。

**注意力层面的隔离是有的**（§4.2 证明了 prefix 看不到 suffix），
**梯度层面的隔离没有** —— action expert 的 loss 会经由联合 attention 的 K/V
反传到整个 Gemma 2B 和 SigLIP。

所以：`pi05_base` 这个 **checkpoint** 是 KI 训出来的；你用 openpi 做的 **finetune** 不是 KI。

### 9.4 分层推理（缺失）

论文运行时是两段：
1. high-level：VLM 自回归生成语义子任务文本（如 `"pick up plate"`）
2. low-level：action expert 以该子任务为条件，10 步去噪出动作

openpi 的 `Policy.infer` 只有第 2 段，`prompt` 由外部给定。
要复现第 1 段需要自己在 `PaliGemma.llm` 上接一个 `embedder.decode` 做 greedy/sampling 解码
（`Embedder.decode` 在 `gemma.py:152-153` 是现成的，权重也在 checkpoint 里，但没有任何调用点）。

### 9.5 数据混合（缺失）

论文的混合配方（MM ~400h/~100 个家庭、ME、CE、HL、WD、VI）在 openpi 里没有对应机制 ——
`TrainConfig` 是单数据集的（`repo_id: str | None`）。多数据集加权只在 DROID 的 RLDS 路径上有
（`DataConfig.datasets: Sequence[RLDSDataset]`，带 weight），LeRobot 路径上没有。

### 9.6 对照表

| 论文组件 | openpi 代码 | 位置 |
|---|---|---|
| 双专家 Gemma、按 token 分权重 | ✅ | `gemma.Attention`, `gemma.Block` |
| 分块注意力（VLM 单向流向 expert） | ✅ | `pi0.make_attn_mask` |
| state 离散化进 prompt | ✅ | `tokenizer.PaligemmaTokenizer.tokenize` |
| adaRMS 时间注入 | ✅ | `gemma.RMSNorm(cond=...)` |
| flow matching + Beta 时间分布 | ✅ | `pi0.compute_loss` |
| 10 步 Euler 去噪 + prefix KV cache | ✅ | `pi0.sample_actions` |
| FAST 离散 token 预训练 | ❌ | — |
| 交叉熵联合损失（α=10） | ❌ | — |
| KI stop-gradient | ❌ | — |
| 分层推理（子任务预测） | ❌ | — |
| 异构数据 co-training 混合 | ❌ | — |

---

## 10. 本 fork 的三个坑

### 10.1 `image_resolution` ≠ 224 会打断预训练权重加载

`Pi0Config.image_resolution` 是本 fork 加的（上游写死 224×224）。但：

- SigLIP 的位置编码是 **learned** 参数，形状 `[1, (H/14)×(W/14), 1152]`（`siglip.py:229`）
- `weight_loaders._merge_params` **不做插值**，只按 key 名对齐
- `scripts/train.py:76` 的 `check_pytree_equality(..., check_shapes=True)` 会直接抛错

所以 `pi05_pick_and_place_260608data_406x406`（29×29=841 token）和
`pi05_0610data_630x476`（45×34=1530 token）两个配置里，JAX 的
`weight_loader=CheckpointWeightLoader(...)` **被注释掉了**，走的是 `pytorch_weight_path`。
PyTorch 侧 `safetensors.torch.load_model` 同样是严格形状匹配。

**结论**：改分辨率前先确认位置编码怎么办 —— 要么接受视觉塔从头训，
要么加一段 pos_embedding 的双线性插值（这是标准做法，ViT 论文和 HF 的
`interpolate_pos_encoding` 都有实现）。另外 630×476 时 prefix 长度会从 968 涨到
`3×1530 + 200 = 4790`，attention 是 $O(L^2)$，显存和时间都是 ~25 倍。

### 10.2 `discrete_state_input=False` + `pi05=True` ⇒ 模型完全看不到本体状态

这是最隐蔽的一个。链路：

1. `pi05=True` ⇒ `Pi0.__init__` **不创建 `state_proj`**（`pi0.py:95-99`）
2. `embed_suffix` 的 `if not self.pi05:` 分支被跳过 ⇒ suffix 里**没有 state token**
3. `discrete_state_input=False` ⇒ `TokenizePrompt` 收到 `state=None`（`transforms.py:255-259`）
   ⇒ prompt 里**没有 state 字符串**

**两条路都断了，proprioception 被完全丢弃**，而且不报错、不警告。

fork 里这样配的有：`pi05_red_cube_right_joint`、`pi05_pick_and_place_260408data`、
`pi05_260521simulationdata`、`pi05_260601data`、`pi05_pick_and_place_260608data`、
`..._406x406`、`pi05_0610data_630x476`、`pi05_260617data_joint`（`config.py:1281–1464`）。

上游的 `pi05_libero` 也是 `discrete_state_input=False` —— 在 LIBERO 那种「视觉基本决定任务阶段」
的仿真基准上这是可行甚至更好的（去掉了 state 上的过拟合捷径）。
但在真机上，**没有 state 的策略难以区分「夹爪已经闭合」和「夹爪即将闭合」这类视觉上近似的状态**，
表现为在抓取瞬间反复犹豫。

`pi05_260626data_eef_quat`、`pi05_marvin_eef_rot6d` 明确设了 `True`，
`pi05_hhw_*` 系列没设（走 `__post_init__` 的默认值 `= pi05 = True`）。
**跑实验前先确认自己在哪一档**，这个开关的影响远大于 lr 调整。

### 10.3 loss 的 padding 地板

见 §6.3。实操建议：如果要看真实动作维上的收敛，别改 loss（会连带改变有效 lr），
而是**额外记录一个截断到 $D$ 维的 metric** 用于观察：

$$\mathcal{L}_{\text{real}} \approx \frac{32}{D}\Big(\mathcal{L} - \frac{32-D}{32}\Big)\quad\text{（假设 pad 维已收敛到预测 }\epsilon\text{）}$$

更稳妥的是直接在 `compute_loss` 外面算一份 `mean((v_t - u_t)[..., :D]**2)`，只上报不回传。

---

## 11. 训练实务：一个真实脚本的逐项拆解

以 `train_sh/run_pi05_tj_clothes_400.sh` + config `pi05_yw_tidy_up`（`config.py:1812-1847`）为例。

### 11.1 它是 README 的哪一步

**是第 2 步「Defining training configs and running training」的后半段**：

```bash
/root/.local/bin/uv run --no-sync scripts/train.py "${CONFIG_NAME}" --exp-name="${EXPERIMENT_NAME}"
```

对应 README:161 的 `uv run scripts/train.py pi05_libero --exp-name=my_experiment`，
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` 也一致。但有三处和字面不符：

| | 情况 |
|---|---|
| **norm stats 不在里面** | README 要求先跑 `compute_norm_stats.py`。这里拆到了同目录的 `prepare_tidy_up.sh`（用 `compute_norm_stats_low_mem.py --direct-lerobot --direct-chunk-size 1024`）。**必须先跑那个**。 |
| **跑的不是本仓库** | `PROJECT_ROOT=/ssd/hhw/openpi-hzh`，`cd` 过去再执行。真正生效的 `pi05_yw_tidy_up` 是**那个 checkout 里的**。两边漂移时，你读的和跑的不是一份代码。 |
| **文件名对不上内容** | 文件名 `run_pi05_tj_clothes_400.sh`，但 `CONFIG_NAME=pi05_yw_tidy_up`、数据集是 `tidy_up_stationery_le`。和 `pi05_hhw_tj_clothes_400_uniform`（`config.py:1782`）没有关系。 |

另外没传 `--overwrite` / `--resume`，checkpoint 目录已存在时会直接报错（`TrainConfig.__post_init__` 之后的目录检查）。

### 11.2 这是全量微调，证据链

- `freeze_filter` **未设置** → 默认 `nnx.Nothing`（`config.py:762`）
- `trainable_filter = nnx.All(nnx.Param, nnx.Not(nnx.Nothing))` = **全部参数**（`config.py:819-821`）
- `model=Pi0Config(pi05=True)`，两个 variant 都是普通 `gemma_2b` / `gemma_300m`，**无 lora**

训练的部分 = 全部 ≈3.35B：

| 模块 | 参数量 | 状态 |
|---|---|---|
| SigLIP So400m/14 视觉塔 | ~414M | 训练 |
| Gemma 2B（expert 0，含 257152×2048 embedder） | ~2.51B | 训练 |
| Action expert 300M + adaRMS 调制层 | ~427M | 训练 |
| `action_in_proj` / `action_out_proj` / `time_mlp_in` / `time_mlp_out` | ~2M | 训练 |

外加一份 EMA 副本：`ema_decay` 未设 → 默认 **0.99**（`config.py:759`），有效窗口仅 100 步
（对比 `pi05_libero` 的 0.999 / 1000 步）。

### 11.3 没有「微调顺序」

**单阶段、单损失、所有参数从第 1 步起同时更新。** 唯一的「顺序」是两件事：

1. **权重血统**：`pi05_base`（PI 用 knowledge insulation 训出来的）→ `CheckpointWeightLoader` 全量加载 → 本次微调。就一跳。
2. **学习率日程**：warmup 2000 步 → 余弦退火，见 §12.3。

损失只有 flow matching L2 一项 —— 没有交叉熵、没有 stop-gradient、没有分层预测（§9）。
所以 VLM 主干在这 150k 步里**没有任何语言监督拉着它**。150k × batch 12 是很长的暴露时间，
这是该配置最值得担心的地方（定量见 §12.3 的累计位移预算）。

### 11.4 显存账：`fsdp_devices=1` 是个隐患

脚本没传 `--fsdp-devices`，默认 `fsdp_devices=1`（`config.py:804`）= **纯数据并行，每张卡各存一份完整副本**。
参数是 fp32（`init_train_state` 只把 `freeze_filter` 命中的参数转 bf16，全量微调时没有命中项，
`train.py:104-105`）：

```
params 13.4GB + Adam m,v 26.8GB + EMA 13.4GB ≈ 53.6GB   （还没算梯度 13.4GB 和激活）
```

4×80GB 也很紧。OOM 时第一件事是加 `--fsdp-devices 4`，而不是降 batch（batch 已经只有 12，见 §12.5）。

顺带一个过时注释：config 里写 "Global batch 12 across six devices gives a per-device batch of 2"，
但脚本给的是 `CUDA_VISIBLE_DEVICES=0,1,2,3` **四张卡** → 实际 per-device 是 **3**。
`12 % 4 == 0` 所以不会触发 `train.py:198` 的整除检查，只是注释没跟上。

### 11.5 改成 LoRA 怎么做

**可以，JAX 路径支持，仓库里已有现成模板**：`pi05_hhw_tj_fangkuai_lora`（`config.py:1670-1712`）。
在 `pi05_yw_tidy_up` 基础上改四处：

```python
model=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",        # rank 16, alpha 16 (attn + ffn)
    action_expert_variant="gemma_300m_lora",  # rank 32, alpha 32 (attn + ffn)
),
freeze_filter=pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
ema_decay=None,                                # LoRA 关掉 EMA
lr_schedule=_optimizer.CosineDecaySchedule(
    warmup_steps=2_000, peak_lr=2.0e-4,        # 比全量高一个数量级
    decay_steps=150_000, decay_lr=1.0e-5,
),
```

**一个反直觉的坑**：`get_freeze_filter`（`pi0_config.py:96-118`）返回的是

```python
nnx.All(PathRegex(".*llm.*"), nnx.Not(PathRegex(".*lora.*")))
```

**只匹配 `llm`**。所以 LoRA 模式下这些**仍然全量训练**：

- **SigLIP 视觉塔 ~414M 全量**（路径是 `PaliGemma/img/...`，不含 `llm`）
- `action_in_proj` / `action_out_proj` / `time_mlp_in` / `time_mlp_out`
- 全部 LoRA A/B（`lora.py:51-52` 命名为 `lora_a` / `lora_b`，被 `.*lora.*` 排除出冻结集）

被冻的是 llm 里的非 LoRA 权重，**含 adaRMS 的 `Dense` 调制层**（它们在 `llm/layers/.../pre_attention_norm_1/` 下）。
冻结部分会被转成 **bf16**（`train.py:105`），这才是 LoRA 省显存的主要来源 —— 不只是优化器状态变小。

想连视觉塔一起省，得自己在 freeze_filter 上再 `nnx.Or` 一个 `PathRegex(".*img.*")`，`get_freeze_filter` 不管这块。

其他注意事项：
- **PyTorch 路径不支持 LoRA**（README:192-198），必须走 `scripts/train.py`（JAX）。
- norm stats 不受影响，可直接复用。
- 换 `exp_name`，否则和全量微调的 checkpoint 目录撞车。

---

## 12. 反向传播时参数到底动多少

「所有参数同时更新」不等于「所有参数动得一样多」，也不等于「动多少由梯度大小决定」。
这一节把更新量算清楚。代码在 `scripts/train.py:137-190`。

### 12.1 一步的四段链路

**步 0 — 只对可训练参数求梯度**

```python
diff_state = nnx.DiffState(0, config.trainable_filter)
loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
```

LoRA 时被 `freeze_filter` 命中的参数**根本不进反向图**（不是算完再丢）。

**步 1 — 全局梯度裁剪**（`optax.clip_by_global_norm(1.0)`）

$$\|g\|_2 = \sqrt{\sum_{\text{所有可训练参数}}\ \sum_{\text{每个元素}} g_i^2},\qquad
\tilde{g} = g\cdot\frac{c}{\max(c,\ \|g\|_2)},\quad c=1.0$$

**这是「所有参数同时更新」唯一真正的耦合点**：SigLIP、Gemma 2B、action expert 的梯度
全在**同一个标量范数**里。任何一个模块出尖峰，整个模型这一步都被同比例缩小。

**步 2 — AdamW**

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\tilde{g}_t, &\beta_1 &= 0.9\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\tilde{g}_t^{\,2}, &\beta_2 &= 0.95\\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},\qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}\\[4pt]
\Delta\theta_t &= -\,\eta_t\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda\,\theta_t\right),
&\varepsilon &= 10^{-8},\ \lambda = 10^{-10}
\end{aligned}
$$

**步 3 — 应用 + EMA**

$$\theta_{t+1} = \theta_t + \Delta\theta_t,\qquad
\theta^{\text{EMA}}_{t+1} = \lambda_{\text{ema}}\,\theta^{\text{EMA}}_t + (1-\lambda_{\text{ema}})\,\theta_{t+1}$$

注意 `train.py:172-174` 是对 `new_params`（**完整** state）做 EMA，包括冻结参数 ——
冻结的那些每步在和自己平均，结果不变。

### 12.2 关键结论：更新量 ≈ 学习率，与梯度大小无关

看 $\hat{m}/(\sqrt{\hat{v}}+\varepsilon)$ 这一项：$\hat m$ 是梯度滑动均值，$\sqrt{\hat v}$ 是梯度滑动 RMS，
**两者量纲相同**，所以比值**无量纲、量级恒为 $O(1)$**，上界约 1。

$$\big|\Delta\theta_i\big| \lesssim \eta_t \qquad\text{（逐元素，与 } g_i \text{ 的绝对大小无关）}$$

极端情况最清楚：**第 1 步**（$t=1$）时 $\hat m_1 = \tilde g_1$、$\hat v_1 = \tilde g_1^2$，于是

$$\frac{\hat m_1}{\sqrt{\hat v_1}} = \operatorname{sign}(\tilde g_1)$$

**第一步就是纯 sign-SGD × 学习率**。梯度是 $10^{-3}$ 还是 $10^{-9}$，位移都是 $\eta_1$。

所以「3.35B 个参数同时更新」**不代表某些模块被更新得更猛** —— AdamW 把每个元素的步长
都归一化到了 $\eta_t$ 这个尺度。真正决定各模块动多快的是**梯度方向的一致性**：

$$\frac{|\hat m|}{\sqrt{\hat v}} \approx \begin{cases}
\approx 1 & \text{梯度方向稳定（真信号）} \Rightarrow \text{每步走满 } \eta_t\\
\ll 1 & \text{梯度在正负间抖动（噪声）} \Rightarrow \text{原地随机游走}
\end{cases}$$

### 12.3 代进 `pi05_yw_tidy_up` 的数字

$\eta_{\text{peak}}=2.5\times10^{-5}$，warmup 2000，cosine 到 $10^{-6}$，`decay_steps=num_train_steps=150{,}000`。

**学习率日程**（`optax.warmup_cosine_decay_schedule`，`optimizer.py:24-30`）：

$$\eta_t = \begin{cases}
\dfrac{2.5\times10^{-5}}{2001} + t\cdot\dfrac{2.5\times10^{-5} - 1.25\times10^{-8}}{2000} & t \le 2000\\[10pt]
10^{-6} + \dfrac{2.5\times10^{-5}-10^{-6}}{2}\left(1+\cos\dfrac{\pi(t-2000)}{148000}\right) & t > 2000
\end{cases}$$

起点 $1.25\times10^{-8}$（`peak_lr/(warmup_steps+1)`），cosine 段恰好 148k 步，
第 150k 步落到 $10^{-6}$ —— **这个配置的余弦是真的走完的**，
不像 `pi05_libero` 那个 `decay_steps=1M ≫ num_train_steps=30k` 退化成恒定 lr 的写法（§7.2）。

**单步绝对位移**：峰值时每个元素最多动 $2.5\times10^{-5}$。

**相对位移**才有意义。Gemma 2B 的 `q_einsum` 用 `lecun_normal`，fan_in = 2048：

$$\sigma_\theta \approx \frac{1}{\sqrt{2048}} \approx 0.022
\quad\Rightarrow\quad
\frac{|\Delta\theta|}{\sigma_\theta} \approx \frac{2.5\times10^{-5}}{0.022} \approx 1.1\times10^{-3}$$

**一步动典型权重的约 0.1%。** Action expert（fan_in 1024，$\sigma\approx0.031$）相对更新还要小 ~30%。

**150k 步的累计上界**（假设梯度符号从不翻转，实际远小于此）：

$$\sum_t \eta_t \approx \bar\eta \cdot 148000 \approx 1.3\times10^{-5}\times1.48\times10^5 \approx 1.9$$

即每个元素理论上最多漂移 1.9 —— 是典型权重尺度 0.022 的 **86 倍**。
**这就是 §11.3 那句担心的定量版本**：150k 步全量微调 + 零语言监督，预算完全够把 VLM 主干
拖离预训练分布，只差梯度方向一致。

**weight decay 项**：$\eta_t\lambda\theta = 2.5\times10^{-5}\times10^{-10}\times0.022 \approx 5\times10^{-17}$。
彻底是 0，证实了 `optimizer.py:71` 那句注释 —— 设成 `1e-10` 只是为了绕开 optax 的 OOM。

### 12.4 全局裁剪什么时候真的起作用

裁剪在 Adam **之前**。如果 $\|g\|$ **持续**大于 1，所有梯度被同一个常数因子 $c/\|g\|$ 缩放，
而 Adam 对全局常数因子是**不变的**（分子分母同时缩放）：

$$\frac{\alpha\hat m}{\sqrt{\alpha^2\hat v}+\varepsilon} \approx \frac{\hat m}{\sqrt{\hat v}}$$

所以**持续裁剪几乎没有效果**。裁剪真正救命的是**间歇性尖峰**：某一步 $\|g\|$ 暴涨 100 倍被压回 1，
而 $\hat v$ 还停留在旧尺度上，这一步就不会把参数轰飞。

### 12.5 梯度不是在「所有数据」上算的

这是最容易误解的一点。**这是 mini-batch SGD，不是 full-batch**：

```python
data_loader = create_data_loader(config, sharding=data_sharding, shuffle=True)   # train.py:220-224
batch = next(data_iter)                                                          # 12 个样本
```

`num_batches` 未传 → `None` → 数据集**无限循环**、每 epoch 重新 shuffle，训练在
`num_train_steps=150_000` 步停。**没有梯度累积**（`train_step` 一次 `value_and_grad` 直接接一次 `tx.update`）。

$$g_t = \nabla_\theta\ \frac{1}{12}\sum_{i=1}^{12}\ell(\theta;\ \xi_i^{(t)})\ \ \neq\ \ \nabla_\theta\ \frac{1}{N}\sum_{i=1}^{N}\ell(\theta;\ \xi_i)$$

$g_t$ 是真实梯度的**无偏但高方差的蒙特卡洛估计**。

**而且噪声比普通 mini-batch 还大 —— 四重随机**。同一帧数据两次被抽到时，训练信号完全不同：

| 随机源 | 位置 | 影响 |
|---|---|---|
| 1. 抽哪 12 帧 | `shuffle=True` | 普通 SGD 噪声 |
| 2. 时间 $t\sim0.999\,\mathrm{Beta}(1.5,1)+0.001$ | `pi0.py:205` | 每样本独立，决定看到的噪声水平 |
| 3. 噪声 $\epsilon\sim\mathcal N(0,I)$，`[12,50,32]` | `pi0.py:204` | **回归目标 $u=\epsilon-a$ 本身被 $\epsilon$ 主导**（归一化后方差为 1） |
| 4. 图像增强（RandomCrop / Rotate / ColorJitter） | `model.py:167-186` | 每样本独立 |

第 3 点最关键：模型要拟合的不是确定标签，而是 $\mathbb{E}[\epsilon - a \mid x_t, o, t]$，
单样本只给出这个条件期望的**一次抽样**。这也是 §6.3 那个 loss 地板的来源 —— 它就是不可约的采样方差。

150k × 12 = **1.8M 次样本抽取**，但因为 $(t,\epsilon,\text{aug})$ 每次重抽，
**没有任何一个样本被以相同方式看过两次**。epoch 数 $= 1.8\times10^6 / N_{\text{frames}}$。

### 12.6 两点串起来的后果：batch 12 是这个配置的真问题

§12.2 说更新量 $\approx\eta_t$ 与梯度大小无关；§12.5 说梯度噪声极大。合起来看，
$\hat m/\sqrt{\hat v}$ 会被压得很小：

$$\frac{|\hat m|}{\sqrt{\hat v}}\ \sim\ \frac{|\text{信号}|}{\sqrt{|\text{信号}|^2+\sigma_{\text{noise}}^2}}\ \ll 1
\quad\text{当噪声占优}$$

即**实际每步走的远不到 $2.5\times10^{-5}$**，靠 150k 步的长时间平均把信号从噪声里积出来。
这就是为什么这个配置要跑 150k 步，而 `pi05_libero`（batch **256**）只要 30k 步 ——
差 21 倍的 batch，用 5 倍的步数补，并不划算。

$\beta_2=0.95$（而非 0.999）在这里也偏激进：$\hat v$ 的有效窗口只有约 20 步 × 12 样本 = 240 个样本的
梯度二阶矩，估计本身就抖。**batch 12 + $\beta_2=0.95$ 是这个配置里最值得调的一对参数，比 lr 更值得调。**

### 12.7 唯一真正「在所有数据上算」的东西

`prepare_tidy_up.sh` 跑的 norm stats：

```bash
compute_norm_stats_low_mem.py --config-name pi05_yw_tidy_up --direct-lerobot --direct-chunk-size 1024
```

$q_{01}, q_{99}, \mu, \sigma$ 是**遍历整个数据集**统计出来的，训练全程固定不变。
它们决定归一化区间（§2.1）和 256 分箱的量程（§2.2）—— 全流程里唯一的全局量。
这也是 CLAUDE.md 里「loss 发散通常是 norm stats 有问题」的原因：它错了，1.8M 次采样每一次都错。

### 12.8 怎么观测

`train.py:178-190` 已经把两个量记进 `info`：

```python
"grad_norm":  optax.global_norm(grads),        # 裁剪【前】的原始梯度范数
"param_norm": optax.global_norm(kernel_params) # 只统计 ndim>1 的 kernel，
                                               # 排除 bias/scale/pos_embedding/input_embedding
```

判读：

| 观察 | 含义 |
|---|---|
| `grad_norm` 长期 $\gg 1$ | 裁剪常态化 → 按 §12.4 等于几乎没生效，说明 lr 或 batch 需要重调 |
| `grad_norm` 偶发尖峰 | 裁剪在正常工作，符合预期 |
| `param_norm` 单调上涨 | 参数在系统性漂移，配合 loss 一起看（结合 §12.3 的 1.9 位移预算判断严重程度） |

---

## 13. 形状速查表

以 `pi05`、`action_dim=32`、`action_horizon=50`、`max_token_len=200`、三相机 224×224、batch $B$ 为例。

| 张量 | 形状 | 产生位置 |
|---|---|---|
| `image[k]` | `[B, 224, 224, 3]` float32 ∈[-1,1] | `Observation.from_dict` / `preprocess_observation` |
| `image_tokens[k]` | `[B, 256, 2048]` | `PaliGemma.img(...)` |
| `tokenized_prompt` | `[B, 200]` int32 | `TokenizePrompt` |
| `lang_tokens` | `[B, 200, 2048]` | `llm(..., method="embed")` |
| `prefix_tokens` | `[B, 968, 2048]` | `embed_prefix` |
| `prefix_ar_mask` | `[968]` bool，全 False | `embed_prefix` |
| `state` | `[B, 32]`（π₀.₅ 下**未被模型使用**） | `PadStatesAndActions` |
| `actions` / `x_t` / `u_t` / `v_t` | `[B, 50, 32]` | — |
| `time` | `[B]` ∈[0.001, 1.0] | `compute_loss` |
| `time_emb` → `adarms_cond` | `[B, 1024]` | `posemb_sincos` → time MLP |
| `suffix_tokens` | `[B, 50, 1024]` | `embed_suffix` |
| `suffix_ar_mask` | `[50]`，`[True] + [False]*49` | `embed_suffix` |
| `attn_mask` | `[B, 1018, 1018]` bool | `make_attn_mask` |
| `positions` | `[B, 1018]` int32 | `cumsum(mask)-1` |
| adaRMS `scale/shift/gate` | 各 `[B, 1, 1024]` | `RMSNorm(cond)` |
| `kv_cache` | `k,v` 各 `[18, B, 968, 1, 256]` | `sample_actions` prefix 前向 |
| `suffix_out` | `[B, 50, 1024]` | `llm(...)` |
| loss（reduce 前） | `[B, 50]` | `compute_loss` |

---

## 14. 一页纸总结

**π₀.₅ 相对 π₀ 的模型改动只有两条**（`pi0_config.py:32-35` 的注释就是这么写的）：

1. **state 从连续 suffix token 搬进离散语言 prompt**（256 分箱，配分位数归一化）
2. **时间从 concat-MLP 注入改为 adaRMSNorm 逐层调制**（+116M 参数，零初始化门控）

其余全部继承 π₀：双专家 Gemma（2B + 300M，共享 attention、按 token 分权重、MQA）、
SigLIP So400m/14 视觉塔、分块注意力（VLM 不看动作 token）、
直线条件流 flow matching（Beta(1.5,1) 时间分布，10 步 Euler，prefix KV cache 复用）。

**π₀.₅ 相对 π₀ 的真正提升不在模型，在训练配方** —— 异构 co-training、离散 token 预训练、
知识隔离、分层推理。而这些**在 openpi 里一个都没有**。
你在这个仓库里做的，是「用 flow matching 回归 finetune 一个 KI 预训练过的 3.35B checkpoint」。

把这句话记住，就不会再对着 loss 曲线困惑为什么模型的语言泛化在训练中越来越差。

**训练侧再补三句**（§11–§12 的浓缩）：

1. **默认 = 全量微调。** `freeze_filter` 不设就是 `nnx.Nothing`，3.35B 参数一个不冻、单阶段、
   从第 1 步起同时更新。没有「先训哪块再训哪块」这回事。
2. **更新量由学习率决定，不由梯度大小决定。** AdamW 的 $\hat m/\sqrt{\hat v}$ 无量纲且 $O(1)$，
   每个元素每步位移 $\lesssim\eta_t$；第 1 步严格等于 sign-SGD $\times\eta_1$。
   模块之间的差异来自**梯度方向的一致性**，不是梯度的绝对大小。
3. **梯度是 12 个样本 + 四重随机的蒙特卡洛估计，不是全数据梯度。** 全流程里唯一在整个数据集上
   算出来的量是 norm stats。想加速收敛，先加 batch，再谈 lr。

---

## 附：关键代码位置索引

| 主题 | 文件:行 |
|---|---|
| 注意力掩码构造 | `src/openpi/models/pi0.py:19` |
| 正弦时间嵌入 | `src/openpi/models/pi0.py:47` |
| 模块构造（π₀ / π₀.₅ 分支） | `src/openpi/models/pi0.py:66` |
| prefix 组装 | `src/openpi/models/pi0.py:107` |
| suffix 组装 + adaRMS 条件 | `src/openpi/models/pi0.py:137` |
| flow matching 训练 | `src/openpi/models/pi0.py:189` |
| 10 步去噪 + KV cache | `src/openpi/models/pi0.py:222` |
| RMSNorm / adaRMSNorm | `src/openpi/models/gemma.py:107` |
| 联合 attention（双专家） | `src/openpi/models/gemma.py:155` |
| GeGLU FFN | `src/openpi/models/gemma.py:239` |
| Block（门控残差） | `src/openpi/models/gemma.py:266` |
| RoPE | `src/openpi/models/gemma.py:425` |
| expert 后缀命名约定 | `src/openpi/models/gemma.py:445` |
| SigLIP 前向 | `src/openpi/models/siglip.py:208` |
| 图像增强 | `src/openpi/models/model.py:167` |
| state 离散化 + prompt 模板 | `src/openpi/models/tokenizer.py:24` |
| 归一化 / 反归一化公式 | `src/openpi/transforms.py:137, 170` |
| ModelTransformFactory | `src/openpi/training/config.py:115` |
| 分位数归一化开关 | `src/openpi/training/config.py:197` |
| 优化器 / 调度 | `src/openpi/training/optimizer.py:17, 66` |
| loss 归约 + EMA | `scripts/train.py:150, 169` |
| PyTorch 镜像实现 | `src/openpi/models_pytorch/pi0_pytorch.py:84` |
| **训练实务（§11–§12）** | |
| `train_step`（梯度→裁剪→AdamW→EMA） | `scripts/train.py:137` |
| 冻结参数转 bf16 | `scripts/train.py:104` |
| `grad_norm` / `param_norm` 上报 | `scripts/train.py:178` |
| batch 整除设备数检查 | `scripts/train.py:198` |
| data loader 创建（`shuffle=True`，无限循环） | `scripts/train.py:220` |
| `trainable_filter` / `freeze_filter` 默认值 | `src/openpi/training/config.py:762, 819` |
| `fsdp_devices` 默认 1 | `src/openpi/training/config.py:804` |
| LoRA freeze filter（只匹配 `.*llm.*`） | `src/openpi/models/pi0_config.py:96` |
| LoRA 参数命名 `lora_a` / `lora_b` | `src/openpi/models/lora.py:51` |
| pi05 LoRA 配置模板 | `src/openpi/training/config.py:1670` |
| 本例配置 `pi05_yw_tidy_up` | `src/openpi/training/config.py:1812` |
| 训练/准备脚本 | `train_sh/run_pi05_tj_clothes_400.sh`, `train_sh/prepare_tidy_up.sh` |
