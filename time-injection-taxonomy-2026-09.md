# 扩散/流匹配策略的时间注入：位置 × 方法，与 patch_policy 的三层核验

**整理日期** 2026-09-04
**范围** (1) π₀.₅ 训练/推理流程的三处更正；(2) 时间注入的两轴分类与 14 个模型的对照表；
(3) patch_policy 时间注入的论文—原码—出处三层核验。
**方法** 全部结论来自逐行读代码；论文结论标注来源页。凡代码与论文不一致处，以代码为准并注明。
**前置** 状态融合谱系与锚点比 `r` 见 `state-action-fusion-and-anchor-ratio-2026-09.md`。

---

## 1. π₀.₅ 训练/推理流程的三处更正

一个常见的复述是：「训练时把单帧状态与图像、文本编码进 VLM 算出 KV cache，再用这个 cache
去给未来 50 步加噪动作做去噪；推理时同样先算 cache 再去噪投影出 50 步动作。」

推理那半正确，训练那半有三处不成立。

### 1.1 推理：正确，但只对 π₀.₅ 成立

`openpi/models/pi0.py:238-244`：

```python
# first fill KV cache with a forward pass of the prefix
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

`embed_prefix`（`:107-135`）只放**图像 token + 语言 token**。π₀.₅ 的 state 被离散成 256 档写进
语言 prompt（`tokenizer.py:22-31`），所以它确实进了 KV cache，10 步去噪只算一次。

**π₀ 不是这样**：state 走 `state_proj` 进 **suffix**（`pi0.py:152-157`），suffix 每个去噪步
都要重算，π₀ 的 state 不在 cache 里。这句复述把 π₀ 和 π₀.₅ 混为一谈了。

### 1.2 训练期不存在 KV cache

`compute_loss:210` 的代码注释直说：

```python
# one big forward pass of prefix + suffix at once
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
    adarms_cond=[None, adarms_cond]
)
```

prefix 与 suffix **一次前向同时算完**，靠 block 注意力掩码隔开（prefix 看不见 suffix，
suffix 能看全部）；cache 返回值被 `_` 丢弃。「先算 cache 再拿去用」只是推理期的省算力手段。

### 1.3 训练期不去噪

```python
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001   # 每样本一个 t
x_t  = time_expanded * noise + (1 - time_expanded) * actions
u_t  = noise - actions
loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

采**一个** t，构一个 `x_t`，回归一次速度场。**单步，无迭代。**`num_steps=10` 的循环只在
`sample_actions` 里。

### 1.4 「加噪」的措辞

`x_t = t·noise + (1−t)·a` 是**插值**不是加噪：t=1 时动作项系数为 0，信号被完全替换，
与 DDPM 的保方差 signal+noise 不同。成文时用 "linear interpolation path between noise and
data"，不要写 "added noise"。

### 1.5 更正后的表述

> **训练**：图像、文本、单帧状态（离散成 256 档写进 prompt）与 50 步动作块在**一次前向**中
> 通过 block-causal 掩码联合编码；每个样本采一个流匹配时刻 t，在噪声与真值动作的线性插值点
> `x_t` 上回归速度场 `u_t = ε − a`。
>
> **推理**：图像、文本、状态先算一遍 prefix KV cache；随后 10 步欧拉积分中，唯一变化的是
> `x_t` 与注入 action expert 的 adaRMSNorm 时间条件，cache 全程复用。

架构含义：**action expert 没有直接的 state 输入端口**，只能通过注意力去读 prefix 里那串数字。
这是 π₀.₅ 与 patch_policy 的 P2（连续 state token 进 prefix）在信息通路上的唯一差别——
π₀.₅ 那条窄一档但省 10× 算力，而按前一份报告算出的量化尺度（一档 ≈ 0.0025 rad，
噪声下限的 1/26），窄的那一档不损失什么。

---

## 2. 时间注入的两轴

编码步骤所有模型一致：标量 t → 正弦编码 → 2 层 MLP → 条件向量。差别全在之后，且可拆成
**位置**与**方法**两个正交的轴。

### 2.1 位置（4 类）

| 位置 | 做什么 | 注入次数 |
|---|---|---|
| **L1 融进动作 token** | 时间向量 broadcast 到动作 token 上，进 transformer 前一次性合并 | 1 |
| **L2 独立 token** | 时间自成一个 token 坐进序列，靠注意力被读到 | 1（间接） |
| **L3 每层归一化内部** | 每个 block 的 norm 用时间生成 scale/shift/gate | 2×depth+1 |
| **L4 每块加性偏置** | 时间向量直接加到隐状态上 | depth |

### 2.2 方法（5 类）

**M1 concat + MLP** — `cat([action_emb, time_emb])` → Linear → swish → Linear。
`pi0.py:172-178`。

**M2 adaLN / adaRMSNorm** — `Dense(cond) → chunk(scale, shift, gate)`，**kernel 零初始化**，
`gate` 乘在残差上：

```python
# openpi/models/gemma.py:128-131
modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
normed_inputs = normed_inputs * (1 + scale) + shift
# :453-459
def _gated_residual(x, y, gate):
    if gate is None: return x + y
    return x + y * gate
```

初始时 gate=0 → 整个 block 恒等 → 训练从「没有 action expert」平滑长出来（AdaLN-Zero）。
**三个实现独立用了同一招**：openpi 的 `kernel_init=zeros`、
`multi_task_dit:612-613` 的 `nn.init.constant_(adaLN_modulation[-1].weight, 0)`、
MolmoAct2 的 `_init_linear(..., zero=True)`。

**M3 FiLM** — 逐通道 scale + bias 加在 conv 输出上，`modeling_diffusion.py:797-805`。
注意 lerobot 默认 `use_film_scale_modulation=False`，**退化成纯 bias**。

**M4 纯加** — `x2 = x2 + time_emb.unsqueeze(1)`，`evo1/flow_matching.py:167-168`。

**M5 独立 memory token** — 时间自成 token 进 cross-attention memory，见 §5。

---

## 3. 对照表

| 模型 | 位置 | 方法 | 注入点数 | 代码 |
|---|---|---|---:|---|
| **π₀** | L1 | M1 concat+MLP | 1 | `pi0.py:172-178` |
| **π₀.₅** | **L3** | **M2 adaRMS** | **37**（18×2+1） | `pi0.py:164-169` → `gemma.py:303,318,410` |
| SmolVLA | L1 | M1 | 1 | `modeling_smolvla.py:753-757` |
| EO-1 | L1 | M1 | 1 | `modeling_eo1.py:388-395` |
| WALL-OSS | L1 | M1 | 1 | `modeling_wall_x.py:212-216` |
| GR00T（W2 分支） | L1 | M1 | 1 | `groot_n1_7.py:242-243` |
| VLA-JEPA | L1 | M1 | 1 | `action_head.py:76-77` |
| X-VLA | L1 | M1（动作+本体+时间三路 concat） | 1 | `soft_transformer.py:378-381` |
| GR00T（DiT 分支） | L3 | M2 AdaLayerNorm（只 scale/shift，无 gate） | depth | `cross_attention_dit.py:64-88` |
| MolmoAct2 | L3 | M2，**9 路** chunk（self/cross/ffn 各 3） | depth | `modeling_molmoact2.py:449,464-475` |
| multi_task_dit | L3 | M2，6 路 chunk | depth | `modeling_multi_task_dit.py:534-548` |
| Diffusion Policy (UNet) | L3 | M3 FiLM（默认只 bias） | 每 ResBlock | `modeling_diffusion.py:797-805` |
| Evo-1 | L4 | M4 加 | depth | `flow_matching.py:167-168` |
| **patch_policy** | **L2** | **M5 memory token** | **0 次直接** | `diffusion_policy.py:289,443-447` |
| ACT | — | 无（不是去噪器） | — | — |

lerobot 侧文件根：`MCap2HDF5_code/third_party/lerobot/src/lerobot/policies/`。

---

## 4. 功能差异：真正的轴是注入频次

L1/M1 的信息在第一层之后要靠网络自己扛完 18 层；L3/M2 每层重新提醒一次，而且是**乘性**的——
直接改变每层的激活尺度，去噪早期（t≈1，纯噪声）与晚期（t≈0，接近真值）可以走完全不同的
计算路径。L1 只能靠加性偏移逼近这一点。

π₀ → π₀.₅ 从 L1/M1 换到 L3/M2，与 state 从 suffix 挪到 prefix **是同一个动作的两半**：
把 state 和 time 都从 suffix token 里清出去，suffix 变成纯粹的 `x_t`。50 个 token 的带宽
全给动作本身，条件信息改走 KV cache（state）与 adaRMS（time）两条旁路。

### 一个具体的坑：正弦基频不能混用

| 实现 | 频率范围 | 配套的 t |
|---|---|---|
| openpi `posemb_sincos` | `min_period=4e-3, max_period=4.0` → 250 ~ 0.25 周期 | **连续 t ∈ [0,1]** |
| patch_policy / DP / DiT | 标准底数 `10000` → 最高 1 周期，最低 1e-4 | **整数 t ∈ [0,100]** |
| X-VLA | `max_period=100` | 连续 |

patch_policy 现在 DDPM 整数时刻配 10000 基频，自洽。**但若改成流匹配（t∈[0,1]），必须同时
换掉 `DiffusionSinusoidalPosEmb` 的底数**——10000 基频下 t∈[0,1] 时除最高一两个通道外全部
退化成常数，时间信息基本消失，且**不报错**，只表现为 loss 卡住。openpi 那组
`4e-3/4.0` 正是为此设计。

---

## 5. patch_policy 的时间注入：三层核验

来源：论文 arXiv:2607.18236v1 · 项目页 patch-policy.github.io · 代码 github.com/gaoyuezhou/patch_policy

### 5.1 论文层：完全没提

*Patch Policy: Efficient Embodied Control via Dense Visual Representations*，
Gaoyue Zhou, Zichen Jeff Cui, Ada Langford, Bowen Tan, Yann LeCun, Lerrel Pinto，2026-07-20。

§2.2 原话：

> "This formulation is **agnostic to the action head architecture and training objective**."

论文只说用了 VQ-BeT 与 Diffusion Policy 两个现成 head，**关于扩散时刻嵌入、时间条件、
时间注入方式没有任何描述**，也不在其消融范围内。时间注入不是 Patch Policy 的贡献。

顺带核到与本项目直接相关的一条，§3.3：

> "All policies receive only visual inputs."

以及为对齐基线 "we remove proprioceptive inputs from ACT's CVAE encoder"。
本项目的 `use_robot_state=false` 是照搬该设定，不是疏漏。

### 5.2 代码层：一个 memory token，硬编码

`models/diffusion_policy/diffusion_policy.py`：

```python
# :289  SinusoidalPosEmb，底数 10000
time_emb = self.time_emb(timesteps).unsqueeze(1)          # (B,1,n_emb)
# :443-447
cond_embeddings = time_emb
if self.obs_as_cond:
    cond_embeddings = torch.cat([cond_embeddings, self.cond_obs_emb(cond)], dim=1)
x = self.drop(cond_embeddings + self.cond_pos_emb[:, :tc, :])
x = self.encoder(x)        # n_cond_layers=0 ⇒ Linear→Mish→Linear
memory = x
# :261
mem_bool[:, 0] = True      # 时间 token 对所有 decoder 位置永远可见
```

四个开关**全部硬编码**在 `DiffusionPolicy.__init__:736-739`，配置文件无法覆盖：

```python
causal_attn=True, time_as_cond=True, obs_as_cond=True, n_cond_layers=0,
```

调度器（`:744-753`）：DDPM，`num_train_timesteps=100`（**不是 1000**）、
`beta_schedule="squaredcos_cap_v2"`、`prediction_type="epsilon"`、`clip_sample=True`。

`_init_weights`（`:302-345`）中 `SinusoidalPosEmb` 属 `ignore_types`（无参数），
`cond_pos_emb` 为 `normal(0, 0.02)`；**全文无任何零初始化**，不存在 AdaLN-Zero 式结构。

**结论：位置 L2 / 方法 M5，零次直接注入。**

### 5.3 出处层：逐字来自 Chi et al. 的 Diffusion Policy

对照 `real-stanford/diffusion_policy` 的
`diffusion_policy/model/diffusion/transformer_for_diffusion.py`：`ModuleAttrMixin`、
`get_optim_groups`、带 `ignore_types` 的 `_init_weights`、
`time_as_cond`/`obs_as_cond`/`encoder_only` 三个 flag、`p_drop_emb`/`p_drop_attn`
——全部同名同构。Chi 版自己的注释即写明时间 token 的位置语义：

```python
mask = t >= (s-1)  # add one dimension since time is the first token in cond
```

**Patch Policy 只加了两样**：`n_patches` 参数，与把上面这行换成逐 patch 分块的 `mem_bool`
构造（`:245-278`）。时间部分一行未改。

即：「时间作为 cross-attention memory 的第 0 个 token」是 **Diffusion Policy (RSS 2023)
的设计**，Patch Policy 原样继承。

### 5.4 本项目移植的忠实度：逐项对上

| 项 | 原实现 | 本项目移植 |
|---|---|---|
| 时间 token 永远可见 | `mem_bool[:,0]=True` | `n_leading_tokens=1` |
| 动作主干因果掩码 | `causal_attn=True` | `causal_mask(horizon)` |
| memory encoder | `n_cond_layers=0` → MLP | 直接写死 MLP |
| 正弦底数 | 10000 | 10000 |
| `num_train_timesteps` | 100 | 100 |
| `beta_schedule` | `squaredcos_cap_v2` | 同 |
| `prediction_type` | `epsilon` | 同 |
| `clip_sample` | True | 同 |

**无偏离。**ACT head 为本项目新增，代码内注释已标明「no counterpart in the reference repo」。

---

## 6. 两条被关掉的分支，与一个块因果陷阱

原代码存在但从未启用的两条路径：

**(a) `time_as_cond=False`** → `encoder_only=True`，BERT 式，时间 token 直接 `cat` 进主干
序列（`:297`），`T = horizon + 1`。这是 L1 位置。硬编码关闭。

**(b) `n_cond_layers>0`** → memory encoder 从 MLP 换成真正的 `nn.TransformerEncoder`
（`:184-196`），**时间 token 会与 patch token 自注意力混合**。硬编码为 0。

(b) 看似是免费的一行升级，**但不能直接开**。原代码 `:476` 是 `x = self.encoder(x)`，
**未传 mask**。cond token 一旦全连接自注意力，第 2 个观测窗口的 patch 信息会渗进第 1 个窗口
的 token，随后 decoder 通过 `memory_mask` 读早期 token 时即拿到未来信息——**块因果性失效**，
而这正是该论文唯一的贡献。

**`n_cond_layers=0` 不是随手设的默认值，是承重的。**

---

## 7. 建议

想加强 patch_policy 的时间条件，两条路，按成本排：

**7.1 给 cond encoder 补 mask（~10 行，更贴近原设计）**
恢复 `n_cond_layers>0` 分支，同时把 `generate_mask_matrix` 那个块因果矩阵
（第 0 行/列给时间 token 开全通）传进 `self.encoder(x, mask=...)`。复用现成掩码构造函数。
**必须留下的检查**：断言 decoder 位置 t 的感受野未越过第 t 个观测窗口——这是上节陷阱的
唯一防线，不能省。

**7.2 decoder 换 adaLN（~30 行）**
`nn.TransformerDecoderLayer` 换成带 adaLN 的自定义 block，零初始化保证不破坏已有权重的
初始行为。ACT head 不需要（无时间维度）。

**7.3 若改流匹配，同时换正弦底数**
见 §4 的坑。10000 → openpi 的 `min_period=4e-3, max_period=4.0`。这个 bug 不报错。

论文既已明说 "agnostic to the action head architecture"，在这一格上做消融不算偏离参考实现，
而是填作者声明为 out-of-scope 的空格——正好接上
`state-action-fusion-and-anchor-ratio-2026-09.md` §3 的四个空格子。

---

## 8. 限制

1. §3 表中 lerobot 侧各模型只核到时间注入的关键行，未通读整个 forward；
   注入点数按 `depth` 推算，未逐模型确认 depth。
2. §5.1 的论文引文取自 arXiv 摘要页与 HTML 全文，未核对 PDF 版式差异。
3. §5.3 的「逐字继承」判据是结构同名同构，未做逐行 diff；`n_patches` 与 `mem_bool` 两处
   差异是确认过的。
4. §7 的两条建议均未实现、未验证，工作量为估算。
5. 本文只覆盖时间注入；状态融合谱系与锚点比 `r` 见前一份报告，两者未做交叉实验。
