# RynnValue：用「时间距离」替代偏好/进度，把机器人价值模型做成基础模型

> 阅读笔记，撰写于 2026-08-13。
> 论文为 arXiv v1（2026-08-10 提交，PDF 标注 Date: August 11, 2026），当前只有一个版本。
> **架构部分以官方开源代码 `alibaba-damo-academy/RynnValue` 的 `rynn_value/` 包为准**，
> 并与 HuggingFace 上 `RynnValue-8B` / `RynnValue-4B` 的 `config.json` 交叉核对；
> 论文正文与代码存在若干不一致（尤其是指令错配样本的监督方式），已在 §模型结构 与 §存疑之处 中显式标注。
> 权重页面已挂出 config，但 README 自称「Checkpoint release is in progress」。

---

## Metadata

- **Title**: RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance
- **Authors**: Dongchi Huang\*, Hongyin Zhang\*, Bohan Hou\*, Siteng Huang†, Zhian Su, Hang Guo, Tong Lu, Zhaofeng Xu, Jiahao Tang, Jianfei Yang, Donglin Wang, Peixi Peng†, Mingxiu Chen, Deli Zhao, Xin Li（\* 共同一作，† 通讯）
- **Affiliations**: DAMO Academy, Alibaba Group；Hupan Lab
- **Venue / year**: arXiv:2608.09853v1（2026-08，cs.RO / cs.CV / cs.LG），未见投稿会议信息
- **Links**:
  - paper: https://arxiv.org/abs/2608.09853
  - project: https://alibaba-damo-academy.github.io/RynnValue.github.io
  - code: https://github.com/alibaba-damo-academy/RynnValue
  - models: https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnvalue （RynnValue-4B / 8B）
- **Tags**: reward model, value foundation model, temporal distance, cost-to-go, robot manipulation, potential-based shaping, VLA, IQL, DSRL, distributional RL, two-hot / symlog

```bibtex
@article{rynnvalue2026,
  title  = {RynnValue: Scaling Robotic Value Foundation Models with Temporal Distance},
  author = {Huang, Dongchi and Zhang, Hongyin and Hou, Bohan and Huang, Siteng and Su, Zhian
            and Guo, Hang and Lu, Tong and Xu, Zhaofeng and Tang, Jiahao and Yang, Jianfei
            and Wang, Donglin and Peng, Peixi and Chen, Mingxiu and Zhao, Deli and Li, Xin},
  journal= {arXiv preprint arXiv:2608.09853},
  year   = {2026}
}
```

---

## 一句话总结

**（机器人奖励建模 / 具身基础模型）** 本文提出 RynnValue：把通用机器人奖励模型的监督目标从「偏好对」和「归一化进度 ∈[0,1]」
换成**时间距离（temporal distance）——即从当前观测到语言指定目标的有向 cost-to-go，单位是物理秒**；
因为这个标签可以**直接从时间戳读出**，训练数据得以扩到 7,000+ 小时 / 3.09M 指令条件片段而**不需要任何偏好或进度标注**；
为防止模型在多帧输入下走捷径（靠采样间隔、序列位置、或从别的 value token 外推），提出
**随机时间采样 + 时序打乱 + value-isolation attention** 三件套；
在 RBM-EVAL-OOD 上 Kendall's τₐ 达 0.675，略超全偏好监督的 SOTA（0.655）；
通过 potential-based shaping（Φ_t = −v_t）接入真机双臂 Franka 的 offline IQL / online DSRL-SAC，
成功率从 52.5%→72.5%（online）、63.8%→82.5%（offline）。

---

## 研究背景与动机

### 领域现状

2025–2026 年通用机器人奖励模型（general-purpose robotic reward model）基本收敛到两条监督路线：

1. **偏好监督**：构造轨迹对/比较集，学一个 pairwise critic。代表：Robometer（RBM-1M）、VLAC、RoboReward。
   问题是偏好对的构造成本随数据规模线性上升，而且比较集是「任务内部」的，跨数据源不可迁移。
2. **归一化进度监督**：把每条轨迹的进度线性映射到 [0,1]，回归这个标量。代表：GVL、RARM、Robometer 的 progress 分支。
   问题更隐蔽——[0,1] 是**轨迹内部坐标**，不是控制意义上的 value：
   同样是「0.5」，在一条 5 秒的轨迹和一条 60 秒的轨迹里代表完全不同的剩余代价；
   执行速度、任务时长、本体形态一变，这个坐标就失去可比性。

### 核心矛盾

> **奖励模型想要「通用」，但它的监督信号（偏好集 / 进度归一化尺度）却是「任务内部定义」的——
> 于是数据源越异构，标签定义就越不一致，越无法把不同来源的数据加到同一个监督接口上。**

论文 §1 明确点出：normalized progress 是 *intra-trajectory coordinate*，而不是 goal-conditioned cost-to-go，
所以它「与控制中的 value 概念对不齐」。

### 本文目标

换一个**尺度不变、跨本体可加**的监督目标，使得「加数据」这件事真正可扩展。
成功标准是三条：(a) 标签零人工标注；(b) 在 OOD 轨迹排序上不输全偏好监督的 SOTA；(c) 能直接当 dense reward 驱动真机 RL。

---

## 核心贡献

论文 §1 结尾自列四条，我按重要性重排：

1. **把监督目标换成时间距离（cost-to-go，单位秒）**。最小时间目标（minimum-time objective）下，
   剩余时间就是 hitting-time cost-to-go，天然带方向性与目标条件性，且可通过 potential-based shaping 转成 dense reward。
   标签来自时间戳 + 子任务切分 + cutoff 重标定，无需偏好对、无需进度标注。
2. **一套 7,000+ 小时 / 1.67M episode → 3.09M 指令条件片段的异构数据配方**（10 个来源，223K 条不同指令）。
3. **两个 shortcut 抑制设计**：temporal-order shuffling（打断「序列位置↔进度」的对应）与
   value-isolation attention（阻断「从别的 value query 外推」），配合双分布头（absolute + relative）。
4. **实证**：无偏好标注即超过偏好监督 SOTA，并作为 zero-shot 奖励标注器提升真机 online/offline RL。

我的判断：真正新的是 (3) 和 (2) 的规模，(1) 在概念上并不新——
TimeRewarder（arXiv 2509.26627，论文引为 [18]）已经在做「从被动视频学 frame-wise temporal distance 作为 dense reward」，
quasimetric RL / hitting-time value 更是老概念。本文的贡献应当被读作
**「把 temporal distance 这个已知目标，第一次推到基础模型规模并解决随之出现的 shortcut 问题」**，而不是提出这个目标本身。
可惜论文没有与 TimeRewarder 做任何实验比较（见 §存疑之处）。

---

## 模型结构详解（论文 §2 + 代码 `rynn_value/`）

### 0. 骨干与规模

| 项 | RynnValue-8B | RynnValue-4B |
|---|---|---|
| backbone | RynnBrain-8B | RynnBrain-4B |
| 实现基类 | `Qwen3VLForConditionalGeneration` | 同左 |
| text hidden_size | 4096 | 2560 |
| text layers / heads / kv_heads | 36 / 32 / 8 | 36 / 32 / 8 |
| vision depth / hidden / out_hidden | 27 / 1152 / 4096 | 24 / 1024 / 2560 |
| vision patch / spatial_merge | 16 / 2 | 16 / 2 |
| rope | mrope interleaved, section [24,20,20], θ=5e6 | 同左 |
| attn_implementation | `pred_slot_isolated_eager` | 同左 |
| tie_word_embeddings | false | true |

RynnBrain（arXiv 2602.14979，论文 [5]）本身就是在 Qwen3-VL 上做具身预训练得到的，
所以 RynnValue 的代码可以直接继承 HF 的 `Qwen3VLForConditionalGeneration`
（`modeling_rynn_value_lang.py:12,50`）。**论文正文没有说骨干是 Qwen3-VL，这一点是从代码 import 确认的**，
README 明确写了 "implemented on the Qwen3-VL architecture"。

### 1. 输入序列布局

给定指令 ℓ、本体元信息 m、K 个观测 𝓘={I_{t_i}}（主设定 **K = 8**），构造的多模态序列为（论文 Eq. 1）：

```
x = [ m, ℓ, I_{t1}, V_1, I_{t2}, R_1, V_2, …, I_{tK}, R_{K-1}, V_K, p_ver ]
```

代码 `conversations.py:177-220`（`InterleavedHistoryConversationBuilder`）给出的实际 prompt 更具体：

```
[meta]  The agent is {robot_description}. The observation is captured from {camera_description}.
        The agent is performing the following task: {instruction}.
Question: For each frame after the first, what is the time delta from the previous frame?
Question: Estimate the minimum remaining time in seconds until the agent completes the task.
for i in 0..K-1:
    <image>
    if i > 0: <relative_value> × 8
    <value> × 8
Analyze this trajectory. Provide:
- Video Description: …
- Match: whether the video matches the stated task (Yes/No).
- Success: whether the agent has completed the task (Yes/No).
Analysis:
  - Video Description: …
  - Match: Yes/No
  - Success: Yes/No
```

注意几点：
- 两个 Question 放在**所有图像之前**，所以 query token 一出现就已经知道要答什么；
- `<relative_value>` 组插在**图像 i 之后、`<value>` 组之前**，即它是对「刚看到的这一帧与上一帧之间的 Δt」的**回溯**估计，不是外推；
- K=8 时共有 8×8=64 个 `<value>` + 7×8=56 个 `<relative_value>` = **120 个 query token**；
- meta 块由 `use_meta=True` 控制，两个发布的 checkpoint 都是 `use_meta: true`，
  所以**不传 `--robot_description` / `--camera_description` 会与训练分布不一致**（README 的 flag 表也提示了这点）。

### 2. Grouped temporal queries（N=8 重复 query token）

论文的说法是「单个 query token 是信息瓶颈」，所以每个预测槽用 N=8 个重复 token，
并**沿特征维拼接**（不是平均，论文 Eq. 2）：

```
h̃^V_i = H_θ(x)_{pos(V_{i,1})} ‖ … ‖ H_θ(x)_{pos(V_{i,N})}      →  维度 N·d
```

代码 `modeling_rynn_value_lang.py:234-244` 完全对应：按 `value_token_id` gather 出所有 `<value>` 位置的 hidden state，
reshape 成 `(total/R, R*D)` 再喂给 head。config 里 `value_token_repeat = relative_value_token_repeat = 8`。

**这里值得多想一层。** 一个槽内的 8 个 token 具有：完全相同的 token embedding、
完全相同的可见 key 集合（因果前缀 + 本槽内双向），唯一的差别是 **mrope 位置编码不同**。
所以它们本质上是「**8 个位置编码互不相同的探针（probe）**」，各自对同一份上下文算出不同的注意力分布，
再把 8 份池化结果拼起来。换句话说，这不是「更大的表征容量」，而是一个
**固定 query 数为 8 的可学习 attention pooling**——比 mean-pool 强的原因是拼接保留了 8 个视角的差异而不是抹平它。
（顺带一提：`conversations.py:106-122` 的 docstring 还写着 "the model mean-pools their hidden states"，
这是与实现不符的**陈旧注释**，实际实现是 concat。）

代价是 head 输入维度爆炸到 `N·d`：8B 是 8×4096 = **32768**。

### 3. 双分布头（BroNet）

```python
z^V_i = ValueHead(h̃^V_i)        # (256,)
z^R_i = RelativeHead(h̃^R_i)     # (256,)
```

代码 `value_heads.py:69-129` 的 `BroNet`（BRO，论文 [21]）结构为：

```
Linear(N·d → 4096) → LayerNorm → ReLU
  → 8 × ResidualBlock{ x + LN(Linear(LN(Linear(x))→ReLU)) }   # 每块两层 Linear + 两层 LayerNorm
  → Linear(4096 → 256)
```

config 确认：`head_type: bro, hidden_dims: 4096, depth: 8, activation: relu`，两个头结构相同、参数独立。

**参数量（我按代码逐层复算，论文未报告）**：

| | 8B（N·d=32768） | 4B（N·d=20480） |
|---|---:|---:|
| input_layer | 134.22 M | 83.89 M |
| 8 × ResidualBlock | 268.63 M | 268.63 M |
| final_layer | 1.05 M | 1.05 M |
| **单头合计** | **≈ 403.9 M** | **≈ 353.6 M** |
| **双头合计** | **≈ 807.8 M** | **≈ 707.2 M** |

也就是说，两个 value head 在 8B 上额外挂了约 **0.81 B** 参数（约骨干的 10%），
在 4B 上约 **0.71 B**（约骨干的 **18%**）。
**我的判断**：论文 §4.2.1 说「4B 已经很强、8B 只有边际提升，说明收益来自 temporal-distance 形式化而非模型规模」——
这个结论应当打个折扣，因为两档模型的 head 容量几乎一样大，而 head 恰恰承担了从 32768/20480 维拼接表征到 256 bin 的全部回归工作。
真正被 scaling 的那部分（骨干）确实没带来多少收益，但「4B≈8B」里有一部分是 head 容量被固定住了。

### 4. symlog + two-hot 分布式回归

`value_tokenizer.py` 实现了一个通用的「标量 ↔ bin 分布」编解码器：

- `symlog(x) = sign(x)·log1p(|x|)`，`symexp(x) = sign(x)·expm1(|x|)`（`value_tokenizer.py:6-11`）；
- bin 中心在 **symlog 空间**均匀（`torch.linspace(symlog(min), symlog(max), 256)`），映回原空间就是对小值密、大值疏；
- two-hot：把连续目标落在相邻两个 bin 上做线性劈分；
- 解码：`v = symexp( Σ_b c_b · softmax(z)_b )`（论文 Eq. 4，代码 `decode_from_bins`）。

发布 config 的实际取值：

| | 绝对头 | 相对头 |
|---|---|---|
| bins | 256 | 256 |
| 支撑区间 | **[0, 512] 秒** | **[−256, 256] 秒** |
| support_transform | symlog | symlog |
| encoding | two_hot（默认） | two_hot（默认） |

代码里还实现了论文未提的两个额外选项：**HL-Gauss 编码**（Farebrother et al. 2024，论文 [7] 同源）
和 **quantile（数据驱动的非均匀 bin_edges）**；发布 config 用的是默认 two_hot + symlog，与论文一致。
这说明作者试过更多的离散化方案，但论文没有报告任何对比消融。

**为什么要分布式回归而不是直接 L2 回归秒数**：论文 §2 的理由是
「symlog binning 压住长尾大值、近零目标几乎不失真；two-hot 让梯度幅度与目标尺度解耦」。
这个理由在这个数据上特别站得住——7000 小时跨 10 个数据源，任务时长从 2 秒到几百秒，
直接回归秒数会被长任务主导梯度。**但论文没有做「two-hot vs 直接回归」的消融**（见 §存疑之处）。

### 5. Value-isolation attention（本文最有意思的结构设计）

论文的动机很直接：如果不加约束，第 i 个 value query 可以**不看图**，直接从前面已经暴露的
value query 表征做线性外推，画出一条平滑的价值曲线——曲线好看，但对回退、失败、非单调事件完全不敏感。

代码 `attention_impl.py` 注册了一个自定义 eager attention `pred_slot_isolated_eager`，
核心逻辑（`attention_impl.py:49-60`）：

```python
same_slot = (slot_q == slot_k) & (slot_q >= 0)          # 同一个预测槽
add_mask  = add_mask.masked_fill(same_slot, 0.0)        # ① 槽内强制可见（覆盖因果 mask ⇒ 槽内双向）
pred_mask = is_pred_key & ~same_slot                    # ② 所有「非本槽」的 query key
add_mask  = add_mask.masked_fill(pred_mask, -inf)       # ③ 对任何 query 都不可见
```

槽 id 的构造在 `modeling_rynn_value_lang.py:341-356`：连续且 token id 相同的 special token 归为一个槽，
`cumsum` 出 `pred_slot_id`，非 query 位置置 −1。

于是形成三条规则（对应论文 Fig. 2(b) 的注意力矩阵）：

| query \ key | 上下文（m, ℓ, 图像, 文本） | 本槽 query | 其他槽 query |
|---|---|---|---|
| 上下文 token | 因果可见 | **不可见** | **不可见** |
| 本槽 query | 因果可见 | **双向可见** | **不可见** |

第一行第二/三列是关键：**上下文 token 看不到任何 query token**，
所以 (a) 后续的语言生成不会被已经算出的 value「污染」，(b) 价值信息也无法经由后续文本 token 间接传播。
生成阶段带 KV cache 时走 `cached_pred_key` 分支（`attention_impl.py:56-59`），
把所有缓存的 pred key 一律屏蔽掉——因为新生成的 token 永远不是 query token。

**两个容易被忽略的点：**

1. **value query 仍然能看到之前的所有图像。** isolation 只切断「query↔query」，不切断「query→更早的帧」。
   所以 V_i 是**历史条件**的估计 v(I_{t_1..t_i}, ℓ)，不是逐帧独立的 v(I_{t_i}, ℓ)。
   这解释了为什么推理时必须按时间顺序喂帧、以及为什么 demo 脚本要用前缀重采样（见 §7）。
2. **这是 eager attention，没有 FlashAttention / SDPA。** 因为需要一个不能表达为标准 causal mask 的
   自定义 additive mask（槽内双向 + 跨槽全屏蔽），`ALL_ATTENTION_FUNCTIONS` 里注册的是手写 matmul + fp32 softmax。
   K=8 帧、每帧约 200 个视觉 token 时序列长度约 1.8k，32 个 head 的注意力矩阵在 fp32 softmax 下每层瞬时约 0.4 GB，
   36 层逐层释放尚可承受，但**吞吐相对 SDPA 有明显损失**——这对「在线 RL 里当奖励服务器实时打分」是个真实成本。
   论文完全没有报告推理延迟或吞吐。

### 6. 语言分支与三项损失

语言分支通过**原始 LM head** 自回归生成 `Analysis` 块（论文 Eq. 5）：

```
y_lang = [ Video Description: y^vid , Match: y^match , Success: y^succ ]
```

监督范围由 `processing_rynn_value_lang.py:256-290` 决定：**在 input_ids 里定位字符串 `"Analysis: \n"`，
其后的所有 token 计 causal LM loss，之前全部置 −100**。
`Video Description` 的监督标签由 **Qwen3-VL-27B** 离线生成的 segment 级描述提供（论文 §3.1.2 末），
即这一路是蒸馏来的、不影响时间目标。

三项损失（论文 Eq. 6–9）：

| 损失 | 目标 | 归一化 | 错配样本 |
|---|---|---|---|
| ℒ_abs | v*_i = max(0, t_G − t_i) | 1/K（K=8） | 乘 ω=0 **屏蔽** |
| ℒ_rel | Δ*_i = t_{i+1} − t_i（可正可负） | 1/(K−1) | **保留** |
| ℒ_lang | Analysis 块 token | 1/\|𝒴\| | **保留**（监督为 Match:No / Success:No） |

```
ℒ = ℒ_abs + ℒ_rel + λ_lang · ℒ_lang ,   λ_lang = 2
```

论文 §3.2.3 末还有一条容易漏掉的实现细节：**LM output projection 被冻结**，
但 ℒ_lang 的梯度仍然通过这个固定投影回传到骨干；三项损失之间**没有任何 stop-gradient**。

> **⚠️ 论文与代码在这里不一致（重要）。**
> 论文说错配样本的绝对损失「乘 ω=0 屏蔽」。代码 `modeling_rynn_value_lang.py:147-153` 做的是**另一件事**：
> ```python
> uniform = torch.full_like(target_dist, 1.0 / n_bins)
> target_dist = torch.where(mask, uniform, target_dist)
> ```
> 即把错配槽的 two-hot 目标**替换成 256 个 bin 上的均匀分布**（注释写得很明白：
> "trained to be maximally uncertain when the instruction doesn't match"）。
> 这**不是零梯度**，而是一个**最大熵正则**——主动把错配样本的 value 分布推平。
> 两者的行为差别不小：ω=0 只是不学，均匀目标是「学会不确定」。
> 后者还让 `out.value.entropy`（模型确实把逐槽熵作为输出字段暴露出来了）成为一个可用的
> **指令-视频错配检测信号**，这是论文完全没有讨论的一个免费能力。
> 我倾向于认为**代码是最终版本，论文 Eq. 6 的描述是简化/过时的**，但作者没有说明。

### 7. 推理与奖励接口

**推理时**（论文 §3.3）：关闭全部训练期采样增强，帧按时间顺序输入，value-isolation mask 保持开启。

**势函数**（论文 Eq. 10）：`Φ_t = −v_t`。剩余时间越长势越负，到达目标时趋近 0。
注意这里**不做 [0,1] 归一化**，保留物理秒的尺度——这正是「跨任务可加」的立足点。

**shaping**（论文 Eq. 11）：

```
r'_t = κ·( γ·Φ_{t+1} − Φ_t ) + { 0    if t=T 且成功
                                { −1   otherwise
```

其中 **κ = 0.1（RynnValue）/ κ = 1.0（Robometer）**，稀疏成功项由人工标注保留。

**⚠️ demo 推理不是「一次前向出整条曲线」。** `rynn_infer/inference.py:210-265` 用的是
**prefix-uniform 采样**：对每个评估步 t，在 [0, t] 上均匀重采 `num_frames` 帧重建一个前缀样本，
跑一次前向，**只读最后一个 value 槽**。所以 T 个时间点 = T 次前向（脚本用 batch 并行）。
这意味着：(a) 复杂度是 O(T) 次 8 帧前向，不是流式；
(b) 每个时刻的观测窗口跨度都在变（早期帧密、后期帧疏），前后打分之间存在细微的非平稳性。
这与论文 §5 把「streaming inference」列为 future work 是一致的，但论文正文没有交代实际打分协议是前缀重采样。

`README` 里那段编程示例 `out.value.pred_value.float().mean(dim=-1)` 注释为 "head-ensemble mean"，
但 `pred_value` 的形状是 `(num_heads, num_slots)`，`dim=-1` 平均的是**帧**不是 head；
`inference.py:247` 用的才是正确的 `pred.mean(dim=0)`。README 这一行是笔误。
（另：发布 config 的 `num_value_heads = 1`，所以这个 ensemble 维度目前恒为 1，代码保留了扩展位。）

### 8. 结构总览（我的重画）

```
 m, ℓ (含两个 Question)
   │
   ├─► I_{t1} ─► [V_1 ×8] ──┐
   ├─► I_{t2} ─► [R_1 ×8][V_2 ×8] ──┤
   │      ⋮                          │  value-isolation:
   └─► I_{tK} ─► [R_{K-1}×8][V_K×8] ─┤   槽内双向 / 跨槽全断 / 上下文看不见槽
                                     │
        RynnBrain (Qwen3-VL, 36L) ───┴──► hidden states
                │                              │
                │                    concat 8×d (=32768 @8B)
                │                       ├──► BroNet(8 blocks,4096) ──► 256 bins symlog[0,512]  ──symexp──► v_i (秒)
                │                       └──► BroNet(8 blocks,4096) ──► 256 bins symlog[-256,256] ─symexp──► Δ_i (秒)
                │
                └──► (冻结投影的) LM head ──► "Video Description / Match / Success"
```

---

## 训练配方

### 数据（论文 Table 1）

1.67M 原始 episode → **3,091,969** 个指令条件片段，223,395 条不同指令，7,000+ 小时。

| 来源 | 原始 episodes | 切分后 | 指令数 | 切分方式 |
|---|---:|---:|---:|---|
| Open X-Embodiment | 693,037 | 693,037 | 180,090 | full trajectory |
| EgoDex（第一人称人手） | 338,234 | 338,234 | 2,038 | full trajectory |
| InternData-A1（合成） | 320,905 | 320,905 | 348 | full trajectory |
| AgiBot | 167,535 | **1,166,042** | 3,741 | coarse task |
| RoboCOIN | 67,420 | 410,877 | 2,124 | coarse task |
| RoboMIND | 32,138 | 32,138 | 184 | full trajectory |
| RoboTwin（仿真） | 27,414 | 27,414 | 23,527 | full trajectory |
| Galaxea Open-World | 16,979 | 95,671 | 11,070 | coarse task |
| RDT | 6,109 | 6,109 | 272 | per-file coarse |
| Soft-FOLD | 1,542 | 1,542 | **1** | per-file coarse |
| **合计** | **1,671,313** | **3,091,969** | **223,395** | |

**标签生成**：每个片段指定一个 completion cutoff（默认取片段末尾，必要时按数据集特定的比例/时长裁剪，
以逼近「第一个语义完成的观测」）；cutoff 之前的帧标为剩余时间，cutoff 及之后标为 0。

**我的观察**：AgiBot 一家就从 167K 扩到 1.166M 片段（×7），占了总量的 38%；
Open X-Embodiment 占 22%。所以「7000 小时 / 3M 片段」这个数字里，
**子任务切分的乘数效应贡献很大，而切分策略是逐数据集手工定的**（Table 1 最后一列三种策略）。
这跟论文「不需要人工定义 dataset-specific progress scale」的卖点略有张力——
progress scale 确实不用定了，但 **cutoff 的裁剪比例还是逐数据集调的**（§3.1.2 承认 "dataset-specific ratio- or duration-based trimming is applied when necessary"）。
换句话说，人工先验从「进度尺度」搬到了「完成点定义」上，只是搬得更薄了。

### 三个抗捷径设计

| 设计 | 做法 | 抑制的捷径 |
|---|---|---|
| **随机时间采样** | K=8 帧在不规则时间戳上采样 | 均匀采样导致的「等差价值序列」 |
| **时序打乱** | 50% 序列不排序；另 50% 走前向偏置随机游走，**rewind 概率 0.3** | 「序列位置 ↔ 任务进度」的对应 |
| **指令错配增强** | 10% 样本换成别的轨迹的指令，监督 Match:No / Success:No | 「无论看到什么都报告平滑进度」 |

时序打乱使得**相对目标 Δ*_i 可以为负**——这正是相对头需要 [−256,256] 支撑的原因。

### 优化超参（论文 §4.1）

| 项 | 值 |
|---|---|
| optimizer | AdamW，lr **1e-6** 常数（无 warmup） |
| β₁ / β₂ / wd / ε | 0.9 / 0.95 / 0.1 / 1e-8 |
| grad clip | 100 |
| precision | bfloat16 + FSDP hybrid sharding |
| per-device batch | **2** |
| K（每样本观测数） | 8 |
| N（每槽重复 query） | 8 |
| shuffling prob / rewind prob | 0.5 / 0.3 |
| mismatch aug | 10% |

**论文未报告**：总训练步数、总 batch size、GPU 数量与型号、训练时长、数据采样权重（10 个来源如何配比）。
lr=1e-6 + 常数调度 + per-device batch 2，暗示这是在一个已经很强的具身预训练骨干上做的轻量微调，
但没有全局 batch size 就无法判断实际的更新规模——**这对复现是硬伤**。

---

## 实验结论

### 1. RBM-EVAL-OOD 轨迹排序（Table 2，Kendall's τₐ ↑）

测试集：Robometer [16] 提出的 OOD 轨迹排序赛道，**976 条轨迹**，横跨 6 个机构/本体/视角。
打分方式：用最后一个查询观测的势 −v_end 作为轨迹分数（只用绝对头，相对头与语言输出不参与）。

| Method | USC Franka | USC Koch | USC Trossen | USC xArm | MIT Franka | UTD SO101 | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|
| GVL | 0.250 | −0.008 | 0.292 | 0.056 | 0.306 | 0.300 | 0.199 |
| VLAC-8B | 0.271 | 0.064 | −0.417 | 0.139 | 0.072 | 0.167 | 0.049 |
| ReWiND | −0.125 | 0.336 | 0.028 | −0.167 | 0.080 | −0.067 | 0.014 |
| Dopamine-GRM-2.0-8B | 0.479 | 0.442 | 0.333 | 0.431 | 0.431 | 0.700 | 0.453 |
| RoboReward-4B | 0.625 | 0.332 | 0.333 | 0.528 | 0.494 | 0.700 | 0.502 |
| Robometer (RoboReward data) | 0.583 | 0.533 | 0.646 | 0.403 | 0.479 | 0.667 | 0.552 |
| **Robometer (RBM-1M)** 偏好 SOTA | 0.646 | 0.471 | 0.653 | **0.694** | **0.601** | 0.867 | 0.655 |
| Robometer (Progress only) | 0.083 | 0.231 | 0.333 | 0.389 | 0.183 | 0.533 | 0.292 |
| RynnValue-4B | 0.542 | 0.488 | 0.917 | 0.667 | 0.473 | **0.933** | 0.670 |
| **RynnValue-8B** | **0.667** | **0.544** | **1.000** | 0.500 | 0.503 | 0.833 | **0.675** |

**读法要小心**：平均分 0.675 vs 0.655 只领先 **0.020**，而且这个领先几乎完全来自
**USC Trossen 一个数据集**（1.000 vs 0.653，+0.347）。在 USC xArm（0.500 vs 0.694）和
MIT Franka（0.503 vs 0.601）上 RynnValue-8B **明显落后于偏好 SOTA**。
τₐ 恰好等于 1.000 / 0.917 这类取值说明该子集的轨迹数极少（τₐ 是离散取值的）。
论文没有报告任何 seed 方差、置信区间或每个子集的轨迹数量。
**我的判断**：「超过全偏好监督 SOTA」这句话在字面上成立，但证据强度撑不住「更好的奖励模型」这个一般性结论；
撑得住的结论是「**在零偏好标注的前提下达到了与偏好 SOTA 同一水平**」——这本身已经足够有价值，不需要夸。

### 2. 消融（Table 3）

| 变体 | USC Franka | USC Koch | USC Trossen | USC xArm | MIT Franka | UTD SO101 | **Avg** |
|---|---:|---:|---:|---:|---:|---:|---:|
| w/o Shuffle（时序打乱） | 0.583 | 0.090 | 0.055 | 0.222 | −0.017 | 0.200 | **0.189** |
| w/o Isolation（隔离注意力） | 0.583 | 0.428 | 0.694 | 0.389 | 0.400 | 0.400 | **0.482** |
| w/o Language（语言监督） | 0.250 | 0.491 | 0.819 | 0.361 | 0.501 | 0.800 | **0.537** |
| Uniform Sampling（去随机采样） | 0.375 | 0.400 | 0.305 | 0.250 | 0.310 | 0.633 | **0.379** |
| w/o Relative（去相对头） | 0.667 | 0.587 | 0.639 | 0.639 | 0.464 | 0.767 | **0.627** |
| **Full (8B)** | 0.667 | 0.544 | 1.000 | 0.500 | 0.503 | 0.833 | **0.675** |

重要度排序清晰：**时序打乱（−0.486）≫ 随机采样（−0.296）> 隔离注意力（−0.193）> 语言监督（−0.138）> 相对头（−0.048）**。

这张表其实是全文最有说服力的证据，而且它讲的故事比标题更犀利：
**去掉时序打乱后 τₐ = 0.189，比 progress-only 基线（0.292）还差**。
也就是说，在多帧设定下，「序列位置」这个捷径的破坏力大到足以让一个 7000 小时的模型退化到不如一个弱基线。
这条经验对任何做多帧价值/进度估计的人都值得直接借鉴。

注意 w/o Relative 只掉 0.048（且在 4 个子集上反而更好），说明**相对头对排序指标贡献很小**；
论文用「USC Trossen 从 1.000 掉到 0.639」来论证它重要，但那恰好是最可疑的那个子集。
相对头真正的价值可能在价值曲线的局部平滑性（§4.4 定性）而非排序精度。

### 3. 指令-轨迹对齐（Fig. 3）

用每条指令 × 每条轨迹打分构成混淆矩阵，看对角线是否集中。归一化对角 margin：

| VLAC-8B | Dopamine-GRM-2.0-8B | RoboReward-4B | RoboReward-8B | Robometer(RBM-1M) | **RynnValue** |
|---:|---:|---:|---:|---:|---:|
| −0.53 | 0.18 | 0.60 | 0.67 | 0.59 | **0.79** |

所有基线都在统一协议下用公开权重重跑。这条结果比 τₐ 更能支撑「语言条件是真的起作用」，
而且与 10% 指令错配增强的设计直接对应，因果链完整。

### 4. Scaling：任务多样性 ≫ episode 数量（Fig. 4）

两组对照，在**总 episode 数可比**的前提下分别缩放：
- 橙线（episode volume，任务集固定）：**1% → 10% 之后几乎立刻饱和**，MAE 卡在 ~1.8 秒不再下降；
- 蓝线（task diversity，每任务 episode 数固定）：从 3.3 秒**单调下降**到 ~1.6 秒，过了中点仍有可观收益。

**这是全文对后续工作最有价值的一条结论**：对 value/reward 模型而言，同一任务重复采数据的边际收益极低，
扩任务覆盖才是关键。（论文没有报告这些子集的绝对规模，也没说 MAE 的评测集大小，所以只能当定性结论用。）

### 5. 真机 RL（Table 4）

平台：双臂 Franka，两个腕部相机 + 左右两个第三人称相机。四个任务，**每个任务 20 次试验**。
四个任务的物体/场景**均未出现在奖励模型训练语料中**，RynnValue 是 **zero-shot 奖励标注器**。

| | Bread Basket | Steak+Spatula | Box-in-Drawer | Bimanual Box | **Avg** |
|---|---:|---:|---:|---:|---:|
| **Online RL** (DSRL-SAC) | | | | | |
| RynnValue | 45.0% | **75.0%** | 70.0% | **100.0%** | **72.5%** |
| Robometer | 35.0% | 45.0% | 65.0% | 65.0% | 52.5% |
| Sparse | 40.0% | 45.0% | 40.0% | 70.0% | 48.8% |
| **Offline RL** (IQL on π₀.₅) | | | | | |
| RynnValue | **100.0%** | **90.0%** | **90.0%** | **50.0%** | **82.5%** |
| Robometer | 80.0% | 80.0% | 50.0% | 45.0% | 63.8% |
| Sparse | 70.0% | 20.0% | 0.0% | 0.0% | 22.5% |
| SFT (π₀.₅) | 70.0% | 25.0% | 0.0% | 0.0% | 23.8% |

最强的证据是 **Box-in-Drawer 和 Bimanual Box Transfer：SFT 与 sparse reward 都是 0%，RynnValue-IQL 做到 90% / 50%**。
这不是「提升几个点」，是从不可解到可解——对「dense reward 有没有用」这个问题是硬证据。
同时 RynnValue 的平均动作 chunk 数也更少（如 Bread Basket 16.8 vs Robometer 18.9 vs SFT 24.8），说明不只是成功率，效率也更好。

**但要注意实验设计里的几个混淆**：
- 每任务仅 20 trials，72.5% vs 52.5% 的差距在四任务合计 80 trials 上大约是 2σ 量级，**没有报告方差或多 seed**；
- online RL 的初始化**逐任务不同**：Bread Basket / Steak 从 SFT checkpoint 起步，
  Box-in-Drawer / Bimanual 从 **Robometer 的 offline-RL checkpoint** 起步。作者说这个初始化在各奖励变体之间保持一致（公平），
  但这意味着后两个任务的 online 数字是「在对手预热过的起点上继续跑」，绝对值不好直接解读；
- shaping 强度 **κ 对两个模型不同（0.1 vs 1.0）**。理由（尺度不同：秒 vs [0,1]）合理，但论文没说这两个 κ 是怎么定的、
  Robometer 的 κ 有没有同等调参预算。

---

## 局限与存疑之处

### 作者承认的（§4.5.2, §5）

1. **精细接触任务上 online 增益有限**。Box-in-Drawer 从共享的 50% 起点，Robometer 到 65%、RynnValue 到 70%——
   作者归因为「奖励模型只看第三人称 RGB，视觉上相似的构型对应完全不同的抓取稳定性和对位精度」。
2. **只看短窗口的采样观测**，无法流式推理，不适合长时程。
3. **目标假设近似最小时间**，没有编码能耗、安全、精度等任务特定代价。
4. 尚未扩展到灵巧手与移动操作。

### 我补充的（基于证据）

5. **⚠️ 论文 Eq. 6 与代码不一致**：错配样本的绝对损失论文说 ω=0 屏蔽，代码是**替换为 256 bin 均匀分布**（最大熵目标）。
   这是行为差异而非表述差异。复现者若按论文实现会得到不同的模型（尤其是 `value.entropy` 这个错配信号会消失）。
6. **最核心的结构贡献没有消融**。Table 3 消融了 4 个训练/结构组件，但**没有消融 N=8 重复 query token**——
   而这恰恰是 §2 第一个被命名的设计（"Grouped temporal queries"），且带来了 0.8B 额外参数。
   同样缺失的还有：K 的取值、two-hot vs 直接回归、symlog vs linear bin、bin 数、BroNet 深度/宽度。
   代码里明明实现了 `hl_gauss` 和 `quantile` 两种替代离散化，却一个数字都没报。
7. **未与最接近的先验工作比较**。TimeRewarder [18]（"Learning dense reward from passive videos via frame-wise
   temporal distance"）在概念上就是本文的目标函数，论文只在引言引用、**没有任何实验对比**。
   RARM [31]、TimeRewarder 都缺席 Table 2。在「temporal distance 是不是比 progress 好」这个核心命题上，
   唯一的对照是 Robometer 自家的 progress-only 分支（0.292），而那是**同一套架构下的消融变体，不是一个认真训练的进度基线**。
8. **主结果的领先幅度脆弱**。0.675 vs 0.655 且优势集中在单个小子集；两个子集上反而落后。无方差、无 seed。
9. **复现信息不足**：没有全局 batch size、训练步数、算力、10 个数据源的采样配比、评测子集规模。
   README 也承认权重「release in progress」。
10. **推理成本未报告**。自定义 eager attention（无 SDPA/Flash）+ 每个时间步一次 8 帧前向的前缀重采样协议，
    对在线 RL 的实时性是真实约束，但全文没有一个延迟或吞吐数字。
11. **子任务 cutoff 仍需逐数据集调**（§3.1.2 的 "dataset-specific ratio- or duration-based trimming"），
    人工先验并未完全消除，只是从「进度尺度」搬到了「完成点定义」。
12. **文档小 bug**：README 编程示例的 `pred_value.mean(dim=-1)` 平均的是帧维不是 head 维（`inference.py:247` 才对）；
    `conversations.py` 的 docstring 仍写 "mean-pools"，实现是 concat。

---

## 与邻近工作的定位

| | 监督信号 | 标注成本 | 尺度语义 | 跨数据源可加 |
|---|---|---|---|---|
| GVL / RARM / Robometer-progress | 归一化进度 ∈[0,1] | 需定义每任务进度尺度 | 轨迹内部坐标 | ✗ |
| Robometer(RBM-1M) / RoboReward / VLAC | 轨迹偏好对 | 需构造比较集 | 相对序 | 弱 |
| ReWiND | 语言引导奖励 | 需语言标注 | — | 弱 |
| TimeRewarder [18] | 帧间 temporal distance（被动视频） | 时间戳 | 时间 | ✓（但规模小） |
| **RynnValue** | **到目标的 cost-to-go（秒）** | **时间戳 + cutoff 重标定** | **物理时间** | **✓** |

真正的差异不在「用时间当监督」（TimeRewarder 已经在做），而在三点：
**(a) 目标是到 goal 的 cost-to-go 而非帧间距离；(b) 规模（7000 小时 / 3M 片段 / 10 个来源）；
(c) 在这个规模下多帧建模暴露出的 shortcut 问题及其解法。** (c) 是我认为最值得引用的部分。

---

## 实践要点：可以借走什么

**值得直接借鉴的：**

1. **时序打乱是多帧进度/价值建模的必需品，不是可选项。** τₐ 0.675 → 0.189 这个跌幅说明，
   只要你的模型一次看多帧且帧按时间排序，它就会去拟合位置而不是内容。任何做 multi-frame progress head 的工作都应该先做这个消融。
2. **value-isolation attention 的实现模式很干净**（`attention_impl.py` 72 行）：
   通过 `ALL_ATTENTION_FUNCTIONS` 注册自定义 additive mask，用 `pred_slot_id` 的 cumsum 划槽，
   槽内置 0（覆盖因果 mask）、跨槽置 −inf。**这是一个通用的「在 LLM 里插入互不干扰的预测槽」的模板**，
   可以直接搬到任何「让 VLM 在序列多处输出独立预测」的场景（多帧动作、多目标检测、多步价值）。
   代价是失去 FlashAttention。
3. **symlog + two-hot 分布头处理长尾连续目标**。跨数据源的时长分布跨两三个数量级时，
   这比直接回归稳得多，且 `value_tokenizer.py` 的实现是自包含的（<300 行，含 two-hot / HL-Gauss / quantile 三种）。
4. **保留物理量纲、不做 [0,1] 归一化**，再用 Φ = −v 转 potential-based shaping。
   这是「跨任务的奖励可加性」的关键一步，比归一化进度干净得多。
5. **扩任务而不是扩 episode**（Fig. 4）。如果你在攒具身数据，这条比任何架构技巧都值钱。

**要小心的：**

- 复现时按**代码**而不是论文 Eq. 6 实现错配监督（均匀分布目标，不是 ω=0）。
- 发布 checkpoint 是 `use_meta=True` 训练的，推理必须传 robot/camera description，否则分布不匹配。
- 打分协议是前缀重采样（每步一次 8 帧前向），不是流式；做在线 RL 要预算好这个开销。
- 两个 value head 加起来 0.7–0.8B 参数，别把它当成「一个轻量头」。

---

## 开放问题

1. **N=8 重复 query 到底贡献多少？** 如果 N=1 掉得不多，那 0.8B 的 head 就该被砍掉，
   整个「grouped temporal queries」的叙事需要重写。这是我最想看到的一个消融。
2. **均匀分布目标（代码版）与 ω=0（论文版）哪个更好？** 前者附带的 entropy 信号能否单独作为
   一个零成本的失败/错配检测器？论文暴露了 `value.entropy` 却完全没有评测它。
3. **cost-to-go 与最小时间假设的错配。** 当演示数据里存在大量「慢而稳」与「快而险」并存的执行时，
   最小时间目标会不会把策略推向不安全的快？§5 提到要加 energy/safety cost，但这其实动摇了
   「时间距离 = value」的等价前提，值得单独研究。
4. **value-isolation 之下，V_i 仍能看到全部历史帧**，所以模型是历史条件的。
   如果改成完全逐帧独立（连历史帧也隔离），τₐ 会掉多少？这决定了这个模型到底是
   「状态价值函数」还是「轨迹进度估计器」——两者在 RL 里的用法很不一样。
5. **能否直接把它当 VLA 的 critic 而非 shaping 项？** 目前是 IQL 里另训一个 PixelIQL critic，
   RynnValue 只提供 shaping。既然它已经输出 cost-to-go，理论上可以直接充当 Q/V 的初始化或正则。
