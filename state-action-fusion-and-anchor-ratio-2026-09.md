# 状态–动作–图像的融合谱系，以及一条可以写进论文的定量结论

> **撰写日期** 2026-09-04
> **前置** `proprioception-code-level-argument-2026-09.md`（ACT / patch_policy 的代码级论证）。
> 本文做两件那份没做的事：**(1) 逐行核对 π₀ / π₀.₅ 在扩散之前到底喂了什么**（你的疑问）；
> **(2) 把本仓库里 20 个策略的「state / action / image 三者怎么融合」拉成一张代码级谱系表**。
> 顺带，在核对 09-03 新权重时算出了一条 **R²=0.96 的定量规律**，写在 §0——
> 我认为这就是你说的「还没有一个能写进论文的结论」的那个结论。
>
> **代码基准**
> `/home/kewei/YING/openpi_yw`（openpi fork）与
> `/home/kewei/YING/robot_data_platform/lerobot/src/lerobot/policies/`（20 个策略实现）。
> 表里每一格都有文件名+行号，不是从论文摘要转述的。
> **文献基准** §5 每条都用 WebFetch 核过 arXiv 摘要页，核不上的当场删除并注明。

---

## 0. 先给结论：一个标量 `r` 预测所有「重锚」干预的正负号

### 0.1 定义

对任一权重，在同一评测集上取两个数：

- `mae@1`：策略预测的 chunk **第 1 步**的 MAE
- `hold@1`：null 基线 `hold_state` 的第 1 步 MAE ——**即机器人在一个 waypoint 里真实移动了多少**

定义 **锚点比 `r = mae@1 / hold@1`**。

`r` 的物理含义很直白：**策略在第一步就已经偏掉了几个「真实运动量」。**
`r ≫ 1` 表示输出里绝对锚点 `s_t` 没对上，误差与要走的距离不是一个量级；
`r ≈ 1` 表示锚点基本对上了，剩下的是形状误差；
`r < 1` 表示策略在第一步比「原地不动」还准，锚点信息已经被完全吃进去了。

### 0.2 规律

把 `eval_policy/runs/` 里**所有**同时有 `policy_raw` 和 `policy_deployed`、
horizon 对齐到 50 的权重全取出来（10 个，横跨 2 个动作空间、4 种 head、3 个模型家族），
横轴 `r`，纵轴「部署栈的状态桥让 MAE 变化了百分之几」：

| run | 空间 | head | `mae@1` | `hold@1` | **`r`** | raw | deployed | **桥的效果** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `vqbet_100k` | joint | vqbet | 0.05353 | 0.01556 | **3.440** | 0.06723 | 0.05897 | **−12.29%** |
| `prev_diffusion` | joint | diffusion | 0.04455 | 0.01556 | **2.863** | 0.06773 | 0.06206 | **−8.37%** |
| `prev_act_head` | joint | act | 0.04107 | 0.01556 | **2.639** | 0.06349 | 0.05844 | **−7.95%** |
| `new_obs2` | joint | diffusion | 0.04022 | 0.01556 | **2.585** | 0.06190 | 0.05838 | **−5.69%** |
| `new_state5` | joint | diffusion | 0.03961 | 0.01556 | **2.546** | 0.06121 | 0.05827 | **−4.80%** |
| `act_baseline` | joint | ACT | 0.02529 | 0.01556 | **1.625** | 0.05112 | 0.05088 | **−0.47%** |
| `pp_eef_nostate` | eef | diffusion | 0.02613 | 0.02098 | **1.245** | 0.03916 | 0.04090 | **+4.44%** |
| `pp_eef_act5` | eef | act | 0.02564 | 0.02098 | **1.222** | 0.03796 | 0.03912 | **+3.06%** |
| `pp_eef_state` | eef | diffusion | 0.02254 | 0.02098 | **1.074** | 0.03721 | 0.04034 | **+8.41%** |
| `acteef_533` | eef | ACT | 0.01694 | 0.02098 | **0.807** | 0.03443 | 0.03794 | **+10.19%** |

**（负号 = 桥让策略变好，正号 = 桥让策略变差。）**

线性拟合：

```
桥的效果(%) = −8.23 · r + 15.15        Pearson = −0.980   R² = 0.961   n = 10
零点：r = 1.84
```

**这不是两个簇拼出来的假象**，两个空间各自单独拟合都成立，而且两条线的零点几乎重合：

| 子集 | n | Pearson | R² | 斜率 (%/单位 r) | 零点 r |
|---|---:|---:|---:|---:|---:|
| 只用 joint 的 6 个 | 6 | −0.975 | 0.950 | −6.59 | **1.615** |
| 只用 eef 的 4 个 | 4 | −0.917 | 0.841 | −15.19 | **1.517** |
| 全部 10 个 | 10 | −0.980 | 0.961 | −8.23 | 1.841 |

唯一的反序是 `pp_eef_nostate`(r=1.245, +4.44%) 与 `pp_eef_act5`(r=1.222, +3.06%)
差了 1.4 个百分点——扩散头自身的采样抖动就有 1.58%，落在噪声里。

### 0.3 这条规律为什么值得写进论文

因为它把三件一直被当成三件事的东西合并成了一件：

| 干预 | 在管线的哪一环 | 做的是同一件事 |
|---|---|---|
| `use_robot_state=true` | **输入端** | 把锚点 `s_t` 交给网络 |
| 相对动作 / EEF 空间 | **目标端** | 把锚点从回归目标里减掉 |
| 部署侧 Hermite 状态桥 | **执行端** | 把锚点在输出上加回去 |

三者都是「**用外部已知的 `s_t` 去替代网络内部重建的 `ŝ_t`**」。
`r` 度量的正是「网络内部那个 `ŝ_t` 还有多差」。所以：

> **`r` 大 → 网络重建的锚点很差 → 任何一种外部重锚都赚；
> `r` 小 → 网络重建的锚点已经比 `s_t` 本身携带的信息更好 → 强行重锚是在扔掉信息，会亏。**

零点 `r ≈ 1.5–1.6` 是一条**可直接执行的部署判据**：
拿任何一个新权重，用一次离线评测（几分钟）算出 `r`，
就能预判要不要开状态桥、要不要打开本体感觉、值不值得换动作空间——
不用真机试错。这在文献里我没有找到对应物（§5.6）。

### 0.4 顺带订正昨天报告里的一句话

昨天我写「执行侧重锚有效（+26–30%）」。
**在关节空间成立，在 EEF 空间号相反**：09-03 的四个 EEF 权重，桥让**每一个**都变差 3–10%。
昨天那份报告成文时 `20260903_pp_eef_state_head/` 还没有结果。
上面这条 `r` 规律把两个方向统一了，不是推翻。

另外昨天说「`use_robot_state` 打开对锚点份额零影响」——
那句话针对的是**关节空间的锚点份额**，仍然成立（39.6% vs 39.9%）。
但 09-03 的 EEF 单变量对照给出了这个开关**在 MAE 上**的干净读数：
**raw +5.2%、@1 +15.9%、位置 +8.7%（11.28→12.27 mm），采样噪声下限 1.58%。**
所以准确的说法是：**本体感觉当输入是有效的，但效应量只有 5–16%，且不改变误差的锚点/形状构成。**

---

## 1. π₀ / π₀.₅：扩散之前到底喂了什么

你的观察——「pi 里面 action 侧的数据在扩散之前是有输入的」——**方向对，但两个模型的答案完全不同，而且 π₀.₅ 恰好是最反直觉的那个**。

### 1.1 π₀：state 是 suffix 里的一个连续 token

`openpi_yw/src/openpi/models/pi0.py:141-151`：

```python
if not self.pi05:
    # add a single state token
    state_token = self.state_proj(obs.state)[:, None, :]
    tokens.append(state_token)
    ar_mask += [True]          # 图像/语言不能看 state，state 之后的 action token 能看
action_tokens = self.action_in_proj(noisy_actions)
```

序列布局：`[图像 tokens][语言 tokens] | [state token][50 个 noisy action token]`。
竖线左边是 prefix（VLM 2B expert，KV-cache），右边是 suffix（action expert 300M）。
**state 是全精度 float，直接坐在 action token 旁边，两者在同一个 softmax 里。**

### 1.2 π₀.₅：`state_proj` **根本不存在**

同一个文件 `pi0.py:92-99`：

```python
if config.pi05:
    self.time_mlp_in  = nnx.Linear(...)
    self.time_mlp_out = nnx.Linear(...)
else:
    self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, ...)
```

lerobot 的移植里把这件事写得更露骨（`policies/pi05/modeling_pi05.py:1101-1103`）：

```python
# Also handle state_proj which shouldn't exist in pi05
if key.startswith("state_proj."):
    logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
```

那 state 去哪了？**被离散成 256 档，以十进制数字的形式写进语言 prompt。**
`openpi_yw/src/openpi/models/tokenizer.py:22-31`：

```python
def tokenize(self, prompt: str, state: np.ndarray | None = None):
    if state is not None:
        # This is the Pi05 format, where the state is part of the discrete language input.
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256+1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

lerobot 逐字复现（`policies/pi05/processor_pi05.py:75-83`）。
`Pi0Config.__post_init__` 里 `discrete_state_input` 默认跟随 `pi05`，
`max_token_len` 也因此从 48 涨到 200（`pi0_config.py:38-41`）——多出来的 152 个 token 位就是给状态数字串的。

**所以 π₀.₅ 的 embed_suffix 的完整入参是（`pi0.py:141-186`）：**

```
embed_suffix(obs, noisy_actions, timestep)
  → tokens      = action_in_proj(x_t)                    # 只有噪声动作
  → adarms_cond = time_mlp_out(swish(time_mlp_in(t_emb)))# 时间走 adaRMSNorm，不占 token
```

**没有 state token，没有时间 token。** action expert 的 token 序列里只有 50 个噪声动作。

### 1.3 三个「有输入」要分清

回到你的原话。「扩散之前有输入」可以指三件事，π₀.₅ 的答案分别是：

| 说法 | π₀ | π₀.₅ | patch_policy |
|---|---|---|---|
| **(a) 噪声动作块 `x_t` 进网络** | ✓ `action_in_proj` | ✓ `action_in_proj` | ✓ `sample` 参数 |
| **(b) 本体感觉 `s_t` 进网络** | ✓ 连续 token，suffix | **✓ 但是 256 档离散数字，在 prefix 语言里** | `use_robot_state` 开关 |
| **(c) 真值未来动作进网络** | ✗ | ✗ | ✗ |

**(a) 对所有去噪模型都成立，不是 π 的特点**——`x_t` 是去噪的自变量，
patch_policy 的 `TransformerForDiffusion.forward(sample, timestep, cond)` 第一个参数就是它。
拿 (a) 去论证「π 有 action 输入而 patch_policy 没有」会被审稿人一句话驳回。

**(c) 才是真正的区分点，而 ACT 是唯一命中的那个**：ACT 的 CVAE encoder
在训练时把**真值未来动作**读进去算 latent（`modeling_act.py:413-418`），推理时置零
（`:456`）。这是训练/推理输入不同构，π₀ / π₀.₅ / patch_policy 三家都没有。

### 1.4 π₀.₅ 的量化有多粗？用你自己的尺子换算

256 档均匀分在归一化后的 [-1, 1]，一档 = `2/256 = 0.0078` 归一化单位。

你的评测 json 同时给了 `mae` 和 `norm_mae`（`new_obs2_h8.json`）：
`0.04253 / 0.13319 = 0.319` rad / 归一化单位。所以

> **π₀.₅ 的一个状态档 ≈ 0.0078 × 0.319 ≈ 0.0025 rad。**

对照你已有的两个尺度：
- 一个 waypoint 的真实运动量 **0.032 rad**（`hold@1` 换算）→ 一档是它的 **1/13**
- 纯视觉重建位姿的误差 **0.066 rad** → 一档是它的 **1/26**

**结论：π₀.₅ 把本体感觉从 float32 降级成 256 档，量化噪声仍然比问题本身的噪声底低一到两个数量级。**
这是一条对你有利的证据：**本体感觉的「精度」不是瓶颈，「有没有」和「在哪一环」才是。**
把 fp32 换成 8 bit 都不影响，说明网络需要的只是一个粗糙的锚点指针，不是一个高精度读数。

### 1.5 openpi / lerobot-pi05 里的「减锚点」原语

这是本次核代码最直接可用的收获——**PI 官方实现里有现成的相对动作开关，而且 gripper 是被单独排除的**。

`openpi_yw/src/openpi/transforms.py:204-243`：

```python
class DeltaActions(DataTransformFn):
    mask: Sequence[bool] | None                       # 逐维开关，None = no-op
    def __call__(self, data):
        state, actions = data["state"], data["actions"]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)

class AbsoluteActions(DataTransformFn):               # 推理侧的逆变换
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
```

lerobot 侧（`policies/pi05/configuration_pi05.py:52-55`）：

```python
use_relative_actions: bool = False
relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
```

**管线顺序（`processor_pi05.py:159-166`）是关键：**

```
raw → relative → normalize → model → unnormalize → absolute
```

**减锚点发生在归一化之前**，所以归一化统计量（π₀.₅ 用 `QUANTILES`，见
`configuration_pi05.py:72-78`，不是 MEAN_STD 也不是 MIN_MAX）是在**增量分布**上算的。
你的 patch_policy 是 `MIN_MAX` 直接压绝对关节角——这两件事叠加起来，
等于把一个动态范围极小的增量信号，塞进一个由绝对关节角的极值撑开的尺子里。

---

## 2. 融合谱系：本仓库 20 个策略，state / action / image 各走哪条路

每行都是从代码读的，不是从论文摘要转述的。「/」表示该策略没有这一项。

### 2.1 六种融合模式

先给分类轴，再给表。**状态进入网络的位置**只有六种：

| 代号 | 模式 | 机制 | 代价 / 特点 |
|---|---|---|---|
| **P0** | **不进** | 状态不是输入 | 锚点必须由视觉重建；`r` 天然大 |
| **P1** | **suffix 连续 token** | 与噪声动作并列，只有 action expert 看得到 | 每步去噪都重算；VLM 主干看不到状态 |
| **P2** | **prefix 连续 token** | 与图像/语言并列，进 KV-cache | 只算一次；但被数百个视觉 token 稀释 |
| **P3** | **prefix 离散文本** | 量化成 256 档数字写进 prompt | 复用 VLM 的语言先验；量化损失（§1.4 算过，可忽略） |
| **P4** | **与动作 token 拼接进自注意力，视觉走交叉注意力** | `cat([state_emb, action_emb])` | 状态与动作在同一序列里，视觉是外部 memory |
| **P5** | **全局条件向量（AdaLN / FiLM）** | 状态 → 调制向量 → 调 norm 的 scale/shift | 不占 token；但只能施加低秩的仿射影响 |
| **P6** | **不当输入，从目标里减掉** | `action -= state`（可逐维 mask） | 网络完全不需要知道锚点；需要推理侧逆变换 |

**P6 与 P0–P5 正交**——可以同时用（π₀.₅ 就是 P3+P6）。

### 2.2 谱系表

| 模型 | 状态 | 位置 | 动作头 / 目标 | 时间注入 | 视觉→动作 | 代码位置 |
|---|---|---|---|---|---|---|
| **ACT** (2304.13705) | 连续 | encoder token | 回归 L1，**绝对** | / | encoder 自注意力 | `act/modeling_act.py:465` |
| ↳ ACT-CVAE 支路 | 连续 | **VAE encoder**（+真值动作） | latent，训练专用 | / | / | `act/modeling_act.py:413-418` |
| **Diffusion Policy** (2303.04137) | 连续 | **P5** FiLM/global cond | DDPM **ε**，绝对 | FiLM | global cond | `diffusion/` |
| **patch_policy** (2607.18236) | 开关 | **P0 / P2**（1 token 拼在 patch 后） | DDPM **ε**，绝对 | 独立 token | block-causal 自注意力 | `patch_policy/modeling_patch_policy.py:431-433` |
| **π₀** (2410.24164) | 连续 | **P1** suffix | flow **v = ε−a**，绝对 | 与动作 concat+MLP | 跨 expert 共享 softmax | `openpi/models/pi0.py:141-151` |
| **π₀.₅** (2504.16054) | **256 档离散** | **P3** prefix 语言 | flow **v**，可选 **P6** | **adaRMSNorm** | 同上 | `pi0.py:92-99`, `tokenizer.py:22-31` |
| ↳ 相对动作原语 | / | / | **P6** 逐维 mask，gripper 排除 | / | / | `transforms.py:204-243`；`configuration_pi05.py:52-55` |
| **π₀-FAST** (2501.09747) | 连续 | P1 | **DCT 离散 token**，自回归 | / | 同上 | `pi0_fast/` |
| **SmolVLA** (2506.01844) | 连续 | **P2** prefix 末尾（`att_mask=1`，视觉/语言看不到它） | flow **v**，绝对 | concat+MLP | prefix KV-cache | `smolvla/modeling_smolvla.py:704-716` |
| **GR00T N1.x** (2503.14734) | 连续 | **P4** `cat([state_emb, action_emb])` | flow matching | AdaLN | **交叉注意力** `encoder_hidden_states=vl_embeds` | `groot/groot_n1_7.py:577,598,602` |
| **Evo-1** (2511.04555) | 连续 | **P4** `cat([context, state_emb])` | flow matching | cross-modulated DiT | 交叉调制 | `evo1/flow_matching.py:222,315` |
| **EO-1** (2508.21112) | 连续 | **P3'** 投影后 `masked_scatter` 进 `<state>` 占位 token | 自回归 **+** flow 双头 | adaRMS | Qwen token 流 | `eo1/modeling_eo1.py:349-364` |
| **MolmoAct2** (2605.02881) | （表里未启用） | / | flow matching expert | AdaLN gate | **逐层 KV-cache 条件** + `cross_attn` | `molmoact2/molmoact2_hf_model/modeling_molmoact2.py:435,482` |
| **X-VLA** (2510.10274) | 连续 | 送进 policy transformer | flow matching | / | soft prompt + 编码器 | `xvla/modeling_xvla.py:267-271` |
| ↳ **X-VLA 的动作空间层** | **gripper 在 proprio 和 action 里都被置零** | / | 关节/EEF 用 MSE，**gripper 用 BCE + sigmoid，权重 0.1** | / | / | `xvla/action_hub.py:160-171, 189-211` |
| **WALL-OSS** (2509.11766) | 连续 **+ DoF mask** | 独立 proprio token | flow **+** FAST 双路 | / | Qwen2.5-VL | `wall_x/modeling_wall_x.py:257-275` |
| **VITA** (2507.13231) | **P0，且无任何条件** | **视觉隐变量 = 流的源分布** | flow，动作自编码器隐空间 | 只有 t | 图像**是** ODE 的起点 | `vita/README.md`, `vita/flow_matching.py` |
| **Fast-WAM** (2603.16666) | 可选 proprio，仅第 0 帧 | proprio encoder | MoT DiT；训练做视频，**推理不做未来预测** | / | 视频 co-training | `fastwam/modeling_fastwam.py:178-183` |
| **VLA-JEPA** | 连续 | action head | DiT + 交叉注意力 | AdaLN | `encoder_hidden_states` | `vla_jepa/action_head.py:131-138` |
| **Multi-task DiT** (2507.05331) | 连续 | **P5** conditioning 向量首位 | DDPM，AdaLN-Zero | AdaLN-Zero | 拼进同一条件向量 | `multi_task_dit/modeling_multi_task_dit.py:342,534` |
| **RTC** (2506.07339) | / | / | **不是策略**，是推理算法 | / | / | `rtc/`；冻结已执行段 + inpaint 其余 |

### 2.3 从这张表能读出的四件事

**(1) 没有任何一个 2025 年之后的模型走 P0。**
唯一的例外 VITA 是把图像隐变量当成流的**源分布**——那不是「不用状态」，
那是「把观测放到了比条件更强的位置上」。
`patch_policy` 的 `use_robot_state=false` 在这张表里是孤例。

**(2) 主流在从 P1 迁往 P2/P3，理由是算力不是精度。**
π₀→π₀.₅ 把 state 从 suffix 挪到 prefix，直接后果是它进了 KV-cache：
10 步去噪只算一次，而不是 10 次。SmolVLA 是同样的选择。
**这条迁移路线跟「精度」无关**，所以不能拿它论证「本体感觉重要」。

**(3) P6（减锚点）在三家实现里独立出现，而且 gripper 都被单独挑出来。**

| 实现 | 对 gripper 的处理 |
|---|---|
| lerobot π₀.₅ | `relative_exclude_joints = ["gripper"]`——gripper **不做**相对化，保持绝对 |
| openpi `DeltaActions` | 逐维 bool mask，gripper 维通常给 `False` |
| X-VLA | proprio 和 action 里**都置零**，另用 **BCE + sigmoid**，损失权重 0.1 |

**三家用三种手段达成同一个判断：gripper 不是一个可以和关节角共用 MSE 的通道。**
你的 patch_policy 用一个 `F.mse_loss` 均匀盖住全部 16 维
（`patch_policy/modeling_patch_policy.py:472`），gripper 在里面。
这是 08 月报告里「夹爪在训练集内就比『不改变握持』差 2.8 倍」的结构性原因，
而现在有三份独立实现给出了修法。

**(4) 时间注入方式与动作头的选择是解耦的，你的矩阵里这一格全空。**
adaRMSNorm（π₀.₅）/ AdaLN-Zero（DiT）/ concat-MLP（π₀）/ 独立 token（patch_policy）——
patch_policy 用的是最弱的那种：时间当一个 token 排在 memory 里，
和几百个 patch token 抢注意力。DiT 系列早就不这么做了。

---

## 3. patch_policy 在这张表里的位置

把 §2 的轴和 §0 的 `r` 合起来看：

| 轴 | patch_policy 现状 | 谱系里的主流 | 你测过吗 |
|---|---|---|---|
| 状态位置 | P0 或 P2（1 token 拼在 patch 后） | P2 / P3 / P4 | ✓ 测过 P0↔P2，值 5–16% |
| **目标参数化** | **绝对关节角** | **P6 相对 + 逐维 mask** | ✗ **从没测过**（EEF 是换空间，不是减锚点） |
| gripper | 混在 16 维 MSE 里 | 单独 BCE / 排除相对化 / 置零 | ✗ 从没测过 |
| 时间注入 | 独立 token | adaRMS / AdaLN-Zero | ✗ 从没测过 |
| 归一化 | MIN_MAX 于绝对角 | QUANTILES 于增量 | ✗ 从没测过 |
| 损失覆盖 | 均匀盖 54 步（85% 永不执行） | 同左（主流也没解决） | ✓ 已量化 |

**空的四格里，前两格是 §0 的 `r` 直接指向的。**
你的关节权重 `r ≈ 2.5–2.9`，远在零点 1.5–1.6 的右边，
说明**锚点误差还有大量剩余**——这正是 P6 该吃掉的那部分。
而 EEF 权重 `r ≈ 1.07–1.25` 已经在零点左边，
说明换 EEF 空间**已经把锚点吃掉了**——所以在 EEF 上再叠状态桥才会变差。

**这解释了一件你之前没有解释的事**：为什么 `pp_eef`（EEF）只追平 `act_joint_361`（关节）
而没有超过（11.28 mm vs 11.28 mm）。因为换 EEF 消掉的是**锚点**那部分误差，
而 `act_joint_361` 的锚点误差本来就小（ACT 有无条件本体感觉，`r=1.63`）。
两条路殊途同归地把 `r` 压到 1 附近，剩下的 1.27× 是**形状误差**，
两条路都没碰它。**形状误差是另一篇文章的题目，不是这一篇的。**

---

## 4. 建议（按「值/成本」排序）

### 4.1 把 §0 那张图做成论文的主图（0 GPU 时）

数据全部已经在 `eval_policy/runs/` 里。需要补的只有：
- 每个点加误差棒（扩散头跑 3–5 个 seed，act/ACT 头是确定性的，误差棒为 0）
- 把 `r` 的定义写严格：`hold@1` 是评测集上的中位数还是均值，两个空间的维度混合怎么处理
- **补 2–3 个 `r` 落在 1.5–2.0 之间的点**——现在零点两侧最近的两个点是 1.625 和 1.245，
  中间是空的，零点位置是外推出来的。`act_baseline` 加不同强度的桥（K=10/20/40）
  就能在同一权重上扫出这个区间，**不用重训**。

这是整份工作里唯一一条「已有数据 → 可发表结论」的路径。

### 4.2 补 P6：相对动作 × {state on, state off} 的 2×2（约 9 GPU 时，2 个 run）

移植路径现在是现成的：`act_delta` 已经有 `use_relative_actions`，
π₀.₅ 的 `RelativeActionsProcessorStep` 给了正确的管线顺序
（**减锚点在归一化之前**，`processor_pi05.py:159-166`），照抄即可。

三种结果都可发表：
- 相对动作把 `r` 从 2.5 压到 1 附近 → §0 的因果链闭合，`r` 从相关变成机制
- 压不下去 → patch_policy 的锚点误差不是目标参数化造成的，是视觉表征的上限，
  这直接回答了「冻结 DINOv2 够不够」
- 相对 + state on 显著优于相对 + state off → P6 和 P2 不可互相替代，也是结论

**注意一个坑**：`relative_exclude_joints=["gripper"]`。
如果你把 gripper 也相对化，它会变成「握持变化量」，
而 gripper 在数据里大部分时间是常数，增量恒为 0——
损失会塌到 0 而策略学不到任何开合。三家实现都排除它是有原因的。

### 4.3 拆 gripper 通道（约 4 GPU 时，1 个 run）

X-VLA 的方案最干净，直接照抄 `xvla/action_hub.py:189-211`：
16 维里 gripper 那一维走 `BCEWithLogitsLoss`，其余走 MSE，gripper 损失乘 0.1，
推理时 sigmoid。改动大约 20 行，落在 `modeling_patch_policy.py:472` 那一句上。

这是唯一一条直通「抓不准」这个真实抱怨的短因果链，
而且它**不被 Hermite 桥覆盖**（`send_next_action_chunk` 只桥关节），
所以 §0 的 `r` 规律对它不适用——它是独立的一格。

### 4.4 不要做的两件事

- **不要拿「π 在扩散前有 action 输入」论证 patch_policy 缺输入**（§1.3 (a)）。
  这个说法对所有去噪模型都成立，包括 patch_policy 自己。
- **不要在 EEF 权重上继续调状态桥。** §0 说得很清楚：那四个权重的 `r` 全在零点左边，
  桥在那个区间的期望贡献是负的，调参数只能改变亏多少。
  该做的是**在 EEF 部署配置里把桥关掉**，预期收益 3–10%，成本 0。

---

## 5. 文献（每条已用 WebFetch 核过 arXiv 摘要页）

### 5.1 直接支持「减锚点」的
- **Demystifying Action Space Design for Robotic Manipulation Policies** — arXiv:2602.23408，
  Feng, Zheng, Wang, Liu, Li, Pang, Wang, Zhan。**13 000+ 真机 rollout、500+ 训练模型、4 个场景**。
  摘要原话：*"properly designing the policy to predict delta actions consistently improves performance"*；
  关节空间与任务空间 *"offer complementary strengths, favoring control stability and generalization, respectively"*。
  **这是目前规模最大的绝对 vs 增量对照，直接支持 §4.2。**
  （注：网上流传的「chunk-wise delta 优于 step-wise、最优 horizon 随抽象层级变化」这条细节
  我在摘要页上没核到，**不要引**，要用得去读正文。）
- **On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer** —
  arXiv:2312.03673，Aljalbout, Frank, Karl, van der Smagt。250+ RL agent × 13 个控制空间。
  结论是动作空间选择对仿真性能与真机迁移都有决定性影响。RL 语境，作为旁证不作主证。
- **Do You Need Proprioceptive States in Visuomotor Policies?** — arXiv:2509.18644。
  昨天已核。**去掉**本体感觉把真机成功率 0%→85%，前提是相对 EEF 动作空间 + 双腕部广角相机。
  你两个前提都不满足，所以**必须**按 §0 的 `r` 框架引用它，不能当反例也不能当正例。

### 5.2 π 系列（§1 的代码依据）
- **π₀: A Vision-Language-Action Flow Model for Robot Control** — arXiv:2410.24164
- **π₀.₅: a Vision-Language-Action Model with Open-World Generalization** — arXiv:2504.16054，
  Physical Intelligence 等 33 人。**注意：摘要页不含「状态以离散 token 进语言前缀」这句**，
  §1.2 那个结论的依据是代码（`tokenizer.py:22-31` + `pi0.py:92-99`），引用时应写「代码实现显示」而非「论文指出」。
- **FAST: Efficient Action Tokenization for Vision-Language-Action Models** — arXiv:2501.09747，
  Pertsch 等。DCT 压缩式动作 token；配 π₀ 训练加速最多 5×。
- **Knowledge Insulation for Vision-Language-Action Models** — arXiv:2505.23705。
  *"naively including such experts significantly harms both training speed and knowledge transfer"*。
  这是 `pi05_base` checkpoint 的实际训练配方，但 openpi 发布版不含该配方（见 `pi05-architecture-deep-dive` §0）。
- **Real-Time Execution of Action Chunking Flow Policies** — arXiv:2506.07339，
  Black, Galliker, Levine，NeurIPS 2025。RTC：执行当前块的同时生成下一块，
  冻结已保证执行的动作、inpaint 其余，**无需重训**。
  **这是你那个 Hermite 桥的正规版本**，而 §0 说明了它在什么条件下该开。

### 5.3 §2 表格里引用的模型
GR00T N1 — arXiv:2503.14734 ｜ SmolVLA — arXiv:2506.01844 ｜ EO-1 — arXiv:2508.21112 ｜
MolmoAct2 — arXiv:2605.02881（*"a flow-matching continuous-action expert onto a discrete-token VLM
via per-layer KV-cache conditioning"*）｜ WALL-OSS — arXiv:2509.11766 ｜
X-VLA — arXiv:2510.10274 ｜ Evo-1 — arXiv:2511.04555（CVPR 2026，0.77B，16.4 Hz）｜
VITA — arXiv:2507.13231 ｜ Fast-WAM — arXiv:2603.16666（Yuan, Dong, Liu, Zhao；
*"Fast-WAM remains competitive with imagine-then-execute variants, while removing video
co-training causes a much larger performance drop"*，190 ms 延迟）｜
Patch Policy — arXiv:2607.18236 ｜ ACT — arXiv:2304.13705 ｜ Diffusion Policy — arXiv:2303.04137 ｜
Multi-task DiT — arXiv:2507.05331。

### 5.4 因果混淆（`r` 大的另一半解释）
- **Causal Confusion in Imitation Learning** — arXiv:1905.11979
- **Fighting Copycat Agents in Behavioral Cloning from Observation Histories** — arXiv:2010.14876

### 5.5 核过但**故意不引**的
- **From Foundation to Application: Improving VLA Models in Practice** — arXiv:2607.06403（LingBot-VLA 2.0）。
  二手来源称它报告了「相对关节动作把成功率从 33.7 提到 55.0」。
  **arXiv 摘要页上没有这个数字**，未经正文核实，不引。若要用，需要读 PDF 确认。
- Qwen-VLA (arXiv:2605.30280) 的本体感觉注入消融——昨天已核不上，仍不引。

### 5.6 我没有找到对应物的部分
用「策略首步误差 / null 首步误差」这个比值去**预测重锚干预的正负号**，
我在上述文献里没有找到已有的提法。
动作空间的论文比的是最终成功率，RTC 的论文讨论的是延迟而不是何时该开。
如果这条 `r` 规律在补点后仍然成立（§4.1），它是本文最有可能被引用的那部分。

---

## 6. 边界

1. **`r` 的两个空间单位不同。** 关节是 rad，EEF 是 m/rad/无量纲的混合。
   `r` 用 null 归一化掉了大部分，但没有归一化掉维度构成的差异。
   §0.2 表里两个空间**各自单独拟合都成立**（R²=0.95 / 0.84），零点也接近（1.615 / 1.517），
   这是目前对该质疑最强的回应，但不是证明。
2. **10 个点，零点是外推的。** `r ∈ [1.25, 1.63]` 区间内没有观测。§4.1 给了不用重训的补点法。
3. **相关不等于因果。** `r` 与桥的效果都由「锚点误差」这个共同因素决定，
   §0.3 的机制解释是推理，不是实验。§4.2 的 2×2 是把它变成因果的最短路径。
4. **全部是离线开环分段指标。** 没有一个真机成功率。
   arXiv:2602.23408 用了 13 000 次真机 rollout，你的对照是 0 次——
   投稿时这会是第一个被问到的问题。
5. **§2 表里 MolmoAct2 的状态一列是「未启用」**，因为 `state_embeddings=None`
   出现在 `modeling_molmoact2.py:1503`。这可能是 lerobot 移植的限制而不是原模型的设计，
   引用前应查原实现。
