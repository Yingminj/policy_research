# ACT 编码器/解码器层数的影响：与 LeRobot 官方实现的对照分析

> 分析对象：`/home/kewei/YING/ACT_rosbag2`（下称**本仓库**），基线文档
> `doc/ACT_深度解析.md`
> 对照实现：`lerobot 0.6.0`
> （`/home/kewei/anaconda3/lib/python3.12/site-packages/lerobot/policies/act/`）
> 实测环境：RTX 4090 / torch 2.11.0+cu130 / 3 相机 480×640 / `hidden_dim=512`
> / `dim_feedforward=3200` / `chunk_size=100` / `action_dim=16`
> 报告日期：2026-08-10

---

## 0. 摘要（先看这里）

1. **本仓库在 2026-07-28 的 commit `6000eb0` 中把 decoder 输出索引从 `[0]` 改成了
   `[-1]`。** 这修掉了上游 ACT 的一个已知 bug，但同时使本仓库**脱离了 ACT 论文
   实际验证过的配置**——论文公布的结果是在"7 层 decoder 但只有第 0 层生效"
   （等效 1 层）下取得的。LeRobot 官方选择了相反的处理方式：保留 bug 的行为，
   把 `n_decoder_layers` 默认值直接设成 **1**。

2. **这是一个静默的 checkpoint 破坏性变更。** `6000eb0` 之前训练的权重，其
   decoder 第 1–6 层从未收到梯度、仍停留在 Xavier 初值；用当前代码
   （`[-1]`）加载时 `load_state_dict` **不会报错**，但推理会经过 6 层随机初始化
   的残差块，输出不可用。任何跨该 commit 复用的 checkpoint 都必须重训。

3. **本仓库的 `enc_layers` 一个参数同时控制两个不同的 encoder**
   （observation encoder 与 CVAE style encoder），代码里还留着
   `# TODO shared with VAE decoder`。LeRobot 把它们拆成了 `n_encoder_layers`
   和 `n_vae_encoder_layers`。二者的每层代价相差 **10.5 倍**（9.47 vs 0.90
   GFLOP），且 VAE encoder 在推理时**完全不运行**——把它们绑在一起调参是纯浪费。

4. **实测结论：层数不是 4090 上的部署瓶颈，但是训练成本的主要来源。**
   全部 11 组配置的单帧推理都在 6.7–11.0 ms 之间；而 encoder 从 4 层加到 8 层，
   训练步时长 +40%、峰值显存 +52%。

5. **"image encoder"在 ACT 里没有层数可调**——图像编码由固定深度的 ResNet
   承担，`enc_layers` 控制的是它之后的 token 级 observation encoder。这三者的
   区分见第 2 节。

---

## 1. 三方实现对照

| 维度 | 上游 ACT（tonyzhaozh/act） | LeRobot 0.6.0 | 本仓库（ACT_rosbag2，当前 HEAD） |
|---|---|---|---|
| decoder 层数配置值 | `dec_layers=7` | `n_decoder_layers=1` | `dec_layers=7`（`configs/apex_real_C*.yaml:22`） |
| decoder **实际生效**层数 | **1**（bug） | **1** | **7** |
| decoder 输出取法 | `transformer(...)[0]` | 循环全部层后 `norm(x)` | `transformer(...)[-1]`（`detr/models/detr_vae.py:134`） |
| obs encoder 层数 | `enc_layers=4` | `n_encoder_layers=4` | `enc_layers=4` |
| VAE style encoder 层数 | 与 `enc_layers` **共用** | 独立 `n_vae_encoder_layers=4` | 与 `enc_layers` **共用**（`detr/models/detr_vae.py:220`） |
| backbone | ResNet18，全相机共享 | ResNet18，全相机共享 | ResNet18，全相机共享（`detr_vae.py:124` `# HARDCODED`） |
| 多相机 token 拼接 | 沿 width 拼 feature map | 每相机 flatten 后 extend 到 token 序列 | 沿 width 拼 feature map |
| 相机身份编码 | 无 | 无 | 无 |
| `return_intermediate_dec` | True（结果被丢弃） | 不存在该机制 | True（只用最后一层） |
| 执行步数 M | eval loop 硬编码 | 配置项 `n_action_steps`，且校验 `≤ chunk_size` | 部署脚本硬编码（3 / 50） |
| `is_pad_head` | 有，无 loss | **已移除** | 有，无 loss（悬空参数） |

LeRobot 在配置文件里对这件事写得很直白
（`configuration_act.py:108-110`）：

```python
# Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
# that means only the first layer is used. Here we match the original implementation by setting this to 1.
# See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
n_decoder_layers: int = 1
```

---

## 2. 澄清："image encoder"在 ACT 中指三个不同的东西

阅读 `doc/ACT_深度解析.md` 第 5、6 节时容易混淆。ACT 的视觉通路上其实有三个可以
被叫做"encoder"的模块，只有其中两个有"层数"这个可调参数：

| 模块 | 代码位置 | 层数由谁控制 | 处理什么 | 推理时是否运行 |
|---|---|---|---|---|
| **① 图像 backbone** | `detr/models/backbone.py` | ResNet 当然有层（18 层卷积、分成 `layer1..layer4`），但它的深度**由 `backbone: resnet18` 这个名字决定，`enc_layers` 完全碰不到它** | 像素 → `[B,512,15,20]` 特征图 | 是 |
| **② observation encoder** | `detr/models/transformer.py:79-100` | `enc_layers` | 902 个 token（900 视觉 + z + qpos）的自注意力融合 | 是 |
| **③ CVAE style encoder** | `detr/models/detr_vae.py:215-229` | **也是 `enc_layers`** | 102 个 token（CLS + qpos + K 个真值 action） | **否** |

所以"增加 image encoder 层数"这个说法在本仓库对应两种完全不同的操作：

- 换更深的 CNN（`resnet18 → resnet34/50`）：改的是 ①，**不改变 token 数量**；
- 增大 `enc_layers`：改的是 ② **和** ③，**不改变图像分辨率或 token 数量**，
  只加深 token 之间的融合。

图像 token 的**数量**由分辨率和相机数决定（`3×15×20 = 900`），跟层数无关；
它才是 observation encoder 二次复杂度的来源。

### 2.1 四个"深度"旋钮各自数的是什么

| 旋钮 | 数的是 | 当前值 | 一层里面是什么 | 每层参数 |
|---|---|---:|---|---:|
| `backbone: resnet18` | ResNet 的卷积层 | 18 | BasicBlock（2×conv3×3 + 残差） | — |
| `enc_layers` → obs encoder | `TransformerEncoderLayer` 个数 | 4 | self-attn → +残差 → LN → FFN(512→3200→512) → +残差 → LN | 4,333,184 |
| `enc_layers` → VAE encoder | `TransformerEncoderLayer` 个数 | 4（**同一个参数**） | 同上，结构完全一样，参数不共享 | 4,333,184 |
| `dec_layers` | `TransformerDecoderLayer` 个数 | 7 | self-attn → LN → **cross-attn** → LN → FFN → LN | 5,384,832 |

decoder 层比 encoder 层多 1,051,648 个参数，差在那个多出来的 cross-attention
（`multihead_attn` 1,050,624 + `norm3` 1,024）。

对账（`hidden_dim=512, dim_feedforward=3200`）：

```
VAE encoder   = 4 × 4,333,184                       = 17,332,736
obs encoder   = 4 × 4,333,184                       = 17,332,736
decoder       = 7 × 5,384,832 + 1,024 (decoder_norm) = 37,694,848
主 Transformer = 17,332,736 + 37,694,848             = 55,027,584   ← 与 doc §4 一致
```

### 2.2 完整数据流图（含张量形状）

`B`=batch（推理时 1），`C`=3 相机，`K`=100，`D`=512，`A`=16，特征图 `15×20`。
双线框 `╔╝` 是**层数可调**的模块。

```
════════════════ 阶段 0：输入 ════════════════

  images  [B,3,3,480,640]        qpos [B,16]        actions [B,100,16]   is_pad [B,100]
  (归一化到[0,1]→ImageNet norm)   (已标准化)          (已标准化, 仅训练)     (仅训练)
        │                            │                     │                  │
        │                            └──────┬──────────────┘                  │
        │                                   │                                 │
        │                    ┌──────────────┴─────────────────────────────────┘
        │                    │
        │        ┌───────────▼──────────────────────────────────────────────────────┐
        │        │  阶段 1：CVAE style encoder  ★推理时整块跳过★                      │
        │        │                                                                  │
        │        │   actions[B,100,16] ─ Linear(16→512) ────────────► [B,100,512]    │
        │        │   qpos   [B,16]     ─ Linear(16→512) ────────────► [B,  1,512]    │
        │        │   cls_embed         ─ Embedding(1,512) ──────────► [B,  1,512]    │
        │        │                                    concat(dim=1)                  │
        │        │                                          ▼                        │
        │        │                                     [B,102,512] ─permute→[102,B,512]
        │        │                            + pos_table(固定正弦,不可学) [102,1,512] │
        │        │                                          ▼                        │
        │        │              ╔═══════════════════════════════════════════╗        │
        │        │              ║ ① CVAE style encoder                      ║        │
        │        │              ║   层数 = enc_layers = 4        ◄── enc=4  ║        │
        │        │              ║   序列长 102，双向 self-attn               ║        │
        │        │              ║   src_key_padding_mask = is_pad（唯一用到）║        │
        │        │              ╚═════════════════════╤═════════════════════╝        │
        │        │                              [102,B,512]                          │
        │        │                          取 [0]（CLS）► [B,512]                   │
        │        │                     latent_proj Linear(512→64)                    │
        │        │                          ┌───────────┴───────────┐                │
        │        │                       mu[B,32]              logvar[B,32]          │
        │        │                          └──── z = mu+e^(logvar/2)·ε ──► z[B,32]  │
        │        └──────────────────────────────────────────┬───────────────────────┘
        │                                                   │
        │              推理: 跳过上面整块, 直接 z := zeros([B,32])
        │                                                   ▼
        │                                  latent_out_proj Linear(32→512) ─► [B,512]
        │                                                   │
        ▼                                                   │
════════════════ 阶段 2：图像 backbone（每相机各跑一次，权重共享）════════════
        │                                                   │
   for cam in [top, wrist_L, wrist_R]:                       │
     image[:,cam] [B,3,480,640]                              │
        │                                                    │
        ├─► ResNet18 (FrozenBN, layer1..layer4)               │   ← 深度由 "resnet18"
        │      └─ 取 layer4 输出 ──────► [B,512,15,20]        │      决定, enc_layers
        │                                                    │      管不着
        ├─► input_proj Conv2d(512→512, k=1) ──► [B,512,15,20]│
        │                                                    │
        └─► PositionEmbeddingSine(256, normalize=True)        │
                 └─► pos [1,512,15,20]  ⚠ 三路相机完全相同    │
                                                             │
   沿 width 拼接 (axis=3):                                    │
     src = [B,512,15,60]        pos = [1,512,15,60]           │
             │                        (= 同一块 15×20 平铺3次) │
             ▼                                                │
   flatten+permute ──► src [900,B,512]   pos [900,B,512]      │
                                                             │
════════════════ 阶段 3：拼 memory 序列 ════════════════       │
                                                             │
   z_token    = latent_out_proj(z)      [B,512] ◄─────────────┘
   qpos_token = input_proj_robot_state(qpos) Linear(16→512)  [B,512]
             │
             ├─ stack ──► [2,B,512] ──┐
             │                        ├─ cat(dim=0) ──► src [902,B,512]
   视觉 token [900,B,512] ────────────┘
                                       additional_pos_embed(可学,2×512)
   pos [900,B,512] ──── cat ────────► pos_embed [902,B,512]
                                       ▲
                                       └─ 只有 z / qpos 两个 token 有可学位置编码

════════════════ 阶段 4：observation encoder ════════════════

              ╔══════════════════════════════════════════════════╗
              ║ ② observation encoder                            ║
              ║   层数 = enc_layers = 4              ◄── enc=4    ║
              ║   序列长 902 = 900视觉 + z + qpos                 ║
              ║   每层: self-attn(902×902) + FFN(512→3200→512)   ║
              ║   mask = None                                    ║
              ╚═══════════════════════╤══════════════════════════╝
                                      ▼
                            memory [902,B,512]

════════════════ 阶段 5：action decoder ════════════════

   query_embed Embedding(100,512) ──► query_pos [100,B,512]
   tgt = zeros_like(query_pos)    ──► [100,B,512]   (内容全 0, 信息全在 query_pos)
                    │
                    ▼
              ╔══════════════════════════════════════════════════╗
              ║ ③ action decoder                                 ║
              ║   层数 = dec_layers = 7               ◄── dec=7   ║
              ║   每层三段:                                       ║
              ║     a) self-attn  100×100   (chunk 内部一致性)     ║
              ║     b) cross-attn 100×902   (读 memory)          ║
              ║        ⚠ 每层都重算 memory 的 K/V 投影            ║
              ║     c) FFN 512→3200→512                          ║
              ║   无 causal mask, 无自回归                        ║
              ║   return_intermediate=True → 存下每层输出          ║
              ╚═══════════════════════╤══════════════════════════╝
                                      ▼
                          stack ──► [7,100,B,512]
                       transpose(1,2) ──► [7,B,100,512]
                                      │
                      ┌───────────────┴────────────────┐
                  本仓库 [-1]                    上游/LeRobot [0]
                  取第6层 → 7层全部训练          取第0层 → 第1~6层永不收梯度
                      │                                │
                      ▼                                ▼
                  hs [B,100,512]                   hs [B,100,512]

════════════════ 阶段 6：输出头 ════════════════

   hs [B,100,512] ─┬─ action_head Linear(512→16) ──► a_hat     [B,100,16]
                   └─ is_pad_head Linear(512→1)  ──► is_pad_hat[B,100,1]  ⚠ 无 loss, 丢弃

════════════════ 阶段 7：训练 vs 推理分叉 ════════════════

  训练:  L = mean(|a_hat − actions|·(1−is_pad)) + 10 · KL(mu,logvar)
         → backward → AdamW

  推理:  a_hat[0] [100,16] ─ 反标准化(dataset_stats.pkl) ─► 关节目标
         → 截取前 M 步 (deploy_act.py: M=3 / deploy_act_test.py: M=50)
         → 可选 cubic 插值 (N=smoothing_steps)
         → max_joint_delta 限幅
         → 发布 Jointcmd + 夹爪 Float32
```

**这张图上和层数直接相关的三处：**

- `enc=4` 出现**两次**（阶段 1 和阶段 4），是同一个 YAML 字段驱动的两个独立权重的
  4 层堆叠。阶段 1 那份在推理时整块不执行。
- `dec=7` 只出现一次（阶段 5），但它的输出在阶段 5 末尾有 `[-1]` / `[0]` 的分叉——
  这就是第 3 节说的那个 32.3M 死参数问题的位置。
- ResNet18 的深度在阶段 2，不受任何 `*_layers` 参数影响。

**另外一个图上才看得清的问题**：阶段 2 里三路相机的 `pos` 是**逐字节相同**的
（`PositionEmbeddingSine` 只依赖特征图尺寸，且 `normalize=True` 把每张图各自归一化
到 `[0,2π]`）。实测验证：

```
pos shape: torch.Size([1, 512, 15, 20])
cam0 block == cam1 block: True
```

所以 observation encoder **无法靠位置编码区分"top 相机的第 (r,c) 格"和"wrist_L 的
第 (r,c) 格"**，只能靠画面内容去猜视角身份。这比 `doc/ACT_深度解析.md` §6.3
"每路相机的 2D 正弦坐标都会从原点重新开始"的说法更严重——不是从原点重新开始，
是三份**完全一样**。这也是 5.1 节说"先补 camera embedding 再谈加深 encoder"的
直接依据。

---

## 3. 核心发现：decoder 层数的 `[0]` / `[-1]` 差异

### 3.1 机理

`TransformerDecoder.forward`（`transformer.py:112-141`）在
`return_intermediate=True` 时返回 `torch.stack(intermediate)`，形状
`[L, K, B, D]`；`Transformer.forward` 再 `transpose(1,2)` 得到 `[L, B, K, D]`。
因此第一维就是**层索引**：

- `[0]` → 第 0 层输出（上游 ACT、LeRobot 复刻的行为）
- `[-1]` → 第 L−1 层输出（本仓库当前行为）

在 `[0]` 下，第 1…L−1 层仍会在 forward 里逐层执行，但它们的输出不参与 loss，
因而**收不到任何梯度**。

### 3.2 实测验证

对同一模型（`enc=4, dec=7`）做一次 backward，打印各 decoder layer
`linear1.weight` 的梯度范数：

```
index [0]  -> L0:9.741e-01  L1:0.000e+00  L2:0.000e+00  L3:0.000e+00  L4:0.000e+00  L5:0.000e+00  L6:0.000e+00
index [-1] -> L0:2.427e-01  L1:3.055e-01  L2:4.112e-01  L3:4.399e-01  L4:5.938e-01  L5:7.541e-01  L6:8.831e-01
```

`[0]` 下 6/7 的 decoder 层梯度**严格为 0**。换算成规模：

| | `[0]`（上游） | `[-1]`（本仓库） |
|---|---:|---:|
| decoder 总参数 | 37,694,848 | 37,694,848 |
| 其中**可训练生效**的 | 5,385,856 | 37,694,848 |
| **死参数** | **32,308,992（占全模型 38.5%）** | 0 |
| 单帧推理中被浪费的时间 | ≈ 2.09 ms（占 23%） | 0 |
| 单帧推理中被浪费的算力 | ≈ 12.7 GFLOP（占 15%） | 0 |

### 3.3 变更来源与 checkpoint 兼容性（P0）

```
commit 6000eb0d3e8c005200fc9ab6b6cf7d8938687b58
Author: snorlaxss   Date: Tue Jul 28 16:14:25 2026 +0800
    decode Parameters updata

-  hs = self.transformer(...)[0]
+  hs = self.transformer(...)[-1]
```

这是本仓库的**本地修改**，不是上游行为。后果：

1. **`6000eb0` 之前训练的所有 checkpoint 在当前代码下失效。** 那些权重的
   decoder L1–L6 停留在 `_reset_parameters()` 的 Xavier 初值；当前代码会让激活
   穿过这 6 层随机残差块再进 `action_head`。因为 shape 完全匹配，
   `load_state_dict(strict=True)` **不会报任何错**，只会输出无意义动作。
2. 反向也不通：现在训练出的权重，用旧代码（`[0]`）评估会只用到 L0，同样错误。
3. **判别方法**：无法从 checkpoint 文件本身区分。只能靠训练时间戳 / 代码版本。
   建议立即在 `save_checkpoint` 里写入一个 `arch_version` 字段，并在部署脚本里
   校验。这是当前**最高优先级**的工程债，优先级高于 `doc/ACT_深度解析.md` 第 16
   节列出的任何一条。

### 3.4 这个"修复"是好事吗

**不能默认为是。** 需要注意：

- ACT 论文报告的仿真/实机成功率，以及 LeRobot 复刻出的对齐结果，都是在**等效
  1 层 decoder** 下取得的。本仓库现在跑的是一个**未经公开验证**的 7 层配置。
- 7 层 decoder 引入了 32.3M 新的可训练参数（模型规模 51.6M → 83.9M，+63%），
  而本仓库的优化设置（`detr/main.py`）**没有 lr scheduler、没有 warmup、没有
  gradient clipping、没有 AMP**，且 `pre_norm=False`（post-norm）。post-norm 的
  深层堆叠在无 warmup 时对初期梯度更敏感——这是需要在你的数据上实测确认的风险，
  不是理论上的必然失败。
- 在演示数据量有限（几十条 episode）的模仿学习场景下，decoder 容量通常不是瓶颈，
  过参数化反而更容易过拟合到演示轨迹。

**建议**：把 `dec_layers ∈ {1, 4, 7}` 作为一组正式消融跑一次，用同一份数据、
同一 seed，比较 validation L1 与 chunk 内分步 MAE 曲线（`MAE(step=0..K-1)`）。
在拿到这组数据之前，`dec_layers=1` 是风险更低的默认值，因为它同时匹配上游和
LeRobot。

---

## 4. 实测数据

### 4.1 层数扫描

单帧推理：batch=1，3 相机 480×640，`torch.no_grad()`，30 次取中位数/P95。
训练步：batch=8，含 forward + backward + AdamW step，12 步取均值。

| enc | dec | vae_enc | 总参数 | obs enc | decoder | vae enc | 推理 P50 (ms) | 推理 P95 (ms) | 训练/步 (ms) | 峰值显存 (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | **1** | 4 | 51.62M | 17.33M | 5.39M | 17.33M | **6.83** | 7.38 | 117.1 | 5.91 |
| 4 | 2 | 4 | 57.00M | 17.33M | 10.77M | 17.33M | 7.27 | 7.76 | 118.3 | 6.14 |
| 4 | 4 | 4 | 67.77M | 17.33M | 21.54M | 17.33M | 7.82 | 8.40 | 124.8 | 6.59 |
| **4** | **7** | **4** | **83.93M** | 17.33M | 37.69M | 17.33M | **8.92** | 9.65 | **135.0** | **7.26** |
| 4 | 10 | 4 | 100.08M | 17.33M | 53.85M | 17.33M | 10.48 | 10.92 | 144.9 | 7.93 |
| 1 | 7 | 1 | 57.93M | 4.33M | 37.69M | 4.33M | 7.49 | 7.76 | 95.7 | 4.46 |
| 2 | 7 | 2 | 66.59M | 8.67M | 37.69M | 8.67M | 7.98 | 8.60 | 108.5 | 5.39 |
| 6 | 7 | 6 | 101.26M | 26.00M | 37.69M | 26.00M | 10.00 | 10.53 | 161.8 | 9.13 |
| 8 | 7 | 8 | 118.59M | 34.67M | 37.69M | 34.67M | 10.98 | 11.71 | 188.6 | 11.00 |
| 4 | 7 | **1** | 70.93M | 17.33M | 37.69M | 4.33M | 8.98 | 9.88 | 132.2 | 6.94 |

加粗行分别是 **LeRobot 默认等效配置**（enc4/dec1）和**本仓库当前配置**
（enc4/dec7）。

**读数：**

- **单层边际成本（推理）**：decoder ≈ **0.40 ms/层**，obs encoder ≈ **0.50 ms/层**。
- **单层边际成本（训练，bs=8）**：decoder ≈ **3.1 ms/层 / +0.22 GB**；
  encoder（因为耦合，一次加两个）≈ **13.3 ms/层 / +0.93 GB**。
  **encoder 深度在训练侧比 decoder 贵 4 倍以上。**
- **VAE encoder 深度对推理完全免费**：最后一行 vae_enc 从 4 降到 1，参数少
  13.0M，推理时间 8.98 ms 与 8.92 ms 在噪声内无差别；但训练省 2.8 ms/步、
  0.32 GB。
- 本仓库当前配置相对 LeRobot 默认配置：**参数 +63%，单帧推理 +2.09 ms（+31%），
  训练步 +15%。**

### 4.2 backbone（真正的"image encoder"）扫描

`enc=4, dec=7`，其余不变：

| backbone | 总参数 | backbone 参数 | 推理 P50 (ms) |
|---|---:|---:|---:|
| resnet18 | 83.93M | 11.17M | **8.99** |
| resnet34 | 94.03M | 21.27M | 12.44 |
| resnet50 | 97.00M | 23.45M | 14.41 |

换 backbone 比加 transformer 层**贵得多**：resnet18→34 只加 10.1M 参数却增加
3.45 ms，相当于 8.6 层 decoder 的时间代价。原因是 backbone 在 480×640 全分辨率
上、且要跑 3 次（每相机一次）。

> 注意：`resnet50` 输出 2048 通道，`input_proj` 会变成 `2048→512` 的 1×1 卷积，
> checkpoint 与 resnet18 不兼容。

### 4.3 理论 FLOP 分解（单帧推理，MAC 按 2 FLOP 计）

| 分辨率 / 相机数 | token 数 N | backbone 合计 | obs enc/层 | decoder/层 | vae enc/层（仅训练） | enc4+dec7 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 480×640 × 3 | 902 | 33.43 G | **9.47 G** | **2.12 G** | 0.90 G | 86.15 G |
| 240×320 × 3 | 242 | 8.91 G | 2.21 G | 1.29 G | 0.90 G | 26.82 G |
| 480×640 × 1 | 302 | 11.14 G | 2.80 G | 1.37 G | 0.90 G | 31.92 G |

关键比值与含义：

- 在当前 480×640×3 配置下，**一个 obs encoder 层 = 4.5 个 decoder 层**的算力。
  实测延迟却是 0.50 vs 0.40 ms/层——因为 batch=1 时 GPU 严重欠占用，处于
  kernel launch 主导区间，FLOP 差异被掩盖。**一旦提高 batch（训练）或换到算力
  更弱的边缘设备，encoder 层的真实代价就会显现**（4.1 节训练列已经印证：
  13.3 vs 3.1 ms/层）。
- decoder 层并不像"只有 100 个 query"想象的那么便宜：每层的 cross-attention 都要
  **重新对 902 个 memory token 做 K/V 投影**（0.946 GFLOP，占该层 45%）。
  这部分在各层间是可以缓存复用的，属于本仓库和 LeRobot 都存在的优化空间。
- **encoder 层的代价随 token 数超线性增长**（9.47 → 2.21 GFLOP，分辨率减半时降
  4.3×，含 N² 项）；decoder 层近似线性（2.12 → 1.29）。所以
  **"高分辨率 + 深 encoder"是最贵的组合**，而"高分辨率 + 深 decoder"要温和得多。

---

## 5. 层数对模型能力的影响机理

以下是机理分析与风险判断，**不是实验结论**；带 ⚠ 的条目需要在你自己的数据上验证。

### 5.1 obs encoder 层数（`enc_layers`）

encoder 做的是 902 个 token 之间的**双向融合**：跨相机对应、视觉与 qpos 对齐、
z 向视觉特征广播。

- **加深的收益**：多视角几何关系越复杂（本仓库 3 路：top + 双腕），需要的融合
  轮次越多。⚠ 如果任务依赖"腕部相机看到的东西要和顶视图中的物体位置对上"，
  encoder 深度可能比 decoder 深度更有价值。
- **加深的代价**：训练侧最贵的一项（13.3 ms/层 + 0.93 GB，因为耦合）；且**每加
  一层同时加深了训练专用的 VAE encoder**，那部分算力在部署时被完全丢弃。
- **一个结构性限制**：本仓库没有 camera-ID embedding，每路相机的 2D 正弦位置编码
  都从原点重新开始（`doc/ACT_深度解析.md` §6.3、§16-P2）。⚠ 在这种情况下加深
  encoder 的边际收益可能受限——模型必须先从内容和拼接顺序里"猜"出视角身份，
  再谈跨视角融合。**先补 camera embedding，再谈加深 encoder**，性价比更高。

### 5.2 decoder 层数（`dec_layers`）

decoder 做两件事：query 之间的 self-attention（chunk 内时序一致性）+ 对 memory 的
cross-attention（把观测读进来）。

**先厘清 decoder 的输入。** `tgt = zeros_like(query_pos)` 容易让人以为 decoder 的
唯一输入是 `query_embed`，实际上有两个入口：

| 进入 decoder 的东西 | 走哪条路 | 性质 |
|---|---|---|
| `memory [902,B,512]` | 每层 cross-attn 的 **K / V** | **数据**——当前帧的图像 + qpos + z |
| `pos_embed [902,B,512]` | 每层 cross-attn 的 K | 正弦常量 + 少量可学参数 |
| `query_pos [100,B,512]` | 每层 self-attn 的 q,k **和** cross-attn 的 q | **参数**，对所有样本恒定 |
| `tgt = 0` | 残差流初值 | 常量 |

`query_embed` 是 `nn.Embedding(100,512)` 的权重，对每个样本、每一帧都是同一份数，
只编码"我是未来第 i 槽"。**所有随样本变化的信息 100% 经由 cross-attention 从
memory 进入**。另外 `query_pos` 不是入口处注入一次，而是通过 `with_pos_embed` 在
**每一层**的 self-attn 与 cross-attn 里重复相加（`transformer.py:236, 241`）。

**`tgt=0` 的一个非平凡后果：第 0 层的 self-attention 是恒等无效的。**
self-attn 的 `value=tgt`，第 0 层 `tgt` 全为 0，所以输出是"一堆 0 的加权和"，与
注意力权重无关。实测（`probe_layer0.py`）：

```
[init]         layer0 self-attn output: max|x| = 0.000e+00
[trained-like] max|x| = 8.901e-02, 100 个 query 之间的差异 = 2.980e-08
[cross-attn]   100 个 query 之间的差异 = 1.325e-01
```

训练后偏置非零，输出变成一个小常量，但在 100 个 query 之间的差异只有 3e-08
（浮点噪声）——第 0 层 self-attention **永远无法让不同 query 产生不同响应**。
真正让 query 分化的是同层紧随其后的 cross-attention（高 6 个数量级）。

**因此：`dec_layers=1` 时，100 个 action query 之间没有任何信息交换。**
上游 ACT 与 LeRobot 默认配置下，`doc/ACT_深度解析.md` §7 所述"query
self-attention 建模 chunk 内部一致性"这一机制**并不存在**；每一步动作都是各自
独立地读同一份 memory 得到的，chunk 连贯性完全来自共享 memory 与学到的 query
embedding。要让 self-attention 真正生效，**至少需要 2 层**（第 1 层的 `tgt` 已非零）。

- **1 层为什么仍然够用**：ACT 的输出是 100 个并行 query 的回归，没有自回归依赖，
  而**观测侧的重活已经在 4 层 encoder 里做完了**。1 层 decoder 提供了完整的
  "读观测 + FFN"回路，只是缺 query 间通信。这解释了为什么上游的 bug 能长期不被
  发现——1 层就足以复现论文指标。
- **加深可能的收益**：⚠ chunk 后段（step 60–99）的动作离当前观测最远、不确定性
  最高，更深的 query self-attention 理论上能改善长程平滑性和末段漂移。这正好是
  值得测的指标——**不要只看总 loss，要看 `MAE(step)` 随 step 的曲线**。
  注意按上面的分析，**1→2 层才是质变点**（query 间通信从无到有），4→7 只是加深
  已有机制。
- **加深的风险**：+32.3M 参数、无 warmup 的 post-norm 堆叠、演示数据量有限
  → 过拟合与优化不稳定的风险同时上升。
- **部署侧的免费午餐**：如果部署只执行前 M=3 步
  （`scripts/deploy_act.py`），那么 chunk 后段的质量对实机行为**几乎没有影响**，
  加深 decoder 换来的那部分收益直接被丢掉。**M 越小，decoder 深度越不值钱。**
  当前 `deploy_act_test.py` 的 M=50 则相反，后段质量会真实影响机器人。

### 5.3 VAE style encoder 层数

- 只在训练时运行，只看 102 个 token，只输出一个 32 维 `z`（经 CLS token 的
  瓶颈）。**信息瓶颈极窄**，深度的边际收益应该很快饱和。
- 在本仓库它被强制等于 `enc_layers`，所以每次为了视觉融合而加深 encoder，都会
  额外付一份**永远不会在部署时用到**的参数和训练算力。
- **这是当前配置里最容易摘的低垂果实**：解耦后把它设成 1–2 层，训练省
  2.8 ms/步和 0.32 GB，推理零影响，⚠ 且大概率不损失策略质量（需一次消融确认）。

---

## 6. 部署侧的时序含义

结合 `doc/ACT_深度解析.md` §13 的频率模型（`T_replan,actual ≳ M·N·d + T_infer`）：

| 场景 | 重规划预算 | 实测 `T_infer`（4090, enc4/dec7） | 结论 |
|---|---:|---:|---|
| `deploy_act.py`，M=3，d=0.05，N=1 | 150 ms | 8.9 ms | 占 5.9%，**层数不是瓶颈** |
| `deploy_act_test.py`，M=50，d=0.05 | 2500 ms | 8.9 ms | 占 0.4%，完全无关 |
| 假想 M=1（真 temporal ensemble） | 50 ms | 8.9 ms | 占 18%，仍有余量 |

**在 4090 上，`enc_layers`/`dec_layers` 怎么调都不会成为实时性瓶颈。**
`doc/ACT_深度解析.md` §16-P1 指出的"chunk 边界无预取导致停顿"才是真正的延迟来源，
量级是 `M·N·d`（150 ms–2.5 s），比 `T_infer` 高一到两个数量级。

⚠ **换到边缘设备则不同**。若目标平台算力约为 4090 的 1/8–1/10（如 Jetson AGX
Orin），按线性外推：

| 配置 | 4090 实测 | Orin 量级估算（×8–10） |
|---|---:|---:|
| enc4 / dec1 | 6.8 ms | 55–68 ms |
| enc4 / dec7（当前） | 8.9 ms | 71–89 ms |
| enc8 / dec7 | 11.0 ms | 88–110 ms |
| enc4 / dec7 + resnet34 | 12.4 ms | 100–124 ms |

在 M=1 或 M=3 的高频重规划下，这些数字就开始逼近甚至突破预算。**如果最终要上
边缘部署，`dec_layers=1` 直接省下 20–25% 的推理时延，且是上游验证过的配置。**

---

## 7. 建议

### 7.1 立刻处理（工程正确性）

1. **给 checkpoint 加架构指纹。** 在保存时写入 `dec_index_mode`（`"first"`/
   `"last"`）、`enc_layers`、`dec_layers`、`chunk_size`、`state_dim`，部署时严格
   校验。当前 `[0]→[-1]` 这类变更完全静默，`load_state_dict` 给不出任何信号。
2. **清点并标记 `6000eb0`（2026-07-28）之前的所有 checkpoint 为不可用**，或用
   旧代码重新评估。
3. **解耦 `enc_layers` 与 VAE encoder 层数。** 在 `build_encoder`
   （`detr_vae.py:215`）改用新参数 `vae_enc_layers`，默认回落到 `enc_layers`
   保持向后兼容。这是对齐 LeRobot 的最小改动，且不影响已有 checkpoint 形状
   （只要值不变）。
4. （可选，收益明确）**缓存 decoder 各层 cross-attention 的 memory K/V 投影**，
   可省下每层 0.946 GFLOP，即 7 层共约 6.6 GFLOP / 单帧 45% 的 decoder 算力。

### 7.2 需要跑的消融（一次性，代价很低）

固定数据、seed、epoch 数，只改层数：

| 编号 | enc | dec | vae_enc | 目的 |
|---|---:|---:|---:|---|
| A（基线，对齐 LeRobot） | 4 | 1 | 4 | 上游验证过的配置；**query 间无通信** |
| **A2（关键对照）** | 4 | **2** | 4 | **query self-attention 从无到有的质变点**（见 5.2） |
| B（当前） | 4 | 7 | 4 | 现状 |
| C | 4 | 4 | 4 | decoder 深度中间点 |
| D | 6 | 1 | 4 | 把预算从 decoder 挪到 encoder |
| E | 4 | 1 | 1 | 验证 VAE 深度是否可裁 |

A→A2 的差异（+5.4M 参数、+0.44 ms 推理）是整组消融里代价最小、机理最明确的一步；
如果 chunk 后段质量有改善，最可能出现在这里。

评估指标（不要只看总 loss）：

- validation L1（**注意 `doc/ACT_深度解析.md` §9.1 指出的固定分母问题，会让
  episode 末尾样本被稀释——比较不同配置时应改用"仅有效元素求均值"的指标**）；
- `MAE(step)` 随 step 0…99 的曲线——decoder 深度的收益如果存在，应该体现在
  **后段**；
- 14 个关节与 2 个夹爪分开统计；
- 离线回放整段 chunk 的轨迹平滑性/振荡。

### 7.3 默认值建议

在拿到 7.2 的结果之前：

- **若目标是复现/对齐已发表结果或要上边缘设备** → `dec_layers=1`（对齐 LeRobot
  与上游实际行为），`enc_layers=4`。
- **若继续用 `dec_layers=7`** → 明确记录这是一个未经公开验证的配置，并至少补上
  warmup 或 gradient clipping（parser 里 `--clip_max_norm` 已存在但未接线，
  见 `doc/ACT_深度解析.md` §11.2）。
- **VAE encoder 无论如何先降到 1–2 层**（需先做 7.1.3 的解耦），推理零成本、
  训练直接省。
- **不要为了"更强的视觉"去换 resnet34/50**——在 480×640×3 相机下它的时延代价
  （+3.5 ms）超过加 8 层 decoder，而参数收益只有 10M。优先考虑补
  camera-ID embedding 或调整分辨率。

---

## 8. 对 `doc/ACT_深度解析.md` 的补充

原文档第 7 节写"当前 decoder 是 7 层……并返回每一层输出；但 `DETRVAE.forward`
最终只取最后一层"——**这个描述对当前代码是准确的**。需要补充的是三点上下文：

1. `[-1]` 是本仓库 `6000eb0` 的本地修改，**不是 ACT 上游行为**；上游是 `[0]`。
2. 因此原文第 4 节给出的 83.93M 参数量，在上游/LeRobot 语义下只有
   **51.62M 是有效的**，另外 32.31M 是死参数。
3. 该修改产生的 checkpoint 不兼容性，应当作为第 16 节的一条 **P0** 条目补入
   （目前该节的三条 P0 均未涉及）。

同时，第 5.3 节的对比表"当前层数：4 / 4"应补充说明：这两个 4 **不是巧合，而是
同一个 `enc_layers` 参数**（`detr_vae.py:220`，代码里带 `# TODO shared with VAE
decoder` 注释）——改一个必然改另一个。

---

## 附：复现脚本

本报告的实测数据由以下脚本产生，已随报告一起存放在
`policy/scripts_act_layer_bench/`（含原始结果 `act_layer_sweep.json`）：

```
bench_act_layers.py   # 层数扫描：参数量 / 推理延迟 / 训练步时长 / 峰值显存
bench_backbone.py     # resnet18/34/50 对比
grad_probe.py         # [0] vs [-1] 的逐层梯度范数验证
probe_layer0.py       # tgt=0 导致 decoder 第 0 层 self-attn 失效的验证
flops.py              # 各模块理论 FLOP 分解
```

运行方式（需在 `/home/kewei/YING/ACT_rosbag2` 下，`sys.path` 已在脚本内处理）：

```bash
cd /home/kewei/YING/ACT_rosbag2 && python <script>.py
```
