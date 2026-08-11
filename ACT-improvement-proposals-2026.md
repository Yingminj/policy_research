# ACT 改进提案（基于 2024-08 ~ 2026-08 文献 + 本仓库实测）

配套阅读：[`ACT-layer-depth-analysis.md`](./ACT-layer-depth-analysis.md)（层数/结构实测），`ACT_rosbag2/doc/ACT_深度解析.md`（原始分析）。

本文所有"现状"均指 `/home/kewei/YING/ACT_rosbag2`（`apex_real_C3_task.yaml`：3 相机、`state_dim=16`、`chunk_size=100`、`enc=4/dec=7`、resnet18、post-norm、无 scheduler）。

---

## 0. 优先级总表

| # | 改进 | 代价 | 预期收益 | 置信度 | 是否需重训 |
|---|------|------|----------|--------|-----------|
| 1 | decoder 深度语义 + checkpoint 守卫 | 半天 | 消除静默失效风险 | ★★★★★ | 否（只加校验） |
| 2 | 动作表示改 delta（相对量） | 1 天 | 成功率 + 平滑度，文献最大单点增益 | ★★★★★ | 是 |
| 3 | 训练配方补齐（warmup/cosine、grad clip、AMP、pre-norm） | 1 天 | 收敛稳定性 + 训练速度 ~1.6× | ★★★★☆ | 是 |
| 4 | 相机身份嵌入（修 PE 逐相机相同） | 2 小时 | 多视角消歧，直击已实测缺陷 | ★★★★☆ | 是 |
| 5 | 推理端聚合：temporal ensemble → BID / TAS | 2–5 天 | 反应性↑，不破坏多模态 | ★★★★☆ | 否（TAS 需训 selector） |
| 6 | 异步执行（real-time chunking / prefix 条件） | 3–7 天 | 直接解你 `T_replan ≳ M·N·d + T_infer` | ★★★★☆ | 部分 |
| 7 | 自适应执行 horizon | 3–7 天 | 精细阶段反应快、空域阶段省算力 | ★★★☆☆ | 部分 |
| 8 | 动作分布建模：KL 扫描 / flow-matching head | 3–7 天 | 多模态保真 | ★★★☆☆ | 是 |
| 9 | backbone 换冻结 DINOv2/v3（分视角决策） | 2–4 天 | 泛化↑，但 wrist 视角可能反而掉 | ★★★☆☆ | 是 |
| 10 | MoE decoder（MEAT） | 1–2 周 | 文献报 +5%~30% | ★★★☆☆ | 是 |

**如果只做三件事：#2 + #3 + #4。** 三者合计约 2~3 天工作量，都是低风险、文献支持强、且和你现有 pipeline 正交的改动。

---

## 1. 先把 decoder 深度的语义和 checkpoint 兼容性钉死（P0）

**现状。** `detr_vae.py:134` 在 commit `6000eb0`（2026-07-28）把 `transformer(...)[0]` 改成了 `[-1]`。改之前，upstream ACT 的 bug 使 decoder L1–L6 从不接收梯度（32.3M 参数 = 全模型 38.5% 是死的）；改之后 7 层全部生效。实测梯度范数：

```
index [0]  -> L0:9.741e-01  L1:0.000e+00 ... L6:0.000e+00
index [-1] -> L0:2.427e-01  L1:3.055e-01 ... L6:8.831e-01
```

**风险。** 这次改动**没有改变 state_dict 的形状**。`6000eb0` 之前训出来的 checkpoint 里 L1–L6 还停在 Xavier 初始值，用现在的代码加载会 `strict=True` 静默通过，然后输出垃圾。这是一个不会报错的失效模式。

**文献依据。** LeRobot 0.6.0 明确把 `n_decoder_layers` 默认设为 1 并在注释里指向 [tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25)，即"为了对齐原始实现的**实际行为**"。也就是说社区共识是 ACT 论文里报的所有数字都是 dec=1 的数字，`dec_layers=7` 从未被验证过有效。

**做法。**
- 在 checkpoint 里写入 `{"decoder_index": -1, "commit": <sha>}`，加载时不匹配就 **raise**，不要 warn。
- 把 `dec_layers=7` 当成一个**未验证的超参**而不是默认值。跑一次对照：A(4/1/4) vs A2(4/2/4) vs B(4/7/4)。
- 实测边际成本：decoder 每层推理 +0.40 ms、训练 +3.1 ms/step +0.22 GB。1→7 层换来 +2.1 ms 推理、+32.3M 参数。

**关键机理（前次分析已验证）。** 因为 `tgt = zeros_like(query_pos)`，decoder 第 0 层的 self-attention 的 value 全为 0，输出对所有 100 个 query 是同一个常量（实测 query 间差异 = 2.98e-08）。所以 **dec=1 意味着 query 之间零通信**——chunk 内 100 个时间步只能各自独立地读观测。真正的质变发生在 1→2，不是 4→7。

---

## 2. 动作表示：绝对关节角 → 相对增量（delta）

**这是本次文献检索里证据最强的一条。**

**文献。** [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/pdf/2602.23408)（ICML 2026，双臂真机 **13,000+ rollout / 500+ 训练模型**）：

- delta 表示 vs 绝对表示：平均成功率 **33.7 → 55.0**。
- 机理：`relQpos` 的标准差只有 `absQpos` 的 **31%–37%**，把"全局关节构型回归"变成"局部运动回归"，目标更居中、方差更低。
- joint-space 与 task-space 各有所长：**joint-space 偏控制稳定性，task-space 偏泛化**。
- 另有多篇 2025–2026 工作（[BEHAVIOR Challenge 2025 冠军方案](https://arxiv.org/pdf/2512.06951)、[From Foundation to Application](https://arxiv.org/pdf/2607.06403)）独立复现了同一结论，并观察到 delta 策略轨迹明显更平滑。

**现状。** 你的 ACT 直接回归绝对 qpos（`state_dim=16`）。对双臂 16 维、`chunk_size=100` 的设定，chunk 末端的绝对角度分布跨度很大，正是 delta 表示收益最大的场景。

**做法。**
1. 训练目标改成 `a_k - qpos_t`（**注意是相对于 chunk 起始的 `qpos_t`，不是逐步差分**——逐步差分会在 100 步积分中累积漂移）。
2. 保留 `qpos_t` 作为 proprio token 输入（已经有了）。
3. 归一化统计量要重算：delta 的 std 远小于 absolute，沿用旧 norm 会让 loss 尺度崩掉。
4. 推理端：`a_k = qpos_t + Δ̂_k`。注意此时 temporal ensembling 混的是**不同 `qpos_t` 基准下的增量**，必须先还原成绝对量再做加权平均，否则会引入偏置。这一点很容易写错。

**验证。** 用现有数据重训一版，比 `MAE(step=0..99)` 曲线。delta 表示的典型特征是 **step 靠后的 MAE 显著改善**（因为长程绝对回归最难）。

---

## 3. 训练配方补齐

**现状（代码审计）。** 无 LR scheduler、无 warmup、无 grad clipping（`--clip_max_norm` 在 argparse 里存在但从未被调用）、无 AMP，且用的是 **post-norm** block（`normalize_before=False`）。post-norm + 无 warmup 是 Transformer 训练里最经典的不稳定组合。

**文献。** [Unpacking the Individual Components of Diffusion Policy](https://arxiv.org/pdf/2412.00084) 的核心结论就是：视觉运动策略的性能差异中，**训练配方与组件级选择的贡献常常大于架构选择**。这条对 ACT 同样成立——你在花预算调 `dec_layers`，但 scheduler 和 clipping 的缺失可能是更大的方差来源。

**做法（按性价比排序）。**
1. **grad clipping**：把已有的 `--clip_max_norm` 接上（`torch.nn.utils.clip_grad_norm_`，默认 1.0）。一行代码。
2. **warmup + cosine decay**：500~1000 step 线性 warmup，之后 cosine 到 0。post-norm 架构对 warmup 尤其敏感。
3. **AMP（bf16）**：实测 batch=8 训练 135.0 ms/step，AMP 通常给 1.5~1.8×，且 4090 上 bf16 无需 loss scaling。
4. **pre-norm**：`normalize_before=True`。这个改动会让深层（dec=7）稳定得多，但**会改变权重语义**，需要从头重训，且和 #1 的 checkpoint 校验要一起做。建议作为 #1 消融实验的一个 arm，而不是直接切换。

**注意。** 1~3 是纯增益、几乎无风险；4 有重训成本，放在你确定要用 `dec>1` 之后再做。

---

## 4. 相机身份嵌入（修一个已实测的确定性缺陷）

**这是前次分析里发现的、比原 doc §6.3 更尖锐的问题。**

**现状。** `PositionEmbeddingSine` 的输出**只依赖 feature map 的形状**，且 `normalize=True` 会把每张图的坐标各自 rescale 到 `[0, 2π]`。三个相机的 feature map 形状相同（15×20），于是三份位置编码**逐字节相同**：

```
cam0 block == cam1 block: True
```

而 `detr_vae.py:132-133` 是把三路特征沿 W 轴 `cat` 起来送进 encoder。结果是：**观测 encoder 拿到 902 个 token，但没有任何信号告诉它哪 300 个来自 top、哪 300 个来自 wrist_L**。模型只能靠图像内容本身的统计差异去隐式区分视角——在两个 wrist 相机之间，这个区分几乎不可能可靠。

**文献。**
- [Multi-View Video Diffusion Policy](https://arxiv.org/html/2604.03181v1)：显式的 view-attention 模块，把 token 序列 reshape 成允许跨视角交互，再 reshape 回来做时序层。
- [Learning Fused State Representations for Control from Multi-View Observations](https://arxiv.org/html/2502.01316)：引入**可学习的 fusion [state] embedding**，避免固定视角带来的偏置；并用 mask token 提升缺视角时的鲁棒性。
- [Cross-Attentive Multiview Fusion](https://arxiv.org/html/2604.12551)：指出简单聚合（concat / average pool）浪费了各视角的独有信息。

**做法（成本递增）。**
1. **最小修复（2 小时）**：加一个 `nn.Embedding(n_cams, hidden_dim)`，逐相机加到对应 token 块上。和 `additional_pos_embed` 完全同构，改动量极小。
2. **中等**：两级注意力——先视角内 self-attn，再跨视角 attn（Multi-View Video Diffusion Policy 的做法）。可以在 token 数不变的前提下降低 N² 项。
3. **附带收益**：有了 view embedding 后，随机 drop 某个相机做训练增广才有意义（对应真机上单相机掉线的鲁棒性）。

**为什么值得优先做。** 这不是"可能有帮助"的猜测，而是一个可以直接打印出来验证的实现缺陷，且修复成本几乎为零。

---

## 5. 推理端聚合：temporal ensembling → BID / TAS

**文献。**
- [Bidirectional Decoding (BID)](https://arxiv.org/abs/2408.17355)（arXiv 2024-08，ICLR 2025）：指出 **temporal ensembling 的加权平均在连续决策落入不同模态时是有害的**。BID 采样多个候选 chunk，按两个准则挑选：**backward coherence**（贴近上一步已选序列）+ **forward contrast**（贴近强策略、远离弱策略）。
- [Temporal Action Selection (TAS)](https://arxiv.org/html/2511.04421v2)（2025-11）：进一步指出 BID 没有充分利用**历史观测下的旧预测**。TAS 缓存不同时刻预测的 chunk，用一个轻量 Space-Aware selector（PPO 训练，cosine 相似度 + softmax）在候选间选择。报告：PushT +42.58%、Image PushT +35.97%、FurnitureBench +41.18%；One Leg 噪声条件下 68.4% vs baseline 22.8%。

**现状。** 你用的是原始 EMA temporal ensembling。对 ACT 这种 **L1 回归（本身就是条件均值估计）** 的策略，EMA 的破坏性其实小于对 diffusion 策略——因为 ACT 输出已经是"平均"过的。但代价是反应性：EMA 让策略对新观测的响应被历史拖慢。

**做法。**
- **低成本先试 BID 的 backward coherence 单项**：ACT 是确定性的，需要靠 CVAE latent `z` 采样多个候选（正好用上你现在训了但推理时固定为 0 的 latent 通路）。这是几乎零额外训练成本的改法。
- TAS 需要额外训一个 selector（PPO），且需要真机/仿真的 reward 信号，成本高一档。
- **务必先做 A/B**：在你的任务上，如果动作分布本来就近似单模态，EMA 不会有明显损害，换掉纯属浪费。先量化"连续两次预测的 chunk 之间是否存在双峰"，再决定。

---

## 6. 异步执行 / real-time chunking

**这一条直接对应你 doc §13 里的部署时序模型 `T_replan,actual ≳ M·N·d + T_infer`。**

**文献。**
- [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/abs/2506.07339)：在执行当前 chunk 的同时生成下一个 chunk，把"保证会被执行的"动作 **freeze**，其余部分 **inpaint**。无需重训，对 diffusion/flow VLA 开箱即用，对推理延迟异常鲁棒。
- [REMAC: Real-Time Robot Execution with Masked Action Chunking](https://arxiv.org/abs/2601.20130)（2026-01）：通过 masked action chunking 学习对预训练策略的修正，并用 **prefix-preserved sampling** 强化 chunk 间连续性，专门处理异步推理下"意图动作 ≠ 实际执行动作"的错配。

**对 ACT 的适配性（重要）。** RTC 的 freeze+inpaint 机制**依赖 diffusion/flow 的迭代去噪过程**——ACT 是单次前向的 L1 回归，没有可以 inpaint 的中间状态。所以：
- 直接套 RTC **不行**，除非你先做 #8（换 flow head）。
- 能直接用的是 **REMAC 式的 prefix 条件**：把"已经确定会执行的前 d 步动作"作为额外输入 token 喂给 decoder，训练时用 mask 模拟。这需要重训，但架构改动小（多加一个 action-prefix projection + mask）。
- 实测你的推理是 8.92 ms（3×480×640、4090）。如果部署平台就是 4090，`T_infer` 可能不是瓶颈，`M·N·d` 才是——先测清楚再决定是否值得做这一条。

---

## 7. 自适应执行 horizon

**文献。**
- [Dynamic Execution Horizon Prediction (DEHP)](https://arxiv.org/abs/2606.11408)：**冻结**预训练 chunk 策略，只用在线 RL 训一个轻量的 execution-horizon 预测分支。学到的行为是：精细阶段预测短 horizon、自由空间运动预测长 horizon。
- [VLA Knows Its Limits](https://arxiv.org/pdf/2602.21445) / [VLA-Corrector](https://arxiv.org/pdf/2607.01804)：事件触发的动态 horizon——稳定阶段保持长 chunk 的效率，局部失败时截断以恢复闭环响应。
- [Adaptive Action Chunking for Robotic Imitation Learning](https://www.mdpi.com/2313-7673/11/5/316)（Biomimetics 2026）：双分支网络，共享视觉编码器 + 并行的动作头和 **chunk-size 预测头**，真机双臂验证。

**现状。** 你的 `chunk_size=100` 且 `n_action_steps=100`——**完全开环执行整个 chunk**。这是反应性最差的配置。

**做法（成本递增）。**
1. **先做最便宜的**：`n_action_steps` 从 100 降到 25/50，其余不变。这是一个纯推理端超参，零训练成本，直接换来 2~4× 的闭环频率，代价是 `T_replan` 压力上升。**先扫这个，再考虑自适应。**
2. DEHP 式的冻结主干 + RL horizon head 是很优雅的方案，但需要在线 RL 基础设施。
3. 最省事的启发式：用 CVAE latent 多次采样的**方差**作为不确定性代理，方差大 → 截短 horizon。不需要 RL，几行代码，值得先试。

---

## 8. 动作分布建模：KL 扫描，或换 flow-matching head

**现状。** ACT 的 CVAE latent 在**推理时被固定为 0**（取先验均值），所以 CVAE 在部署时唯一的作用是"训练时吸收演示中的多模态性，让 L1 回归不至于被平均掉"。而 `kl_weight` 是一个高敏感超参。

**文献（CVAE 侧）。** KL weight 过高会导致 **posterior collapse**——模型学会忽略 latent、预测平均行为。有工作报告 KL weight 从 0 增到 1e-4 就使某任务成功率从 100% 掉到 30%。常见建议是从 10 起，在小数据集上扫 1/5/10/50。

**文献（生成式 head 侧）。** 多篇 2025–2026 工作指出，**回归式方法会把多模态 latent 分布坍缩成确定性预测**，而 flow matching 在分布层面工作，能保留动作多样性；flow 相比 diffusion 训练更易、推理更快。[MARS Policy](https://arxiv.org/pdf/2605.29766)（"Multimodality Only When It Matters"）提出只在真正需要多模态的状态上付出生成式建模的代价——对 ACT 这种延迟敏感的场景是很务实的折中。

**做法。**
1. **先扫 `kl_weight`**（1/5/10/50），成本最低。同时实测：VAE encoder 深度在推理时**完全免费**（vae4→vae1 省 13.0M 参数、2.8 ms/step、0.32 GB，推理耗时 8.98 vs 8.92 ms 属噪声），所以 `n_vae_encoder_layers` 可以放心地单独调——LeRobot 正是把它从 `n_encoder_layers` 里拆出来的。
2. **再考虑换 head**：把 action head 从 `Linear(512→16)` 换成 flow-matching（几步 Euler）。这是本清单里最伤推理延迟的改动，且会使 #5/#6 的方案空间变大（RTC 就能用了）。**除非你有证据表明你的演示数据确实多模态，否则不建议做。** 判据：同一初始状态下多条人类演示的轨迹是否明显分叉。

---

## 9. 视觉 backbone

**现状与实测。** resnet18 全量微调，ImageNet 预训练。backbone 占推理 FLOP 的绝对大头（480×640×3 相机下 33.43 GFLOP，而 obs-encoder 9.47 GFLOP/层、decoder 2.12 GFLOP/层）。实测换 backbone 的代价：**resnet18 8.99 ms / resnet34 12.44 ms / resnet50 14.41 ms**——换 backbone 比加 8 层 decoder 还贵。

**文献。**
- [DINOv3-Diffusion Policy](https://arxiv.org/pdf/2509.17684)：自监督大视觉模型用于视觉运动策略，对比标准监督式 backbone 的性能与泛化。
- [StereoPolicy](https://arxiv.org/html/2605.09989v1)：**关键的细节结论**——拼接冻结 DINOv2 的特征能提升性能，但**收益依赖视角：外部视角受益，wrist 视角反而变差**，推测是域不匹配（DINOv2 的预训练分布里没有近距离手腕视角）。
- 三种编码器使用范式（从零训 / 冻结预训练 / 端到端微调）在文献里各有胜出场景，没有统一答案。

**做法（针对你的 3 相机配置）。**
- **分视角决策**，不要一刀切：top（外部视角）用冻结 DINOv2/v3，wrist_L / wrist_R 保留可微调的 resnet18。这正是 StereoPolicy 的实证结论所建议的。
- 顺带修一个 repo 问题：`detr_vae.py:124` 是 `self.backbones[0](...)  # HARDCODED`——**三个相机共享同一套 backbone 权重**。这和 #4 的相机身份缺失是同一个根因的两面。分视角 backbone 会自然解决它，但也会让参数量上升。
- 冻结 backbone 的额外好处：反向传播不经过 backbone，训练显存和时间大幅下降，可以把省下的预算给 batch size。

---

## 10. MoE decoder（MEAT）

**文献。** [Mixture-of-Experts Action Chunking Transformer (MEAT)](https://www.sciencedirect.com/science/article/abs/pii/S0921889026002885)（*Robotics and Autonomous Systems*, 2026）：在 ACT 架构里引入**稀疏专家路由**，让不同专家模块针对不同 token、不同任务阶段、不同时间区段专门化。报告在仿真与真机操作任务上**比标准 ACT 提升 5%–30%**，同时保持相对 diffusion 策略的推理效率优势。

**为什么这条排在最后但值得记住。** 它和 #1 是同一个问题的两种答案：**"decoder 该有多大容量、容量该放在哪"**。你现在的 `dec=7` 是"稠密地堆 7 份相同结构"，MEAT 的主张是"同样的参数量，按 chunk 内的时间阶段稀疏路由"。考虑到 ACT 的 100 步 chunk 天然横跨"接近 → 接触 → 操作"多个阶段，按阶段专门化在直觉上比均匀加深更合理。

**实施建议。** 这是本清单里工程量最大的一条（1–2 周）。**先做完 #1 的消融**——如果 A2(4/2/4) 已经追平 B(4/7/4)，说明你的任务根本不需要额外 decoder 容量，那 MoE 也不会有收益，直接跳过。

---

## 附：另外三个方向（未列入 10 条，但和你的硬件相关）

- **力/触觉输入。** [Haptic-ACT](https://arxiv.org/pdf/2409.11925) 把力矩通过 MLP 映射成观测 token，在易损物体操作上 80% vs 纯视觉 ACT 50%；Bi-ACT / Bi-LAT 融合本体感觉 + 力矩 + 视觉；CATCH-FORM-ACTer 让 transformer 同时预测运动和**柔顺参数**（刚度、阻尼）。如果你的 apex 平台有力矩读数，这是一个 token 数几乎不增加、收益明确的改法。
- **Q-chunking RL 微调。** [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)：直接在 chunked action space 里跑 RL，利用离线数据的时序一致行为改善探索，并允许无偏的 n-step backup。适合"BC 已经能做到 60~70%，想推到 90%"的阶段。
- **立体/3D 感知。** StereoPolicy 等工作显示双目/深度输入对精细操作有帮助。你已有 3 个相机，如果其中有 baseline 已知的一对，这是几乎免费的信息。

---

## 建议的执行顺序

**第 1 周（低风险、可并行）**
- #1 checkpoint 守卫（半天，先做，防止后面所有实验被污染）
- #3.1–3.3 grad clip + warmup/cosine + AMP（1 天）
- #4.1 相机身份嵌入（2 小时）
- #7.1 `n_action_steps` 从 100 降到 25/50 扫一遍（纯推理端，零训练成本）

**第 2 周（一次重训解决）**
- #2 delta 动作表示 + #8.1 `kl_weight` 扫描，和 #1 的层数消融 A/A2/B 合并成一个正交实验矩阵

**之后（按第 2 周结果决定）**
- 若长程 MAE 仍差 → #6 prefix 条件 / #7 自适应 horizon
- 若发现明确的多模态演示 → #8.2 flow head → 解锁 #5/#6 的完整方案
- 若容量确实不够（B 显著优于 A2）→ #10 MoE
- 若泛化是瓶颈而非拟合 → #9 分视角 backbone

**统一的评估口径。** 不要看 total loss，看 **`MAE(step = 0..99)` 曲线**。ACT 的改进几乎总是表现为"曲线尾部下压"而不是整体平移；total loss 会被前 20 步主导，掩盖真正的差异。

---

## 参考文献

**动作表示 / 空间设计**
- [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/pdf/2602.23408) — ICML 2026
- [Task adaptation of VLA: 1st Place, 2025 BEHAVIOR Challenge](https://arxiv.org/pdf/2512.06951)
- [From Foundation to Application: Improving VLA Models in Practice](https://arxiv.org/pdf/2607.06403)

**Chunk 执行 / 推理端聚合**
- [Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling](https://arxiv.org/abs/2408.17355) — ICLR 2025
- [Temporal Action Selection for Action Chunking](https://arxiv.org/html/2511.04421v2)
- [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/abs/2506.07339)
- [Real-Time Robot Execution with Masked Action Chunking (REMAC)](https://arxiv.org/abs/2601.20130)
- [Dynamic Execution Horizon Prediction for Chunk-based Robot Policies](https://arxiv.org/abs/2606.11408)
- [VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies](https://arxiv.org/pdf/2602.21445)
- [VLA-Corrector: Lightweight Detect-and-Correct Inference for Adaptive Action Horizon](https://arxiv.org/pdf/2607.01804)
- [Adaptive Action Chunking for Robotic Imitation Learning](https://www.mdpi.com/2313-7673/11/5/316) — Biomimetics 2026

**架构 / 容量**
- [Mixture-of-experts action chunking transformers for high-precision robot imitation learning (MEAT)](https://www.sciencedirect.com/science/article/abs/pii/S0921889026002885) — RAS 2026
- [Unpacking the Individual Components of Diffusion Policy](https://arxiv.org/pdf/2412.00084)
- [MARS Policy: Multimodality Only When It Matters](https://arxiv.org/pdf/2605.29766)

**视觉 backbone / 多视角**
- [DINOv3-Diffusion Policy](https://arxiv.org/pdf/2509.17684)
- [StereoPolicy: Improving Robotic Manipulation Policies via Stereo Perception](https://arxiv.org/html/2605.09989v1)
- [Multi-View Video Diffusion Policy](https://arxiv.org/html/2604.03181v1)
- [Learning Fused State Representations for Control from Multi-View Observations](https://arxiv.org/html/2502.01316)
- [Cross-Attentive Multiview Fusion of Vision-Language Embeddings](https://arxiv.org/html/2604.12551)

**力/触觉 · RL**
- [Haptic-ACT: Bridging Human Intuition with Compliant Robotic Manipulation via Immersive VR](https://arxiv.org/pdf/2409.11925)
- [Reinforcement Learning with Action Chunking (Q-chunking)](https://arxiv.org/abs/2507.07969)
- [SERNF: Sample-Efficient Real-World Dexterous Policy Fine-Tuning](https://arxiv.org/pdf/2602.09580)
