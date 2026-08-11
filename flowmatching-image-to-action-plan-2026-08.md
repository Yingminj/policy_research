# Flow Matching 动作生成:文献综述、LeRobot 仓库审计与「图像 → 动作」十方案

- **日期**:2026-08-11
- **代码基线**:`/home/kewei/YING/robot_data_platform/lerobot`
- **目标**:构建一个直接从图像生成动作(image → action)的模型
- **文献时间范围**:2024-10 ~ 2026-08

> 说明:2026 年的条目来自本次网络检索到的摘要 / HTML 页面,未逐篇通读全文。
> π₀ / π₀.₅ / GR00T / RTC / ReinFlow / VITA 这几条细节可确认。
> 仓库审计部分的所有 `file:line` 均为实际 grep / 阅读所得。

---

## 目录

1. [近两年 Flow Matching 动作/运动生成模型综述](#一近两年-flow-matching-动作运动生成模型综述)
2. [LeRobot 仓库现状审计](#二lerobot-仓库现状审计)
3. [「图像 → 动作」十个可行方案](#三图像--动作直接生成10-个可行方案)
4. [推荐路线与里程碑](#四推荐路线与里程碑)
5. [参考文献](#五参考文献)

---

## 一、近两年 Flow Matching 动作/运动生成模型综述

Flow Matching(FM)在 2024–2026 期间基本取代 DDPM 成为动作生成的默认生成范式。原因有三:

1. **训练目标更简单** —— 直接回归速度场 `v = x₁ - x₀`,无需噪声调度表;
2. **概率路径更直** —— 线性插值路径 `x_t = t·x₀ + (1-t)·x₁`,ODE 求解误差小;
3. **采样步数大幅下降** —— 从 DDPM 的上百步降到 4–10 步,甚至 1 步。

以下按六条研究主线归纳。

### 1.1 VLA + Flow Matching Action Expert(主流范式)

当前最大的一条线:预训练 VLM 负责感知与语义理解,后面挂一个用 FM 训练的
"action expert" 输出动作块(action chunk)。

| 模型 | 关键设计 |
|---|---|
| **π₀** ([2410.24164](https://arxiv.org/abs/2410.24164)) | PaliGemma + 独立参数的 action expert,双向注意力,条件 FM,10 步去噪,50 步 chunk |
| **π₀.₅** | 引入非均匀(Beta)时间采样,解决前几步去噪最难的问题;离散状态 token + knowledge insulation |
| **GR00T N1 / N1.5** | 双系统:VLM 慢系统 + DiT FM 动作头快系统,proprio 条件,推理仅 4 步 |
| **SmolVLA** | 小型化 SmolVLM + FM expert,交错跳层注意力,面向消费级硬件 |
| **WALL-OSS** | 指令推理 / 子目标分解 / 细粒度动作合成的耦合架构 |
| **X-VLA** | soft-prompt 多本体统一,注意力形式的 FM(直接回归 clean action) |
| **GraspVLA** | 把自回归感知任务与 FM 动作生成统一进一条 CoT |

**共同结论**:动作表示已收敛到「chunk 级 flow matching / diffusion」,而非自回归
next-token;π₀-FAST 那条离散 token 路线主要作为对比基线存在。

### 1.2 采样加速:少步 → 一步(2025–2026 最热)

多步去噪是 VLA 反应性(reactivity)的核心瓶颈,这一年的工作几乎都在打这个点。

- **FlowPolicy**(2024):consistency flow matching + 3D 点云条件,一步生成。
- **MP1** ([2507.10543](https://arxiv.org/abs/2507.10543)):首个把 **MeanFlow**
  (学习区间平均速度场,靠 MeanFlow Identity)用到操作策略,1-NFE 出整条轨迹,
  配 Dispersive Loss,毫秒级推理。
- **MVP** ([2602.13810](https://arxiv.org/abs/2602.13810),ICLR 2026 Oral):
  在平均速度场上加**瞬时速度约束(IVC)**,修正 MeanFlow 训练的自举误差,
  当前一步生成的 SOTA。
- **Mean-Flow One-Step VLA** ([2603.01469](https://arxiv.org/html/2603.01469v1)):
  把 MeanFlow 搬到 VLA 尺度,处理噪声诱导的退化。
- **Implicit Drifting Policy** ([2606.01098](https://arxiv.org/pdf/2606.01098)):
  用条件专家几何做一步生成。
- **EfficientFlow** ([2512.02020](https://arxiv.org/pdf/2512.02020)):
  等变(equivariant)FM,用对称性换样本效率与算力。
- **Shortcut Model**:把步长 `d` 作为额外条件输入,同一网络支持 1/2/4/… 步推理;
  ReinFlow 已验证其在 1 步下仍可 RL 微调。

### 1.3 实时执行与异步分块(部署侧)

FM 策略的第二个瓶颈是「算下一块时机器人在动」。

- **RTC / Real-Time Chunking** ([2506.07339](https://arxiv.org/html/2506.07339),
  Physical Intelligence):把保证会执行的动作「冻结」,其余部分做 **inpainting**,
  免训练、可直接套到任意 diffusion/flow VLA 上。**本仓库已实现。**
- **πR²** ([2607.26055](https://arxiv.org/html/2607.26055v1)):反应式实时流策略。
- **Denoising-Variance Adaptive Chunking**
  ([2606.03847](https://arxiv.org/pdf/2606.03847)):用去噪方差判断何时该重新规划。
- **Initial Noise Selection** ([2606.19774](https://arxiv.org/pdf/2606.19774)):
  异步执行下选一个好的初始噪声,让轨迹「起点对、终点也对」。
- **Action-Prior Denoising** ([2605.25537](https://arxiv.org/pdf/2605.25537))、
  **ABPolicy** ([2602.23901](https://arxiv.org/pdf/2602.23901),B 样条流策略)、
  **Prior-Corrected Orthogonal Trust-Region Guidance**
  ([2605.24433](https://arxiv.org/pdf/2605.24433)):都在解决块间接缝抖动。

### 1.4 Flow 策略的 RL 后训练

FM 策略的似然不可直接计算,这一年出现了多种绕法。

- **ReinFlow** ([2505.22094](https://arxiv.org/abs/2505.22094),NeurIPS 2025):
  向确定性流路径注入**可学习噪声**,把流转成离散时间马尔可夫过程,从而精确算似然。
  腿式运动奖励净增 135%,相比 DPPO 省 82.6% wall time;
  Shortcut 模型在 1–4 步下成功率净增 40%。
- **FPO(Flow Policy Optimization)**:用逐样本的条件 FM 目标变化量重构重要性采样。
- **Q-VGM** ([2606.08015](https://arxiv.org/html/2606.08015v1)):
  Q 值梯度匹配,离策略微调 flow-matching VLA。
- **FlowDPG** ([2606.22303](https://arxiv.org/pdf/2606.22303)):
  在 FM 策略上做确定性策略梯度,真机操作。
- **RL with Density Transport** ([2606.08602](https://arxiv.org/html/2606.08602v1))、
  **Score-Based One-step MeanFlow PO** ([2605.23365](https://arxiv.org/html/2605.23365v1))。

### 1.5 表示与几何改进

- **Frequency-Aware FM** ([2606.20135](https://arxiv.org/abs/2606.20135v1)):
  DCT 变换到频域,在**余弦系数上**做 flow matching,再基展开回连续动作,
  并正则一阶时间导数 → 动作天然平滑连续。
- **WarmPrior** ([2605.13959](https://arxiv.org/pdf/2605.13959)):
  用时序先验「拉直」流路径,减少所需步数。
- **VFP**:变分 flow matching,显式建模多模态。
- **Riemannian FM**:在 SE(3) / 流形上做 FM,适合末端位姿动作。
- **测试时组合** ([2510.01068](https://arxiv.org/pdf/2510.01068)):
  分布级组合多个 diffusion/flow 策略。

### 1.6 视觉直接到动作(与本项目目标最相关)⭐

- **VITA: Vision-to-Action Flow Matching Policy**
  ([2507.13231](https://arxiv.org/abs/2507.13231),**ICLR 2026**,
  代码 [ucd-dare/VITA](https://github.com/ucd-dare/VITA)):
  **无噪声、无条件模块**的 FM。核心洞见:既然流的源分布可以是任意分布,
  那就**直接把视觉 latent 当作流的源**,流向动作 latent。
  因此生成过程中不再需要反复注入视觉条件(省掉 cross-attention / AdaLN),
  1.5–2× 更快。为解决动作与视觉 latent 的结构差异,引入 action autoencoder
  把动作映射到与视觉 latent 对齐的空间,并提出 **flow latent decoding** ——
  把动作重建损失反传穿过整个 ODE 求解步骤。
  在 ALOHA / Robomimic 的 9 仿真 + 5 真机任务上验证。
- **Patch Policy** ([2607.18236](https://arxiv.org/html/2607.18236)):
  DINOv2 dense patch 特征直接驱动控制,6.5 GPU-hours 收敛,远低于 OpenVLA-OFT。

### 1.7 人体运动生成(motion synthesis)分支

若 "motion model" 也包含人体运动生成:

- **Motion Flow Matching** ([2312.08895](https://arxiv.org/abs/2312.08895)):
  最早把 FM 用于人体运动合成与编辑,采样从上千步降到 10 步。
- **FlowMotion** ([2504.01338](https://arxiv.org/abs/2504.01338)):
  target-predictive 条件 FM,专门抑制抖动。
- **MotionFlux** ([2508.19527](https://arxiv.org/pdf/2508.19527)):
  rectified FM + 偏好对齐,解决文本-动作细粒度语义对齐,实时合成。
- **MotionHiFlow** ([2604.23264](https://arxiv.org/pdf/2604.23264),ICME):
  层次化 FM 文生动作。
- **Riemannian Motion Generation** ([2603.15016](https://arxiv.org/pdf/2603.15016)):
  黎曼流形上统一表示与生成。
- **MotionGPT3** ([2603.26747](https://arxiv.org/html/2603.26747v2)):
  从 diffusion 迁移到 flow。
- Rectified Flow 多模态交互 / 反应式 3D 运动生成。

---

## 二、LeRobot 仓库现状审计

**结论:有,而且非常多。** `/home/kewei/YING/robot_data_platform/lerobot` 是一个大幅
扩展过的 LeRobot fork,20 个策略里 **12 个是 flow-matching 系**。

### 2.1 属于 Flow Matching 的策略

路径前缀均为 `src/lerobot/policies/`。

| 策略 | 关键代码位置 | FM 形式 |
|---|---|---|
| `pi0` | `pi0/modeling_pi0.py:596`(`u_t = noise - actions`)、`:641` MSE、`:644` `sample_actions`、`:705` `denoise_step` | 标准条件 FM,10 步;Beta(1.5, 1.0) 时间采样(`configuration_pi0.py:44-48`) |
| `pi05` | `pi05/modeling_pi05.py`(结构同 π₀) | 同上 + 离散状态 token |
| `smolvla` | `smolvla/modeling_smolvla.py:546` `sample_noise`、`:549` `sample_time`、`:726` `sample_actions`、`:775` `denoise_step` | 10 步(`configuration_smolvla.py:63`) |
| `groot` | `groot/groot_n1_7.py:590` `velocity = actions - noise`、`:621` loss、`:662-684` Euler 积分 | DiT 动作头,**仅 4 步**(`groot_n1_7.py:134`) |
| `evo1` | `evo1/flow_matching.py:173` `FlowmatchingActionHead`、`:437` `predict_velocity` | 多本体 FM 头,32 步(`configuration_evo1.py:71,78`) |
| `vla_jepa` | `vla_jepa/action_head.py:289, 303, 335` | DiT + FM,JEPA 世界模型 |
| `xvla` | `xvla/modeling_xvla.py:223-234`(训练)、`:255-271`(采样) | **注意:预测 clean action 而非 velocity**,`x_t = t·noise + (1-t)·action` |
| `multi_task_dit` | `configuration_multi_task_dit.py:38` `objective: "diffusion" \| "flow_matching"`、`:241` `is_flow_matching` | **可切换开关** —— 最适合做对照实验的基线 |
| `wall_x` | `wall_x/modeling_wall_x.py:315` `flow_loss` | float32 稳定化 |
| `lingbot_va` | `lingbot_va/modeling_lingbot_va.py:260` `_flow_matching_loss`;`lingbot_va/utils.py:933` 双流训练 | 20 / 50 步 |
| `molmoact2` | `molmoact2/modeling_molmoact2.py:883, 1009, 1641` | **连续 FM 与离散 AR 双模式** |
| `eo1` | `configuration_eo1.py:68-71` Beta 时间采样 | π₀ 家族变体 |
| `fastwam` | `configuration_fastwam.py:202` 10 步;`fastwam/wan/` | Wan 视频世界模型 |
| `rtc` | `rtc/modeling_rtc.py:126-229`(guidance 计算) | **不是策略,是推理时包装器**,实现 Real-Time Chunking |

### 2.2 非 Flow Matching 的策略

`act`(L1 回归)、`diffusion`(DDPM)、`vqbet`(VQ + BeT)、`tdmpc`、
`pi0_fast`(FAST tokenizer + 自回归)、`gaussian_actor`(RL 高斯策略)。

### 2.3 仓库空白 = 机会点

对 `meanflow` / `mean_flow` / `shortcut` / `consistency`(FM 语义下)全仓 grep,
**零命中**。也就是说:

1. **没有一步 / 少步生成范式**(MeanFlow、Shortcut、Consistency FM)——
   全部策略都是 4–50 步 Euler。
2. **没有 VITA 式「视觉 latent 作为流源」的无条件 FM** ——
   所有策略都靠 cross-attn / AdaLN 反复注入视觉条件。
3. **没有 flow 策略的 RL 后训练**(ReinFlow / FPO / Q-VGM)。
4. **没有纯视觉、无语言的轻量 FM 策略** —— 所有 FM 策略都背着一个 VLM。

### 2.4 新增策略的落地方式

```
src/lerobot/policies/<name>/
    configuration_<name>.py    # @PreTrainedConfig.register_subclass("<name>")
    modeling_<name>.py         # 继承 PreTrainedPolicy
    processor_<name>.py        # make_<name>_pre_post_processors
    __init__.py
```

工厂是动态查表,见 `src/lerobot/policies/factory.py:120`
(`PreTrainedConfig.get_choice_class(policy_type)`)。
完整规范见 `docs/source/bring_your_own_policies.mdx`。

---

## 三、「图像 → 动作」直接生成:10 个可行方案

### 方案 1:最小基线 —— `multi_task_dit` 直接切 flow_matching

- **做法**:`objective="flow_matching"`,视觉编码器换 DINOv2 / SigLIP,
  去掉语言分支,DiT 输出 chunk。
- **改动**:仅配置 + 视觉 backbone 替换,几十行。
- **价值**:两小时内拿到 diffusion vs FM 的**同架构对照实验**,
  是后面所有方案的度量基准。
- **难度**:★☆☆☆☆
- **定位**:第一步必做。

### 方案 2:VITA 式 —— 视觉 latent 直接作为流的源分布 ⭐

- **做法**:抛弃高斯噪声。视觉编码器输出 latent `z_img` 作为 `x₀`,
  动作 autoencoder 把动作编码到同维度 latent `z_act` 作为 `x₁`,
  学 `v(x_t, t)`,**速度场网络里完全不再有条件注入模块**。
  训练时用 flow latent decoding:动作重建损失反传穿过 ODE 求解步。
- **价值**:1.5–2× 推理加速,参数量大幅下降,而且这正好是
  「直接从图像生成动作」的字面实现。仓库里完全空白。
- **风险**:ODE 步数上反传显存开销大(需 gradient checkpointing
  或只反传最后几步);视觉 / 动作 latent 维度对齐需要调。
- **难度**:★★★☆☆
- **定位**:核心方案,优先级最高。

### 方案 3:MeanFlow 一步策略(1-NFE)

- **做法**:新建 `policies/meanflow/`。网络输入 `(x_t, r, t)`,
  学区间平均速度 `u(x_t, r, t)`,用 MeanFlow Identity 训练;
  推理一次前向 `x₁ = x_t - (t-r)·u`。参考 MP1;
  若不稳定,叠加 MVP 的瞬时速度约束(IVC)。
- **价值**:把 π₀ 的 10 步 / GR00T 的 4 步压到 1 步,控制频率直接上一个数量级。
- **风险**:MeanFlow 训练需要 JVP(`torch.func.jvp`),
  对 bf16 和 FlashAttention 不友好,建议速度场头用 fp32。
- **难度**:★★★★☆
- **定位**:与方案 2 正交,可叠加(VITA + MeanFlow)。

### 方案 4:Shortcut Model —— 一个模型支持任意步数

- **做法**:把步长 `d` 作为额外条件嵌入,训练时混合 FM 损失 + self-consistency
  损失(`f(x, t, 2d) ≈ 平均两次 f(·,·,d)`)。
- **价值**:训练一次,部署时按算力 / 延迟预算在 1/2/4/8 步之间自由切换,
  不用为每个平台单独训模型。也是 ReinFlow 验证过最适合后续 RL 微调的形式。
- **难度**:★★★☆☆

### 方案 5:从现成 π₀ / GR00T 蒸馏出少步学生模型

- **做法**:不从头训。用仓库里已有的 `pi0` / `groot` 作为 teacher(10 / 4 步),
  按 consistency distillation 或 rectified-flow reflow
  (在 teacher 生成的 noise-action 配对上重训,把路径拉直)训一个纯视觉学生。
- **价值**:**投入产出比最高** —— 直接继承大模型的泛化能力,不需要海量数据,
  且能干净地剥掉语言分支得到「纯图像 → 动作」模型。
- **风险**:上限受 teacher 约束。
- **难度**:★★☆☆☆
- **定位**:数据量不足时的首选。

### 方案 6:冻结视觉基座 + 轻量 FM 头(Patch Policy 路线)

- **做法**:冻结 DINOv2 / SigLIP,取 dense patch tokens,
  后面接一个 ~50M 的 DiT FM 头(可直接复用
  `groot/action_head/cross_attention_dit.py` 或 `vla_jepa/action_head.py`)。
- **价值**:单卡 L40S ~6.5 小时收敛;最容易复现、最容易做消融的配置。
- **难度**:★☆☆☆☆
- **定位**:方案 2 / 3 的对照组。

### 方案 7:频域 Flow Matching(DCT 系数上做流)

- **做法**:动作块经 DCT 变到频域,在前 K 个余弦系数上做 FM,
  推理时基展开回连续轨迹;附加一阶时间导数正则。
- **价值**:① 动作**天生平滑**,消除块内抖动;
  ② 只对低频系数建模 → 生成维度从 `T×A` 降到 `K×A`(K ≪ T),显著提速。
  对真机部署极友好。
- **难度**:★★☆☆☆
- **定位**:低风险高回报,可作为任意方案的插件。

### 方案 8:接上仓库已有的 RTC,做异步实时执行

- **做法**:新策略只要暴露 `denoise_step` 接口,就能直接套
  `policies/rtc/modeling_rtc.py` 的 guidance 包装器(`:126-229`),免训练。
  再叠加 denoising-variance 自适应重规划决定何时换块。
- **价值**:解决「算下一块时机器人在动」的接缝问题,
  这是真机上最直观的体验提升。**几乎零成本** —— 仓库已经写好了。
- **难度**:★☆☆☆☆
- **定位**:真机阶段必做。

### 方案 9:世界模型联合训练(image → future latent → action)

- **做法**:复用 `fastwam` 的 Wan 视频 VAE / latent 或 `vla_jepa` 的 JEPA 世界模型。
  用同一个 FM 网络联合生成「未来视觉 latent + 动作块」,
  或者先流到未来视觉 latent、再从中解码动作。
- **价值**:未来帧预测提供极强的稠密自监督信号,
  能用大量无动作标注的视频数据预训练,大幅缓解机器人数据稀缺。
- **风险**:训练成本最高;视频生成分支容易主导损失,需要仔细配权重。
- **难度**:★★★★★
- **定位**:中长期方向,不适合作为起点。

### 方案 10:RL 后训练(ReinFlow / FPO / Q-VGM)

- **做法**:模仿学习收敛后,用 ReinFlow 向确定性流路径注入可学习噪声,
  转成离散时间 MDP 从而可精确计算似然,再用 PPO 微调;
  或用 Q-VGM 做离策略 Q 值梯度匹配。
  仓库里 `gaussian_actor` + `src/lerobot/rewards/sarm`(已有奖励模型)
  可直接提供 critic 与奖励信号。
- **价值**:文献报告成功率净增 40%(1–4 步)、腿式奖励净增 135%,
  且在**少步 / 一步**模型上依然有效 —— 与方案 3 / 4 天然互补。
- **难度**:★★★★☆
- **定位**:成功率卡在瓶颈时的破局手段。

### 方案速查表

| # | 方案 | 难度 | 主要收益 | 阶段 |
|---|---|---|---|---|
| 1 | multi_task_dit 切 FM 基线 | ★☆☆☆☆ | 对照基准 | 起步 |
| 2 | VITA 视觉 latent 作流源 | ★★★☆☆ | 1.5–2× 加速 + 新颖度 | 主攻 |
| 3 | MeanFlow 一步生成 | ★★★★☆ | 1-NFE,频率提升一个量级 | 主攻 |
| 4 | Shortcut 任意步数 | ★★★☆☆ | 一模型多部署预算 | 主攻(备选) |
| 5 | 从 π₀/GR00T 蒸馏 | ★★☆☆☆ | 数据不足时的最优解 | 兜底 |
| 6 | 冻结基座 + 轻量 FM 头 | ★☆☆☆☆ | 6.5 GPU-h 收敛 | 起步 |
| 7 | 频域 DCT FM | ★★☆☆☆ | 平滑 + 降维提速 | 插件 |
| 8 | RTC 异步实时执行 | ★☆☆☆☆ | 消除块间接缝,零成本 | 部署 |
| 9 | 世界模型联合训练 | ★★★★★ | 无标注视频预训练 | 中长期 |
| 10 | RL 后训练 | ★★★★☆ | 成功率 +40% | 提升上限 |

---

## 四、推荐路线与里程碑

### 阶段 A:起步(1–2 周)

- 方案 1(`multi_task_dit` FM 基线)
- 方案 6(冻结基座 + 轻量 FM 头)
- **产出**:可信的评测协议与对照基准;确认 diffusion → FM 的增量。

### 阶段 B:主攻(1–2 月)

- 方案 2(VITA 式视觉源分布)+ 方案 7(频域)叠加
- 再用方案 3(MeanFlow)压到一步

这三条正交,组合起来就是一个新颖度足够、且填补本仓库真实空白的技术方案:

> **以视觉 latent 为源、在频域上做一步 MeanFlow 的图像到动作策略**

### 阶段 C:兜底与部署

- 数据不足 → 方案 5(从 π₀ 蒸馏)先拿到可用模型
- 真机部署 → 方案 8(RTC,零成本)
- 成功率瓶颈 → 方案 10(RL 后训练)

### 阶段 D:中长期

- 方案 9(世界模型联合训练),用无动作标注视频扩展数据规模

---

## 五、参考文献

### VLA + Flow Matching

- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)

### 视觉直接到动作

- [VITA: Vision-to-Action Flow Matching Policy (ICLR 2026)](https://arxiv.org/abs/2507.13231) · [代码](https://github.com/ucd-dare/VITA)
- [Patch Policy: Efficient Embodied Control via Dense Visual Representations](https://arxiv.org/html/2607.18236)

### 少步 / 一步生成

- [MP1: MeanFlow Tames Policy Learning in 1-step](https://arxiv.org/abs/2507.10543)
- [Mean Flow Policy with Instantaneous Velocity Constraint (ICLR 2026 Oral)](https://arxiv.org/abs/2602.13810)
- [Mean-Flow based One-Step VLA](https://arxiv.org/html/2603.01469v1)
- [Implicit Drifting Policy: One-Step Action Generation](https://arxiv.org/pdf/2606.01098)
- [EfficientFlow: Efficient Equivariant Flow Policy Learning](https://arxiv.org/pdf/2512.02020)

### 实时执行 / 异步分块

- [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/html/2506.07339)
- [πR²: Reactive Real-time Flow Policies](https://arxiv.org/html/2607.26055v1)
- [Denoising-Variance Adaptive Chunking](https://arxiv.org/pdf/2606.03847)
- [Start Right, Arrive Right: Initial Noise Selection](https://arxiv.org/pdf/2606.19774)
- [Action-Prior Denoising for Smooth Real-Time Chunking](https://arxiv.org/pdf/2605.25537)
- [ABPolicy: Asynchronous B-Spline Flow Policy](https://arxiv.org/pdf/2602.23901)
- [Smoother Action Chunking via Prior-Corrected Orthogonal Trust-Region Guidance](https://arxiv.org/pdf/2605.24433)

### RL 后训练

- [ReinFlow: Fine-tuning Flow Matching Policy with Online RL](https://arxiv.org/abs/2505.22094)
- [Q-VGM: Q-Value-Gradient Matching for Flow-Matching VLA](https://arxiv.org/html/2606.08015v1)
- [FlowDPG: Deterministic Policy Gradient on Flow Matching Policies](https://arxiv.org/pdf/2606.22303)
- [RL for Flow-Matching Policies with Density Transport](https://arxiv.org/html/2606.08602v1)
- [Score-Based One-step MeanFlow Policy Optimization](https://arxiv.org/html/2605.23365v1)
- [Reinforcement Fine-Tuning of Flow-Matching Policies for VLA](https://arxiv.org/abs/2510.09976)

### 表示与几何

- [Frequency-Aware Flow Matching](https://arxiv.org/abs/2606.20135v1)
- [WarmPrior: Straightening Flow-Matching Policies with Temporal Priors](https://arxiv.org/pdf/2605.13959)
- [Compose Your Policies! Test-time Distribution-level Composition](https://arxiv.org/pdf/2510.01068)

### 人体运动生成

- [Motion Flow Matching for Human Motion Synthesis and Editing](https://arxiv.org/abs/2312.08895)
- [FlowMotion: Target-Predictive Conditional Flow Matching](https://arxiv.org/abs/2504.01338)
- [MotionFlux: Rectified Flow Matching + Preference Alignment](https://arxiv.org/pdf/2508.19527)
- [MotionHiFlow: Hierarchical Flow Matching Text-to-Motion](https://arxiv.org/pdf/2604.23264)
- [Riemannian Motion Generation](https://arxiv.org/pdf/2603.15016)
- [MotionGPT3: From Diffusion to Flow](https://arxiv.org/html/2603.26747v2)

### 综述

- [Towards a Unified Understanding of Robot Manipulation: A Comprehensive Survey](https://arxiv.org/pdf/2510.10903)
- [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/html/2605.00080v1)
