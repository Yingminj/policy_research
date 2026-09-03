# 离线评测是否引入阈值型精度指标（acc@τ）：调研、实测与改造方案

**范围。** Dyna-2 网页报告使用 MSE / L1 / acc@0.1 / acc@0.5 四个指标评测动作预测。本文回答三件事：
（1）这类指标在 policy / VLA 领域的谱系与真实用途；（2）它相对本仓库 `eval_policy/offline_chunk_eval.py`
现有口径（`mae` / `rmse` / `norm_mae` / 分 horizon / 分关节 / 双 null 基线 / 部署滤波消融）到底新增了什么信息；
（3）如果要加，怎么加、加哪一种、阈值取多少、有哪些坑——以及**哪些部分建议不加**。

编撰于 2026-09-03。

**核实方法。** 文献部分：全部通过 WebFetch 抓取 `arxiv.org/abs/<id>` 摘要页或正文 HTML 逐篇核对（标题、作者、v1 日期、
摘要原文），未使用任何来自模型记忆的引用；无法逐字核实的条目在 §11 明确标注。
**实测部分：本文所有数字均来自本仓库已有产物**，没有跑任何新的 GPU 任务：
`runs/**/*.json`（111 份 aggregate 报告）与 `runs/scripts_act_eval_test/test_img/test_s53_trace.npz`
（`--trace` 导出的 200 个 anchor 的原始 pred/gt/state）。复现脚本见 §5.1 与 §8.1。

---

## 0. 结论先行

| # | 结论 | 依据 |
|---|---|---|
| **C1** | **不要照搬 Dyna-2 的 acc@0.1 / acc@0.5 当主指标。** 在本仓库实测数据上，阈值型精度**分辨 policy 与 null 基线的能力明显弱于现有的 MAE 比值**：MAE 口径 null/policy = **2.66×**，而 acc@0.05 口径 policy 34.0% vs null 32.0%，只差 **2.0 个百分点**（剔除夹爪后只差 **0.7 pp**）。 | §6.2 实测 |
| **C2** | 阈值型精度**唯一**新增的信息是误差分布的**形状**（CDF），而不是水平。若被比较的对象形状相同，acc@τ 与 `norm_mae` 是单调等价的、零新增信息。 | §3.2 推导 |
| **C3** | **形状确实在变，而且变得有规律**：本仓库 39 组 `policy_raw` vs `policy_deployed` 配对中，部署滤波栈让尾部比 `R = rmse/mae` **24/24 全部上升**（中位 +0.146）；其中 21 组滤波器**降低了 MAE 却同时加重了尾部**。这是 MAE/RMSE 现有口径**没有报出来过**的事实。 | §5.2 实测 |
| **C4** | 因此正确的做法不是加两个固定阈值，而是**加一个归一化误差直方图**：一次评测之后 acc@任意 τ、中位数、p90、CDF 曲线全部可事后计算，不必为改 τ 重跑。阈值型指标不可从现有 `abs_sum`/`sq_sum` 事后还原，这是本改造的**唯一**结构性约束。 | §3.3 / §8.3 |
| **C5** | **先做零成本的两步**：①`summarise()` 加 `norm_rmse`（1 行），让 `R` 成为尺度无关的一等公民；②用已有的 `--trace` npz 跑 §8.1 的分析脚本看 CDF。这两步不动评测主流程、不花 GPU，且足以决定要不要做 C4。 | §8.1 / §8.2 |
| **C6** | 若最终要报 acc@τ：**必须把两个夹爪维度从主数字里剔除**。实测夹爪 `R = 4.2/4.3`（臂关节约 1.5），acc@0.1 高达 92–94%，把 16 维平均的表头数字**虚高了 6.2 个百分点**（49.4% → 剔除后 43.2%）。 | §6.3 实测 |

**一句话建议。** acc@τ 不是升级，是**换了一个更苛刻的问法**。它会让本项目现有的结论看起来更差（这恰恰是它的价值：
它揭示了"策略只是平均更接近，并没有更精确"），但它在**判别力**上不如你已有的 MAE-vs-null 比值。
真正值得加的是**分布本身**（直方图 / 中位数 / 分位数），acc@τ 只是它的一个读数。
实测上信息量最大的单个新数字不是 acc@0.1，而是 **`p90`**：policy 0.455 vs null 1.630（3.6×），
而中位数只差 1.55× —— 一句话说清了"策略学到的是把大错压下去，不是把小错做得更小"。

---

## 1. Dyna-2 的四个指标：原文口径

来源：`https://www.dyna.co/dyna-2`（WebFetch 核实，2026-09-03）。该工作**没有 arXiv 论文**，只有网页技术报告，
因此下列口径的精确程度以网页原文为限。

| 指标 | 网页定义 | 备注 |
|---|---|---|
| MSE | 预测 action chunk 与 GT 的平方差，在所有动作维度与预测 horizon 上取平均 | 与本仓库 `rmse²` 同族 |
| L1 | 同上，取绝对差 | 与本仓库 `mae` 同族 |
| acc@0.5 | "the fraction of action dimensions falling within τ of the ground truth"，τ 以**归一化动作单位**计；0.5 档被描述为衡量 "general motion intent"，用于 human-to-robot 迁移研究 | **逐维度**，不是逐时间步 |
| acc@0.1 | 同上定义，0.1 档被描述为 "movement precision and in-domain scaling trends" 的指示 | |

**评测场景**：1,000→1,000,000 小时人类视频的嵌套数据集；机器人侧 39 个操作任务（叠布、打结、装箱、装配等），
含内部 benchmark 与外部数据集；后训练评测覆盖 14 个任务、多本体，每任务最多 10 小时演示。
网页原文称多指标并用是为了 "prevents scaling artifacts, ensuring findings aren't dependent on any single measurement's characteristics"。

**三个网页没有说清、但直接决定数字大小的事**（本文后续均按最坏情况处理）：

1. **"normalized action units" 是哪一种归一化**：z-score（除以训练集 σ）还是 min-max 到 [-1,1]，还是分位数裁剪。
   两者相差数倍，τ=0.1 在不同归一化下不是同一个量。**结论：Dyna-2 的绝对数值与本仓库的数值不可直接比较。**
2. **是否含夹爪/离散维**：近二值维度在阈值指标下会系统性拉高平均（§6.3 实测证实）。
3. **padding / episode 末尾如何处理**。

---

## 2. 谱系：阈值型精度在 policy / VLA 及邻域中的位置

结论：**"MSE + L1" 是普遍口径；"阈值型精度"不是 Dyna 的发明，但也远没有标准化**，
它的直系祖先是自回归 VLA 的 token 容差准确率，旁系近亲是自动驾驶轨迹预测的 Miss Rate。

| 工作 | 指标 | 定义 | 与 acc@τ 的关系 |
|---|---|---|---|
| **OpenVLA**（arXiv:2406.09246，2024-06-13） | 动作离散化 | "we discretize each dimension of the robot actions separately into one of 256 bins"；"we set the bin width to uniformly divide the interval between the 1st and 99th quantile of the actions in the training data"（正文 HTML 核实） | 提供了阈值指标的**归一化基准**：分位数裁剪区间，而非 min-max |
| **VQ-VLA**（arXiv:2507.01016，ICCV 2025） | Top-1 acc / k-Bin 容差 acc | k-Bin Acc = (1/N)Σ𝟙(\|aᵢ−âᵢ\| ≤ k)，k=0 为 Top-1，k=5 为 5-bin 容差 | **acc@τ 的离散版本**。5 bins / 256 bins ≈ q1–q99 区间的 **1.95%** |
| **Dyna-2**（网页，2026） | acc@0.1 / acc@0.5 | 见 §1 | 连续版本 |
| **Argoverse 2**（arXiv:2301.00493，2023-01-02） | minADE / minFDE / **Miss Rate** | MR = minFDE 超过 2 m 的场景比例 | **跨领域先例**：ADE/FDE 是矩，MR 是 CDF 上的一点。轨迹预测领域二十年来一直是"矩 + 阈值"并报 |
| Waymo Open Motion | 分横向/纵向、**随速度自适应**的阈值 | — | 说明固定 τ 是已知缺陷，业界解法是让 τ 随运动量缩放 |
| **CI-MSE**（arXiv:2606.29898，2026-06-29） | Critical Interval MSE | 只在任务关键区间上算误差，配 temporal ensembling / RTC 对齐 + DTW；Spearman ρ 从 raw MSE 的 **−0.61** 提升到 **−0.87**（27 个 checkpoint，π₀.₅ / X-VLA / GR00T N1.7） | **竞争方案**：不改指标形状，改**在哪里算** |
| **VLA-FAIL**（arXiv:2606.21386，2026-06-19） | **ACC = Action Chunk Consistency** | 利用 receding-horizon 的 chunk 重叠，检测相邻 chunk 不一致 | ⚠️ **命名冲突**：本领域 "ACC" 已被占用为 Action Chunk Consistency。本仓库若引入 accuracy，**不要缩写成 ACC** |
| **Eval-Actions**（arXiv:2601.18723，2026-01-26） | EG / RG / CoT 专家分级 | 13K+ 真机 episode，AutoEval SRCC 0.81/0.84 | 说明领域共识：二值成功率与单一标量误差都不够，需要过程级诊断 |
| **How VLAs Fail Differently**（arXiv:2605.28726，2026-05-27） | 方向反转率 / jerk / 速度违规 | 方向反转率 AUROC 0.79–0.93 为通用失败预测子；速度违规 AUROC 0.41–0.69 基本无效 | 说明**动作层面的形状类指标**确有判别力，但要选对 |

**一个可以做的换算**（本文推导，非引用）：OpenVLA 的 5-bin 容差 = q1–q99 区间的 1.95%。
若动作近似正态，q1–q99 ≈ 4.65σ，则 5-bin 容差 ≈ **0.091σ** —— 与 Dyna-2 的 acc@0.1 **几乎重合**。
这提示 Dyna 的 τ=0.1 大概率是 z-score 口径，且 0.1σ 这个量级在两条独立的技术路线上被独立选中，
不是随手取的数。**但见 §6，这个量级对本项目的数据未必合适。**

---

## 3. 数学定位：矩 vs CDF —— acc@τ 到底新增什么

### 3.1 现有口径都是矩

设归一化误差 $e = |pred - gt| / \sigma_j$，本仓库现在报的是：

- `norm_mae` = $E[e]$ —— 一阶矩
- `rmse` / `norm_rmse` = $\sqrt{E[e^2]}$ —— 二阶矩
- `mae_per_horizon` / `mae_per_joint` —— 一阶矩在两个轴上的切片

`acc@τ` = $P(e < \tau)$ —— 是 CDF 在一点的取值。这是**不同类**的量。

### 3.2 关键推论：形状不变时 acc@τ 零新增信息

设 $e = m \cdot z$，其中 $m = E[e]$ 是水平，$z$ 是均值为 1 的**形状**分布。则

$$\text{acc@}\tau = P(z < \tau/m) = F_z(\tau/m)$$

$F_z$ 单调递增，故只要形状 $F_z$ 在被比较的对象之间相同，**acc@τ 就是 `norm_mae` 的一个单调变换**，
排序完全一致，除了改变刻度以外不提供任何新信息。

**这给出了一个严格的判据：acc@τ 值不值得加，取决于形状在你要比较的对象之间是否真的不同。**
而形状可以用一个现成的、零成本的量来度量：

$$R = \frac{\text{rmse}}{\text{mae}} \qquad (\text{正态 } R=1.2533,\ \text{Laplace } R=1.4142,\ R \uparrow \Rightarrow \text{尾更重})$$

§5 用本仓库 111 份已有报告直接测了这件事。

### 3.3 结构性约束：阈值不可事后还原

`ChunkErrorAccumulator` 只保存 `abs_sum` / `sq_sum` / `count`（`offline_chunk_eval.py:268-291`）。
`norm_mae` 之所以能在 `summarise()` 里**事后**除以 `a_std`（`:668`），是因为 MAE 对尺度是线性的。
**阈值是非线性的**：$P(e<\tau)$ 无法从 $\sum|e|$ 与 $\sum e^2$ 还原。因此：

> 任何阈值型指标都必须在 `update()` 内部、拿到 `a_std` 之后计算；
> 且**一旦 τ 写死，改 τ 就要重跑整个评测**（六个数据集 × ~10⁴ anchor）。

这就是 §0-C4 主张用直方图而非固定阈值的全部理由：直方图同样在 `update()` 里算，但一次评测覆盖所有 τ。

---

## 4. 与本仓库现有 harness 的逐项对照

| 能力 | Dyna-2 报告 | `offline_chunk_eval.py` 现状 | 结论 |
|---|---|---|---|
| L1 / MAE | ✅ | ✅ `mae` | 平 |
| MSE | ✅ | ✅ `rmse`（单调等价） | 平 |
| 尺度归一 | ✅（口径未说明） | ✅ `norm_mae`，除以 checkpoint 自带的训练集 σ（`action_stats:396`） | **本仓库更明确** |
| 分 horizon | ❌ 单标量 | ✅ `mae_per_horizon` / `mae_at_horizon` | **本仓库更强** |
| 分关节 | ❌ | ✅ `mae_per_joint` / `norm_mae_per_joint` | **本仓库更强** |
| null 基线 | ❌ | ✅ `hold_state` / `train_mean` | **本仓库决定性更强**（见下） |
| 训练集去污染 | ❌ | ✅ action 指纹剔除 | **本仓库更强** |
| 部署忠实度 | ❌ | ✅ `policy_deployed` + 逐级滤波消融 | **本仓库更强** |
| 采样噪声下界 | ❌ | ✅ `--seed-repeat` | **本仓库更强** |
| **误差分布形状** | ✅ acc@0.1/0.5 | ❌ **只有矩** | **唯一缺口** |
| 统计不确定度 | ❌ | ❌ | 两边都缺（§8.5） |

**关于 null 基线**：这是两套口径最大的差距。一个 0.03 的 MAE 本身没有意义；
本仓库 `null/policy = 4.2×` 这种读数才是"策略确实学到了东西"的判据。Dyna-2 全套四个指标都没有 null，
**照搬它的指标集会丢掉本项目已经解决的问题**。任何新增指标都必须同时对 `hold_state` / `train_mean` 计算。

---

## 5. 零成本证据（一）：现有 111 份报告里已经藏着形状信息

### 5.1 复现

```bash
cd /home/kewei/YING/paper/eval_policy
python3 - <<'EOF'
import json, glob, statistics as st
pairs = []
for f in sorted(glob.glob('runs/**/*.json', recursive=True)):
    try: a = json.load(open(f)).get('aggregate')
    except Exception: continue
    if not a or 'policy_raw' not in a or 'policy_deployed' not in a: continue
    r, p = a['policy_raw'], a['policy_deployed']
    pairs.append((f, r['mae'], p['mae'], r['rmse']/r['mae'], p['rmse']/p['mae']))
diff = [x for x in pairs if abs(x[1]-x[2]) > 1e-9]
print(f"配对 {len(pairs)}，其中滤波器实际生效 {len(diff)}")
print(f"  R 上升: {sum(x[4] > x[3] for x in diff)}/{len(diff)}")
print(f"  MAE 下降且 R 仍上升: {sum(x[2] < x[1] and x[4] > x[3] for x in diff)}")
print(f"  中位 ΔR = {st.median([x[4]-x[3] for x in diff]):+.3f}")
EOF
```

### 5.2 结果

```
配对 39，其中滤波器实际生效 24
  R 上升: 24/24
  MAE 下降且 R 仍上升: 21
  中位 ΔR = +0.146
```

各累加器的 `R = rmse/mae` 中位数（n 为样本数）：

| 累加器 | n | 中位 R | 范围 |
|---|---|---|---|
| `train_mean` | 63 | **1.236** | 1.205 – 1.472 |
| `policy`（一代 harness） | 24 | 1.609 | 1.466 – 2.180 |
| `policy_raw` | 39 | 1.870 | 1.544 – 2.316 |
| `policy_deployed` | 39 | **1.968** | 1.754 – 2.319 |
| `hold_state` | 63 | 2.083 | 1.828 – 2.493 |

差距最大的几组：

| 报告 | MAE | R |
|---|---|---|
| `intrain_control_deployed.json` | 0.01175 → 0.02219 | 1.811 → **2.181** |
| `prev_diffusion_h8.json` | 0.04738 → **0.03351** | 1.638 → **1.968** |
| `new_obs2_h8.json` | 0.04253 → **0.03084** | 1.649 → **1.960** |
| `vqbet_100k_h8.json` | 0.05501 → **0.03572** | 1.544 → **1.837** |

### 5.3 读法与必须声明的 caveat

**读法。** 部署滤波栈（rollback 移除 / gripper loop 移除 / 二项平滑 / excursion 线性化 / Hermite 桥 / gripper clip）
在 21 组里**降低了平均误差，同时加重了误差的尾部**。也就是说滤波器把"到处都差一点"换成了
"大部分更准、少数地方错得更狠"。这与 `experiment_report` 中"Hermite 桥只覆盖约 65% 的指令位移"的实测完全自洽：
桥在大部分低速片段几乎无害，在大位移片段造成集中的大误差。

**这是现有 MAE/RMSE 口径从未报出来的事实**，而它正好落在本项目最关心的轴 A（部署侧执行层扭曲动作）上。

**Caveat（必须写进任何引用这组数字的地方）。** 这里的 `R` 是在 aggregate 上算的，
分子分母都混合了 16 个尺度不同的关节和 100 个 horizon 步。
**尺度混合本身就会把 R 抬到 1.2533 以上，即使每个关节各自都是正态的。**
因此：

- ❌ 不要用 R 的**绝对值**论证"重尾"；
- ✅ 可以用**同一份报告内 raw vs deployed 的配对比较**（同关节、同数据、同混合），这是合法的相对陈述。

要让 R 的绝对值有意义，需要先做 §8.2 的 `norm_rmse`（1 行）。

---

## 6. 零成本证据（二）：在真实 trace 上直接算 acc@τ

`--trace` 已经会把前 N 个 anchor 的原始 `pred/gt/state/valid` 存成 npz。
本仓库存有 6 份，因此**不写一行 harness 代码、不花一分钟 GPU，就能得到真实的 acc@τ**。

数据：`runs/scripts_act_eval_test/test_img/test_s53_trace.npz`（200 anchor × 100 步 × 16 关节，ACT baseline，
对应报告 `runs/scripts_act_eval_test/eval53_act_baseline.json`）。
每关节 σ 从报告反解：`σ_j = mae_per_joint[j] / norm_mae_per_joint[j]`（已验证 trace 的 `joint_names` 与报告一致）。
该 trace 的 `norm_mae = 0.1834`（报告 aggregate 为 0.2109，差异来自 trace 只覆盖首个数据集的前 200 个 anchor）。

### 6.1 完整误差分布

| 分位 | p10 | p25 | **p50** | p75 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|---|
| 归一化 \|误差\| | 0.0043 | 0.0247 | **0.1020** | 0.2439 | 0.4554 | 0.6362 | 1.1734 | 2.0814 |

**均值 0.1834 落在第 67 百分位。** 也就是说 `norm_mae` 报的那个数，比一半以上的实际误差都大 —— 
**典型精度比 MAE 显示的好 44%**（中位 0.102 vs 均值 0.183），代价全部由上面三分之一承担。
这是矩类指标的固有性质，不是缺陷，但它意味着"MAE 降低 5%"可能来自完全不同的两件事。

### 6.2 acc@τ 的判别力：**弱于现有的 MAE 比值**

同一 trace，policy 与 `hold_state` null 的对照（逐维度口径，与 Dyna 定义一致）：

| 口径 | policy | null | 判别力 |
|---|---|---|---|
| **`norm_mae`** | **0.1834** | **0.4884** | **null/policy = 2.66×** |
| acc@0.05 | 34.0% | 32.0% | **+2.0 pp**，错误率比 1.03× |
| acc@0.1 | 49.4% | 42.6% | +6.8 pp，错误率比 1.14× |
| acc@0.2 | 69.7% | 53.9% | +15.8 pp，错误率比 1.52× |
| acc@0.3 | 80.3% | 62.8% | +17.5 pp，错误率比 1.89× |
| acc@0.5 | 91.6% | 71.9% | +19.7 pp，错误率比 **3.34×** |
| acc@1.0 | 98.4% | 82.3% | +16.1 pp，错误率比 11.27× |

**剔除夹爪后（14 个臂关节，即 §6.3 要求的主数字口径）判别力更差：**

| 口径 | policy | null | 判别力 |
|---|---|---|---|
| acc@0.05 | 25.7% | 25.0% | **+0.7 pp** |
| acc@0.1 | 43.2% | 37.1% | +6.1 pp |
| acc@0.2 | 66.2% | 49.8% | +16.4 pp |
| acc@0.5 | 90.9% | 70.2% | +20.7 pp |
| `p50` / `p90` | 0.102 / 0.455 | 0.158 / **1.630** | 中位只差 1.55×，**p90 差 3.6×** |

**这是本文最重要的一组数字，含义有四层：**

1. **在 Dyna 声称衡量"精度"的紧阈值上（0.05–0.1σ），本项目训练出来的 ACT 与"什么都不做"几乎无法区分。**
   acc@0.05：34.0% vs 32.0%。策略的优势**完全不在精细精度上**，而在于"大错更少"。
2. **因此 acc@0.1 作为主指标会削弱、而不是增强本项目已有的论证。** 现有 `null/policy = 2.66×` 是一个干净有力的读数；
   换成 acc@0.1 的 "+6.8pp" 既更弱也更难解释，而且在 §8.5 的统计不确定度下很可能不显著。
3. **但这个"坏消息"本身是真信息。** 它是对轴 D（执行窗口选错、null 胜过策略）和轴 E（离线口径与真机脱节）的
   独立佐证：一个在紧阈值上打不过 null 的策略，在真机上"看起来动了但不到位"是完全可以预期的。
   **MAE 把这件事藏起来了，acc@τ 把它挖出来了。** 这是加它的理由，但不是把它当主指标的理由。
4. **策略真正赢在哪里，分位数说得比 acc@τ 更清楚**：中位误差 0.102 vs null 0.158（只好 1.55×），
   而 p90 是 0.455 vs **1.630**（好 3.6×）。**策略学到的东西几乎全部是"把大错压下去"，不是"把小错做得更小"。**
   如果只加一个新数字，`p90`（或 `p50`/`p90` 一对）比 acc@0.1 信息量更大、也更好解释 —— 
   而它们与 acc@τ 出自同一个直方图，不需要额外成本（§8.3）。

### 6.3 夹爪陷阱（实测确证）

逐关节 acc@0.1 与形状（policy，全 horizon）：

| 关节 | acc@0.1 | norm_mae | R |
|---|---|---|---|
| Joint1_L … Joint7_R（14 个臂关节） | 28.7% – 55.9% | 0.129 – 0.298 | **1.38 – 1.72** |
| `gripper_L` | **92.3%** | 0.061 | **4.22** |
| `gripper_R` | **94.2%** | 0.042 | **4.34** |

- 16 维平均 acc@0.1 = **49.4%**；剔除两个夹爪后 = **43.2%**。**夹爪把表头数字虚高了 6.2 pp。**
- 夹爪的 `R` 是臂关节的近 3 倍 —— **§5 观察到的"重尾"有相当一部分就是夹爪**。
  夹爪训练态 95% 恰好是 0.0/1.0（见 harness 文件头注释），σ 很小，绝大多数时刻误差≈0（拉高 accuracy），
  切换瞬间误差巨大（拉高 R）。它在阈值指标下本质上是**分类准确率**，不该和连续关节平均在一起。

**硬性要求：任何 acc@τ 的主数字必须只含 14 个臂关节；夹爪单列，并且应当换成开合状态的分类口径。**

### 6.4 逐维度 vs 全关节（conjunctive）

| τ | 逐维度 acc（Dyna 口径） | 全 16 关节同时满足 |
|---|---|---|
| 0.1 | 49.4% | **4.2%** |
| 0.2 | 69.7% | 13.7% |
| 0.5 | 91.6% | **50.4%** |

对一条 14 自由度双臂，**位姿是否到位是"与"关系，不是平均关系**。逐维度口径说"91.6% 的维度在 0.5σ 内"，
全关节口径说"只有一半的时间步整条位姿在 0.5σ 内"。后者更贴近任务，但 4.2% 这种数字作为主指标过于苛刻、
分辨率也差。**建议：全关节口径只在 τ=0.5 一档作为辅助读数报出，不进主表。**

（注：若各关节独立，0.916¹⁶ = 24%，实测 50.4% ⇒ 关节间误差高度相关，符合"整条 chunk 被桥替换"的机理。）

### 6.5 一个被证伪的顾虑：时间对齐

紧阈值指标对 GT 时间对齐远比 MAE 敏感 —— 这是先验上最该担心的坑（CI-MSE 正是为此上了 DTW）。
**实测结果是：在本项目数据上这个顾虑不成立。**

| GT 平移 | acc@0.05 | acc@0.1 | acc@0.2 | acc@0.5 | norm_mae |
|---|---|---|---|---|---|
| −3 tick | 25.0% | 42.3% | 65.0% | 90.3% | 0.2089 |
| **0** | **25.3%** | **42.6%** | **65.8%** | **90.8%** | **0.2038** |
| +3 tick | 24.9% | 41.8% | 65.1% | 90.6% | 0.2070 |

（14 个臂关节，去掉首尾 3 步的公共窗口。）±3 tick 的错位只让 acc@0.1 变化 0.8 pp。

原因可测：**每 tick 的动作增量中位数只有 0.0027σ**（均值 0.0110σ，p90 0.0315σ）——
机械臂大部分时间几乎不动，运动是突发的。所以 `--latency-steps` 的估计误差不会污染 acc@τ。

⚠️ **但这个结论有边界**：一旦你只在高速/关键片段上算指标（CI-MSE 那条路），
per-tick 位移就不再是 0.0027σ 而是接近 p90 的 0.0315σ，对齐敏感度要重新测。

---

## 7. 落地前必须先定的八个设计决策

| # | 决策 | 选项 | 本文建议 | 理由 |
|---|---|---|---|---|
| D1 | **归一化基准** | (a) 训练集 σ（现 `norm_mae` 基准）<br>(b) q1–q99 区间（OpenVLA 口径）<br>(c) 物理单位 rad/m | **(a)**，并在 JSON 里记下基准来源 | 与现有 `norm_mae` 同基准，才能和历史报告对齐；(b) 需要额外统计量且 checkpoint 里没有 |
| D2 | **维度聚合** | 逐维度 / 全关节与 | **逐维度为主，全关节 @0.5 为辅** | §6.4 |
| D3 | **夹爪** | 含 / 剔除 / 单列 | **主数字剔除，单列，另报开合分类准确率** | §6.3 |
| D4 | **固定阈值 vs 直方图** | 写死 τ / 存直方图 | **直方图** | §3.3：改 τ 要重跑六个数据集 |
| D5 | **τ 取值** | 照搬 0.1/0.5 / 自校准 | **报 0.05 / 0.1 / 0.2 / 0.5 四档**；0.05 与 0.5 分别用于确认"无精度优势"和"与文献可比" | §6.2：0.05 几乎无判别力（这本身是结论），0.5 接近饱和 |
| D5b | **不要取的 τ** | 0.01 / 0.03 | **不报**。p25 = 0.0247，这两档主要在测数值噪声与 σ 反解精度 | §6.1 |
| D6 | **padding mask** | — | **必须显式排除**。现有 `err` 在 `update()` 里已经乘过 `valid`（`:283`），padding 位置的误差**恰好是 0**；直方图若照抄这个 `err`，**每一个 padding 槽都会被记成"完美预测"**，acc@τ 会被系统性虚高 | ⚠️ 这是本改造最容易踩的 bug |
| D7 | **哪些累加器要算** | 只算 policy / 全部 | **全部**，包括 `hold_state`、`train_mean`、5 个滤波消融、`seed_*` | §4：没有 null 的 accuracy 没有意义 |
| D8 | **缩写** | ACC / acc@τ | **不要用 ACC** | §2：ACC 已被 VLA-FAIL 占用为 Action Chunk Consistency |

---

## 8. 分阶段改造方案

按"先零成本、后改代码"排序。**P0 与 P1 建议立刻做；P2 在 P0/P1 的结果支持时再做；P3 可选；P4 明确不做。**

### 8.1 P0（零成本，今天就能跑）：trace 分布分析脚本

不动 harness。放到 `runs/<日期>_metric_cdf/cdf_probe.py`：

```python
#!/usr/bin/env python3
"""从 --trace 导出的 npz 直接读误差 CDF，用来决定要不要给 harness 加阈值指标。

用法: python cdf_probe.py <trace.npz> <对应的 aggregate.json>
每关节 sigma 从报告反解: sigma_j = mae_per_joint[j] / norm_mae_per_joint[j]
"""
import json, sys
import numpy as np

trace, report = sys.argv[1], sys.argv[2]
a = json.load(open(report))["aggregate"]
a = a.get("policy") or a["policy_raw"]
names = list(a["mae_per_joint"])
sig = np.array([a["mae_per_joint"][n] / max(a["norm_mae_per_joint"][n], 1e-12) for n in names])

d = np.load(trace)
assert list(d["joint_names"]) == names, "trace 与报告的关节顺序不一致"
gt, val = d["gt"], d["valid"]
arm = [i for i, n in enumerate(names) if not n.startswith("gripper")]   # D3
TAUS = (0.05, 0.1, 0.2, 0.5)

def row(label, pred):
    ne = np.abs(pred - gt) / sig
    m = val[..., None] & np.ones_like(ne, bool)                        # D6: 显式 mask
    e = ne[m]
    ea = ne[:, :, arm][val[..., None] & np.ones_like(ne[:, :, arm], bool)]
    print(f"{label:<12} norm_mae={e.mean():.4f} R={np.sqrt((e**2).mean())/e.mean():.3f} "
          f"p50={np.median(e):.4f} p90={np.percentile(e, 90):.4f} | "
          + " ".join(f"@{t}={100*(ea < t).mean():.1f}%" for t in TAUS) + "  (臂关节)")

row("policy", d["pred"])
row("hold_state", np.repeat(d["state"][:, None, :], gt.shape[1], axis=1))
```

自检：对 `test_s53_trace.npz` + `eval53_act_baseline.json`，应逐字复现（已验证）：

```
policy       norm_mae=0.1834 R=1.680 p50=0.1020 p90=0.4554 | @0.05=25.7% @0.1=43.2% @0.2=66.2% @0.5=90.9%  (臂关节)
hold_state   norm_mae=0.4884 R=1.786 p50=0.1582 p90=1.6295 | @0.05=25.0% @0.1=37.1% @0.2=49.8% @0.5=70.2%  (臂关节)
```

**产出**：在你关心的每个 checkpoint 上跑一遍，如果 policy 与 null 的 acc@τ 差距在所有 τ 上都不如 MAE 比值
（§6.2 已在 ACT baseline 上验证是这样），那么 P2 可以直接不做，改做 §8.5。

### 8.2 P1（1 行）：`norm_rmse`，让 R 变成合法的一等指标

`offline_chunk_eval.py:665` `summarise()`：

```python
 def summarise(acc, a_std, horizon, names):
     mae_hj = acc.mae_per_horizon_joint()
     norm_hj = mae_hj / a_std.double().clamp_min(1e-8)
+    norm_sq_hj = acc._safe(acc.sq_sum) / a_std.double().clamp_min(1e-8).pow(2)
+    norm_rmse = norm_sq_hj.mean().sqrt().item()
     cuts = [c for c in (1, 10, 25, 50, horizon) if c <= horizon]
     return {
         "mae": acc.mae(),
         "rmse": acc.rmse(),
         "norm_mae": norm_hj.mean().item(),
+        "norm_rmse": norm_rmse,
+        "tail_ratio": norm_rmse / max(norm_hj.mean().item(), 1e-12),
         ...
```

`tail_ratio` 是**逐关节归一后**的 R，去掉了 §5.3 那个尺度混合的 caveat，
可以直接和 1.2533（正态）/ 1.4142（Laplace）比。成本 3 行、零运行时代价、不改变任何既有字段。

**这是全案性价比最高的一步**：它把 §5 那条"部署滤波器加重尾部"的观察从"只能配对比较"升级为"可以绝对陈述"。

顺带在 `:903` 的打印里加一列：

```python
-        print(f"  {k:<18} mae={v['mae']:.5f}  rmse={v['rmse']:.5f}  "
-              f"norm_mae={v['norm_mae']:.5f}  {delta}")
+        print(f"  {k:<18} mae={v['mae']:.5f}  rmse={v['rmse']:.5f}  "
+              f"norm_mae={v['norm_mae']:.5f}  tail={v['tail_ratio']:.2f}  {delta}")
```

### 8.3 P2（约 25 行）：归一化误差直方图

**只有在 P0 显示 acc@τ 对你要比的对象确有增量时才做。** 做就做直方图，不要做固定阈值（D4）。

`ChunkErrorAccumulator`（`:268`）：

```python
class ChunkErrorAccumulator:
    #: 归一化 |误差| 的直方图边界。对数网格保证跨尺度可读；把要报的 τ 显式并进去，
    #: 使 acc@τ 落在 bin 边界上、无量化误差（D5）。
    _LOG = torch.logspace(-3, 1, 129, dtype=torch.float64)
    EDGES = torch.tensor(sorted(set(_LOG.tolist()) | {0.05, 0.1, 0.2, 0.5, 1.0}), dtype=torch.float64)

    def __init__(self, horizon, n_joints, device="cpu", a_std=None):
        self.abs_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.sq_sum = torch.zeros(horizon, n_joints, dtype=torch.float64, device=device)
        self.count = torch.zeros(horizon, 1, dtype=torch.float64, device=device)
        # a_std 为 None 时不建直方图，--selftest 与旧调用点保持可用
        self.std = None if a_std is None else a_std.double().clamp_min(1e-8).to(device)
        self.hist = None if a_std is None else torch.zeros(
            horizon, n_joints, len(self.EDGES) + 1, dtype=torch.float64, device=device)

    def update(self, pred, gt, valid):
        err = (pred.double() - gt.double()) * valid.unsqueeze(-1).double()
        self.abs_sum += err.abs().sum(dim=0)
        self.sq_sum += err.pow(2).sum(dim=0)
        self.count += valid.double().sum(dim=0).unsqueeze(-1)
        if self.hist is None:
            return
        # D6: err 已经被 valid 乘过, padding 位置恰好是 0.0 —— 若不显式剔除,
        # 每个 padding 槽都会落进第一个 bin, 被算成"完美预测"。
        m = valid.unsqueeze(-1).expand_as(err)                       # (B, H, J)
        b = torch.bucketize(err.abs() / self.std, self.EDGES)        # (B, H, J)
        H, J, B = self.hist.shape
        flat = (torch.arange(H, device=b.device).view(1, H, 1) * J * B
                + torch.arange(J, device=b.device).view(1, 1, J) * B + b)
        self.hist.view(-1).scatter_add_(
            0, flat[m], torch.ones(int(m.sum()), dtype=torch.float64, device=b.device))

    def acc_at(self, tau, upto=None, joints=None):
        """P(|e|/sigma < tau)。tau 必须是 EDGES 中的值, 否则向下取到最近的边界。"""
        h = slice(0, upto) if upto else slice(None)
        j = slice(None) if joints is None else joints
        c = self.hist[h, j].sum(dim=(0, 1))
        # bucketize 的 bin i 覆盖 (EDGES[i-1], EDGES[i]], 所以 tau=EDGES[k] 对应 c[:k+1]。
        # 用 c[:k] 会漏掉一个 bin —— 在 tau=1.0 上实测差 3.4pp。
        k = int(torch.searchsorted(self.EDGES, torch.tensor(tau, dtype=torch.float64)))
        assert abs(float(self.EDGES[k]) - tau) < 1e-12, f"tau={tau} 不在 EDGES 上"
        return (c[:k + 1].sum() / c.sum().clamp_min(1.0)).item()

    def quantile(self, q, upto=None, joints=None):
        """归一化 |误差| 的第 q 分位 (0-1), 用 bin 上界作保守估计。"""
        h = slice(0, upto) if upto else slice(None)
        j = slice(None) if joints is None else joints
        c = self.hist[h, j].sum(dim=(0, 1))
        cum = c.cumsum(0) / c.sum().clamp_min(1.0)
        k = int(torch.searchsorted(cum, torch.tensor(q, dtype=torch.float64)))
        return float(self.EDGES[min(k, len(self.EDGES) - 1)])
```

两个构造点（`:505` 与 `:518`）传入 `a_std`；`a_std` 在 `:474` 已经取到，早于两处构造，无需调序。
`summarise()` 增加：

```python
    arm = [i for i, n in enumerate(names) if not n.lower().startswith("gripper")]  # D3
    out["acc"] = {str(t): acc.acc_at(t, joints=arm) for t in (0.05, 0.1, 0.2, 0.5)}
    out["acc_at_horizon"] = {str(c): {str(t): acc.acc_at(t, upto=c, joints=arm)
                                      for t in (0.1, 0.5)} for c in cuts}
    out["acc_gripper"] = {str(t): acc.acc_at(t, joints=[i for i in range(len(names))
                                                        if i not in arm]) for t in (0.1, 0.5)}
    out["norm_quantiles"] = {f"p{int(q*100)}": acc.quantile(q, joints=arm)
                             for q in (0.5, 0.75, 0.9, 0.99)}
```

**内存**：`hist` 是 `(horizon, n_joints, |EDGES|+1)` float64 = 100 × 16 × ~134 ≈ 214k 元素 ≈ 1.7 MB/累加器。
典型一次跑 ~10 个累加器 × (per-dataset + overall) ≈ 34 MB。相对现在的显存/内存占用可忽略。
**运行时**：每 batch 一次 `bucketize` + 一次 `scatter_add_`，相对一次策略前向可忽略。

**为什么直方图而不是 4 个计数器**：计数器是 4 行、更省事，但 τ 一旦要改就得重跑六个数据集 × 10⁴ anchor；
直方图多 20 行，换来 acc@任意 τ、中位数、p90、CDF 曲线全部事后可算。§6 的全部分析都只需要这一个结构。

### 8.4 P2 的自检（必须写，跟着 `--selftest` 一起跑）

`_selftest()`（`:685`）追加：

```python
    # 直方图: padding 不得被记成完美预测 (D6)
    std = torch.ones(2)
    a = ChunkErrorAccumulator(horizon=3, n_joints=2, a_std=std)
    pred = torch.zeros(1, 3, 2)
    gt = torch.full((1, 3, 2), 5.0)          # 每个有效槽的归一化误差都是 5.0
    valid = torch.tensor([[True, True, False]])
    a.update(pred, gt, valid)
    assert a.hist.sum() == 4, a.hist.sum()               # 2 步 x 2 关节, padding 那步不计
    assert a.acc_at(0.1) == 0.0, a.acc_at(0.1)           # 误差 5.0 远大于 0.1
    assert a.acc_at(0.1, upto=2) == 0.0
    # 完美预测应当 100%
    b = ChunkErrorAccumulator(horizon=1, n_joints=1, a_std=torch.ones(1))
    b.update(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.tensor([[True]]))
    assert b.acc_at(0.05) == 1.0
    # 分位数单调
    assert a.quantile(0.5) <= a.quantile(0.9)
    print("selftest OK (histogram)")
```

**回归检查**：加完 P1/P2 后，对任意一个已归档的报告重跑同样的命令，
`mae` / `rmse` / `norm_mae` / `mae_per_joint` **必须逐位不变**。这几个字段的计算路径没有被触碰，
任何变动都说明改错了。

### 8.5 P3（可选，但可能比 acc@τ 更值钱）：按 episode 自助法置信区间

本仓库现在比较的 checkpoint 之间差距常在 3–5%（例：`eef_200k` 0.03323 vs `acteef_533_200k` 0.03443），
而**所有报告都没有任何不确定度**。同时 anchor 之间高度相关（`--stride 20`、chunk 长 100，重叠率 80%），
所以朴素的 $\sqrt{p(1-p)/n}$ 会把置信区间低估一个数量级。

最小实现：`update()` 额外按 `episode_index` 累一份 per-episode 的 `abs_sum`/`count`，
`summarise()` 里对 episode 做 1000 次有放回重采样，报 `mae` 与 `acc@τ` 的 2.5%/97.5% 分位。
约 20 行。**如果 P0 的结论是"acc@τ 判别力不如 MAE"，那么把预算花在这里的回报更高**：
它直接回答"这两个 checkpoint 的差距是不是噪声"，而这是本项目当前所有对比表都答不了的问题。

### 8.6 P4：明确**不做**的三件事

| 不做 | 原因 | 什么条件下再考虑 |
|---|---|---|
| **CI-MSE 式关键区间**（arXiv:2606.29898） | 它需要 (a) VLM 标注关键区间，(b) **N 个 checkpoint 配对的真机成功率**来验证相关性。本仓库有失败**演示**标注（`batch_success_361_fail_72`），但那是演示的成败，不是策略 rollout 的成败，不能用来算 Spearman | 攒够 ≥10 个 checkpoint 的真机成功率读数之后。届时这条路的预期收益（ρ −0.61 → −0.87）远高于换指标 |
| **per-anchor 灾难性异常检测** | 先验上最诱人的假设是"平时很好、偶尔灾难"，**实测被证伪**：per-anchor 平均归一化误差 中位 0.165 / p99 0.416 / max 0.485，只有 3× 的跨度，没有灾难性 anchor。重尾在 anchor **内部**（特定步、特定关节），不在 anchor 之间 | trace 只有 200 个非随机 anchor（首个数据集的前 200 个）。P2 上线后用全量数据重新验一次；若结论反转再做 |
| **把全关节 conjunctive 口径进主表** | τ=0.1 只有 4.2%，分辨率差 | 只在 τ=0.5 作辅助读数 |

---

## 9. 建议的最终报告表头

P1 + P2 之后，`aggregate` 每个累加器建议长这样（新增字段以 **粗体** 标出）：

```
                    mae      rmse     norm_mae  **norm_rmse** **tail**  **acc@0.1** **acc@0.5** **p50** **p90**
policy_raw        0.05112  0.09132   0.1575      0.2646      1.68        43.2%       90.8%    0.102   0.455
policy_deployed   0.05088  0.09535   0.1577      0.2911      1.85        ...         ...      ...     ...
hold_state        0.09705  0.20211   0.2988      ...         ...         42.6%       71.9%    0.158   ...
train_mean        0.31868  0.38410   0.8819      ...         ...         ...         ...      ...     ...
```

配套三条读法约定，建议写进 `eval_policy/README.md`：

1. **acc@τ 永远与 null 并排报**，单独一个 acc 数字没有意义（§4）。
2. **acc 主数字只含臂关节**，夹爪单列（§6.3，D3）。
3. **τ 与归一化基准必须同时声明**（"acc@0.1σ，σ 为 checkpoint 训练集 action std"），
   否则与任何外部数字都不可比（§1）。

---

## 10. 反方意见（devil's advocate）

诚实列出反对本方案的论点：

1. **"你在用一个 checkpoint 的 200 个 anchor 否定一整类指标。"**
   成立。§6 的全部结论建立在 `test_s53_trace.npz` 上（ACT baseline、单数据集、非随机采样的前 200 个 anchor）。
   §8.1 的 P0 脚本存在的意义就是把这个证据基础扩到全部 6 份 trace 与所有关心的 checkpoint。
   **在跑完 P0 之前，§6.2 应当被当作一个强烈的先验，而不是定论。**

2. **"acc@τ 的价值在跨本体/跨数据集可比，本项目只有一个本体，所以看不出价值。"**
   成立且重要。Dyna-2 用它是为了在 1k→1M 小时、39 个任务、多本体之间画 scaling 曲线，
   有界指标在那个场景下不可替代。**本项目是单本体、单任务族的 checkpoint 调试**，
   `norm_mae` 已经提供了跨数据集可比性。**如果将来要写跨本体或 scaling 的论文，这个结论要重新评估。**

3. **"tail_ratio 只是 acc@τ 的一个更差的替代品。"**
   部分成立。`R` 只用两个矩概括形状，acc@τ 与分位数信息更全。
   但 `R` 的成本是 3 行且能立刻重算全部 111 份历史报告，acc@τ 要重跑所有评测。
   **本方案的立场是：R 是筛子（P1），直方图是答案（P2），先用筛子决定要不要付答案的钱。**

4. **"你把 acc@0.05 上 policy 打不过 null 说成是 acc 的缺点，其实是策略的缺点。"**
   完全成立，而且这正是 §6.2 第 3 点想说的。区别只在于**该指标是否适合当主指标**：
   一个在你关心的操作点上把两个对象压缩到 2 个百分点之内的指标，判别力差是客观事实，
   哪怕它揭示的事实是真的。**建议的处理是：acc@0.05 作为一条"精度诊断"结论写进报告正文，
   而不是作为一列排序用的主指标。**

---

## 11. 参考文献与核实状态

核实分三级：**A** = WebFetch 抓 `arxiv.org/abs/` 摘要页逐字核对元数据与摘要；
**B** = 另抓正文 HTML 核对具体口径；**C** = 仅来自检索摘要，未逐字核实（引用时需注明）。

| # | 文献 | 核实 |
|---|---|---|
| 1 | **Dyna-2**, Dyna Robotics, `https://www.dyna.co/dyna-2`（无 arXiv 论文，网页技术报告） | **B**（网页正文，2026-09-03 抓取） |
| 2 | **OpenVLA: An Open-Source Vision-Language-Action Model**. Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn · arXiv:2406.09246 · v1 2024-06-13。256 bins / 1st–99th quantile 口径引自正文 HTML | **A + B** |
| 3 | **VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers**. Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, Hao-Shu Fang, Tong He · arXiv:2507.01016 · v1 2025-07-01 · ICCV 2025。k-Bin Acc 公式 | **A**（元数据）+ **C**（k-Bin 公式） |
| 4 | **Critical Interval MSE: Toward Reliable Offline Validation for Robot Manipulation Policies**. Haoxu Huang, Tongsam Zheng, Yifan Chen, Jiacheng You, Yang Gao · arXiv:2606.29898 · v1 2026-06-29。CI-MSE 定义、DTW/TE/RTC 对齐、ρ −0.87 vs −0.61、27 个 checkpoint（π₀.₅ / X-VLA / GR00T N1.7）均引自正文 HTML | **A + B** |
| 5 | **VLA-FAIL: Efficient Task Failure Detection for Finetuned Vision-Language-Action Models**. Florian Seligmann, Emiliyan Gospodinov, Enes Ulas Dincer, Gerhard Neumann · arXiv:2606.21386 · v1 2026-06-19。ACC = Action Chunk Consistency | **A** |
| 6 | **Eval-Actions: Fine-Grained Execution Quality Evaluation for Robotic Manipulation**. Mengyuan Liu, Juyi Sheng, Peiming Li, Ziyi Wang, Tianming Xu, Tiantian Xu, Hong Liu · arXiv:2601.18723 · v1 2026-01-26 | **A** |
| 7 | **How VLAs Fail Differently: Black-Box Action Monitoring Reveals Architecture-Specific Failure Signatures**. Krishnam Gupta · arXiv:2605.28726 · v1 2026-05-27。方向反转率 AUROC 0.79–0.93；速度违规 0.41–0.69 | **A** |
| 8 | **Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting**. Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, James Hays · arXiv:2301.00493 · v1 2023-01-02 | **A**（元数据）；**Miss Rate = minFDE > 2 m 的场景比例** 未能逐字核实（arXiv HTML 404、PDF 抓取失败），属领域常识，引用时请自行复核 |
| 9 | **Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success**（OpenVLA-OFT）. Moo Jin Kim, Chelsea Finn, Percy Liang · arXiv:2502.19645 · v1 2025-02-27。以 L1 回归取代离散 token 分类 | **A** |
| 10 | **ALOE: Action-Level Off-Policy Evaluation for VLA Model Post-Training**. Rushuai Yang 等 14 人 · arXiv:2602.12691 · v1 2026-02-13 | **A**。检索时命中，但其 off-policy value estimation 与本文离线动作误差口径无关，**不构成本方案依据**，列此仅为排除 |
| 11 | **Real-Time Execution of Action Chunking Flow Policies (RTC)**. Kevin Black, Manuel Y. Galliker, Sergey Levine · arXiv:2506.07339 | 见 `accuracy-root-cause-literature-survey-2026-09.md` §2.1（已核实） |

**本地产物引用**

| 出处 | 用于 |
|---|---|
| `eval_policy/runs/**/*.json`（111 份） | §5 的 R 配对统计 |
| `eval_policy/runs/scripts_act_eval_test/test_img/test_s53_trace.npz` | §6 全部实测 |
| `eval_policy/runs/scripts_act_eval_test/eval53_act_baseline.json` | §6 的每关节 σ 反解 |
| `eval_policy/offline_chunk_eval.py:268-311, 396-406, 474, 505, 518, 665-680, 685` | §8 全部改动点 |
| `policy/accuracy-root-cause-literature-survey-2026-09.md` §2（轴 A）、§附表 | §5.3 的机理解释 |
