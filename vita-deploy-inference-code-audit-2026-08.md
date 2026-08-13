# VITA 真机部署链路代码审计（针对 `vita-deploy-vibration-2026-08.md` 的抖动/顿挫）

> 审计日期：2026-08-13（夜）
> 审计对象：`lerobot_vlahost/workflows/robot_interaction/deploy.py` 及其驱动的整条链路
> → `rollout/strategies/base.py` → `rollout/strategies/core.py:send_next_action_chunk`
> → `rollout/inference/chunk.py` → `robots/marvain_m6_http/marvain_m6_http.py`
> → HTTP `/action_chunk` → `Apex_Deploy_new/robot_node/vlahost/vlahost/vla_node.py`
> 参照文档：[`vita-deploy-vibration-2026-08.md`](./vita-deploy-vibration-2026-08.md)
> 复现脚本：[`scripts_deploy_audit/`](./scripts_deploy_audit/)（数据用 `scripts_vita_chunk/scripts_vita_chunk_0813pm/A200k.npy`）

---

## 0. 一页结论

`deploy.py` 本身没有问题（它只是个拼命令行的 wrapper）。**问题全部在它启动的那条链路上，而且不是模型问题。**

**最重要的一条：机器人从来没有执行过 VITA 输出的轨迹。**
`core.py:474-533` 的"chunk 拒绝 + Hermite 降级"在本部署下是**每块必然触发**的，
而 chunk 长度（8 或 4）远小于代码里的 `K=40`，于是走的是 `else` 分支——
**整条 chunk 被替换成一条从实测位置到"模型第 8 帧"的三次 Hermite S 曲线，
起点速度 0、终点速度 0**。模型中间 7 帧的形状被整体丢弃。

实测（52 个 200k 原始 chunk，`scripts_deploy_audit/sim.py`）：

| 量 | 值 |
|---|---|
| 触发阈值 `threshold_rad` | 0.0087 rad（0.5°） |
| 实际 `max\|action0 − state\|`（14 关节取 max） | 中位数 **0.1178 rad（6.7°）**，最小 0.0097 |
| 超阈值的 chunk 比例 | **100.0 %**（n=4 那次录制也有 75.2 %） |
| 每块里被整列替换的手臂关节比例 | 平均 **78 %** |

叠加服务端"整块播完才要下一块、播完就停止发布"（`vla_node.py:434-461` + `939-961`），
执行流就是：**0.27 s 的 S 曲线 → 零速停住 → 60~100 ms 空档 → 下一条 S 曲线**，
周期 ≈ 3 Hz。这正是肉眼看到的"顿挫/波动"，与模型残差无关。

**排序（按"修了能立刻见效"）**

| # | 问题 | 位置 | 性质 |
|---|---|---|---|
| P0-1 | 每块 chunk 被整条换成零速起止的 Hermite | `core.py:474-533` | 执行的不是模型轨迹 |
| P0-2 | 播完才请求下一块 + 播完停止发布 → 每周期硬停 | `vla_node.py:451-461`, `939-961` | 走-停-走-停 |
| P0-3 | "连续拒绝 2 次"在确定性策略上是死仪式，第二次必然重复第一次 | `core.py:474-537` | 白扔一次推理，且保证走到 P0-1 |
| P1-4 | 客户端把首帧钉在**实测位置**，服务端 blend 起点却是**上次指令位置** → 每块开头一段反向 | `core.py:551` + `vla_node.py:824-836` | §5.4 那 83% 的"回退接缝" |
| P1-5 | 夹爪观测与训练不是同一个信号（不是"放大 1.25×"） | `marvain_m6_http.py:528-544` vs 训练 profile `state_topic: null` | 分布外输入，证实 §9.1 |
| P2-6 | 图像下采样核不一致：训练 `INTER_LINEAR`，部署 `INTER_AREA` | `marvain_m6_http.py:272` vs `profiles/schema.py:229` | 头相机域偏移 |
| P2-7 | 每 tick 新开一条 MJPEG 连接抓帧、解码 1280×1440；而推理只要 2.8 ms | `marvain_m6_http.py:172-214` | 死区几乎全是观测开销 |
| P2-8 | `source_hz` 从不下发，靠服务端默认值 30 兜底 | `marvain_m6_http.py:729` | 潜伏 bug（改 fps 即静默错速） |
| P2-9 | `_has_reached_target` 阈值 200° 恒真，且**持 `_chunk_lock` 打日志**，与 500 Hz 发布线程抢锁 | `vla_node.py:443-506` | 发布抖动（真·高频） |
| P2-10 | 500 Hz 播放线程无绝对时基，实际频率必然偏低 → 轨迹被系统性拉长 | `vla_node.py:939-961` | 整体变慢 |

---

## 1. P0-1：执行的轨迹是合成的 S 曲线，不是模型输出

### 1.1 代码

`core.py:414-419` 先算首帧偏差：

```python
threshold_rad = 0.0087                       # 0.5°
for i, (current, target) in enumerate(zip(current_positions, first_action)):
    diff = abs(float(target) - float(current))
    if diff > max_diff: max_diff = diff
```

`core.py:474-537`：

```python
elif max_diff > threshold_rad:
    reject_count += 1
    if reject_count >= 2:
        K = 40
        for i, joint_key in enumerate(arm_joint_keys[:14]):
            if diff > threshold_rad:
                if chunk_size > K:  ...          # 只重建前 K 帧，末速度对齐——从不走这里
                else:                            # chunk_size = 8 或 4，永远走这里
                    chunk[:, i] = cubic_hermite_segment(
                        current_val, float(original_chunk[-1, i]), chunk_size,
                        start_velocity=0.0, end_velocity=0.0)   # ← 零速起、零速止
    else:
        if_continue = True
        return None                              # 第一次直接丢弃这次推理
```

`chunk_size` 是 `inference.n_action_steps`（8，实验四是 4），**永远 ≤ K=40**，
所以 `if chunk_size > K` 那条"只重建前 K 帧、末速度对齐接入点"的正确分支**一次都没执行过**。
`§5.1` 表格里写的"用三次 Hermite 重建前 K=40 帧"是按代码字面读的，
实际行为是**整块替换 + 两端零速**，这一条要更正。

第一块还有一条无条件路径（`core.py:422-471`，`K=10`），同样 `8 ≤ 10` → 整块替换。

### 1.2 阈值和实际偏差差了一个数量级

`threshold_rad = 0.0087` rad = 0.5°。而 §9 已经测出 `|action0 − state|` 手臂均值 0.0378 rad；
取 14 个关节的 **max** 后中位数是 **0.1178 rad = 6.75°**，是阈值的 **13.5 倍**。
52 块里 **52 块**超阈值，n=4 那次 137 块里 103 块超阈值。

这个阈值不可能被满足：模型是在**实测位置**上做的预测，而 `action0` 天然是"下一拍的指令"，
在训练集里它和 `state` 就差 0.0203 rad（§9），已经是阈值的 2.3 倍。
**换句话说，即使模型完美复现训练分布，这个门也照样每块都关。**

### 1.3 后果（实测）

把整条后处理栈（Hermite 降级 → `remove_boundary_rollbacks` → `remove_small_rollbacks` →
`remove_open_gripper_loops` → `smooth_action_chunk` → `smooth_large_excursions`）
原样打到 200k 的原始输出上（`scripts_deploy_audit/sim.py`）：

```
chunk 内归一化速度剖面（手臂 14 关节）
  200k 原始模型输出      1.158 1.072 1.052 0.903 1.077 0.779 0.960   峰/边 = 1.09
  过完部署后处理栈       0.718 0.978 1.211 1.285 1.219 0.947 0.641   峰/边 = 1.89
  §5.4b 里 100k 实测录制 0.634 0.903 1.187 1.311 1.286 0.997 0.684   峰/边 = 1.99   ← 几乎重合
```

消融（`scripts_deploy_audit/abl.py`）：

```
只做二项式平滑         1.237 1.130 1.018 0.945 0.909 0.835 0.926   峰/边 = 1.14
只做 Hermite 降级      0.589 1.038 1.232 1.327 1.298 0.990 0.525   峰/边 = 2.38
```

**钟形是 Hermite 造的，不是二项式滤波造的。** §5.4b 把这个钟形归因于
"②把开头的边界回退删平，⑤又固定首尾帧只压中间"——从消融看，二项式那一遍
只能把峰/边从 1.09 抬到 1.14，做不出 1.99 的钟。这条归因要更正。

同一份模拟还给出两个数字：

```
每块净位移（14 手臂关节求和）  模型要求 0.6734 rad → 实际执行 0.4385 rad   （65%）
每块路径长度                   模型 0.7809 rad → 执行 0.4570 rad
执行流每步速度                 首步 0.0337 / 中段 0.0886 / 末步 0.0348 rad/step
```

- **每块只走了模型要求的 65%**。因为 Hermite 起点是**落后于指令的实测位置**，终点却只到模型第 8 帧。
  §11.4 观察到的"每 chunk 净位移 ≈ 每 chunk 首帧偏移，比值 1.01，没有净进展"就是这个。
- **末步速度只有中段的 39%**，且下一块又从 0 起速——**每 0.3 s 完整地加速-减速一次**。

### 1.4 对原报告结论的影响

- §2 / §9 / §11 里对**原始 chunk**（`record_chunk.txt`）算出来的模型指标（lag-1 −0.080、
  反转率 27.1% 等）**依然成立**，那是对模型的诊断，没问题。
- 但"真机上看到的振动/顿挫"**不能**归到这些指标上：那些高频成分在下发前就被整条抹掉了，
  真机执行的是 S 曲线序列。
- §9 第 3 条测的 `seam/intra = 4.40×` 是**原始输出**的接缝；**执行流**的接缝在模拟里是
  `1.31×`（反转率 2.7%）。也就是说，真机上的问题不是"接缝跳变大"，而是**"每块之间完整停一次"**。
- §10 第 3 条（改读出头 / 加二阶差分平滑损失）作为**模型侧**的判断仍然成立，
  但它**解决不了现在肉眼看到的顿挫**，而且在 P0 修好之前，真机是无法用来验证它的。

---

## 2. P0-2 / P0-3：每个周期都有一段硬停

### 2.1 服务端：播完才要，播完就停

`vla_node.py:443-461`：

```python
if traj is not None and len(traj) > 0:
    remaining = len(traj) - idx
    if remaining <= 0:                       # ← 必须整块播完
        if self._has_reached_target(target_position, joint_threshold_deg=200.0):
            return 1
```

`vla_node.py:944-951`：轨迹耗尽后 `sp is None`，**不再发布任何 setpoint**，
底层控制器 hold 在最后一个指令位。

所以 "需要新块" 这个事件**发生在运动已经结束之后**，而不是"快结束了"。
从这一刻到下一条 setpoint 出现，中间是纯静止。

### 2.2 客户端：这段静止有多长

链路里各段实测/估算（推理延迟是本机 4090 实测，`scripts_deploy_audit/lat.py`）：

```
VITA generate_actions                       2.8 ms   ← 完全不是瓶颈
1280×1440 JPEG 解码 + 拆分 + resize        ~10 ms
GET /state + 新开一条 MJPEG 连接抓一帧      两次 HTTP 往返 + 等一个帧边界
控制循环 tick 量化（fps=30）                平均 16 ms 检测延迟 + 33 ms/次
被白扔的那次推理（P0-3）                    多花整整一个 tick ≈ 33 ms
写 record_chunk.txt（全量重读计数）         文件线性增长，长跑后变慢
```

保守估计每周期 **60~100 ms 静止**，占 0.30 s 播放周期的 20~33 %。
叠加 §1 的"零速起止"，就是完整的**走-停-走-停**。

### 2.3 P0-3：拒绝重试在这里是死仪式

`core.py:534-537` 第一次拒绝 → `if_continue = True; return None`，
下一 tick `core.py:368-371` 强制 `need_new_chunk = 1` 再推一次。
但 §2.3 已经证明 VITA 在同一观测下**逐比特确定**，而此时**手臂是停着的**，
观测几乎不变 → 第二次输出和第一次一样 → `reject_count` 必然到 2 → 必然降级。

**这个"两次机会"永远不会救回任何一块**，它唯一的作用是每周期多花一个 tick，
并且保证走到 P0-1。（可以在日志里数：`⚠️ Chunk rejected` 的行数应当恰好等于
`=== Chunk` 的块数。）

### 2.4 建议

1. **服务端提前请求**：把 `remaining <= 0` 改成 `remaining <= lead_setpoints`，
   `lead_setpoints = ceil(chunk_playback_hz × 实测端到端延迟)`（先按 0.1 s → 50 拍）。
   新块到达时旧块还在播，`submit_chunk` 的 blend 正好接上，**空档消失**。
   这是一行改动，收益最大。
2. **删掉 P0-3 的重试**（阈值判断保留与否见下），至少省一个 tick。
3. 客户端的 `chunk_interval_s` 依然是死代码（§11.1 结论不变），不要在它上面花时间。

---

## 3. P0-1 的修法（三选一，按侵入性排序）

**(a) 最小改动：把阈值改成物理上合理的值，并让降级走"只桥接前几帧"的分支。**

```python
threshold_rad = math.radians(3.0)            # 0.052 rad，比训练集的 0.0203 高一档
K = max(2, chunk.shape[0] // 2)              # 8 步 chunk → 桥接前 4 帧，必然走 chunk_size > K 分支
```

走 `chunk_size > K` 分支时，`end_velocity = original_chunk[K] - original_chunk[K-1]`
是**对齐接入点速度**的，桥接段结束后回到模型自己的轨迹，末尾不再归零。

**(b) 更对的做法：桥接的锚点用"上一次下发的指令"而不是"实测位置"。**
见下面 P1-4——现在这个锚点选错了，是接缝回退的直接原因。

**(c) 结构上正确的做法：换 RTC 引擎**（`deploy_config_vita.yaml` 已有
`execution_horizon` 字段，`rollout/inference/rtc.py` 已实现）。
RTC 按设计就是让新块与在执行的旧块在重叠区对齐，首帧偏移由算法吸收，
不需要任何手写桥接。§11.5 的结论（"接缝要真解决只剩 RTC"）在这里得到代码层面的支持。

不管选哪条，**都要先把 `record_chunk.txt` 之外再记一份"实际下发的 chunk"**，
否则评估仍然看不到执行流。现在 `core.py:645-646` 已经同时算出了
`processed_actions`（下发）和 `raw_actions`（原始），只写了后者——把前者也写一份即可。

---

## 4. P1-4：接缝回退的确切来源（客户端锚实测位置 / 服务端锚指令位置）

两侧各自"从当前位置平滑接入"，但**两侧对"当前位置"的定义不同**：

| | 起点 | 位置 |
|---|---|---|
| 客户端 Hermite / `remove_boundary_rollbacks` | `obs_raw[...pos]` = **实测关节角** | `core.py:396`, `core.py:551` |
| 服务端 `submit_chunk` blend | `self._last_published_sp` = **上一次发布的指令** | `vla_node.py:824-834`, `900-908` |

而手臂天然落后指令 **0.025 rad**（§5.4a 实测）。于是每块开头，服务端要在
**1/30 s 内**从"上次指令位置"线性插到"实测位置"——一段**方向与运动相反**的段，
速度 ≈ 0.025 × 30 = **0.75 rad/s**，是块内典型速度（0.008 × 30 = 0.24 rad/s）的 3 倍。

这与 §5.4 测到的"83% 的接缝是回退、`|seam|/|v_prev|` 中位数 3.18×"完全对上。
**这不是模型的首帧偏移造成的，是两侧锚点定义不一致造成的。**

修法：客户端的桥接锚点改用"上一块最后一帧下发的指令"（客户端自己就有，
或从 `/state` 暴露 `last_published_sp`）；或者关掉服务端 `chunk_blend_from_current`，
由客户端单方面负责连续性。**两侧同时"平滑接入"是最坏的组合。**

---

## 5. P1-5：夹爪观测的根因（证实并修正 §9.1）

§9.1 推断"真机上报的夹爪 state 大约被放大了 1.25×"。代码层面看，**不是缩放问题，是接错了信号**：

- **训练侧**：`batch_3/meta/conversion_manifest.json` 里 profile `marvin-gripper-quadtile` 的
  两个末端执行器都是 `"state_topic": null`。按 `CLAUDE.md` 的语义，
  末端执行器观测退化为 **`command_echo`**——训练时 `observation` 的夹爪列
  **就是 `/control/gripperValue*` 指令本身**，单位是 DM_gripper 的开合比例 **[0, 1]**。
  （这也解释了 §9.1 里 "action ≡ state，差 0.00012"。）
- **部署侧**：`marvain_m6_http.py:528-544` 读 `/state` 的 `gripper_left.position`，
  它来自 `end_effector.py:284-303` → `/info/gripper_feedback_L` 的 `Float32MultiArray.data[0]`，
  是**驱动器标定后的电机实测位置**。`GripperAdapter` 自己的注释写明指令侧才是
  "clip 到 [0,1] 再 lerp 到 [min_pos, max_pos]"。

两者是**不同量纲的不同信号**，只是碰巧比值 ≈1.25。`STATE` 用 `MIN_MAX` 归一化，
所以 1.21/1.30 的实测值归一化后越界 → 分布外输入，且**每一帧都越界**。

修法（按保真度）：

1. **回显指令**：客户端自己保存上一次下发的 `gripper_left/right`，
   在 `get_observation()` 里用它填夹爪观测列。与训练**逐比特同源**，改动 5 行。
2. 用 DM_gripper 的 `[min_pos, max_pos]` 把反馈仿射映回 [0,1]（需要拿到标定值，
   `1/0.798 ≈ 1.25` 只是回归出来的近似）。
3. 长期：录数据时把 `/info/gripper_feedback_*` 一起录，profile 里填上 `state_topic`，
   让训练观测就是实测值。

---

## 6. P2 级问题

### 6.1 图像下采样核不一致（训练 vs 部署）

`robot_data_platform/tool/robot_data/profiles/schema.py:227-229` 明确写着：

```python
# INTER_LINEAR mirrors the deployment splitter's `_resize_camera`, so
# training frames and live inference frames go through the same filter.
return cv2.resize(crop, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
```

它对齐的是 `vlahost/lebot_client.py:88` 的 `_resize_camera`（`INTER_LINEAR`）。
但**真机推理走的不是 `lebot_client`**，而是
`marvain_m6_http.py:272`：

```python
top_resized = cv2.resize(top_img, (w_half, h_bottom), interpolation=cv2.INTER_AREA)
```

头相机是 1280×960 → 640×480 的 2:1 下采样，`INTER_LINEAR`（会明显混叠）和
`INTER_AREA`（盒式抗混叠）产生的高频纹理**完全不同**。训练集每一帧都是前者，
推理每一帧都是后者。改成 `INTER_LINEAR` 即可对齐（一处改动）。
（腕部两块是 640×480 原尺寸，`schema.apply` 直接返回 crop，不受影响。）

### 6.2 观测链路开销远大于推理

- `_grab_mjpeg_frame`（`marvain_m6_http.py:172-214`）**每 tick 新开一条 HTTP 长连接**，
  读到一个完整 JPEG 就关掉。30 Hz × 每次一条连接，还要等帧边界。
- 每 tick 都解码 1280×1440 JPEG 并拆分（本机实测 ~10 ms/帧），
  但**只有触发推理的那 1/9 个 tick 需要图像**。
- `_split_quad_image` 还多产出一个没人用的 `top2` 副本。
- `deploy_config_vita.yaml: show_cameras: true` 会再起一个进程轮询同一个 `/state` 和 MJPEG。

建议：常驻一个后台 MJPEG 读取线程（保持连接，只留最新帧），
或者只在要推理的 tick 才抓图。修完 P0-2 之后这部分决定残余空档的下限。

### 6.3 `source_hz` 从不下发

`marvain_m6_http.py:729` 的 body 只有 `{"actions": [...]}`，
而 `ACTION_CHUNK_HANDLER.md` §2/§3 明确要求带 `source_hz`。
服务端 `vla_node.py:839` 回退到 `chunk_waypoint_hz = 30.0`，
**当前恰好等于 `inference.fps = 30`，所以没暴露**。
改 `fps` 或换机器人配置就会静默按错速度回放。建议补上：

```python
body = {"actions": chunk_payload, "source_hz": float(self.config.fps)}
```

### 6.4 服务端 `_check_need_new_chunk` 的两个问题

`vla_node.py:456`：`joint_threshold_deg=200.0` —— 200° 的阈值让 `_has_reached_target`
**恒为真**，"确认机器人真的到位了"这个设计意图完全没生效（注释还写着 2.0）。

更要紧的是它**在持有 `_chunk_lock` 的情况下**做：

```python
self.get_logger().info(f"Target NOT reached: ... errors={[f'{e:.2f}' for e in errors_deg]}")
```

每次 `GET /state`（30 Hz）都格式化 14 个浮点数并打一条 ROS 日志，**锁是握着的**。
而 500 Hz 的 `_chunk_player_loop` **每一拍**都要拿同一把 `_chunk_lock`
（`vla_node.py:945-950`）。一次日志调用 50~200 µs（写盘/转发时更长），
于是每 33 ms 就给 setpoint 流插一个抖动。这是**真正意义上的高频抖动**候选来源，
和 chunk 级顿挫是两回事。修法：把日志挪出锁外，阈值改回 2°（或直接删掉这个恒真检查）。

### 6.5 500 Hz 播放线程没有绝对时基

`vla_node.py:957-961`：

```python
elapsed = time.perf_counter() - tick_start
sleep_for = period - elapsed
if sleep_for > 0: time.sleep(sleep_for)
```

`time.sleep` 在 2 ms 周期上的超调（Linux 上典型 50~150 µs）**不会被后续拍补偿**，
实际发布频率必然低于 500 Hz。而插值密度是按 `round(500/30) = 17` 拍/waypoint 定死的，
所以**整条轨迹被系统性拉长**——等价于以低于 30 fps 的速度回放训练轨迹。
建议改成按 monotonic 截止时间调度（`next_t += period`），
并在日志里打一次实测频率确认。

### 6.6 杂项（不影响抖动，但会咬人）

- `deploy_config_vita.yaml:19` 仍指向 **10k 的 batch_2** checkpoint。
  不带 `--policy-path` 直接跑 `deploy.py` 会静默加载旧模型。
- 同文件 `inference.rtc.execution_horizon: 30` 在 `type: chunk` 下完全不生效，容易误导。
- `core.py:51-53` 的 `if_continue` / `reject_count` / `is_first_chunk` 是**模块级全局**，
  `engine.reset()` 不清它们；同一进程里跑第二个 episode 时状态是脏的。
- `core.py:659-666` 每块都把整个 `record_chunk.txt` 读一遍来数块号，
  文件只追加不轮转，长跑后这一步会进到发送路径的关键延迟里。
- `core.py:433` `arm_joint_keys[:16]` 实际只有 14 个元素；
  `core.py:440` 等处 `print(f"当前={current_val:.2f}°")` 打的是弧度，标注是度。
- `deploy.py:267-288` 一批 `if args.xxx:` 用真值判断，`--fps 0` / `--duration 0`
  这类"显式传 0"会被静默忽略（`--return-to-initial` 用 `type=bool` 也永远为真）。

---

## 7. 建议的执行顺序

1. **P0-2**：服务端 `remaining <= lead_setpoints` 提前请求（一行）。→ 空档消失。
2. **P0-1 + P0-3**：删掉重试；阈值抬到 3°；`K = chunk_len // 2` 让降级走"桥接前几帧、
   末速度对齐"的正确分支。→ 执行流回到模型轨迹，两端不再归零。
3. **P1-4**：桥接锚点统一到"上一次下发的指令"，或关掉服务端 blend。→ 消掉回退接缝。
4. **P1-5**：夹爪观测改为回显上一次下发的指令值。→ 消除分布外输入。
5. 复测：此时再用 `record_chunk.txt` 的原始输出 + **新增的"实际下发 chunk"记录**
   分别算 §4 的三个指标。只有到这一步，§10 第 3 条（读出头 / 二阶差分平滑损失）
   才具备可验证的前提。
6. 剩下的 P2 依次处理；`INTER_AREA → INTER_LINEAR`（6.1）和日志出锁（6.4）都是一行。

---

## 8. 复现

```bash
cd /home/kewei/YING/paper/policy/scripts_deploy_audit
python3 sim.py     # 阈值触发率 + 整条后处理栈模拟 + 速度剖面对比
python3 abl.py     # Hermite vs 二项式的消融，净位移/路径长度
conda run -n lerobot env PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src python lat.py   # VITA 推理延迟
```

`traj_standalone.py` 是 `lerobot_vlahost/src/lerobot/rollout/trajectory.py` 的逐字副本
（直接 import 该包会被 base 环境里坏掉的 `transformers`/`huggingface-hub` 版本挡住，
见原报告 §1.6）。数据取自 `../scripts_vita_chunk/scripts_vita_chunk_0813pm/A200k.npy`、
`S200k.npy`（200k 原始输出）和 `../scripts_vita_chunk/scripts_vita_chunk_0813_n4/`（n=4 那次）。

现场核对（不用改代码）：日志里 `⚠️ Chunk rejected` 的行数应当恰好等于
`record_chunk.txt` 里 `=== Chunk` 的块数——若相等，即证实"每块都走降级"。
