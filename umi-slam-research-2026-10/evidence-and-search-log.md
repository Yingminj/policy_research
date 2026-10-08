# UMI 与 SLAM：证据、检索与核验记录

配套主报告：[UMI 与 SLAM 深度研究](../umi-slam-deep-research-2026-10.md)

核验截止：2026-10-08。本文档记录本次综述的来源范围与分析边界，便于继续研究；不把检索流程包装成已注册的系统综述。

## 1. 研究范围与输入

研究问题：

1. UMI 及其衍生系统如何将人类示教变为可部署的机器人动作监督？
2. 近期 SLAM、视觉惯性与学习几何路线分别改善了哪些能力？
3. 哪些定位指标能解释 UMI 数据质量，哪些问题必须由本体约束或控制语义解决？
4. 如何设计公平的工程选型与下游验证？

本地背景资料：

- 本目录的 robot-data-quality-curation-survey-2026-09.md；
- survey_and_discuss/no-embodiment-to-real-robot-2026-08.md；
- paper_note/vista-2606.04708.md。

以上作为问题与线索来源；本文涉及的新进展、关键方法与数值重新查阅公开原始来源。本次未改写原有文件。

## 2. 检索策略与时间

检索和核验跨 2026-10-07 至 2026-10-08；近期重点为 2024—2026，保留必要的早期基础方法。使用公开网络检索，随后回到 arXiv、正式论文集、作者项目和官方仓库。未运行 Semantic Scholar/OpenAlex 批量 API、没有购买或绕过访问限制。

以下是按实际检索主题归并的复检词组，**不是逐条工具请求的完整转录**：

| 主题 | 可复检词组 |
|---|---|
| UMI 基础与工程 | Universal Manipulation Interface；FastUMI；UMI dataset pose tracking |
| 高保真与传感器 | HiFi-UMI；UMI-3D LiDAR IMU；RDT2 tracking system |
| 数据可执行性与筛选 | FeasibleCap；VISTA UMI physical validation |
| 策略相关采集 | HIL-UMI；RoboPocket policy feedback |
| 触觉/灵巧/接触 | DexUMI；RealDexUMI；exUMI；TacUMI；OmniUMI；WT-UMI；BRIDGE handheld teleoperation |
| 移动与主动观察 | ActiveUMI；MV-UMI；UMI on Legs；UMI-on-Air；HoMMI |
| 几何基础模型 | DUSt3R MASt3R VGGT SLAM；CUT3R；SLAM3R；LingBot-Map；SURE-Map |
| 混合与失效 | ScaRF-SLAM；MASt3R-Fusion；FoundationSLAM；Failure or Drift monocular SLAM |
| 其他地图/传感器 | SplaTAM；Gaussian Splatting SLAM；FAST-LIO2；FAST-LIVO2；LIO-SAM；ConceptGraphs；AERO-VIS |

补充查找包括官方实现入口、论文版本差异、方法限制、实验分母以及“calibration-free / metric / real-time / robot-free”的具体含义。对 SLAM 路线同时查找支持和限制性证据，避免只检索排行榜提升。

## 3. 纳入、排除与计数

**纳入：**与 UMI 示教、位姿恢复、监督迁移、接触、可执行性、数据反馈或相关 SLAM 路线直接有关；有可核验原始论文或官方来源；截至截止日已经公开。

**排除/降级：**只有宣传性二手介绍而无法回到原始出处；与 UMI 缩写无关的主题；题名/标识不匹配；无法确认条件的数值；仅凭演示视频推断一般性能；截止日之后才有的信息。

最终纳入 **54 项研究工作：25 项 UMI/示教，29 项 SLAM/几何**。另列 9 个正式出版补充页或官方资源；其中 FastUMI 的正式出版页与预印本属于同一项工作，不重复计数。并非记录了所有命中、去重、排除数量，因此不提供伪精确的 PRISMA 流程统计。

来源明显偏向 arXiv；主报告中的多数论文链接指向 arXiv 的稳定版本。它是发布平台，不等于所有论文来自同一研究团队，也不等于全部未评审。不过这种入口集中会带来公开发表与可访问性偏差。正式状态在可核验处补充，不用引用次数筛掉新论文，不声称检索穷尽。

## 4. 阅读层级与用途约束

- **F：**读取论文全文中的选定方法、实验或限制章节。不是声称从第一页到最后一页逐字精读，也不是完整代码审计。
- **A：**原始论文摘要及元数据；少数为搜索工具呈现的原始摘要。只支持研究方向与作者主张的有限描述。
- **P：**正式出版页或作者官方项目/论文入口。
- **C：**官方仓库说明。未安装或运行；仓库存在不等于全部材料可复现。

11 项论文达到 F 层级。其余主要用于技术版图、补充方向和阅读索引。主报告尽量只对 F 层级论文使用有条件的具体实验数值；外围 A 层级论文不用于构建性能排名。

“来源存在且题名一致”“方法主张已核对”“数值上下文已核对”“结果已独立复现”是四个不同等级。本次做到前三者中的相应部分，**没有独立复现任何一项结果**。

## 5. 逐项证据登记

完整机器可读目录见 [sources.json](./sources.json)。下表的状态仅表示本次能核验的内容；“未核验正式发表”不等于断言论文未被接收。

| 编号 | 论文 | 层级 | 版本/发表状态 | 本次用途与边界 |
|---|---|---|---|---|
| U01 | [Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots](https://arxiv.org/abs/2402.10329v3) | F+C | RSS 2024；读 arXiv v3 | 方法、相对动作、时延与补充实现；2024-03-06 版本 |
| U02 | [FastUMI: A Scalable and Hardware-Independent Universal Manipulation Interface with Dataset](https://arxiv.org/abs/2409.19499v2) | F+P | CoRL 2025；方法读 v2，规模另核正式版 | 硬件、采集、质量处理；v2 与正式版数据规模不同 |
| U03 | [RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization](https://arxiv.org/abs/2602.03310v1) | F | 预印本；本次未核验正式发表 | 摘要、训练框架与采集硬件附录；万小时为作者报告 |
| U04 | [HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone](https://arxiv.org/abs/2607.25895v1) | F | 技术报告/预印本 | 硬件、重建、质量、后训练实验及局限；3 mm 为局部条件结果 |
| U05 | [UMI-3D: Extending Universal Manipulation Interface from Vision-Limited to 3D Spatial Perception](https://arxiv.org/abs/2604.14089v1) | F | 预印本；本次未核验正式发表 | LiDAR/IMU 恢复、标定及策略接口；区分采集三维与策略三维 |
| U06 | [FeasibleCap: Real-Time Embodiment Constraint Guidance for In-the-Wild Robot Demonstration Collection](https://arxiv.org/abs/2603.07580v1) | F | 预印本；本次未核验正式发表 | 方法与可执行性实验；具体实验比例不得泛化 |
| U07 | [VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training](https://arxiv.org/abs/2606.04708v2) | F | 预印本；本次未核验正式发表 | 视觉适应与物理验证、评分定义；v2 2026-06-04 |
| U08 | [HIL-UMI: Bringing Human-in-the-Loop Post-Training of Vision-Language-Action Models to Universal Manipulation Interface](https://arxiv.org/abs/2609.20659v1) | A | 2026-09 预印本 | 策略相关采集/后训练方向；未核验完整实验和实现 |
| U09 | [RoboPocket: Improve Robot Policies Instantly with Your Phone](https://arxiv.org/abs/2603.05504v2) | A | 预印本；本次未核验正式发表 | 手机交互与策略反馈；未将数据效率数字用于横向排名 |
| U10 | [UMI-Bench 1.0: An Open and Reproducible Real-World Benchmark for Tabletop Robotic Manipulation with UMI Data](https://arxiv.org/abs/2606.10382v1) | A | 预印本；本次未核验正式发表 | 真实任务流程与可复现评测范围 |
| U11 | [DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation](https://arxiv.org/abs/2505.21864v3) | A | arXiv v3；本次未单独核验会议版 | 灵巧示教的运动学和视觉差距；不作性能排名 |
| U12 | [RealDexUMI: A Wearable Universal Manipulation Interface for Dexterous Robot Learning](https://arxiv.org/abs/2606.06033v2) | A+P | 项目页标注 CoRL 2026；读 arXiv v2 | 摘要和官方项目的方法解释；命令/实际状态与共享末端 |
| U13 | [exUMI: Extensible Robot Teaching System with Action-aware Task-agnostic Tactile Representation](https://arxiv.org/abs/2509.14688v1) | A | arXiv 元数据标注 CoRL 2025 | 可扩展硬件与触觉表征；不把 usability 外推为通用保证 |
| U14 | [TacUMI: A Multi-Modal Universal Manipulation Interface for Contact-Rich Tasks](https://arxiv.org/abs/2601.14550v1) | A | 预印本；本次未核验正式发表 | 多模态采集；分割结果不是策略成功率 |
| U15 | [OmniUMI: Towards Physically Grounded Robot Learning via Human-Aligned Multimodal Interaction](https://arxiv.org/abs/2604.10647v3) | A | 预印本；本次未核验正式发表 | 物理交互、多模态和部署控制对应 |
| U16 | [WT-UMI: Tactile-based Whole-Body Manipulation via Force-Supervised Contact-Aware Planning](https://arxiv.org/abs/2606.13232v1) | A | 预印本；本次未核验正式发表 | 力监督和目标姿态修正；不声称没有机器人训练数据 |
| U17 | [ActiveUMI: Robotic Manipulation with Active Perception from Robot-Free Human Demonstrations](https://arxiv.org/abs/2510.01607v1) | A | 技术报告 | 主动观察方向；摘要级方法归类 |
| U18 | [MV-UMI: A Scalable Multi-View Interface for Cross-Embodiment Learning](https://arxiv.org/abs/2509.18757v1) | A | 预印本；本次未核验正式发表 | 多视角上下文；不外推任务收益百分比 |
| U19 | [UMI on Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers](https://arxiv.org/abs/2407.10353v1) | A | arXiv；本次未单独核验会议版 | 末端意图与全身控制分层 |
| U20 | [UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies](https://arxiv.org/abs/2510.02614v3) | A | 读 2026-03 修订版；未单独核验会议论文集 | 本体相关引导；不解释为形式化可行性保证 |
| U21 | [HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations](https://arxiv.org/abs/2603.03243v2) | A | 预印本；本次未核验正式发表 | 头部上下文与全身移动操作 |
| U22 | [UMIGen: A Unified Framework for Egocentric Point Cloud Generation and Cross-Embodiment Robotic Imitation Learning](https://arxiv.org/abs/2511.09302v1) | A | 预印本；本次未核验正式发表 | 点云与跨本体数据生成；不声称无需任何跟踪 |
| U23 | [UMI-Bridge: Action-Anchored Latent Alignment across Human and Robot Manipulation Data](https://arxiv.org/abs/2609.18232v2) | A | 2026-09 预印本 | 动作关联的跨域对齐；与 BRIDGE 是不同论文 |
| U24 | [Bridging Handheld and Teleoperated Supervision for Contact-Rich Manipulation via State-Gated Experts](https://arxiv.org/abs/2606.26603v2) | A | 预印本；读 2026-09-21 修订版 | 接触任务中手持/遥操作监督差异；避免泛化收益幅度 |
| U25 | [Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](https://arxiv.org/abs/2603.02139v1) | A | 论文来源标注 CVPR 2026 | 相机选择、视场与泛化；不解释为鱼眼普遍最优 |
| S01 | [ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898v2) | A | IEEE T-RO 2021；读 arXiv v2 | 视觉/惯性/多地图路线；未审计源代码 |
| S02 | [VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator](https://arxiv.org/abs/1708.03852) | A | IEEE T-RO 2018；读 arXiv 页面 | 单目惯性与滑窗估计；未审计源代码 |
| S03 | [DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras](https://arxiv.org/abs/2108.10869v2) | A | 读 arXiv v2 | 学习对应与可微几何优化；不比较原始 FPS |
| S04 | [Deep Patch Visual Odometry](https://arxiv.org/abs/2208.04726v2) | A | 读 arXiv v2 | patch 视觉里程计，区别于完整 SLAM |
| S05 | [Deep Patch Visual SLAM](https://arxiv.org/abs/2408.01654) | A | 原始论文摘要检索记录 | DPVO 的全局 SLAM 扩展；无独立运行验证 |
| S06 | [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132v3) | A | CVPR 2024；读 arXiv v3 | 两视图点图与三维几何表征 |
| S07 | [Grounding Image Matching in 3D with MASt3R](https://arxiv.org/abs/2406.09756) | A | 原始论文页面 | 三维先验与密集匹配 |
| S08 | [MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](https://arxiv.org/abs/2412.12392v2) | F+C | CVPR 2025；读 arXiv v2 | 系统方法与局限；核对鱼眼畸变和全局地图更新边界 |
| S09 | [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651) | A | CVPR 2025 | 多视图前馈几何；不是独立完整 SLAM |
| S10 | [VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold](https://proceedings.neurips.cc/paper_files/paper/2025/file/bc65ab11abfbad890171109686233f4e-Paper-Conference.pdf) | P+C | NeurIPS 2025 | 官方论文与仓库元数据；架构级归类，不承担量化比较 |
| S11 | [VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction](https://arxiv.org/abs/2601.19887v1) | F+C | 官方仓库标注 RSS 2026；读 arXiv v1 | 因子图、运行效率与限制；嵌入式速度有配置条件 |
| S12 | [SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](https://arxiv.org/abs/2412.09401v3) | A | CVPR 2025；读 arXiv v3 | 点图与局部/全局重建；未验证 TCP 输出接口 |
| S13 | [Continuous 3D Perception Model with Persistent State](https://arxiv.org/abs/2501.12387) | A+C | CUT3R；官方仓库标注 CVPR 2025 | 持续状态与点图；不把 learned metric scale 等同物理尺度约束 |
| S14 | [LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction](https://arxiv.org/abs/2604.14141v3) | A+C | 官方仓库标注 ECCV 2026；读 2026-09 修订版 | 多层上下文流式重建；采用最新核验标题 |
| S15 | [SURE-Map: Self-Correcting Streaming Geometric Foundation Models](https://arxiv.org/abs/2609.15795v2) | A | 2026-09 预印本 | 一致性、纠错、尺度校正；不外推为 UMI 毫米级能力 |
| S16 | [MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM](https://arxiv.org/abs/2509.20757v3) | A | 读 arXiv v3；未单独核验会议版 | 视觉先验与惯性/GNSS 因子融合 |
| S17 | [ScaRF-SLAM: Scale-Consistent Reconstruction with Feed-Forward Models and Classical Visual SLAM](https://arxiv.org/abs/2606.00307v4) | F | 页面标注 RA-L；读 2026-09-25 版本 | 经典跟踪与基础模型建图；核对标定、尺度对齐和限制 |
| S18 | [FoundationSLAM: Unleashing the Power of Depth Foundation Models for End-to-End Dense Visual SLAM](https://arxiv.org/abs/2512.25008v2) | A | arXiv 元数据标注 AAAI 2026 | 深度先验、流与一致性优化 |
| S19 | [M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM](https://arxiv.org/abs/2603.16844v1) | A | 预印本；本次未核验正式发表 | 匹配、多视图几何与 Gaussian 地图组合 |
| S20 | [NICE-SLAM: Neural Implicit Scalable Encoding for SLAM](https://arxiv.org/abs/2112.12130v2) | A | CVPR 2022；读 arXiv v2 | 分层隐式场路线 |
| S21 | [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://arxiv.org/abs/2312.02126v3) | A | CVPR 2024；读 arXiv v3 | RGB-D Gaussian 跟踪与建图 |
| S22 | [Gaussian Splatting SLAM](https://arxiv.org/abs/2312.06741v2) | A | CVPR 2024；MonoGS | 单目 Gaussian SLAM；未独立评价接触几何 |
| S23 | [MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion](https://arxiv.org/abs/2410.03825v2) | A | ICLR 2025；读 arXiv v2 | 动态几何方向；不等同完整动态物体状态估计 |
| S24 | [FAST-LIO2: Fast Direct LiDAR-inertial Odometry](https://arxiv.org/abs/2107.06829) | A | 原始论文页面；未单独核验期刊版 | 原始点配准与激光惯性里程计 |
| S25 | [FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://arxiv.org/abs/2408.14035v2) | A | 读 arXiv v2；未单独核验期刊版 | 激光、惯性与视觉融合 |
| S26 | [LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping](https://arxiv.org/abs/2007.00258v3) | A | IROS 2020 | 激光惯性因子图路线 |
| S27 | [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://arxiv.org/abs/2309.16650) | A+P | 原始摘要与官方论文入口；未单独核验会议版 | 语义三维场景图；更正过一次错误标识，不采用错误来源 |
| S28 | [AERO-VIS: Asynchronous Event-based Real-time Onboard Visual-Inertial SLAM](https://arxiv.org/abs/2605.07885v2) | A | 页面标注 RA-L 2026；读 2026-09 修订版 | 事件/双目/惯性；验证场景非 UMI |
| S29 | [Failure or Drift? Evaluating Monocular SLAM under Synthetic and Real-World Corruptions](https://arxiv.org/abs/2608.30690v1) | F | ECCV 2026 NeuSLAM workshop，非主会论文 | 比较 ORB-SLAM2、DPVO、DROID；核对扰动与存活偏差 |


## 6. 关键核验与纠错

| 核验点 | 处理结果 | 主报告位置 |
|---|---|---|
| 原始 UMI 的传感器与部署链条 | 不简化为纯单目；区分示教轨迹恢复与机器人本体状态 | 第 3、7 节 |
| FastUMI 数据规模 | 同时保存早期版本与正式出版差异，数字各自归属对应来源 | 第 6.2 节 |
| HiFi-UMI 精度与通过率 | 保留局部测量条件；区分各阶段通过率；后训练与预训练分开 | 第 6.3、8.3 节 |
| VISTA 碰撞验证 | 自碰撞不扩写为场景接触/碰撞完整验证 | 第 6.4 节 |
| UMI-3D 三维信息 | 区分采集传感器与策略实际输入 | 第 6.3 节 |
| MASt3R-SLAM 鱼眼 | 放宽内参假设不等于没有镜头畸变域差距 | 第 5.3 节 |
| VGGT-SLAM 速度 | 保留硬件/子地图条件，不当作高频控制延迟 | 第 5.3 节 |
| ScaRF-SLAM 比较 | 保留标定与尺度对齐条件，不宣称经典方法普遍胜出 | 第 5.4 节 |
| 扰动研究的比较对象 | 是 ORB-SLAM2，未写成 ORB-SLAM3；workshop 与主会区分 | 第 4.4 节及参考文献 |
| TacUMI 任务指标 | 分割性能不当作策略成功率 | 第 6.5 节 |
| UMI-Bridge / BRIDGE | 独立登记，不混为同一篇 | 第 6.7 节 |
| ConceptGraphs 标识 | 检索中发现一个不相关医学论文标识，排除后改用 2309.16650 | S27 |
| TUM 数据参考范围 | 使用成功访问的 TUM-VI 官方说明；未采用访问失败页面支持结论 | 第 11.2 节 |
| 最新标题/版本 | LingBot-Map、SURE-Map 使用本次核验的修订版标题 | S14、S15 |

## 7. 跨论文张力：候选综合，不是已裁定的争论

下面的“张力”指需要通过共同实验连接的主张，不假设论文作者相互反驳。所有候选均为 **scholar_confirmation: pending**，未获人类专家确认。

| 候选 | 对照来源 | 表面张力 | 条件化解释 | 优先实验 |
|---|---|---|---|---|
| T01 | RDT2 U03 / HiFi-UMI U04 | 规模与保真哪个更决定成败 | 改变的变量不同，不能跨论文归因 | 固定模型与任务，交叉改变数据量和噪声 |
| T02 | HiFi-UMI U04 / BRIDGE U24、WT-UMI U16 | 手持数据能否替代真实控制数据 | 后训练任务范围与接触控制语义不同 | 分自由空间/接触阶段比较监督来源 |
| T03 | MASt3R-SLAM S08 / ScaRF-SLAM S17 | 学习几何是否替代经典估计 | 标定、相机模型、输入传感器与任务不同 | 相同 UMI 原始数据和对齐协议比较 |
| T04 | CUT3R S13、LingBot-Map S14 / SURE-Map S15 | 持续记忆是否足够保证长时一致 | 记忆容量和纠错约束是不同能力 | 长序列、重复访问、尺度与漂移测试 |
| T05 | MASt3R-SLAM S08 / 相机选择 U25 | 未标定几何与鱼眼策略优势能否直接相加 | 策略视场收益与定位模型域适应不同 | 保持策略图像，分别改变定位输入/模型 |
| T06 | FeasibleCap U06、VISTA U07 / HIL-UMI U08 | 可执行/高质量数据是否就是最有价值数据 | 物理有效性与策略覆盖是两条轴 | 可信度加权与策略价值采样的对照 |

结构化版本见 [cross-paper-tensions.yaml](./cross-paper-tensions.yaml)。

## 8. 反向审查记录

本次为同一研究流程内的批判性自查，没有独立审稿人或外部交叉模型审查。

- 检查是否把多个任务、硬件和模型上的成功率拼成排名：未这样比较。
- 检查是否把预印本、项目宣称与独立复现混同：主报告明确区分。
- 检查数学推导中的假设：固定左乘刚体变换抵消；时变/尺度/外参不作同样保证；协方差推导注明局部线性化。
- 检查时间单位：示意时延计算采用 m/s 与 s，结果换算为 mm；非论文实测。
- 检查“无机器人”范围：采集、预训练、后训练、控制校正与评估分开。
- 检查理论推断与实验事实：工程选型和实验设计均标为建议。
- 检查参考材料成熟度：摘要级论文不承担精密接口实现和横向量化结论。
- 检查可访问性：引用可核验源；不声称数据、权重和硬件均已实际获取。

## 9. 尚待核验，按优先级排序

1. 用目标任务的真实原始数据比较现有定位与学习几何；公开基准不能代替近场腕部数据。
2. 独立评估局部位姿、时间、双臂和接触误差对策略的敏感性。
3. 精读并运行较新 HIL-UMI、SURE-Map、UMI-Bridge 的实验与实现；当前仅作前沿方向引用。
4. 针对每个候选核查代码提交、数据公开子集、许可证、模型权重和硬件可获得性。
5. 对候选跨论文张力进行专家确认；完成更严格数据库检索后才能升级为系统性证据综合。
6. 如果要形成论文研究问题，另做新颖性检索和先验实验，不能把本文提出的问题直接当作尚无人研究的空白。

## 10. 交付文件

- 主报告：umi-slam-deep-research-2026-10.md。
- 本记录：evidence-and-search-log.md。
- 文献登记：sources.json。
- 跨论文张力候选：cross-paper-tensions.yaml。

没有额外创建 PDF、Word 或演示稿；没有修改既有报告。

