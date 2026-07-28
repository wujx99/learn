---
title: Query-based BEV 的时序融合方法
topic: query-based-bev-temporal-fusion
topics:
  - query-based-bev-temporal-fusion
  - autonomous-driving-coordinate-transforms
framework: framework-agnostic
status: learning
updated: 2026-07-27
---

# Query-based BEV 的时序融合方法

## 学习目标

读完本文后，应当能够：

1. 判断一个方法保存的是历史图像特征、稠密 BEV queries，还是稀疏 object/instance queries；
2. 用“状态、对齐、读取、更新、传播”五个步骤拆解时序模块；
3. 解释 BEVFormer、PETRv2、Sparse4D 和 StreamPETR 的时序机制为什么不同；
4. 根据任务稠密性、延迟、显存和长时稳定性选择方案；
5. 识别训练序列长度、坐标系和失效重置等常见工程问题。

## 前置知识与范围

需要了解 Transformer 的 query/key/value、deformable attention、自动驾驶坐标变换和多视角 3D 检测。坐标变换可先参考[自动驾驶坐标变换与鱼眼相机几何](../autonomous-driving-coordinate-transforms/README.md)。

本文以 camera-only、在线 3D 感知为主。“Query-based BEV”在文献中有两种常见含义：

- **BEV query**：规则 BEV 网格上的稠密 queries，代表空间位置；
- **object/instance query**：数量有限的稀疏 queries，代表目标假设。

二者的时序状态、成本和适用任务都不同，不能只因都使用 Transformer decoder 就归为同一机制。

## 核心心智模型：时序融合就是维护一个隐状态

把所有方案统一写成：

$$
\tilde{M}_{t-1}=A(M_{t-1},T_{t\leftarrow t-1},\Delta t),
$$

$$
Y_t=R(Q_t,F_t,\tilde{M}_{t-1}),\qquad
M_t=U(Y_t,\tilde{M}_{t-1}).
$$

其中：

- $F_t$：当前帧多相机图像特征；
- $M_{t-1}$：历史状态，可以是图像特征、BEV feature 或 instance queries；
- $A$：利用 ego pose、目标速度或时间间隔完成对齐；
- $R$：用 attention 或 feature sampling 读取当前帧与历史；
- $U$：选择、融合并写回下一帧状态。

分析一个新方法时，依次问五个问题即可：**存什么、怎么对齐、怎么读、怎么更新、何时清空**。

## 方法版图

| 类别 | 历史状态 | 代表方法 | 主要对齐 | 融合位置 | 典型特点 |
| --- | --- | --- | --- | --- | --- |
| 历史 perspective feature | 前帧图像特征及其 3D PE | PETRv2 | 把历史 3D 坐标变到当前 ego 系 | decoder cross-attention 的 K/V | 实现直接，但成本随显式帧数增长 |
| 稠密 BEV recurrence | 上一帧完整 BEV feature | BEVFormer | ego-motion 对齐 BEV reference points | encoder temporal self-attention | 对稠密任务友好，状态规模为 $H_{bev}W_{bev}$ |
| 跨时刻稀疏采样 | 多帧 image feature pyramid | Sparse4D v1 | 3D anchor/keypoints 投影到每一时刻 | query 对多帧图像特征采样后融合 | 无稠密 BEV，但显式访问 $T$ 帧，成本约随 $T$ 增长 |
| object-query memory | 历史 object queries、reference points、pose/time | StreamPETR | ego 与 object motion 编码、reference point 变换 | decoder 中 query-to-memory attention | 长序列、低额外开销，但只保存对象级信息 |
| recurrent instance bank | instance feature 与结构化 3D anchor | Sparse4Dv2 | 对 anchor 做坐标变换，特征与几何解耦 | 当前 instance 与 temporal instance 交互 | 推理时序成本由 $O(T)$ 降到 $O(1)$ |

这里的 $O(1)$ 是指每帧相对历史长度的增量成本近似固定，不表示模型总计算量为常数。

## 1. 历史图像特征：PETRv2

[PETRv2](https://arxiv.org/abs/2206.01256) 不传播 object query 本身。它缓存上一帧的 2D image features，并通过 3D position embedding 把历史位置转换到当前帧坐标系。对齐后的历史特征与当前特征拼接，作为 object queries cross-attention 的 key/value。

可以概括为：

```text
previous image features + transformed historical 3D PE ─┐
                                                          ├─ decoder K/V ← object queries
current image features  + current 3D PE ──────────────────┘
```

它属于“**query 读取时序特征**”，不是“历史 query 传播”。优点是保持 PETR 的 perspective-space 表示；缺点是若直接扩展更多历史帧，K/V token 数、显存和 cross-attention 成本都会增加。

## 2. 稠密 BEV query 递归：BEVFormer

[BEVFormer](https://arxiv.org/abs/2203.17270) 使用规则网格 BEV queries。上一帧输出的 BEV feature $B_{t-1}$ 被保存为递归状态；当前帧先做 temporal self-attention，再通过 spatial cross-attention 从多相机图像读取信息：

$$
Q_t^{temp}=\operatorname{TSA}(Q_t,A(B_{t-1})),
$$

$$
B_t=\operatorname{SCA}(Q_t^{temp},F_t).
$$

TSA 并非先生成一张完美 warp 后的历史 BEV 再普通相加，而是根据 ego-motion 移动 reference points，再用 deformable attention 在历史和当前 BEV 上稀疏采样。官方论文明确描述了先按 ego-motion 对齐历史 BEV，再递归融合；[官方实现](https://github.com/fundamentalvision/BEVFormer)也保留 `prev_bev` 作为流式状态。

优势：

- 每个 BEV cell 都有历史，适合 map segmentation、occupancy 等稠密任务；
- 递归状态使推理成本不随已运行的总帧数线性增长；
- 遮挡区域仍可能从历史 BEV 中恢复信息。

代价：

- 状态量为 $H_{bev}W_{bev}C$，高分辨率 BEV 的显存和 attention 成本较高；
- 动态物体不能仅靠 ego-motion 完全对齐，需由 attention 隐式学习残差运动；
- 递归误差、错误位姿和长期未重置的状态可能累积。

## 3. 对多帧图像做稀疏 4D 采样：Sparse4D v1

[Sparse4D v1](https://arxiv.org/abs/2211.10581) 为每个 3D anchor 生成多个 4D keypoints，把它们投影到不同 camera、scale 和 timestamp 的 image feature 上，再分层融合 view、scale、time 和 keypoint 维度。

它绕过了稠密 BEV：

$$
f_i=\operatorname{Fuse}\left(
\left\{\operatorname{Sample}(F_{\tau,v,l},\pi_{\tau,v}(p_{i,k}))\right\}
_{\tau,v,l,k}\right).
$$

其中 $i$ 是 instance，$k$ 是 3D keypoint，$\tau$ 是时间，$v$ 是相机，$l$ 是特征层级。

这种方案的核心是“**当前 query/anchor 回看历史 image features**”。空间访问稀疏、几何约束强，但需要保存并访问多个时间戳的 image feature pyramid，计算与存储随窗口长度 $T$ 增长。

## 4. Object query memory：StreamPETR

[StreamPETR](https://arxiv.org/abs/2303.11926) 将前景分数较高的历史 object queries 连同 reference points、时间和 pose 信息写入 memory queue。当前 queries 在 decoder 中读取这组 memory，同时继续读取当前图像特征。

```text
history object queries ─ ego/object motion encoding ─┐
                                                      ├─ temporal decoder ─ top-k ─ memory queue
newly initialized queries ────────────────────────────┘
current image features ───────────────────────────────┘
```

Motion-Aware Layer Normalization 把 ego motion、object motion 与时间间隔注入 query feature 的调制过程，以缓解只变换 reference point 而特征语义未随运动更新的问题。

优势是历史状态从稠密 $H_{bev}W_{bev}$ 降为有限个 object queries，适合在线 3D detection/tracking；长时历史的额外开销较小。局限是没有对象 query 承载的背景、路面和自由空间信息容易丢失，因此它不能直接替代面向 occupancy 或地图的稠密 BEV memory。

## 5. Recurrent instance bank：Sparse4Dv2

[Sparse4Dv2](https://arxiv.org/abs/2305.14018) 把 Sparse4D v1 的显式多帧采样改为递归 temporal fusion。历史只传递少量 instance features 和结构化 anchors；anchors 通过位姿变换进入当前坐标系，feature 与 anchor 的几何变换解耦。

$$
(f_{t-1}^{temp},a_{t-1}^{temp})
\xrightarrow{\text{pose transform}}
(f_{t-1}^{temp},\tilde a_{t-1}^{temp})
\xrightarrow{\text{instance interaction}}
(f_t,a_t).
$$

因此每帧不必重新访问全部历史 image features，历史长度相关的融合复杂度从 $O(T)$ 变为近似 $O(1)$，同时仍能通过递归累积长时信息。与 StreamPETR 相同，它本质上是 instance-level state；二者差异主要在 instance 的几何参数化、图像采样方式、memory 更新和 decoder 结构，而不是“一个有时序、一个没有”。

## 两条正交的分类轴

只按论文名记忆容易混乱，更实用的是沿两条轴定位方法。

### 轴一：状态粒度

$$
\text{image feature}
\rightarrow \text{dense BEV}
\rightarrow \text{sparse instance}.
$$

- 越靠左/中，保留的场景信息越完整，适合多任务和稠密预测；
- 越靠右，状态越紧凑，适合 detection/tracking 和低延迟部署；
- 稀疏状态的上限受 query selection、目标召回和 memory eviction 影响。

### 轴二：时间访问模式

- **显式窗口**：一次读取 $T$ 帧，如 PETRv2 扩展、多帧 Sparse4D v1；并行训练自然，但成本随 $T$ 增长。
- **递归状态**：只读取上一时刻的压缩状态，如 BEVFormer、StreamPETR、Sparse4Dv2；流式高效，但更易出现训练—推理时长差异和误差累积。
- **有限 memory bank**：保留固定数量的关键 token/帧，是窗口与严格递归之间的折中。

## 对齐不是一个单独步骤

时序对齐至少包含三层：

1. **静态场景的 ego-motion 对齐**：

   $$
   {}^{E_t}\mathbf p={}^{E_t}\mathbf T_{E_{t-1}}{}^{E_{t-1}}\mathbf p.
   $$

2. **动态目标补偿**：可用预测速度作显式外推，或让 attention/MLN 学习残差；仅做 ego warp 无法对齐运动车辆。
3. **feature 语义对齐**：几何坐标变换后，feature embedding 本身未必已表达新的朝向、时间和运动状态，因此常需 time embedding、pose encoding、motion-conditioned normalization 或 attention 更新。

这解释了为什么“我已经 warp 过了”仍不等于完成时序融合。

## 如何选型

| 需求 | 更自然的起点 | 原因 |
| --- | --- | --- |
| 检测 + BEV map/occupancy 多任务 | BEVFormer 类 dense BEV recurrence | 历史覆盖整个 BEV 空间 |
| 纯 3D detection，追求低延迟 | StreamPETR / Sparse4Dv2 类 | 状态与交互均为 instance-level |
| 希望少改动单帧 perspective decoder | PETRv2 类 | 主要扩展历史 feature 与 3D PE |
| 短窗口内充分利用原始历史图像证据 | Sparse4D v1 类 | query 可直接从多个 timestamp 重新采样 |
| 超长在线序列 | 固定大小 recurrent state 或 memory bank | 单帧增量成本不随总时长增长 |

不存在无条件最优方案。稠密 BEV memory 用计算换场景完整性，object memory 用召回和背景信息换效率；显式窗口保留原始证据，递归状态则用压缩和误差累积换流式性能。

## 面向规划的 40–60 帧 dense BEV

如果传感器频率为 10 Hz，40–60 帧对应约 4–6 秒。规划需要的是这个时间跨度内的**充分状态**，而不一定是 40–60 份等分辨率、可被任意查询的历史 feature。

### 静态与动态需要不同的时间分辨率

自动驾驶场景不能严格只分成“OD 与静态”两项，但可以先用三类状态理解：

| 状态 | 长历史的作用 | 适合的时间分辨率 |
| --- | --- | --- |
| 静态结构：道路、边界、固定障碍物 | 补全遮挡、扩大已观测区域、稳定几何与语义 | 长寿命、低频更新；旧观测仍有价值 |
| 动态交通参与者 | 估计速度、加速度、意图和遮挡后的存在概率 | 近期高频；越旧的精确位置越快失效 |
| 场景状态：信号灯、临时施工、可通行性 | 既可能长期稳定，也可能离散切换 | 需显式置信度、年龄和状态转移 |

因此，不是只有动态障碍物需要 40–60 帧，也不是所有信息都需要保存 40–60 帧。更准确的说法是：

- 规划可能需要 4–6 秒的上下文；
- 动态分支需要最近若干帧的高频细节，以及更长时间尺度的压缩运动状态；
- 静态分支需要长时累计结果，但通常不需要保留每一帧；
- 对所有历史帧做全量 dense attention 是一种数据结构选择，不是规划需求本身。

### Instance query 是否必须有检测监督

Instance query 本身只是一个槽位，并不天然等于动态障碍物。但 StreamPETR、Sparse4D 等检测式方法通常通过分类分数、3D box 回归和 top-k selection 让 query 获得“一个 query 对应一个对象”的语义，其 memory 更新也依赖这套检测监督。

只用 occupancy loss 并非数学上不能学习 instance slots，但监督存在欠定性：多个 query 可以共同解释同一片 occupancy，一个 query 也可以覆盖多个对象；query 的置换、合并和拆分通常不影响最终 dense loss。因此纯 occupancy loss 很难稳定地产生跨帧一致的 object identity。

如果只有 occupancy 监督，有三条路径：

1. **不强求 instance 化**：维护 dense scene memory，再增加 motion/uncertainty-aware update。这是风险最低的起点。
2. **弱 instance 化**：从 occupancy connected components、语义类别、ego-warp 后的变化区域或预测 flow 产生 pseudo instances，再监督 slot assignment 和跨帧一致性。
3. **slot-based occupancy**：让多个 slots 解码并合成 occupancy，同时加入 exclusivity、coverage、temporal consistency、slot persistence 和 motion prediction loss。它不需要 3D box 标注，但已经超出“只有一个最终 occupancy CE loss”。

如果 occupancy 标注包含 semantic class、future occupancy 或 occupancy flow，弱 instance 化更可行；如果只有当前帧 binary occupancy，直接采用检测式 instance bank 的风险很高。

### Dense recurrent memory 的主要问题

递归更新

$$
H_t=U(H_{t-1},B_t)
$$

能把推理状态从 $O(T H W C)$ 降为 $O(HWC)$，但 40–60 步会暴露以下问题：

- **信息瓶颈与覆盖写**：同一个 $C$ 维 cell 必须同时保存静态结构、动态状态、遮挡证据和不确定性；新观测可能覆盖仍有用的旧证据。
- **静动态对齐冲突**：ego warp 可对齐静态世界，却不能对齐运动目标；统一 warp 后，动态部分会拖影。
- **递归漂移**：pose 噪声、错误 occupancy 和错误更新会被下一步继续读取，形成 ghost occupancy。
- **历史年龄不可辨**：若没有 time/age embedding，网络不易判断某个 feature 来自刚刚观测还是数秒前。
- **训练与推理不一致**：训练只 unroll 4–8 帧而推理 60 帧，会出现 feature norm、置信度和误差分布漂移。
- **长程梯度不足**：完整 60 帧 BPTT 显存高；截断 BPTT 又使早期状态很难收到规划或 occupancy loss 的有效梯度。
- **场景切换与失效恢复**：必须定义 reset、位姿跳变、掉帧和长时间无观测时的衰减策略。

[VideoBEV](https://arxiv.org/abs/2303.05970) 说明简单 recurrent BEV 可以有效利用长历史，但也指出 BEVFormer 式递归不一定自然地从更多帧持续获益。较新的 [OnlineBEV](https://arxiv.org/abs/2507.08644) 将长时增益受限归因于动态目标造成的对齐困难，并额外学习 motion-guided alignment。

### Deformable attention 能解决什么，不能解决什么

Deformable attention 将每个 query 的读取从全局 $HW$ 个位置降到 $K$ 个采样点：

$$
y_i=\sum_{k=1}^{K}a_{ik}H\bigl(p_i+\Delta p_{ik}\bigr).
$$

它适合修正小范围错位和按内容选择局部历史，但不是长时 memory 的完整答案：

- 若 dense queries 对 60 份历史逐一采样，复杂度仍为 $O(THWK)$；只是空间读取变稀疏，时间维没有消失。
- 若只从一个 recurrent state 采样，早期历史必须已被压进该 state；deformable attention 无法恢复被覆盖的信息。
- 大位移、快速运动、转弯、掉帧或错误 pose 可能使真实对应点落到采样范围之外。
- 每个 BEV cell 都预测 offset 时，空区域也产生计算；dense query 数量大时成本仍明显。
- 同一 cell 中混合静态背景和动态目标时，一组 offsets 未必能同时对齐两者。
- 长期不可见区域需要显式 age/confidence，否则模型可能把陈旧记忆当作当前事实。

所以 deformable attention 更适合作为**局部读取与残差对齐算子**，不应单独承担历史压缩、遗忘和不确定性管理。

### 推荐的分层时空 memory

对“dense BEV query + occupancy-only + 40–60 帧”更合适的起点是三层结构：

```text
当前 dense BEV B_t
  ├─ 短期高频 memory：最近 4–8 帧，高分辨率，motion-aware deformable fusion
  ├─ 中期 recurrent state：约 1–2 秒，门控更新，保存 occupancy dynamics
  └─ 长期静态 memory：覆盖 4–6 秒或更久，低频/低分辨率，confidence-age update
                         ↓
              gated multi-scale fusion
                         ↓
                  occupancy + planning
```

一种具体参数化为：

$$
H_t^{short}=\operatorname{Fuse}(B_{t-s+1:t}),
$$

$$
H_t^{dyn}=\operatorname{GRU}_{bev}
\left(\operatorname{MotionAlign}(H_{t-1}^{dyn},B_{t-1:t}),B_t\right),
$$

$$
H_t^{static}=g_t\odot B_t^{static}
+(1-g_t)\odot\operatorname{EgoWarp}(H_{t-1}^{static}),
$$

再令当前 dense queries 分别读取三个 memory，并用 uncertainty、age 和 occupancy change 预测门控权重。

实践上可以先做以下最小版本：

- 最近 4 帧保留全分辨率 BEV；
- 第 5–20 帧压成一个 motion-aware recurrent state；
- 第 20–60 帧只保留下采样后的 static/slow memory，并记录 `age` 与 `confidence`；
- 短期分支使用 deformable attention，中长期分支使用 gated convolution 或少量 cross-attention；
- 训练先 unroll 8–12 帧，并随机注入由更长序列预计算或 stop-gradient 得到的历史状态，而不是一开始完整反传 60 帧。

在只有 occupancy 监督时，可增加不需要 box 标注的辅助目标：ego-warp 后的 temporal consistency、future occupancy、occupancy flow、visibility/uncertainty、随机遮帧重建和 memory age calibration。这些信号比强行引入 object queries 更贴合现有标注。

## 工程风险与排查信号

### 坐标变换方向写反

- **症状**：静态背景随 ego 运动向错误方向漂移，转弯时尤其明显。
- **检查**：用一个静止世界点验证 ${}^{E_t}\mathbf T_{E_{t-1}}$；不要只看矩阵变量名。

### 只平移 BEV，没有正确处理旋转

- **症状**：直行正常，转弯时历史轮廓出现重影或圆弧状错位。
- **修复**：确认 reference point、BEV tensor rotation 与坐标原点约定一致。

### 动态目标被当成静态场景 warp

- **症状**：静态地图对齐良好，但车辆拖影，速度估计不稳。
- **修复**：增加 velocity-aware reference propagation、motion encoding，或让 object query 单独建模。

### 训练只看很短序列，推理却长期递归

- **症状**：离线短 clip 指标正常，连续运行后置信度、位置或 feature norm 漂移。
- **修复**：训练中增加连续帧长度、随机截断状态、状态 dropout/噪声，并做长序列 replay 测试。

### 场景切换后没有重置 memory

- **症状**：新 scene 开头出现上一场景目标或幽灵检测。
- **修复**：以 scene token、时间倒退、位姿跳变或数据流重启为条件清空状态。

### 把 top-k memory 当成可靠 tracking

- **症状**：遮挡后 query identity 交换，历史错误被不断强化。
- **修复**：检查 selection/eviction 策略、query 去重、置信度衰减及 track consistency；检测 query propagation 不自动等价于显式数据关联。

## 自测与练习

1. 为什么 PETRv2 虽然使用 object queries，却不应被归类为 object-query propagation？
2. BEVFormer 已使用 ego-motion 对齐，为什么动态目标仍可能产生拖影？
3. 将 Sparse4D v1 的历史窗口从 2 帧增到 8 帧，会增加哪些张量的存储和访问成本？
4. 如果任务从 3D detection 扩展到 occupancy，为什么只传播 top-k object queries 往往不够？
5. 为一个 recurrent instance bank 设计三个必须触发 reset 的条件。

小练习：选取一个模型的推理代码，把 state 的 tensor shape、坐标系、写入规则、淘汰规则和 reset 条件列成表；若其中任何一项无法回答，说明对其时序机制的理解还不完整。

## 相关论文与代码

- BEVFormer：[论文](https://arxiv.org/abs/2203.17270)；[官方代码](https://github.com/fundamentalvision/BEVFormer)
- PETRv2：[论文](https://arxiv.org/abs/2206.01256)；[官方代码](https://github.com/megvii-research/PETR)
- Sparse4D v1：[论文](https://arxiv.org/abs/2211.10581)
- Sparse4Dv2：[论文](https://arxiv.org/abs/2305.14018)；[官方代码](https://github.com/HorizonRobotics/Sparse4D)
- StreamPETR：[论文](https://arxiv.org/abs/2303.11926)；[官方代码](https://github.com/exiawsh/StreamPETR)

## 下一步

建议按以下顺序继续：

1. 画出 BEVFormer TSA 和 StreamPETR object memory 的逐 tensor 数据流；
2. 对照官方代码定位 pose transform、memory update 和 reset；
3. 在同一组符号下比较 dense BEV 与 sparse instance 的显存和复杂度；
4. 再扩展到 occupancy、online mapping 或端到端规划中的混合 memory。
