---
title: 自动驾驶静态地图与 3D 目标检测的监督构造
topic: autonomous-driving-map-od-supervision
topics:
  - autonomous-driving-map-od-supervision
  - autonomous-driving-coordinate-transforms
  - query-based-bev-temporal-fusion
framework: framework-agnostic
status: learning
updated: 2026-08-13
---

# 自动驾驶静态地图与 3D 目标检测的监督构造

## 问题

自动驾驶领域中，静态地图和 OD（这里指 3D Object Detection）的监督是怎么做的？有哪些比较好的学习资料？

## 核心结论

理解“监督”时应当区分两层：

1. **标签生产**：地图和 3D box 的真值是怎样采集、自动生成并校验的；
2. **训练目标构造**：数据集标签如何转换为网络直接优化的 target 和 loss。

地图监督和 OD 监督的主要差别不在 backbone，而在输出对象的数学表示：OD 标签天然是一组有限的 3D box；静态地图既可以表示成稠密 BEV mask，也可以表示成折线、多边形或拓扑图。矢量地图因此还必须处理实例匹配、点序等价性、采样密度和拓扑关系。

## 1. 静态地图标签如何生产

工业上的地图标签通常不是逐帧在相机图像上绘制出来的，而是先建立全局地图：

```text
多车、多趟采集
  → LiDAR、相机、GNSS/IMU 标定与定位
  → 多帧点云拼接
  → 动态物体过滤
  → 提取车道线、路沿、停止线等元素
  → 矢量化与拓扑连接
  → 人工校验和规则 QA
  → 全局 HD Map
```

实际生产通常采用“自动预标注 + 人工修正 + 规则校验”，而不是完全手工标注。地图自动标注系统可参考 [VMA](https://arxiv.org/abs/2304.09807) 和 [THMA](https://arxiv.org/abs/2212.11123)。

训练在线地图模型时，根据当前车辆位姿从全局地图中裁剪局部区域。若采用列向量和左乘变换，地图点从全局坐标系转换到当前 ego 坐标系可写为：

$$
{}^{E}\tilde{\mathbf p}
= {}^{E}\mathbf T_W\,{}^{W}\tilde{\mathbf p}.
$$

随后进行感兴趣区域裁剪、几何 clipping、类别映射和采样，生成当前帧的地图 GT。所有针对传感器数据的 BEV 旋转、翻转和缩放增强，都必须同步作用在地图标签上。

适合理解公开数据格式的资料包括：

- [nuScenes Map Expansion 教程](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html)：适合学习全局地图如何按照 ego pose 查询和裁剪为局部地图；
- [Argoverse 2 HD Map 文档](https://argoverse.github.io/user-guide/api/hd_maps.html)：包含 3D lane boundary、lane graph、drivable area 和 pedestrian crossing；
- [Waymo Open Dataset](https://waymo.com/open/)：可用于理解 3D box、轨迹和地图在真实自动驾驶数据集中的组织方式。

## 2. 静态地图在网络中的监督方式

### 2.1 BEV 栅格监督

首先把地图元素栅格化到固定范围、固定分辨率的 BEV 网格：

```text
lane divider    → channel 0
road boundary   → channel 1
ped crossing    → channel 2
drivable area   → channel 3
```

模型输出 $H\times W\times C$ 的语义概率图，常用 BCE、Focal、Dice 或 IoU loss。栅格监督实现简单且训练稳定，但会损失地图实例身份、精确矢量几何和连接关系，而且通常还需要后处理才能供规划模块使用。

[HDMapNet](https://arxiv.org/abs/2107.06307) 是理解 BEV 语义地图学习、矢量化和评测方式的合适起点。

### 2.2 矢量地图监督

每个地图实例被表示为一条折线或多边形，并重采样为固定数量的控制点：

$$
P=\{(x_1,y_1),\ldots,(x_N,y_N)\}.
$$

DETR 风格的地图模型一般输出固定数量的 map queries，训练包含两层匹配：

1. 在预测实例与 GT 实例之间执行 Hungarian matching；
2. 在匹配实例内部寻找预测点与 GT 点的等价点序。

地图点序并不唯一：开放折线可以正向或反向表示；闭合多边形可以从任意顶点开始，并可以顺时针或逆时针排列。因此点监督可以概括为：

$$
L_{\mathrm{pts}}
=\min_{\pi\in\Pi}
\sum_i\left\|\hat{\mathbf p}_i-\mathbf p_{\pi(i)}\right\|_1,
$$

其中 $\Pi$ 是该地图实例所有允许的等价排列。

常见损失包括：

- 地图实例分类的 Focal loss；
- 控制点坐标的 L1 loss；
- 相邻线段方向的 cosine loss；
- 可选的 bbox、IoU 或 Chamfer loss；
- 栅格分割或点级预测等 dense auxiliary supervision。

推荐阅读：

- [MapTR](https://arxiv.org/abs/2208.14437)：重点关注 permutation-equivalent representation 和 hierarchical bipartite matching；
- [MapTR 官方代码](https://github.com/hustvl/MapTR)：重点追踪 dataset pipeline、GT vectorization、assigner 和 loss；
- [MapTRv2](https://arxiv.org/abs/2308.05736)：在一对一匹配之外加入 one-to-many matching 和 dense supervision，以缓解监督稀疏和收敛慢的问题；
- [VectorMapNet](https://proceedings.mlr.press/v202/liu23ax.html)：另一种端到端 polyline 生成方案。

### 2.3 拓扑监督

如果任务还需要预测“哪条车道连接哪条车道”或者“哪个交通灯控制哪条车道”，GT 会进一步包含 lane-lane、lane-traffic 或 element-element 的邻接矩阵。

模型通常对候选实例对预测连接概率，并使用 BCE 或 Focal loss 监督图中的边。这里必须注意：拓扑 GT 依赖稳定的实例匹配；如果几何实例本身匹配错误，拓扑 loss 也会被错误地分配。

相关基准可参考 [OpenLane-V2](https://arxiv.org/abs/2304.10440)。

## 3. 3D Object Detection 的监督方式

3D OD 的单个 GT box 通常可以表示为：

$$
b=(x,y,z,w,l,h,\theta,v_x,v_y,c),
$$

其中包含三维中心、尺寸、yaw、速度和类别。不同数据集对属性、可见性、检测范围和无 LiDAR 点物体的过滤规则可能不同，不能只对齐 box 字段而忽略标注协议。

### 3.1 Center-based 监督

以 CenterPoint 为代表，target 构造过程为：

1. 将物体中心投影到 BEV feature map；
2. 在对应类别通道上绘制 Gaussian heatmap；
3. 使用 Focal loss 监督中心 heatmap；
4. 仅在 GT 中心位置监督 offset、$z$、长宽高、朝向和速度；
5. 回归分支通常使用 L1 或 Smooth-L1 loss。

朝向常写为 $(\sin\theta,\cos\theta)$，以避免角度在 $-\pi$ 和 $\pi$ 附近不连续。详细 target generation 可参考 [CenterPoint](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html)。

### 3.2 Query-based 监督

DETR3D、BEVFormer 一类模型使用固定数量的 object queries：

```text
预测 object queries
        ↕ Hungarian matching
GT 3D boxes
```

匹配 cost 通常由类别误差与 box 参数误差共同组成。匹配完成后，分别计算分类、中心、尺寸、朝向和速度等损失；未匹配的 query 被监督为 no-object。

推荐阅读：

- [DETR3D](https://tsinghua-mars-lab.github.io/detr3d/)：较直接的 camera-only、query-based 3D 检测框架；
- [BEVFormer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690001.pdf)：适合理解多相机和历史图像如何形成 BEV feature，并同时支持 OD 与 map head；
- [nuScenes Detection 官方说明](https://www.nuscenes.org/object-detection)：包含标签字段、类别、过滤规则以及 NDS、mAP 的定义。

## 4. OD 标签在工业中如何自动生产

OD 自动标注通常使用车端在线模型无法使用的完整前后文和较大计算量：

```text
整段 LiDAR/视频
  → 多帧高精度检测器
  → 跨帧 tracking
  → 按 object track 聚合点云
  → object-centric box refinement
  → 时序一致性检查
  → 人工抽检或返修
```

离线标注模型不要求实时，也不要求因果，因此能够利用未来帧、更多传感器信息和长时间窗口，生成质量高于车端实时 detector 的伪标签。

Waymo 的 [Offboard 3D Object Detection from Point Cloud Sequences](https://openaccess.thecvf.com/content/CVPR2021/html/Qi_Offboard_3D_Object_Detection_From_Point_Cloud_Sequences_CVPR_2021_paper.html) 是理解 3D OD auto-labeling 的代表性资料。

## 5. 容易忽略的问题

### 5.1 位姿和标定误差就是标签误差

地图 GT 依赖 global-to-ego 位姿，OD 的多帧融合依赖 ego motion 和传感器外参。位姿、时间同步或外参有误时，即使原始地图和 3D box 完全正确，转换到训练帧后也会成为带系统偏差的标签。

### 5.2 地图 GT 可能包含当前传感器不可见的内容

从全局 HD Map 裁剪出的局部 GT 可能包含被车辆遮挡、磨损或当前相机视野外的地图元素。此时模型学习的不只是逐像素识别，还会利用场景先验补全局部地图。需要明确任务究竟监督完整局部地图，还是只监督当前可观测区域。

### 5.3 数据增强必须同步作用于几何标签

相机图像增强、BEV 旋转缩放、时序对齐和坐标归一化之间必须保持一致。只增强输入而没有变换 3D box、map points、velocity 或历史 ego pose，会产生隐蔽且严重的训练错误。

### 5.4 训练匹配方式与评测指标不是同一回事

训练可以使用 L1、Focal、IoU 或 Hungarian cost，评测则可能使用中心距离 AP、Chamfer AP、拓扑分数或 NDS。设计 loss 时应理解它和最终指标之间的代理关系，而不能把两者视为完全等价。

## 6. 推荐学习顺序

1. 使用 nuScenes devkit 同时画出某个 sample 的相机图像、3D box 和局部地图；
2. 阅读 HDMapNet，理解 raster map supervision；
3. 阅读 MapTR，并在代码中追踪一条 GT polyline 如何变成采样点、等价点序和 matching target；
4. 阅读 CenterPoint，自己实现一次 Gaussian heatmap target generation；
5. 阅读 DETR3D 和 BEVFormer，对比 center-based 与 query-based OD；
6. 最后阅读 [UniAD](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)，理解 map、tracking、motion、occupancy 和 planning 的联合监督。

建议完成的最小实验，是在 nuScenes mini 上同时可视化：

```text
多相机图像
3D box GT
全局地图裁剪结果
BEV raster mask
MapTR polyline 采样点
```

把这五种数据放在同一个 sample 和统一坐标系下检查，是打通标签生产、坐标变换与训练监督最有效的方法。

## 相关知识

- [自动驾驶坐标变换与鱼眼相机几何](../autonomous-driving-coordinate-transforms/README.md)
- [Query-based BEV 时序融合](../query-based-bev-temporal-fusion/README.md)

