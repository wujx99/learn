# Camera-based 3D Occupancy Prediction

从多相机图像预测稠密三维空间中的占用与语义，包括显式深度提升、query-based 反向投影、时序融合和体素监督。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| FB-OCC: 3D Occupancy Prediction based on Forward-Backward View Transformation | 2023 | read | [note](2023-fb-occ/README.md) |

## 推荐顺序

先区分 3D box detection 与 voxel occupancy 的输出粒度，再沿 forward depth lift、semantic BEV query、backward image sampling、3D fusion 的顺序阅读 FB-OCC；实验部分务必把单模型架构、联合预训练、TTA 与七模型挑战赛集成拆开。
