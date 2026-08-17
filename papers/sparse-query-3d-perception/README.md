# Sparse Query 3D Perception

面向多相机 3D 感知的稀疏实例表征、几何采样、递归时间融合与检测—跟踪一体化。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion | 2022 | read | [note](2022-sparse4d/README.md) |
| Sparse4D v2: Recurrent Temporal Fusion with Sparse Model | 2023 | read | [note](2023-sparse4d-v2/README.md) |
| Sparse4D v3: Advancing End-to-End 3D Detection and Tracking | 2023 | read | [note](2023-sparse4d-v3/README.md) |
| Sparse4D: Sparse-based End-to-End Multi-Sensor Temporal Perception | 2026 | read | [note](2026-sparse4d-tpami/README.md) |

## 推荐顺序

按 v1 → v2 → v3 阅读最清楚：v1 回答“如何围绕稀疏 3D anchor 从多视角、多尺度、多帧图像取证”；v2 把随历史长度线性增长的多帧采样改成递归实例记忆；v3 保留 v2 的推理结构，重点修复 one-to-one 训练、置信度排序和 query attention，并利用 query 的跨帧连续性直接分配跟踪 ID。最后读 TPAMI 正式版，把三代设计放回统一接口，并补齐 camera/LiDAR fusion、传感器鲁棒性与实车验证。
