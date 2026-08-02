# Query-based BEV Temporal Fusion

面向相机 3D 感知的 query-based BEV 表征、跨视角几何采样、时间对齐与记忆融合。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers | 2022 | read | [note](2022-bevformer/README.md) |
| BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision | 2023 | read | [note](2023-bevformer-v2/README.md) |

## 推荐顺序

先读 BEVFormer v1，建立 BEV queries、Spatial Cross-Attention 和 Temporal Self-Attention 的数据流；再读 v2，重点理解 perspective supervision 如何改变 backbone 的梯度路径，以及 hybrid object queries 和离线时间融合带来的收益与代价。
