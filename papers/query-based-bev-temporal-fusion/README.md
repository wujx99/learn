# Query-based BEV Temporal Fusion

面向相机 3D 感知的 query-based BEV 表征、跨视角几何采样、时间对齐与记忆融合。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| PETR: Position Embedding Transformation for Multi-View 3D Object Detection | 2022 | read | [note](2022-petr/README.md) |
| BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers | 2022 | read | [note](2022-bevformer/README.md) |
| BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision | 2023 | read | [note](2023-bevformer-v2/README.md) |
| Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection (StreamPETR) | 2023 | read | [note](2023-streampetr/README.md) |

## 推荐顺序

可先把 PETR 与 BEVFormer v1 作为两条并行起点：前者用稀疏 object queries 全局查询带 3D 射线编码的图像 token，后者用稠密 scene-level BEV queries 做几何引导的局部采样。随后读 StreamPETR，看 PETR 的 object queries 如何变成在线时序记忆；再读 BEVFormer v2，理解 perspective supervision 如何改变 backbone 的梯度路径，以及 hybrid object queries 和离线时间融合的收益与代价。
