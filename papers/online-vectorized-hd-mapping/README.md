# Online Vectorized HD Mapping

使用车载传感器在线构建实例级折线/多边形 HD 地图，关注结构化表示、集合匹配、长距离几何与时序一致性。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction | 2023 | read | [note](2023-maptr/README.md) |
| StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction | 2023 | read | [note](2023-streammapnet/README.md) |
| MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction | 2024 | read | [note](2024-maptr-v2/README.md) |

## 推荐顺序

先读 MapTR，掌握 permutation-equivalent shape、层级 query 和层级匹配这套基本语言；再读 MapTRv2，看 one-to-many、dense supervision 与按两个轴解耦的 attention 如何改善收敛和扩展性；最后读 StreamMapNet，关注它如何改成“一实例一 query + 多点采样”，并用稀疏 query 与稠密 BEV 两条递归记忆获得大范围、跨帧稳定地图。若重点是时间建模，也可以在 MapTR 后直接读 StreamMapNet。
