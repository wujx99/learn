# Camera-based BEV Temporal Fusion

面向多相机 3D 感知的显式深度提升、稠密 BEV 对齐、固定窗口融合与递归长期记忆。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection (SOLOFusion) | 2023 | read | [note](2023-solofusion/README.md) |
| Exploring Recurrent Long-Term Temporal Fusion for Multi-View 3D Perception (VideoBEV) | 2024 | read | [note](2024-videobev/README.md) |

## 推荐顺序

先读 SOLOFusion，把时间跨度理解成动态 stereo baseline，并区分高分辨率短时深度匹配与低分辨率长时 BEV 融合；再读 VideoBEV，比较固定窗口显式保存历史与单状态递归压缩在容量、延迟、遗忘和多任务适用性上的取舍。可并行阅读 query-based topic 中的 StreamPETR，比较稠密 scene memory 与稀疏 object memory。
