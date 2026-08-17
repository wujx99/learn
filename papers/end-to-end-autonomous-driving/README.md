# End-to-End Autonomous Driving

从传感器输入联合学习场景表征、行为预测与 ego planning，重点关注任务之间的信息接口、候选轨迹表达、开环/闭环评估和安全决策。

## Papers

| Paper | Year | Status | Note |
| --- | --- | --- | --- |
| SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation | 2025 | read | [note](2025-sparsedrive/README.md) |
| SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving | 2026 | read | [note](2026-sparsedrive-v2/README.md) |

## 推荐顺序

先读 SparseDrive，理解动态 agent、静态 map 与 ego 如何统一为 sparse instances，以及 prediction/planning 为什么可以并行；再读 SparseDriveV2，观察作者如何把研究焦点从全栈任务接口转向 planner，并用 path–velocity factorization 扩展静态轨迹库。两篇的主要 benchmark 不同，因此更适合比较设计思想，不能直接用表中分数判断 V2 对 V1 的净提升。
