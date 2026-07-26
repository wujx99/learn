---
title: 深度学习分布式训练全景与技术选型
topic: distributed-training
topics: [distributed-training, training-loop]
framework: PyTorch
status: learning
updated: 2026-07-26
---

# 深度学习分布式训练全景与技术选型

## 一句话心智模型

分布式训练不是一种单独的算法，而是在多个设备之间分配训练循环中的四类东西：**数据、训练状态、模型计算和激活**；每少复制一份状态、每多切开一段计算，就会引入相应的通信、调度和故障复杂度。

学习时始终问四个问题：

1. 每个 rank 保存哪些参数、梯度、优化器状态和激活？
2. 每个 rank 计算哪一部分样本、层、张量或 token？
3. 它在什么时候与谁通信，通信量是多少？
4. 最终是否仍等价于目标中的单卡训练语义？

## 学习目标

读完后，你应该能够：

- 解释分布式训练为什么同时受到算力、显存和通信限制；
- 理解 rank、world size、process group 和集合通信；
- 区分 DDP、ZeRO/FSDP、TP、PP、SP/CP 解决的问题；
- 看懂 DP、TP、PP、CP 组成的混合并行拓扑；
- 根据模型是否能放入单卡、序列长度和网络拓扑选择起点；
- 正确计算 global batch size，并识别常见的性能与正确性问题；
- 知道 PyTorch 分布式程序相对普通训练循环改变了什么。

## 前置知识

建议先掌握仓库中的 [PyTorch 训练循环](../training-loop/training-loop.ipynb)，尤其是 batch、梯度累积、`backward()`、`optimizer.step()` 和 checkpoint。

本文以**同步、数据中心内、多 GPU 训练**为主。异步参数服务器、联邦学习和跨互联网去中心化训练是不同的问题，不在本文范围内。

---

## 1. 为什么需要分布式训练

单卡训练可能遇到三个不同瓶颈，不能用同一种并行方式解决。

### 1.1 时间不够：需要更多吞吐量

模型能放入单卡，但训练一遍数据需要数月。最自然的方案是让多张卡各算不同数据，再同步梯度，即**数据并行**。

### 1.2 模型状态放不下：需要切分参数相关状态

训练内存远大于权重文件大小。以混合精度 Adam 为例，一个参数可能同时对应：

- 低精度参数：约 2 bytes；
- 低精度梯度：约 2 bytes；
- FP32 master parameter：约 4 bytes；
- 两个 FP32 Adam moment：约 8 bytes。

粗略合计可达 **16 bytes/parameter**，还没有计算激活、临时 buffer 和内存碎片。具体数值取决于精度、优化器和实现，不能把 16 当作定律。

如果模型计算本身适合单卡，但这些训练状态的复制放不下，优先考虑 **ZeRO/FSDP**。

### 1.3 单个算子或激活放不下：需要切分模型计算

如果某层权重、矩阵乘法或长序列激活连临时聚合到单卡都不可接受，仅切分持久状态还不够，需要 **Tensor Parallel、Pipeline Parallel 或 Context Parallel**。

### 1.4 分布式并不自动更快

总步时可以粗略写成：

$$
T_{step} = T_{compute} + T_{communication} + T_{input} + T_{idle} - T_{overlap}
$$

更多 GPU 会减少每张卡的计算，却可能增加通信、流水线气泡、负载不均和同步等待。分布式训练的目标不是“使用最多 GPU”，而是满足显存和训练时限约束后，让昂贵设备保持有效工作。

---

## 2. 分布式程序的基本坐标系

### 2.1 一张 GPU 一个进程

PyTorch DDP 的常见部署方式是**每张 GPU 一个进程**。每个进程独立运行同一份 Python 程序，拥有自己的模型对象、优化器和 dataloader。

关键术语：

| 术语 | 含义 |
| --- | --- |
| rank | 一个进程在通信组中的编号，通常是 $0,1,\ldots,W-1$ |
| world size | 通信组中的总进程数 $W$ |
| local rank | 进程在当前节点内的编号，常用于选择本机 GPU |
| node rank | 当前机器在多机作业中的编号 |
| process group | 一组可以参与集合通信的 rank |
| backend | 实际通信后端；NVIDIA GPU 训练通常使用 NCCL |
| rendezvous | 多个进程发现彼此并建立通信组的初始化过程 |

不要默认 `rank == GPU 编号`。global rank 标识整个作业中的进程，local rank 才通常映射到当前节点设备。

### 2.2 单机多卡与多机多卡

- **单机多卡**：GPU 往往通过 NVLink/NVSwitch 或 PCIe 连接；节点内通信通常更快。
- **多机多卡**：跨节点依赖 InfiniBand/RoCE 或以太网；网络带宽、网卡绑定和拓扑影响显著。

因此常见的拓扑感知策略是：通信频繁的 TP 放在高速节点内，通信相对较少的 DP/FSDP 扩展到节点间。它是经验起点，不是永远正确的规则。

### 2.3 Process group 不只是“所有 GPU”

混合并行会建立多个相互正交的组。例如 16 张卡采用 `DP=2, TP=4, PP=2`：

- 每个 TP group 有 4 个 rank，共同切一个层；
- 每个 PP group 有 2 个 stage，共同切一串层；
- 每个 DP group 有 2 个模型副本，处理不同数据。

一个 rank 同时属于一个 DP group、一个 TP group 和一个 PP group。**world size 是物理进程数，并行度是这些进程的逻辑坐标。**

---

## 3. 集合通信：分布式训练的积木

设有 $W=4$ 个 rank，每个 rank 持有不同张量。常见原语如下：

| 原语 | 输入到输出 | 典型用途 |
| --- | --- | --- |
| broadcast | root 的一份张量复制给所有 rank | 同步初始化、元数据 |
| reduce | 所有 rank 的张量归约到 root | 汇总指标 |
| all-reduce | 所有张量归约，结果发回所有 rank | DDP 梯度同步 |
| all-gather | 每个 rank 的分片拼接后发给所有 rank | 临时重建完整参数/激活 |
| reduce-scatter | 先归约，再把结果分片给各 rank | 分片梯度同步 |
| all-to-all | 每个 rank 给每个其他 rank 发送不同分片 | MoE token dispatch、部分上下文并行算法 |
| send/recv | 两个 rank 点对点传输 | pipeline stage 间传激活/梯度 |

一个重要等价关系是：

$$
\text{AllReduce} \equiv \text{ReduceScatter} + \text{AllGather}
$$

这个视角能帮助理解为什么 DDP 常用 all-reduce，而 fully sharded 方法围绕 reduce-scatter 和 all-gather 组织。

### 3.1 带宽与延迟

通信时间的简化模型是：

$$
T_{comm} \approx \alpha + \frac{n}{B}
$$

其中 $\alpha$ 是启动延迟，$n$ 是消息量，$B$ 是有效带宽。大量小消息受延迟支配；少量大消息更容易接近带宽上限。因此 DDP 会把许多梯度装入 bucket，而不是每个小参数单独通信。

### 3.2 集合通信必须匹配

通信组中的 rank 必须以兼容的顺序、形状和次数进入 collective。某个 rank 少执行一次、提前报错或走了不同控制流，其他 rank 可能一直等待，表现为 hang。这是分布式故障中最重要的心智模型之一。

---

## 4. 数据并行：复制模型，切分数据

### 4.1 从单卡训练循环到同步数据并行

假设两个 rank 的模型参数初始相同：

1. rank 0 处理 batch A，得到局部梯度 $g_0$；
2. rank 1 处理 batch B，得到局部梯度 $g_1$；
3. 对梯度做 all-reduce 并平均：

$$
\bar g = \frac{g_0+g_1}{2}
$$

4. 两个 rank 都用同一个 $\bar g$ 执行 optimizer step，因此参数继续保持一致。

数据并行增加的是**有效 batch**和总吞吐量；它不会把单个样本的 forward 自动分给多张卡。

### 4.2 PyTorch DDP 的核心行为

`DistributedDataParallel` 包装模型，并通过 autograd hook 在 backward 过程中触发梯度同步。实现通常会把参数梯度分 bucket，并让已经计算完成的 bucket 通信与剩余反向计算重叠。

DDP 每个 rank 仍保存完整的：

- 模型参数；
- 梯度；
- 优化器状态；
- 当前 micro-batch 的激活。

所以 DDP 适合“模型能放进单卡，只是想更快”，不解决模型状态无法放入单卡的问题。

### 4.3 Global batch size

只有独立数据副本数会扩大 global batch。若每个 DP rank 的 micro-batch 为 $b$，梯度累积 $K$ 次，DP degree 为 $D$：

$$
B_{global}=b\times K\times D
$$

TP 和 PP 通常共同计算同一批样本，**不能**乘进 global batch。混合并行里把 world size 误当 DP degree，是常见错误。

global batch 改变后，优化超参数和收敛行为可能改变。更高系统吞吐不等于相同训练语义下更快收敛。

### 4.4 DistributedSampler

模型由 DDP 负责同步，数据划分由用户负责。训练集通常使用 `DistributedSampler`，并在每个 epoch 调用 `sampler.set_epoch(epoch)`，使所有 rank 以一致但逐 epoch 变化的方式重排和分片数据。

如果每个 rank 都遍历完整 dataset，计算会被重复；如果随机种子和 sampler 使用错误，可能出现样本重复或遗漏。

---

## 5. ZeRO 与 FSDP：切分数据并行中的冗余状态

普通数据并行的关键浪费是：每个 rank 都复制了相同训练状态。ZeRO 按阶段逐步消除冗余：

| 方法 | 优化器状态 | 梯度 | 参数 | 直观结果 |
| --- | --- | --- | --- | --- |
| DDP | 每 rank 完整 | 每 rank 完整 | 每 rank 完整 | 通信简单，显存开销最大 |
| ZeRO-1 | 分片 | 完整 | 完整 | 先消除最大的 optimizer state 冗余 |
| ZeRO-2 | 分片 | 分片 | 完整 | 进一步减少梯度冗余 |
| ZeRO-3 | 分片 | 分片 | 分片 | 参数也不长期完整复制 |

PyTorch FSDP 的 full sharding 在目标上接近 ZeRO-3，但 API、执行细节和功能不能简单视为完全相同。

### 5.1 Fully sharded 一次 layer 计算发生什么

一个简化过程是：

1. 每个 rank 平时只保留参数 shard；
2. forward 某个 FSDP unit 前，通过 all-gather 临时物化该 unit 的完整参数；
3. 计算 forward，并在可释放时重新只保留 shard；
4. backward 需要时再次 all-gather 参数；
5. 梯度通过 reduce-scatter 归约并分片；
6. 每个 rank 只更新自己持有的 optimizer state 与参数 shard。

它仍是数据并行语义：各 rank 处理不同数据，只是训练状态不再完整复制。

### 5.2 它省了什么，又付出什么

收益：

- 参数、梯度和优化器状态长期占用可接近按 sharding degree 缩小；
- 用户不必手写每个矩阵乘法如何切分。

代价：

- forward/backward 中增加参数 all-gather；
- 某个 unit 计算时仍需临时完整参数，峰值内存取决于 wrapping 粒度和预取；
- checkpoint、初始化、state dict 和调试更复杂；
- 小模型可能因为额外通信反而更慢。

### 5.3 CPU/NVMe offload

Offload 把某些参数或 optimizer state 移到 CPU，甚至 NVMe。它扩展了容量，但把瓶颈转移到 PCIe、内存带宽或存储 I/O。优先级通常是：先避免不必要状态、使用合适精度和分片，再在确有容量需求时 offload。

### 5.4 Activation checkpointing 不是 ZeRO stage

Activation checkpointing 丢弃一部分 forward 中间激活，在 backward 时重新计算，属于“用额外计算换激活显存”。它与 DDP、FSDP、TP、PP 都可组合。

需要区分：

- **模型 checkpoint**：为了故障恢复而写磁盘；
- **activation checkpointing**：为了省显存而在 backward 重算。

---

## 6. Tensor Parallel：切分层内张量与计算

当一个层或其计算无法由单卡高效承担时，可以把矩阵沿维度切分。

考虑线性层：

$$
Y=XW
$$

### 6.1 Column parallel

按输出维切权重：

$$
W=[W_1\;W_2], \qquad Y=[XW_1\;XW_2]
$$

每个 rank 计算不同输出特征，输出天然是 sharded。若下一层接受这种布局，可以避免立即 all-gather。

### 6.2 Row parallel

按输入维切分：

$$
X=[X_1\;X_2], \qquad W=\begin{bmatrix}W_1\\W_2\end{bmatrix}
$$

各 rank 计算部分和 $X_iW_i$，再通过归约得到完整或适当分片的输出。

Megatron 风格 TP 会成对安排 column-parallel 与 row-parallel 层，使 Transformer MLP 和 attention 中的布局转换与 collective 数量可控。

### 6.3 TP 的特征

- 参数、计算和部分激活都在层内被切分；
- 每层或每几层就可能通信，通信频率高；
- 很依赖低延迟、高带宽互连；
- TP degree 越大，每张卡矩阵越小，可能降低 kernel 效率；
- 通常优先限制在 NVLink/NVSwitch 连接的节点内。

PyTorch 当前的 `torch.distributed.tensor.parallel` 基于 DTensor，提供 column-wise、row-wise 和 sequence parallel 等布局，但官方仍将部分 TP API 标为 experimental，使用时应核对当前版本文档。

---

## 7. Pipeline Parallel：切分层，调度 micro-batch

Pipeline Parallel 把连续层分为多个 stage：

```text
micro-batch -> stage 0 -> stage 1 -> stage 2 -> stage 3
```

stage 之间传递激活，backward 时反向传梯度。为了避免只有一个 stage 工作，把 global batch 切成多个 micro-batch，在不同 stage 上交错执行。

### 7.1 Pipeline bubble

流水线开始填充和结束排空时会有设备空闲，称为 bubble。设 stage 数为 $P$、micro-batch 数为 $M$，简单 GPipe 式调度的 bubble 比例可直观近似为：

$$
\text{bubble fraction} \approx \frac{P-1}{M+P-1}
$$

增加 $M$ 能减小 bubble，但会改变激活驻留、调度和 batch 语义。1F1B、interleaved 1F1B 等调度用于改善内存或利用率。

### 7.2 切分点和负载均衡

PP 不仅要让各 stage 层数相近，还要让：

- forward/backward 计算时间接近；
- 激活通信量可接受；
- 每个 stage 显存不过载；
- tied weights、skip connection 和不规则控制流能正确处理。

最慢的 stage 决定整条 pipeline 的节拍。平均分层不等于平均分计算。

### 7.3 PP 适合什么

- 模型天然按层顺序可切分；
- 单个 stage 能放入设备，但完整模型不能；
- 跨节点带宽较慢，希望减少相较 TP 的高频 collective；
- 能接受更复杂的调度和 bubble。

PyTorch 的 `torch.distributed.pipelining` 提供拆分和多种 schedule，但截至本文更新时官方仍标记为 alpha，应隔离 API 依赖并核对版本。

---

## 8. Sequence Parallel 与 Context Parallel

这两个术语在论文和框架中有重叠，阅读时要先看“到底沿哪个张量维度切了什么计算”。

### 8.1 Megatron/PyTorch TP 语境中的 Sequence Parallel

基础 TP 往往只切 attention/MLP 内部矩阵，而 block 边界处的激活仍复制。Sequence Parallel 进一步沿 sequence 维切分 LayerNorm/RMSNorm、Dropout 等逐 token 计算和对应激活，以减少 activation memory，并通常与 TP 配套。

它不必然意味着 attention 可以在每个 rank 只看局部上下文。

### 8.2 面向超长序列的 Context Parallel

Context Parallel 通常把同一个样本的 sequence/context token 分给不同 rank，并通过 ring、all-gather、all-to-all 或专门 attention 算法处理跨分片的 K/V 依赖。

它主要解决：

- 单条超长序列激活过大；
- attention 计算与 KV 张量无法由单卡承担。

### 8.3 选择判断

- 已经使用 TP，想减少 block 边界激活复制：考虑其配套 SP；
- 单样本序列本身太长：考虑 CP/长序列 attention 分片；
- 只是 global batch 很大：用 DP，不要因为数据多而使用 CP。

---

## 9. 额外的一维：Expert Parallel

Mixture-of-Experts 模型会把不同 expert 放到不同 rank。router 决定 token 去向，通常通过 all-to-all 分发 token，再收集 expert 输出。

Expert Parallel 的核心挑战不是普通 dense TP：

- token 路由可能负载不均；
- all-to-all 对网络和消息布局敏感；
- capacity factor、drop token 或 load-balancing loss 会影响模型与系统行为。

它常与 DP、TP、PP 组合，但只有 MoE 架构需要，因此不作为通用模型的默认方案。

---

## 10. 混合并行：把不同瓶颈映射到不同维度

大模型训练通常使用多维 device mesh。若忽略 EP，可写成：

$$
W = D \times T \times P \times C
$$

其中：

- $D$：Data/FSDP degree；
- $T$：Tensor Parallel degree；
- $P$：Pipeline Parallel degree；
- $C$：Context Parallel degree。

### 10.1 一个 64 GPU 示例

假设 `TP=8, PP=2, CP=1, DP=4`：

$$
64 = 8 \times 2 \times 1 \times 4
$$

- 8 张卡协作完成每个 stage 内的层；
- 2 个 stage 切开层深度；
- 上述 16 张卡构成一个完整模型副本；
- 共有 4 个数据副本处理不同 batch。

如果每个 DP rank 组对应的 micro-batch 为 2，累积 8 次，则：

$$
B_{global}=2\times8\times4=64
$$

不是 $2\times8\times64$。

### 10.2 常见组合

| 组合 | 适用场景 | 主要通信 |
| --- | --- | --- |
| DDP | 模型能放单卡，追求吞吐 | 梯度 all-reduce |
| FSDP | 状态复制放不下，单个 FSDP unit 可临时物化 | 参数 all-gather、梯度 reduce-scatter |
| TP + DP/FSDP | 单层较大，且希望扩展吞吐 | 层内 collective + 数据并行通信 |
| PP + DP/FSDP | 模型很深，按 stage 切分自然 | stage 间 send/recv + 数据并行通信 |
| TP + PP + DP | 超大 dense Transformer | 三类通信与 pipeline 调度 |
| CP + TP/DP | 单样本上下文极长 | attention 相关跨 context 通信 |

### 10.3 拓扑映射原则

通常把通信最频繁的维度映射到最快链路：

1. TP 往往放在节点内 NVLink/NVSwitch；
2. PP 只在 stage 边界通信，可跨节点；
3. DP/FSDP 可扩展到节点间，但仍受梯度或参数通信影响；
4. 实际映射应根据消息量、网络拓扑和 profile 修正。

---

## 11. 技术选型：从最简单方案逐步增加复杂度

### 11.1 决策树

```text
模型训练状态 + 激活能否放入单卡？
├── 能
│   ├── 单卡能否在期限内完成？
│   │   ├── 能：单卡；先不要分布式
│   │   └── 不能：DDP
│   └── DDP 扩展效率差：检查 batch、数据、通信重叠和网络
└── 不能
    ├── 主要是参数/梯度/optimizer state 复制导致？
    │   └── FSDP/ZeRO + activation checkpointing
    ├── 单层或临时完整参数仍放不下？
    │   └── TP
    ├── 模型很深且可按层均衡切分？
    │   └── PP
    └── 单条序列激活/attention 放不下？
        └── CP/长序列并行

仍需更多设备或多种约束同时存在？
└── 组合成混合并行，并按硬件拓扑映射 process groups
```

### 11.2 先问清的输入

不要从“某框架支持什么”开始，而应收集：

- 参数量、层数、hidden size、attention heads；
- sequence length 与 micro-batch；
- 参数/梯度/优化器/激活分别占多少峰值显存；
- GPU 数、单卡显存和计算能力；
- 节点内与节点间带宽、拓扑；
- 目标 global batch、训练 token 数和时限；
- 是否能改变精度、重计算、optimizer 或模型结构；
- 容错和 checkpoint 时间要求。

### 11.3 推荐的增加复杂度顺序

1. 单卡正确性基线；
2. AMP、activation checkpointing 等单卡优化；
3. DDP；
4. 状态内存不够时 FSDP/ZeRO；
5. 单个计算单元放不下时 TP；
6. 深度和集群规模继续增加时 PP；
7. 超长上下文时 CP；
8. 最后才组合并调优多维并行。

这个顺序不是能力排名，而是**调试复杂度控制**。每增加一个维度，都应保留可以比较 loss、梯度和吞吐的前一阶段基线。

---

## 12. PyTorch DDP：普通训练循环改变了什么

本文不把 DDP 代码作为主线，但下面的骨架能把概念映射回训练循环：

```python
# train.py（结构示意，使用 torchrun 启动）
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = build_dataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, shuffle=False)

    model = build_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()      # DDP hooks 在 backward 中同步梯度
            optimizer.step()

        if dist.get_rank() == 0:
            save_for_inference(model.module.state_dict())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

单机 8 卡通常类似：

```bash
torchrun --standalone --nproc-per-node=8 train.py
```

相对单卡循环，关键变化只有：

1. 初始化 process group 并选择 local device；
2. 用 `DistributedSampler` 切数据；
3. 用 DDP 包装模型；
4. backward 自动同步梯度；
5. 控制日志和普通文件写入，避免所有 rank 重复执行；
6. 正确清理 process group。

多机启动还需要一致的 rendezvous 配置、节点数量和 node rank。不要直接复制版本不明的启动参数，应以当前 PyTorch `torchrun` 文档和集群调度器约定为准。

---

## 13. Checkpoint 与可恢复性

### 13.1 需要保存哪些状态

为了真正恢复训练，通常至少包括：

- model state；
- optimizer state；
- scheduler state；
- AMP scaler state；
- 已完成的 epoch/global step；
- 随机数状态和 sampler 进度（要求精确重现时）；
- 数据、模型和训练配置。

### 13.2 DDP 与 sharded checkpoint

- DDP 模型在各 rank 参数相同，简单场景可由 rank 0 保存未包装模型的 state dict；
- FSDP/TP 等状态是 sharded 的，先聚合到 rank 0 可能造成内存峰值和 I/O 瓶颈；
- PyTorch Distributed Checkpoint 支持多 rank 并行保存和加载时 reshard，更适合大规模 sharded state。

### 13.3 Checkpoint 也是系统瓶颈

大模型 checkpoint 可能达到 TB 级。需要衡量：

- 保存间隔与可接受的丢失工作量；
- 同步保存造成的训练暂停；
- 存储带宽和小文件数量；
- 异步保存的额外内存与一致性；
- 换 world size 恢复时能否 reshard。

---

## 14. 性能分析：先分解 step time

### 14.1 吞吐与扩展效率

常见指标：

- samples/s 或 tokens/s：系统吞吐；
- model FLOPs utilization（MFU）：有效模型计算相对理论峰值；
- step time：完成一次 optimizer update 的时间；
- communication time、exposed communication：通信总时间和未被计算隐藏的部分；
- data stall：等待 dataloader 的时间。

从 1 个设备扩到 $N$ 个设备，speedup 与效率定义为：

$$
S_N=\frac{T_1}{T_N}, \qquad E_N=\frac{S_N}{N}
$$

实际大模型可能没有单卡可运行基线，此时要明确使用哪一个较小规模作为相对基线。

### 14.2 Strong scaling 与 weak scaling

- **Strong scaling**：总问题规模不变，增加设备；每卡工作变小，最终更容易被通信和延迟支配。
- **Weak scaling**：每卡工作规模大致不变，增加设备同时扩大总问题；更容易保持效率，但训练语义也可能变化。

### 14.3 通信计算重叠

仅看到 NCCL kernel 占时间并不代表它全部是额外开销。若通信与 backward GEMM 同时执行，真正影响 step time 的是没有被隐藏的 exposed tail。

提高重叠可能涉及：

- DDP bucket 大小和梯度就绪顺序；
- FSDP prefetch、wrapping 粒度；
- TP collective 与矩阵乘法调度；
- 独立 CUDA streams；
- 避免不必要的同步点，如频繁 `.item()`。

### 14.4 最小 benchmark 纪律

比较两个方案时固定：

- 模型、精度、sequence length；
- global batch 或明确记录其变化；
- gradient accumulation；
- activation checkpointing；
- optimizer 与编译选项；
- warm-up steps 和统计窗口。

同时验证 loss 曲线或梯度，不能只比较吞吐。一个漏同步梯度的程序可能“快得惊人”。

---

## 15. 常见故障：症状 → 优先检查

| 症状 | 优先检查 | 常见原因 |
| --- | --- | --- |
| 作业一直 hang | 最后一个 collective、各 rank 控制流 | collective 次数/顺序不一致，某 rank 先报错，网络配置错误 |
| 单卡正常，多卡 loss 不对 | global batch、loss reduction、sampler | 学习率未适配、样本重复、loss 缩放错误 |
| 多卡反而更慢 | 每卡 batch、通信占比、输入流水线 | batch 太小、网络慢、频繁同步、数据加载不足 |
| 某些 rank OOM | 每 rank 输入形状和分片布局 | 动态 batch、长尾序列、stage 不均衡、rank 0 额外聚合 |
| GPU 利用率呈周期性空洞 | pipeline timeline、checkpoint/I/O | pipeline bubble、同步保存、straggler |
| DDP 在 backward 卡住 | unused parameters、动态图分支 | 不同 rank 使用了不同参数集合或不同控制流 |
| NCCL 初始化失败 | rank/device 映射、网卡、端口 | local rank 重复绑卡、接口选择错误、端口冲突 |
| 恢复后结果突变 | 完整训练状态 | optimizer/scheduler/scaler 或 RNG 未恢复 |
| 指标被重复记录 | rank 条件 | 所有 rank 都写同一文件或上报相同指标 |

调试顺序建议：

1. 单卡跑通并记录基线 loss；
2. 单机两卡、极小数据复现；
3. 打印带 rank 的阶段日志，定位最后共同到达的位置；
4. 检查数据索引、tensor shape、collective 和梯度；
5. 再扩到整机和多机；
6. 正确性稳定后才 profile 和调性能。

---

## 16. 最容易混淆的判断

### “用了 8 张卡，所以 global batch 乘 8”

不一定。只有 DP degree 乘入 global batch；8-way TP 可能只是 8 张卡共同算同一批样本。

### “FSDP 就是模型并行”

从“参数分散在多卡”看很像，但其核心仍是数据并行语义：不同 rank 处理不同数据，通过临时 all-gather 执行模块。TP 则直接定义一个算子的分片计算。

### “`no_sync()` 等于没有梯度”

DDP 的 `no_sync()` 通常用于梯度累积，只跳过该轮跨 rank 同步；本地 backward 和梯度累加仍会发生。

### “更多并行维度一定更省显存”

每种并行会引入 buffer、通信临时张量、调度状态或 padding。峰值内存必须 profile，不能只用参数量除以 GPU 数。

### “训练结果不同就是分布式错了”

浮点归约顺序、随机数和数据顺序可能带来非 bitwise-identical 结果。应先定义可接受的数值与收敛容差，再判断是否错误。

---

## 17. 自测与场景题

### 概念自测

1. DDP 为什么能保证所有 rank 在 step 后参数一致？
2. AllReduce 为什么可以拆成 ReduceScatter + AllGather？
3. FSDP 在 forward 前 all-gather 参数，为什么仍然能降低显存？
4. TP 与 PP 分别沿模型的哪个方向切分？
5. 为什么 TP 通常比 PP 更依赖节点内高速互连？
6. activation checkpointing 和 distributed checkpoint 分别解决什么？
7. `DP=8, TP=4, PP=2` 使用多少 GPU？micro-batch 为 2、累积 16 次时 global batch 是多少？

第 7 题答案：总 GPU 数为 $8\times4\times2=64$，global batch 为 $2\times16\times8=256$。

### 技术选型题

尝试先独立回答：

1. **ResNet 能放单卡，但训练太慢**：从 DDP 开始。
2. **70B 模型状态放不下，但每个 Transformer block 可临时物化**：优先评估 FSDP/ZeRO 与 activation checkpointing。
3. **某个超大线性层连临时物化都放不下**：需要 TP。
4. **模型很深，单层可放下，集群跨节点链路较慢**：评估 PP，并把 stage 做负载均衡。
5. **单条超长序列的 attention 激活放不下**：评估 CP/长序列并行，而不是只增加 DP。
6. **MoE expert 能分散，但 token 路由通信巨大**：评估 EP 的负载均衡与 all-to-all 拓扑。

---

## 18. 最终速查表

| 技术 | 切分什么 | 主要解决 | 主要通信 | 首要代价 |
| --- | --- | --- | --- | --- |
| DDP | 数据 | 吞吐 | gradient all-reduce | 状态完整复制 |
| ZeRO/FSDP | 参数相关训练状态 + 数据 | 状态显存 | all-gather、reduce-scatter | 通信和状态管理复杂 |
| TP | 层内 tensor/算子 | 单层计算与显存 | 高频 all-reduce/all-gather/reduce-scatter | 强依赖高速互连 |
| PP | 连续层/stage | 模型深度与容量 | activation/gradient send-recv | bubble 与负载均衡 |
| SP | sequence 维上的部分激活/逐 token 计算 | TP 场景的激活显存 | 布局转换 collective | 通常依赖 TP 设计 |
| CP | 单样本上下文 token | 超长序列 | attention 相关 ring/gather/all-to-all | 算法和通信复杂 |
| EP | experts 与 token 路由 | MoE 容量 | all-to-all | 负载不均 |
| Activation checkpointing | 保存的中间激活 | 激活显存 | 无必然跨卡通信 | backward 重计算 |
| Offload | GPU 上的状态 | GPU 容量 | CPU/NVMe 数据搬运 | 低速链路/I/O |

选择时先定位瓶颈：

```text
吞吐不足        -> DP
训练状态放不下  -> FSDP/ZeRO
单层放不下      -> TP
模型太深        -> PP
单序列太长      -> CP
MoE experts      -> EP
激活太大        -> activation checkpointing / SP / CP
```

如果只记住一个原则：**从最小、最容易验证正确的并行方案开始，只在明确瓶颈要求时增加新的并行维度。**

---

## 相关知识与论文

- [PyTorch 训练循环](../training-loop/training-loop.ipynb)：分布式系统最终仍围绕本地 forward、backward 和 optimizer step 组织。
- [ZeRO 原始论文](https://arxiv.org/abs/1910.02054)：按阶段切分 optimizer state、gradient 和 parameter 的来源。
- [Megatron-LM 原始论文](https://arxiv.org/abs/1909.08053)：Transformer 层内 Tensor Parallel 的经典设计。
- [Megatron-LM 混合并行论文](https://arxiv.org/abs/2104.04473)：DP、TP、PP 组合及大规模训练权衡。

## 当前官方资料

以下链接用于核对当前 PyTorch/NCCL 能力；API 状态可能随版本变化：

- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)
- [PyTorch Tensor Parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)
- [PyTorch Pipeline Parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 下一步

这份文档负责全景和选型。实践部分继续阅读 [PyTorch DDP：从单进程到多机](../pytorch-ddp/README.md)，其中提供了可用 `python` 和 `torchrun` 执行的同一份训练脚本。
