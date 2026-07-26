---
title: PyTorch DDP：从单进程到多机
topic: pytorch-ddp
topics: [pytorch-ddp, training-loop, distributed-training]
framework: PyTorch
status: learning
updated: 2026-07-26
---

# PyTorch DDP：从单进程到多机

## 一句话心智模型

DDP 启动多个相互独立的训练进程：每个进程在一张设备上保存完整模型、读取不同数据并计算局部梯度；`backward()` 期间，DDP 把各 rank 的梯度同步为相同结果，于是每个进程执行 `optimizer.step()` 后仍得到相同参数。

从单卡扩展到多机，训练循环的数学主体没有变化。真正新增的是：

```text
进程启动与会合
    + rank 到设备的绑定
    + 数据分片
    + backward 中的梯度同步
    + 指标显式聚合
    + rank-aware 的日志与 checkpoint
    + 多机网络和故障处理
```

## 学习目标

完成后，你应该能够：

- 从普通 PyTorch 训练脚本改造出 DDP 程序；
- 解释 `torchrun`、rank、local rank、world size 和 process group；
- 正确使用 `DistributedSampler` 和 `set_epoch()`；
- 区分 DDP 自动同步的梯度与需要手动同步的指标；
- 正确计算 global batch size，使用梯度累积与 `no_sync()`；
- 处理 AMP、日志、checkpoint 和恢复训练；
- 写出单机多 GPU 与多机多 GPU 启动命令；
- 系统排查 hang、NCCL、数据重复和性能退化。

## 前置知识和配套文件

- [PyTorch 训练循环](../training-loop/training-loop.ipynb)：单进程训练基础。
- [分布式训练全景与技术选型](../distributed-training/README.md)：DDP 与 FSDP、TP、PP 等方法的边界。
- [可运行训练脚本](assets/train.py)：同一个文件支持普通 `python` 和 `torchrun`。
- [Slurm 模板](assets/slurm.sbatch)：仅作为集群适配起点，不是跨集群通用配置。

示例使用小型合成回归数据，不下载数据集。当前仓库环境已验证：

- 单进程 CPU；
- 两个 CPU 进程的 Gloo DDP；
- 非整除 iteration 数下的梯度累积与 `no_sync()`；
- 单 GPU CUDA AMP；
- checkpoint 保存与恢复。

当前机器只有一张 GPU，因此无法实际验证多 GPU NCCL 和多机网络；相应命令来自 PyTorch 当前官方 `torchrun` 约定，并在本文明确标记环境前提。

---

## 1. 先运行单进程基线

从仓库根目录执行：

```bash
python knowledge/pytorch-ddp/assets/train.py \
  --device cpu \
  --epochs 6 \
  --output-dir /tmp/ddp-single
```

你会看到类似：

```text
mode=single backend=none device=cpu world_size=1 global_batch=16 amp=False
epoch=01 train_loss=... val_loss=...
...
checkpoint=/tmp/ddp-single/checkpoint.pt
```

这一步的重要性不是性能，而是建立**正确性基线**：数据、模型、loss、optimizer 和 checkpoint 在不涉及通信时必须先工作。

### 1.1 单进程训练的核心

脚本中的核心仍是熟悉的五行：

```python
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(features), targets)
loss.backward()
optimizer.step()
```

DDP 不会替换 optimizer，也不会把一个 forward 自动拆成多个设备上的算子。它主要介入 `backward()` 的梯度同步。

### 1.2 为什么同一份脚本能支持两种模式

脚本通过 `WORLD_SIZE` 判断它是否由多进程 `torchrun` 启动：

```python
def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1
```

- 普通 `python train.py` 没有这些分布式环境变量，走单进程路径；
- `torchrun --nproc-per-node=2 ...` 注入 `RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等变量，走 DDP 路径。

生产项目也可以要求始终由 `torchrun` 启动，从而减少条件分支。这里保留双模式是为了清楚比较“加入 DDP 前后到底改变了什么”。

---

## 2. 用 CPU 理解真正的多进程 DDP

即使只有一张 GPU，也可以用两个 CPU 进程和 Gloo 验证 DDP 的控制流：

```bash
torchrun --standalone --nproc-per-node=2 \
  knowledge/pytorch-ddp/assets/train.py \
  --device cpu \
  --epochs 6 \
  --output-dir /tmp/ddp-cpu
```

此时 `torchrun` 做两件事：

1. 启动一个本地 rendezvous 服务；
2. 启动两个独立 Python worker，并为它们设置不同 rank。

脚本输出应包含：

```text
mode=DDP backend=gloo device=cpu world_size=2 global_batch=32 amp=False
```

只有 rank 0 打印训练日志，所以不会看到每行重复两次。

> CPU/Gloo 测试用于学习和正确性验证，不代表 GPU/NCCL 的性能。

---

## 3. `torchrun`、agent 和 worker

### 3.1 进程结构

单机四卡启动时：

```text
torchrun agent
├── worker: rank 0, local rank 0 -> cuda:0
├── worker: rank 1, local rank 1 -> cuda:1
├── worker: rank 2, local rank 2 -> cuda:2
└── worker: rank 3, local rank 3 -> cuda:3
```

多机时通常每个节点运行一个 torchrun agent，每个 agent 再启动本节点 worker。所有 worker 合在一起构成一个 worker group。

### 3.2 四个 rank 相关变量

| 变量 | 含义 | 典型用途 |
| --- | --- | --- |
| `RANK` | 全局 worker 编号 | 日志、全局主进程判断 |
| `WORLD_SIZE` | 全部 worker 数 | global batch、collective |
| `LOCAL_RANK` | 当前节点内 worker 编号 | 绑定本地 GPU |
| `LOCAL_WORLD_SIZE` | 当前节点的 worker 数 | 节点内布局 |

例如两台机器、每台四卡：

```text
node 0: global ranks 0,1,2,3; local ranks 0,1,2,3
node 1: global ranks 4,5,6,7; local ranks 0,1,2,3
world size = 8
```

所以不能用 global rank 直接选择 GPU：node 1 的 rank 4 应绑定 `cuda:0`，不是不存在的 `cuda:4`。

### 3.3 Rendezvous 是什么

rendezvous 让不同节点发现彼此，并对同一次作业的成员达成一致。单机的 `--standalone` 会自动创建本地 rendezvous；多机需要所有节点提供相同的：

- `--rdzv-id`：本次作业的唯一标识；
- `--rdzv-backend`：通常使用 `c10d`；
- `--rdzv-endpoint`：所有节点可访问的 `host:port`。

它解决的是“谁属于这次训练、去哪里会合”，不是模型梯度通信算法本身。

---

## 4. 初始化 process group 与绑定设备

示例脚本根据设备选择 backend：

```python
if use_cuda:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = "nccl"
else:
    device = torch.device("cpu")
    backend = "gloo"

dist.init_process_group(backend=backend)
```

经验规则：

- CUDA GPU 训练使用 NCCL；
- CPU 分布式使用 Gloo；
- 每个 GPU worker 只管理一张 GPU；
- 在创建 CUDA model 和 DDP wrapper 前完成设备绑定。

`init_process_group()` 使用 `torchrun` 注入的环境变量完成初始化，因此脚本不需要手工传 `rank` 和 `world_size`。

### 4.1 为什么初始化失败常表现为等待

process group 是集体系统：world size 声明有 8 个 worker，就必须有 8 个兼容 worker 加入。同一个 rendezvous 配置下少一个节点、端口不可达或 job id 不一致，其余进程只能等待或超时。

---

## 5. 数据分片：DDP 不会替你切 dataset

### 5.1 `DistributedSampler`

DDP 只同步模型梯度，不知道 dataset。训练数据通常这样处理：

```python
sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=seed,
)

loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=False,
)
```

注意：设置 `sampler` 后不要再让 DataLoader `shuffle=True`。shuffle 的控制权已经交给 sampler。

### 5.2 每个 epoch 必须调用 `set_epoch()`

```python
for epoch in range(start_epoch, num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

`DistributedSampler` 使用 epoch 参与确定性 shuffle。忘记调用时，每个 epoch 可能重复相同顺序。各 rank 必须使用相同 epoch，否则数据分片可能重叠或不一致。

### 5.3 padding 与 validation

当 dataset 大小不能被 world size 整除时，默认 `DistributedSampler(drop_last=False)` 会补充索引，使各 rank 样本数相同。这适合需要各 rank iteration 数一致的训练，但验证指标可能重复统计少量样本。

示例脚本在验证阶段采用 rank-strided subset：

```python
val_indices = range(rank, len(val_dataset), world_size)
val_subset = Subset(val_dataset, val_indices)
```

这样每个验证样本只出现一次，然后通过 all-reduce 汇总 `loss_sum` 和 `sample_count`。代价是各 rank 验证 batch 数可能相差一个；因为验证没有 DDP backward collective，本例可以安全处理。

生产代码也可以实现不 padding 的 distributed eval sampler。

---

## 6. DDP 包装模型后发生了什么

CPU 和 CUDA 的包装方式略有不同：

```python
if device.type == "cuda":
    model = DDP(model, device_ids=[local_rank])
else:
    model = DDP(model)
```

关键过程：

1. 构造 DDP 时，模型状态以 rank 0 为基准保持一致；
2. DDP 为需要梯度的参数注册 autograd hook；
3. backward 产生梯度时，梯度被装入 bucket；
4. bucket 就绪后触发 collective，通常与剩余 backward 计算重叠；
5. 同步完成后，各 rank 对应参数拥有相同的平均梯度；
6. 各 rank 独立执行相同 optimizer step，参数继续一致。

### 6.1 DDP 自动同步什么

- 参数梯度：在 backward 中同步；
- buffer：根据配置在 forward 时广播，例如 BatchNorm running statistics 的相关行为。

DDP**不会自动同步**：

- Python 变量；
- `loss.item()`；
- 自定义日志计数器；
- dataset 状态；
- 每个 rank 任意修改的非参数对象；
- optimizer state（它依靠相同梯度和更新保持一致）。

### 6.2 不要在包装后只改一个 rank 的模型

如果 rank 0 在 DDP 构造后替换层、注册新参数或单独修改训练控制流，其他 rank 不会自动获得相同结构。模型结构和参与 backward 的参数集合应在各 rank 上一致。

---

## 7. Global batch 与学习率

设：

- 每 rank micro-batch 为 $b$；
- 梯度累积步数为 $K$；
- DDP world size 为 $W$。

则：

$$
B_{global}=b\times K\times W
$$

示例默认 `batch_size=16`：

| 启动方式 | world size | accumulation | global batch |
| --- | ---: | ---: | ---: |
| 普通 Python | 1 | 1 | 16 |
| 2 个 DDP worker | 2 | 1 | 32 |
| 8 个 DDP worker | 8 | 1 | 128 |
| 8 个 worker，累积 4 次 | 8 | 4 | 512 |

这解释了为什么同一学习率下单卡与多卡 loss 曲线不一定相同：你可能同时改变了 global batch。

比较单卡与 DDP 正确性时，可以保持 global batch 不变。例如单卡 batch 32 对比两卡每 rank batch 16。

学习率是否按 world size 线性增大取决于 optimizer、模型、batch 范围和 warmup，不是 DDP 的硬性规则。

---

## 8. 指标需要显式 all-reduce

每个 rank 的 `loss.item()` 只反映它处理的数据。若只打印 rank 0 的局部 loss，会把“避免重复打印”和“得到全局指标”混为一谈。

示例累计两个量：

```python
loss_and_count[0] += loss * batch_size
loss_and_count[1] += batch_size
dist.all_reduce(loss_and_count, op=dist.ReduceOp.SUM)
global_loss = loss_and_count[0] / loss_and_count[1]
```

为什么不直接平均各 rank 的平均 loss？如果 rank 的样本数不同，平均的平均会产生偏差。使用总 loss 和总样本数能正确加权。

分类准确率同理：先汇总 `correct_count` 和 `sample_count`，再相除。

---

## 9. 梯度累积与 `no_sync()`

普通梯度累积连续执行多次 backward，再 step 一次。若直接使用 DDP，每次 backward 都会同步梯度，数学上可能仍可用，却产生不必要通信。

DDP 提供 `no_sync()`：

```python
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(loader):
    should_step = ...
    sync_context = nullcontext() if should_step else model.no_sync()

    with sync_context:
        loss = loss_fn(model(x), y) / group_size
        loss.backward()

    if should_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

重要事实：

- `no_sync()` 不会跳过本地 backward；
- 梯度仍累加到 `.grad`；
- 它只让中间 micro-batch 暂不跨 rank 同步；
- 最后一次同步 backward 会同步累计梯度；
- forward 必须也位于 `no_sync()` context 内，示例脚本遵守这一点。

### 9.1 最后不足 $K$ 个 micro-batch

若 loader 有 10 个 iteration、累积 4 次，分组为 `4 + 4 + 2`。最后一组 loss 应除以 2，而不是固定除以 4，否则最后一组梯度被低估。

示例通过当前 group 的真实大小处理残余 batch。这是许多最小示例为了简洁而忽略的边界条件。

---

## 10. AMP 与 DDP 的组合

AMP 仍发生在每个 rank 本地：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = loss_fn(model(x), y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

DDP 在 backward hook 中同步相应梯度。各 rank 的 scaler 状态通常因为输入、梯度和更新行为一致而保持一致，但 checkpoint 仍应保存 scaler state。

本例的 `--amp` 只在 CUDA 上启用；CPU/Gloo 教学路径会自动关闭它：

```bash
torchrun --standalone --nproc-per-node=gpu \
  knowledge/pytorch-ddp/assets/train.py \
  --device cuda --amp
```

`--nproc-per-node=gpu` 会按可见 GPU 数启动 worker。若只想使用部分 GPU，应先正确设置作业资源或 `CUDA_VISIBLE_DEVICES`，然后确保每个 local rank 都有唯一可见设备。

---

## 11. Checkpoint 与恢复训练

### 11.1 为什么只让 rank 0 保存

DDP 各 rank 拥有相同模型参数和 optimizer state。所有 rank 同时写同一路径会竞争并损坏文件，因此简单场景只让 rank 0 写：

```python
if rank == 0:
    torch.save(...)
```

保存 DDP 模型时通常使用未包装模块：

```python
raw_model = model.module if isinstance(model, DDP) else model
state = raw_model.state_dict()
```

这样 checkpoint 不会带额外 `module.` 前缀依赖。

### 11.2 为恢复训练保存的内容

示例保存：

- 完成的 epoch；
- model state；
- optimizer state；
- AMP scaler state；
- 序列化后的参数配置。

配置中的 `Path` 被转成字符串，使 checkpoint 能使用 `torch.load(weights_only=True)` 的安全模式读取。不要对不可信 checkpoint 使用允许任意 pickle 对象的加载方式。

恢复示例：

```bash
python knowledge/pytorch-ddp/assets/train.py \
  --device cpu \
  --epochs 10 \
  --resume /tmp/ddp-single/checkpoint.pt \
  --output-dir /tmp/ddp-single
```

`--epochs=10` 表示最终训练到第 10 个 epoch，不是再训练 10 个。

### 11.3 多机路径要求

示例让所有 rank 从相同 `--resume` 路径加载，并由 rank 0 保存。因此多机时 output/resume 路径必须位于所有节点可见的共享文件系统。

如果各节点只有本地盘，需要改为 rank 0 加载后 broadcast，或使用适合分布式 checkpoint 的系统。

示例只恢复到 epoch 边界；精确恢复到任意 iteration 还需要 sampler/data pipeline 位置和随机数状态。

---

## 12. 单机多 GPU

环境前提：

- Linux；
- 至少两张对当前进程可见的 CUDA GPU；
- PyTorch 构建包含 NCCL；
- 每个 worker 绑定唯一 GPU。

启动：

```bash
torchrun --standalone --nproc-per-node=gpu \
  knowledge/pytorch-ddp/assets/train.py \
  --device cuda \
  --amp \
  --epochs 6 \
  --output-dir /tmp/ddp-gpu
```

也可以明确指定进程数：

```bash
torchrun --standalone --nproc-per-node=4 ...
```

前提是四张 GPU 都可见。进程数多于可见 GPU 时，示例会在初始化后检查 local rank 并报错；真实程序不应让多个 DDP worker 意外共享同一 GPU。

### 12.1 不建议用 `nn.DataParallel` 作为过渡目标

`DataParallel` 是单进程、多线程模型；DDP 是多进程模型，通常有更好的扩展性和更明确的设备隔离。学习现代 PyTorch 多 GPU 训练应直接理解 DDP。

---

## 13. 多机多 GPU：通用手动 `torchrun`

假设：

- 两台机器：`node0.example.com` 与 `node1.example.com`；
- 每台 8 张 GPU；
- 两台机器都能访问 `node0.example.com:29400`；
- 仓库、Python 环境和共享 checkpoint 路径一致；
- 防火墙和容器网络允许通信。

在**两台机器上执行相同命令**：

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-id=ddp-demo-001 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=node0.example.com:29400 \
  knowledge/pytorch-ddp/assets/train.py \
  --device cuda \
  --amp \
  --output-dir /shared/checkpoints/ddp-demo-001
```

所有节点必须使用相同的 `nnodes`、`nproc-per-node`、`rdzv-id` 和 endpoint。`rdzv-id` 应对每个并发作业唯一，避免不同作业错误地加入同一 worker group。

### 13.1 两个网络平面

需要同时考虑：

1. **控制面**：节点能否解析并连接 rendezvous endpoint；
2. **数据面**：NCCL collective 实际选择了哪个网卡、RDMA 或 GPU Direct 路径。

有多张网卡时，自动选择可能不符合预期：

```bash
export NCCL_SOCKET_IFNAME=eth0
```

接口名必须替换成真实高速网络接口，不要盲目复制 `eth0`。CPU/Gloo 对应 `GLOO_SOCKET_IFNAME`。

### 13.2 固定规模与弹性不是一回事

`--nnodes=2` 表示固定两个节点。`--nnodes=1:4 --max-restarts=...` 属于 elastic 作业：成员变化时 worker group 会整体重启，rank 和 world size 可能变化。

要实现有意义的容错，训练脚本必须定期 checkpoint，并能从 checkpoint 恢复。`torchrun` 重启进程不会自动恢复模型进度。

---

## 14. Slurm 模板如何理解

[slurm.sbatch](assets/slurm.sbatch) 使用常见模式：

```text
srun：每个节点启动一个 torchrun agent
torchrun：每个 agent 按本节点 GPU 数启动 worker
```

提交前至少要适配：

- partition、account、QoS 和 wall time；
- 每节点 GPU 数与 `--nproc-per-node`；
- Python/conda/module 环境；
- 仓库工作目录；
- 共享 checkpoint 路径；
- 集群推荐的 rendezvous 和网络接口；
- Slurm 对 `SLURM_NODEID`、容器和 task binding 的具体设置。

Slurm 集群策略差异很大，因此模板没有被标为“可直接通用运行”。确定实际环境后，应优先阅读该集群管理员提供的 PyTorch 示例。

---

## 15. 常见错误与排查顺序

### 15.1 所有进程在初始化阶段等待

优先检查：

- 声明的节点/worker 是否全部启动；
- `rdzv-id`、endpoint 和端口是否一致；
- hostname 是否可解析；
- 端口和网卡是否被防火墙阻断；
- 不同节点的 PyTorch/CUDA/NCCL 环境是否一致。

### 15.2 backward hang

DDP 要求各 rank 以兼容顺序参与梯度 collective。常见原因：

- 某 rank 先抛异常，其他 rank 仍在等待；
- 不同数据触发不同模型分支，参与 backward 的参数集合不同；
- 某个 rank 少执行一次 iteration；
- 用户自定义 collective 的顺序或 tensor shape 不一致；
- `no_sync()` 在各 rank 的使用模式不同。

不要把 `find_unused_parameters=True` 当万能修复。先确认 unused parameter 是否是预期模型语义；该选项还会引入额外 autograd graph 遍历。

### 15.3 数据重复或每个 epoch 顺序一样

检查：

- 是否使用 `DistributedSampler`；
- DataLoader 是否错误地再次 `shuffle=True`；
- 每个 epoch 是否调用 `sampler.set_epoch(epoch)`；
- rank/world size 是否正确传给自定义 sampler；
- dataset 是否在不同 rank 上构造出不同长度。

### 15.4 指标不可信

检查打印的是 rank 0 局部指标还是 all-reduce 后的全局指标；检查验证 sampler 是否 padding 重复样本；检查分母是 batch 数还是样本数。

### 15.5 多卡比单卡慢

优先检查：

- 每 rank batch 是否太小；
- dataloader 是否喂不满多张卡；
- global batch 是否被意外改变；
- 是否频繁 `.item()` 或 barrier；
- gradient accumulation 是否忘记 `no_sync()`；
- GPU 是否通过预期高速互连通信；
- 通信是否能与 backward 重叠；
- 日志和 checkpoint 是否造成同步 I/O。

---

## 16. 调试环境变量

遇到初始化、collective 不匹配或 NCCL 问题时，可临时启用：

```bash
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

进一步缩小 NCCL 范围：

```bash
export NCCL_DEBUG_SUBSYS=COLL   # collective 类型/大小/顺序
# 或
export NCCL_DEBUG_SUBSYS=GRAPH  # 拓扑检测
```

`DETAIL` 会增加检查和日志开销，只用于排错，不应作为正式性能测试配置。

推荐在日志中为每行加 rank，并在关键阶段打印：

```text
before init -> after init -> data ready -> before forward
-> before backward -> after backward -> after step -> before checkpoint
```

最后一个所有 rank 都能到达的阶段，通常能快速缩小 hang 的范围。

---

## 17. 不均匀输入与 `DDP.join()`

训练时如果某些 rank 提前耗尽数据，它们不再进入后续 gradient collective，其他 rank 可能 hang。首选方案通常是让 sampler 保证相同 iteration 数。

当不均匀输入是业务要求时，可以研究 DDP join context：它通过 shadow collectives 匹配仍在训练的 rank。使用前必须核对模型中的额外 collective；如果 forward 还有 SyncBatchNorm 等非 DDP collective，配置不当仍可能出问题。

这属于进阶工具，不应代替修复意外的数据长度不一致。

---

## 18. 性能实验应该怎样比较

### 18.1 固定问题定义

比较单卡和 DDP 时记录：

- world size；
- per-rank micro-batch；
- accumulation steps；
- global batch；
- precision；
- 数据 worker 数；
- warm-up 与统计 step；
- 是否包含 validation/checkpoint 时间。

### 18.2 两种合理实验

**固定 global batch**：观察增加设备后完成同一个 optimizer step 能快多少，适合 strong scaling。

**固定 per-rank batch**：global batch 随设备数增加，观察系统总吞吐，适合 weak scaling；但优化语义也随之变化。

### 18.3 不要用本例衡量 GPU 扩展效率

本例模型极小，通信和 Python 开销远大于计算，多卡很可能更慢。它用于验证控制流。性能 benchmark 应使用接近真实工作负载的模型、tensor shape 和 dataloader。

---

## 19. 从单卡到多机的迁移检查表

### 单进程基线

- [ ] loss 能下降；
- [ ] checkpoint 能保存和恢复；
- [ ] 固定 seed 后行为可解释；
- [ ] 明确 per-rank batch 和目标 global batch。

### 单机 CPU 双进程

- [ ] Gloo DDP 能完成；
- [ ] rank 0 独占普通日志与 checkpoint；
- [ ] 全局 loss 经过 all-reduce；
- [ ] sampler 每 epoch 调用 `set_epoch()`。

### 单机多 GPU

- [ ] 一进程一 GPU；
- [ ] NCCL backend；
- [ ] AMP 与 checkpoint 恢复正常；
- [ ] 检查吞吐和显存，而不只看 GPU utilization。

### 多机

- [ ] 软件、代码、数据视图一致；
- [ ] rendezvous endpoint 所有节点可达；
- [ ] job id 唯一；
- [ ] NCCL 使用预期网络接口；
- [ ] checkpoint 路径共享或实现了跨节点分发；
- [ ] 先两节点小作业，再扩完整规模；
- [ ] 保存 rank-aware 日志以便定位失败节点。

---

## 20. 自测与练习

### 概念题

1. 两节点、每节点 8 个 DDP worker，per-rank batch 4，累积 2 次，global batch 是多少？
2. 为什么只让 rank 0 打印不能保证打印的是全局 loss？
3. DDP 为什么不需要同步 optimizer state？在哪些人为修改下这个结论会失效？
4. 为什么 `LOCAL_RANK` 适合选择 GPU，而 global `RANK` 不适合？
5. 忘记 `sampler.set_epoch()` 会怎样？
6. `no_sync()` 跳过了什么，又保留了什么？
7. `torchrun --max-restarts=3` 为什么不能取代 checkpoint？

第 1 题答案：

$$
B_{global}=4\times2\times(2\times8)=128
$$

### 动手练习

1. 分别运行单进程 batch 32 和双进程每 rank batch 16，比较 loss 曲线。
2. 暂时删掉 `train_sampler.set_epoch(epoch)`，打印前几个样本索引，观察 epoch 间变化。
3. 故意只打印 rank 0 的局部验证 loss，再与 all-reduce 后结果比较。
4. 加入 `--log-every`，确保普通日志只由 rank 0 输出，但调试日志包含 rank。
5. 在 checkpoint 中保存 CPU、CUDA 和 Python RNG state，并验证 epoch 边界恢复。
6. 构造一个不同 rank 走不同分支的模型，用调试环境变量观察 DDP 报错。

---

## 21. 最终速记

```text
python train.py
    -> 一个进程，一份数据，一份模型

torchrun --nproc-per-node=N train.py
    -> N 个进程
    -> local rank 绑定设备
    -> DistributedSampler 切数据
    -> DDP 包模型
    -> backward 同步梯度
    -> all-reduce 同步指标
    -> rank 0 写日志/checkpoint

多机
    -> 每个节点运行 torchrun agent
    -> 相同 rendezvous 参数
    -> 节点间网络必须可达
    -> checkpoint 必须可恢复
```

最核心的三个区分：

1. **DDP 自动同步梯度，不自动同步你的指标。**
2. **global batch 由 per-rank batch、累积步数和 world size 共同决定。**
3. **单机到多机主要增加部署与网络复杂度，不改变 DDP 的训练数学。**

## 当前官方资料

- [torchrun / Elastic Launch](https://docs.pytorch.org/docs/stable/elastic/run)
- [PyTorch distributed communication package](https://docs.pytorch.org/docs/stable/distributed)
- [DistributedDataParallel API](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [PyTorch DDP tutorial series](https://docs.pytorch.org/tutorials/beginner/ddp_series_intro.html)

## 下一步

建议先依次亲手运行单进程 CPU、双进程 CPU/Gloo、单机多 GPU 三个阶段。获得真实集群后，再根据调度器和网络拓扑修改 Slurm/多机启动部分；不要在没有单机正确性基线时直接调试多机 NCCL。
