---
title: PyTorch 到 ONNX 与 TensorRT 部署
topic: pytorch-onnx-tensorrt-deployment
topics: [pytorch-onnx-tensorrt-deployment]
framework: PyTorch
status: learning
updated: 2026-07-30
---

# PyTorch 到 ONNX 与 TensorRT 部署

## 学习目标

完成本文后，应当能够：

- 解释 `.pth`、ONNX model 与 TensorRT engine 分别保存了什么；
- 将一个 PyTorch 模型以明确的输入契约导出为 ONNX；
- 逐级验证 PyTorch、ONNX Runtime 与 TensorRT 的数值一致性；
- 为动态 batch 或动态分辨率设计 TensorRT optimization profile；
- 使用 `trtexec` 构建、验证和分析 FP32/FP16 engine；
- 根据错误出现的阶段，定位权重加载、图导出、算子支持、shape、精度或性能问题。

本文先打通 FP32/FP16 主链路，并提供一套可以逐步执行的小型 CNN 实验。INT8/PTQ/QAT 和自定义算子会分别作为后续专题，因为它们各自包含独立的训练、图转换和部署问题。

## 前置知识与环境边界

- 熟悉 `torch.nn.Module`、tensor shape、`model.eval()` 和 checkpoint；
- 理解 CPU/GPU、dtype 和 batch 的基本含义；
- 示例默认 Linux、NVIDIA GPU、PyTorch、ONNX、ONNX Runtime 与 TensorRT；
- TensorRT API 迭代较快，本文采用现代 `torch.export`-based ONNX exporter 和 TensorRT name-based runtime API 的思路。实际项目必须记录 PyTorch、ONNX、ONNX Runtime、TensorRT、CUDA、cuDNN、GPU 型号与 driver 版本。

## 核心心智模型：不是文件格式转换，而是逐层编译

常说的“`.pth` 转 ONNX 再转 TensorRT”容易让人误以为只是改后缀。更准确的过程是：

```text
Python 模型定义 + checkpoint + 输入/预处理契约
                         │ load_state_dict
                         ▼
              可执行的 PyTorch Module
                         │ capture / normalize / translate
                         ▼
                 ONNX 计算图 + 权重
                         │ parse / optimize / tactic selection
                         ▼
             TensorRT serialized engine (.plan)
                         │ execution context + CUDA buffers/stream
                         ▼
                       输出
```

三个阶段的职责不同：

| 产物 | 主要内容 | 仍然缺少什么 |
| --- | --- | --- |
| `.pth` / `.pt` checkpoint | 通常是参数、buffer、优化器等 Python 对象 | 未必包含模型结构和输入契约 |
| `.onnx` | 静态数据流图、operator、tensor type/shape 信息与 initializer | 不负责为某张 GPU 选择最快 CUDA kernel |
| TensorRT engine / plan | 已优化网络、选定 tactics、权重和运行所需元数据 | 通常不能当作跨 TensorRT/GPU 环境的通用交换格式 |

真正的部署契约还包括：

$$
f_\text{system}=\text{postprocess}\circ f_\text{model}\circ\text{preprocess}.
$$

如果训练使用 RGB、`[0,1]`、特定 mean/std，而部署使用 BGR、`[0,255]`，即使三个后端的裸模型完全一致，系统输出仍会错误。

## 1. 先正确恢复 PyTorch 模型

### 1.1 `.pth` 不是统一格式

`.pth` 只是惯用后缀。文件可能保存：

```python
# 推荐的可维护方式
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pth")

# 也有人只保存 state_dict
torch.save(model.state_dict(), "weights.pth")

# 保存整个 Python 对象，强依赖原类定义和环境，不适合作为长期部署接口
torch.save(model, "model.pth")
```

导出 ONNX 前通常需要重新构造模型并严格加载权重：

```python
import torch

model = MyModel(...)                         # 必须与训练结构一致
checkpoint = torch.load("checkpoint.pth", map_location="cpu", weights_only=True)
state_dict = checkpoint.get("model", checkpoint)

# 仅在确认训练使用 DataParallel/DDP 前缀时处理
state_dict = {
    key.removeprefix("module."): value
    for key, value in state_dict.items()
}
missing, unexpected = model.load_state_dict(state_dict, strict=True)
assert not missing and not unexpected

model.eval()
```

不要为了“先跑起来”长期使用 `strict=False`。它可能让新层保持随机初始化，却直到部署精度异常才暴露。

### 1.2 固定一份黄金输入和黄金输出

导出前保存一份能复现的输入、输出及预处理元数据：

```python
torch.manual_seed(42)
x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

with torch.inference_mode():
    y_pt = model(x)

torch.save({"input": x, "output": y_pt}, "golden_io.pt")
```

真实项目最好同时保存一小组真实样本，覆盖空场景、极值、小目标和不同 shape。随机输入适合检查图和数值，不足以衡量任务指标。

## 2. 导出 ONNX：明确图与输入契约

### 2.1 最小静态 shape 导出

现代 PyTorch 推荐基于 `torch.export` 的 exporter：

```python
import torch

model.eval()
example = (torch.randn(1, 3, 224, 224),)

onnx_program = torch.onnx.export(
    model,
    example,
    input_names=["images"],
    output_names=["logits"],
    opset_version=18,       # 应按目标 runtime 的支持情况选择
    dynamo=True,
    verify=True,
    report=True,
)
onnx_program.save("model.onnx")
```

关键点：

- example input 不只是“占位数据”，它帮助 exporter 捕获计算并推导约束；
- Python `if`、循环、list 操作和数据相关控制流不一定能直接变成普通 ONNX 数据流；
- 非 tensor 参数可能在导出时被固化为常量；
- `opset_version` 表示 ONNX operator 语义版本，不等于 ONNX 包版本；
- 大模型可能把权重保存为 external data，移动文件时必须一起移动外部权重文件。

### 2.2 动态 shape 是带约束的变量，不是“任意尺寸”

若只希望 batch 动态，可显式声明：

```python
batch = torch.export.Dim("batch", min=1, max=16)

onnx_program = torch.onnx.export(
    model,
    (torch.randn(4, 3, 224, 224),),
    input_names=["images"],
    output_names=["logits"],
    dynamo=True,
    dynamic_shapes={"images": {0: batch}},
)
onnx_program.save("model_dynamic.onnx")
```

动态维度会传播到 shape 运算。以下代码很危险：

```python
# 将 example 的 H、W 读成 Python 整数后写死在图中的风险较高
h, w = int(x.shape[-2]), int(x.shape[-1])
```

应尽量让 shape 保持为图中的符号值，并用 tensor operator 表达 reshape、slice 和 padding。导出后要用最小、典型和最大 shape 分别执行，而不是只复测 example shape。

### 2.3 导出失败时先判断是哪一类边界

| 症状 | 常见原因 | 优先动作 |
| --- | --- | --- |
| graph capture 失败 | Python 控制流、数据结构或副作用无法被捕获 | 阅读 exporter report，缩小到最小复现，改写 forward |
| 找不到 ONNX translation | ATen/custom operator 没有对应 ONNX 表达 | 用等价基础算子分解；后续再考虑自定义 translation |
| ONNX 能生成但 shape 错 | reshape、index 或动态轴被固化 | 检查 graph input/value info，并测试多组 shape |
| 输出名称/数量变化 | dict、list、optional output 被展开或消除 | 为部署写薄 wrapper，固定纯 tensor 输入输出 |

建议为部署单独增加一个不含预处理框架对象的 wrapper：

```python
class DeployModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        return outputs["logits"]  # 固定为明确的 tensor 接口
```

## 3. ONNX 不是导出成功就结束

### 3.1 先做结构检查

```python
import onnx

model_onnx = onnx.load("model.onnx")
onnx.checker.check_model(model_onnx)
print(onnx.helper.printable_graph(model_onnx.graph))
```

可配合 Netron 查看输入输出、initializer、shape 子图和意外的 `Cast`/`Transpose`。结构检查只能证明图满足 ONNX 规则，不证明语义与 PyTorch 相同。

### 3.2 用 ONNX Runtime 做第一道数值隔离

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)
y_ort = session.run(["logits"], {"images": x.cpu().numpy()})[0]

np.testing.assert_allclose(
    y_pt.cpu().numpy(),
    y_ort,
    rtol=1e-4,
    atol=1e-5,
)
```

不要机械套用一个 tolerance。合理阈值取决于 dtype、输出尺度、累积深度和任务：

$$
e_\text{abs}=|y_a-y_b|,
\qquad
e_\text{rel}=\frac{|y_a-y_b|}{\max(|y_b|,\epsilon)}.
$$

对于分类还应比较 top-k；检测、分割、生成模型应比较最终任务指标。接近决策阈值的微小数值误差可能导致离散后处理产生明显变化。

调试时采用逐级比较：

```text
PyTorch FP32 ≈ ONNX Runtime CPU FP32
             ≈ TensorRT FP32
             ≈ TensorRT FP16
```

哪一级首次发生偏差，问题通常就在该级转换或精度策略，而不是原始 checkpoint。

## 4. TensorRT：解析、优化、构建 engine

### 4.1 先用 `trtexec` 建立基线

`trtexec` 可以解析 ONNX、构建 engine、运行正确性烟雾测试和测量性能，是隔离 Python runtime 代码问题的最快工具之一。

下面的 FP32、动态 shape 和 benchmark 参数适用于 TensorRT 10.x/11.x；精度控制存在重要版本差异。

静态 shape、FP32：

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp32.plan \
  --profilingVerbosity=detailed \
  --dumpLayerInfo
```

TensorRT 10.x 的 FP16：

```bash
trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp16.plan \
  --fp16 \
  --profilingVerbosity=detailed
```

TensorRT 10.x 的动态 batch FP16：

```bash
trtexec \
  --onnx=model_dynamic.onnx \
  --saveEngine=model_dynamic_fp16.plan \
  --fp16 \
  --minShapes=images:1x3x224x224 \
  --optShapes=images:8x3x224x224 \
  --maxShapes=images:16x3x224x224
```

用真实 shape 测量已有 engine：

```bash
trtexec \
  --loadEngine=model_dynamic_fp16.plan \
  --shapes=images:8x3x224x224 \
  --warmUp=500 \
  --duration=10 \
  --useCudaGraph
```

TensorRT 11.x 使用 strongly typed network，已经移除 `--fp16`、`--int8`、`--best` 等精度开关。11.x 中应先用离线转换工具把 dtype/quantization 表达到 ONNX 图里，再由 `trtexec` 构建；不能原样照搬上面的 10.x FP16 命令。实际使用前运行 `trtexec --help`，并把命令与工具版本一起纳入部署记录。

### 4.2 Optimization profile 的真正含义

对于 runtime dimension，TensorRT 需要：

```text
min shape ≤ runtime shape ≤ max shape
                   ↑
           opt shape 是重点调优点
```

- `min`/`max` 定义 engine 接受的范围；越宽不一定越好，可能限制 tactic 或增加资源需求；
- `opt` 不是默认输入，也不是范围平均值，而是 builder 重点选择高性能 tactic 的 shape；
- 线上 shape 分布如果有两个明显峰值，两个较窄的 profile 往往比一个极宽 profile 更合理；
- runtime shape 超出 profile 会报错，不会自动重新构建 engine；
- 切换 profile 或改变 shape 后，首次 enqueue 可能有额外 shape/resource 更新开销。

### 4.3 Python 构建骨架

理解下面的阶段即可；初学时优先用 `trtexec` 验证 ONNX：

```python
from pathlib import Path
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

onnx_bytes = Path("model_dynamic.onnx").read_bytes()
if not parser.parse(onnx_bytes):
    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    raise RuntimeError(errors)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

profile = builder.create_optimization_profile()
profile.set_shape(
    "images",
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(16, 3, 224, 224),
)
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("TensorRT engine build failed")
Path("model_dynamic_fp16.plan").write_bytes(serialized_engine)
```

builder 会为 layer 搜索可用 tactic。构建耗时不等于推理耗时；生产构建可使用 timing cache 减少重复搜索，但 cache 与硬件、TensorRT/CUDA 版本及 builder 配置相关。

### 4.4 Runtime 还要负责什么

engine 文件本身不会替你完成：

1. 选择 optimization profile；
2. 设置每个动态输入的 runtime shape；
3. 分配足够的 device input/output memory；
4. 绑定 tensor name 与 device address；
5. 在正确的 CUDA stream 上异步拷贝、enqueue、取回结果；
6. 保证 buffer 生命周期覆盖异步执行。

现代 TensorRT runtime 的核心流程可概括为：

```python
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(Path("model.plan").read_bytes())
context = engine.create_execution_context()

context.set_input_shape("images", runtime_shape)
# 根据 context.get_tensor_shape(name) 分配 device buffers
# context.set_tensor_address(name, int(device_pointer))
# context.execute_async_v3(stream_handle=stream)
```

这里省略 CUDA 内存管理细节，避免把某个 CUDA Python binding 的版本接口误当成 TensorRT 核心概念。实践中可以使用官方 `cuda-python`，或让 PyTorch/CuPy tensor 提供 device memory，并仔细管理连续性、dtype、地址与 stream 同步。

## 5. 正确性验证：建立可追责的误差阶梯

### 5.1 测试矩阵

至少记录：

| Backend | Precision | Shapes | 比较对象 |
| --- | --- | --- | --- |
| PyTorch | FP32 | min/opt/max + 真实样本 | 黄金基线 |
| ONNX Runtime | FP32 | 同上 | PyTorch FP32 |
| TensorRT | FP32/TF32 配置明确 | 同上 | ORT/PyTorch FP32 |
| TensorRT | FP16 | 同上 | TensorRT FP32 + 任务指标 |

对于 stochastic、NMS、排序、并行 reduction 等操作，还要判断它们是否具有确定性。只比较一次随机输入的最大绝对误差，不能证明模型可上线。

### 5.2 二分定位中间层

最终输出不一致时：

1. 先确认输入 bytes、layout、dtype、shape 完全一致；
2. 在 PyTorch wrapper 和 ONNX 中临时暴露一组中间 tensor；
3. 找到第一个明显偏离的节点；
4. 检查该节点前后的 `Cast`、broadcast、padding、resize、normalization 和 reduction；
5. 对 FP16 敏感层尝试保留更高精度，以验证是否为数值问题。

重点检查：

- `Resize` 的 coordinate transformation 与 align-corners 语义；
- padding 的方向和数值；
- `Softmax`、归一化、指数和除法的溢出/下溢；
- 很大或很小激活进入 FP16；
- 相同值排序、NMS 阈值附近的离散变化；
- implicit broadcast 和错误的动态维度。

## 6. 性能分析：先定义指标，再谈加速比

### 6.1 Latency 与 throughput 不是同一个目标

若处理 $N$ 个样本总耗时为 $T$：

$$
\text{throughput}=\frac{N}{T},
\qquad
\text{average latency}\neq\frac{1}{\text{throughput}}
$$

当存在 batch、并发 stream、排队和异步 pipeline 时，单请求端到端 latency 不能由 throughput 简单倒推。报告中至少说明：

- batch size、input shape、precision；
- warmup、测量轮数、同步位置；
- GPU 型号、功耗/时钟状态、TensorRT/CUDA 版本；
- 是纯 engine compute、GPU latency，还是包含 H2D/D2H、预处理、后处理的端到端 latency；
- 平均值之外的 P50/P95/P99。

### 6.2 常见“TensorRT 没变快”的原因

| 观察 | 可能原因 | 检查方法 |
| --- | --- | --- |
| GPU latency 很低但端到端慢 | CPU 预处理、同步拷贝或 Python 调度占主导 | Nsight Systems；分别计时各阶段 |
| FP16 几乎不加速 | 网络受 memory bandwidth/launch overhead 限制，或发生格式转换 | `--dumpLayerInfo`、layer profile |
| 动态 shape 首次特别慢 | shape/profile 切换产生一次性更新 | 固定 shape 复测并预热 |
| batch=1 GPU 利用率低 | 小 kernel、频繁 launch | CUDA Graph、融合、并发请求 |
| engine build 很慢 | tactic 搜索 | 开发期降低 optimization level；生产使用 timing cache |
| profile 很宽且性能差 | opt 与线上主流 shape 不一致，部分 tactic 不适用 | 按真实 shape 分布拆 profile |

优化顺序应是：建立可靠基线 → 找到瓶颈 → 只改变一个变量 → 复测正确性和性能。

## 7. 可运行实验：从 checkpoint 到 TensorRT benchmark

这一实验使用一个很小的 `Conv2d → BatchNorm2d → ReLU → pooling → Linear` 网络。模型输入契约是 `float32`、NCHW、`[N, 3, 32, 32]`，输出是 `[N, 10]`；只有 batch 是动态维度，允许范围为 $[1,16]$。

实验文件位于 [`assets/`](assets/)：

| 文件 | 职责 |
| --- | --- |
| [`demo_model.py`](assets/demo_model.py) | 定义可复现的小模型和明确的 tensor 接口 |
| [`export_onnx.py`](assets/export_onnx.py) | 保存 checkpoint 与 batch 1/8/16 黄金数据，严格恢复权重并导出动态 ONNX |
| [`validate_onnx.py`](assets/validate_onnx.py) | 运行 ONNX checker，并逐个 batch 比较 ORT 与 PyTorch |
| [`run_trtexec.sh`](assets/run_trtexec.sh) | 构建动态 TensorRT engine，并测量 batch 1/8/16 |

### 7.1 准备隔离环境

在 `assets/` 目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch onnx onnxruntime onnxscript
```

TensorRT 需要与 NVIDIA driver/CUDA 环境匹配。可以使用 NVIDIA 官方容器，或按对应 TensorRT 版本的安装说明配置 `trtexec`。不要仅为了运行 ORT 阶段就安装 TensorRT。

### 7.2 导出 checkpoint、黄金数据和 ONNX

```bash
python export_onnx.py --output-dir artifacts
```

脚本刻意完整执行：构造模型 → 保存 `state_dict` → 创建同结构模型 → `strict=True` 恢复 → 生成黄金数据 → 声明 batch 约束 → 导出 ONNX。`artifacts/export_report/` 会保留 exporter report，失败时应先读它，而不是反复随机改 opset。

现代 exporter 可能依次尝试多种 graph capture 策略。日志中某一次尝试显示 `❌`，但后续策略成功并最终保存 ONNX，并不表示整个导出失败；应结合最终返回值、report、ONNX checker 和数值验证判断。反过来，即使文件已经生成，也不能跳过后两项验证。

预期产物：

```text
artifacts/
├── demo_checkpoint.pth
├── demo_dynamic.onnx
├── export_report/
└── golden_io.pt
```

### 7.3 隔离验证 ONNX

```bash
python validate_onnx.py --artifacts artifacts
```

预期 batch 1、8、16 都打印 `PASS`。这里同时测试了动态范围边界与重点 shape；若 batch 8 通过而 1 或 16 失败，首先检查符号 shape 是否在 `reshape`、索引或 Python 整数转换中被固化。

### 7.4 构建并测量 TensorRT

FP32 基线：

```bash
chmod +x run_trtexec.sh
./run_trtexec.sh artifacts
```

TensorRT 10.x 还可以测试 builder 的 FP16 精度开关：

```bash
PRECISION=fp16 ./run_trtexec.sh artifacts
```

脚本会检查 `trtexec --help`。若当前是 TensorRT 11.x，它会拒绝使用已移除的 `--fp16` 并解释下一步，而不会悄悄构建一个实际为 FP32 的 engine。11.x 的离线 dtype 转换和 INT8 图量化放在后续量化专题中系统学习。

`trtexec` 输出中的 `GPU Compute Time` 更接近 engine compute；`Latency` 还会受到 enqueue 与传输配置影响。记录 batch 1/8/16 的 P50/P95、throughput 和显存后，再尝试改变 `optShapes`，观察 profile 如何影响 tactic 与性能。

### 7.5 有意制造三个故障

1. 把 `validate_onnx.py` 的 input name 改错，观察 runtime 接口错误；
2. 把 `run_trtexec.sh` 的运行 batch 改为 17，观察 profile 越界；
3. 在 `DemoNet.forward` 中加入依赖 tensor 值的 Python 分支，阅读 exporter report，并尝试用可导出的 tensor control flow 重写。

这三个故障分别对应接口契约、后端 shape 约束和 graph capture 边界，能够练习“先判断故障阶段，再处理细节”。

## 8. 一套可复用的工程目录与门禁

```text
deployment/
├── export_onnx.py
├── validate_onnx.py
├── build_engine.sh
├── validate_tensorrt.py
├── benchmark.sh
├── golden/
│   ├── inputs.*
│   └── expected.*
└── manifests/
    └── model-version.yaml
```

manifest 建议记录：

```yaml
model_commit: "..."
checkpoint_sha256: "..."
input_names: [images]
input_dtype: float32
input_layout: NCHW
preprocess: "RGB, x/255, normalize(mean, std)"
dynamic_shapes:
  images:
    min: [1, 3, 224, 224]
    opt: [8, 3, 224, 224]
    max: [16, 3, 224, 224]
opset: 18
pytorch: "..."
onnx: "..."
onnxruntime: "..."
tensorrt: "..."
cuda: "..."
gpu: "..."
precision: fp16
```

CI 或发布门禁至少包括：

1. checkpoint 严格加载；
2. ONNX checker 通过；
3. ORT 与 PyTorch 在样本集上通过数值/任务指标阈值；
4. TensorRT engine 构建成功且 parser 无被忽略的错误；
5. TensorRT min/opt/max shape 均通过；
6. FP16 任务指标下降不超过预算；
7. 延迟、吞吐、峰值显存不超过回归阈值。

## 9. 常见误区与排查决策树

```text
checkpoint 无法加载
  └─ 先检查模型定义、key 前缀、missing/unexpected keys

PyTorch 正确，但 ONNX 导不出
  └─ 看 exporter report：capture 问题还是 operator translation 问题

ONNX 导出成功，但 ORT 不一致
  └─ 查输入契约、动态 shape 固化、operator 语义和中间输出

ORT 正确，但 TensorRT parse/build 失败
  └─ 查 TensorRT operator/parser 支持、shape tensor、plugin 与版本矩阵

TensorRT FP32 正确，FP16 不正确
  └─ 查数值范围、敏感层、Cast/format conversion，局部提高精度验证

数值正确，但速度不理想
  └─ 分离 H2D/compute/D2H/预后处理，查 profile、融合、batch 与同步
```

额外提醒：

- `model.eval()` 不等于关闭 autograd；导出/基准时仍应使用 `torch.inference_mode()`；
- ONNX `dynamic_shapes` 与 TensorRT profile 是两层约束，两边都要配置；
- engine build 成功不等于结果正确，结果正确也不等于性能测量可信；
- 不要把在开发 GPU 上生成的 plan 文件默认复制到所有机器；优先在目标环境构建、缓存并验证；
- unsupported operator 不一定必须写 plugin：先尝试改写或分解成后端支持的基础算子，plugin 是维护成本更高的边界。

## 10. 自测与练习

### 概念自测

1. 为什么只有 `state_dict.pth` 时不能直接恢复模型结构？
2. ONNX opset、软件包版本和 TensorRT 版本分别解决什么问题？
3. 为什么 ONNX 输入已经是动态 batch，TensorRT 构建时仍需要 optimization profile？
4. `optShapes` 为什么不应随便设成 `min` 与 `max` 的平均值？
5. 为什么 FP16 的 `allclose` 通过后仍要测任务指标？
6. 为什么异步 GPU 推理直接用 Python 墙钟计时可能得到错误结果？

### 动手练习

选择一个包含 `Conv2d`、`BatchNorm2d`、`ReLU` 和 pooling 的小模型：

1. 保存并严格加载其 checkpoint；
2. 导出静态 ONNX，使用 ORT 与 PyTorch 比较；
3. 将 batch 改为 $[1,16]$ 的动态维度，测试 batch 1、8、16；
4. 用 `trtexec` 构建 FP32 和 FP16 engine；
5. 记录两种精度在 batch 1/8/16 下的误差、GPU latency 与 throughput；
6. 故意把部署输入从 RGB 改为 BGR，观察“后端数值一致但任务语义错误”的现象；
7. 故意用 batch 17 运行，解释错误来自 ONNX 图还是 TensorRT profile。

## 11. 相关知识与官方资料

- 仓库内：[训练循环](../training-loop/training-loop.ipynb)解释了 `train/eval`、autograd 与 checkpoint 的训练侧语义。
- PyTorch 官方：[ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)。
- ONNX Runtime 官方：[Python API quickstart](https://onnxruntime.ai/docs/get-started/with-python.html)。
- NVIDIA 官方：[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/python-api.html)、[Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)、[`trtexec` benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)、[10.x 到 11.x 的 `trtexec` 迁移](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)与[性能优化](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。

## 下一步

1. **训练算子扩展到部署 Plugin**：从 `torch.autograd.Function`、C++/CUDA Extension，到 ONNX translation 和 TensorRT Plugin V3，重点理解 forward/backward 与部署 forward 的边界。
2. **模型量化与 TensorRT INT8**：从数值表示、scale/zero-point、PTQ calibration，到 QAT、敏感层和逐层精度定位。
3. 回到本文练习，选择自己的真实模型建立 golden dataset、manifest 和自动回归门禁。
