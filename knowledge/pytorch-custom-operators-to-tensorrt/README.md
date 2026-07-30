---
title: 从 PyTorch 训练算子到 ONNX 与 TensorRT Plugin
topic: pytorch-custom-operators-to-tensorrt
topics: [pytorch-custom-operators-to-tensorrt, pytorch-onnx-tensorrt-deployment]
framework: PyTorch
status: learning
updated: 2026-07-30
---

# 从 PyTorch 训练算子到 ONNX 与 TensorRT Plugin

## 学习目标

完成本文后，应当能够：

- 区分 `nn.Module`、`autograd.Function`、custom operator、C++/CUDA Extension 与 TensorRT Plugin；
- 为自定义训练算子推导并验证 backward；
- 用 `torch.library` 注册 schema、实现、FakeTensor kernel 和 autograd formula；
- 理解 dispatcher、dispatch key、mutation/aliasing contract 与 `torch.compile` 的关系；
- 判断 ONNX 导出时应分解为标准算子，还是保留自定义节点；
- 描述 `IPluginV3` 的 build/runtime 职责与 ONNX parser 发现插件的过程；
- 建立 eager、compiled、ONNX、TensorRT 的逐级正确性与性能门禁。

本文使用标量偏置 Swish 作为贯穿示例：

$$
y=x\sigma(\beta x)+b,
$$

其中 $x,y\in\mathbb{R}^{*}$，$b\in\mathbb{R}$，$\beta$ 是非 tensor 标量。这个算子本可由标准算子组合实现，因此特别适合学习：先验证完整扩展链路，再判断真实项目是否值得维护 kernel/plugin。

## 前置知识与边界

- 熟悉 tensor、broadcast、chain rule 和 PyTorch 训练循环；
- 已阅读[PyTorch 到 ONNX 与 TensorRT 部署](../pytorch-onnx-tensorrt-deployment/README.md)；
- Python 示例可在 CPU 上运行，C++/CUDA 与 TensorRT 部分需要编译工具链和 NVIDIA 环境；
- 本文不讨论 INT8/QAT，它们属于下一篇量化专题；
- API 随版本变化，生产代码应固定 PyTorch、CUDA、编译器、TensorRT 和 GPU 架构矩阵。

## 核心心智模型：一个数学算子，多层契约

```text
数学定义
  │
  ├─ autograd contract：forward 保存什么，backward 返回哪些梯度
  ├─ dispatcher contract：schema、alias/mutation、CPU/CUDA/Fake 实现
  ├─ compiler contract：FakeTensor、shape/dtype、可追踪性
  ├─ interchange contract：分解成标准 ONNX，或自定义 domain::op
  └─ runtime contract：TensorRT Plugin 的 format、shape、workspace、enqueue
```

“训练时能跑”只证明第一小段链路。部署失败通常不是 CUDA 公式本身，而是某层契约没有表达清楚。

## 1. 五个常被混淆的扩展层级

| 层级 | 解决的问题 | 不自动提供什么 |
| --- | --- | --- |
| `nn.Module` | 组织参数、子模块和 forward | 新 operator identity 或自定义 kernel |
| `autograd.Function` | 自定义 forward/backward 记录 | dispatcher schema、FakeTensor、ONNX/TRT 支持 |
| `torch.library` custom op | 给 dispatcher 注册一等 operator | 高性能 C++/CUDA kernel |
| C++/CUDA Extension | 实现和打包 native kernel | ONNX translation 与 TensorRT Plugin |
| TensorRT Plugin | 扩展 TensorRT 可执行 layer | 训练 backward；Plugin 只服务推理 forward |

若表达式由 PyTorch 原生算子组成且性能足够，优先写普通函数或 `nn.Module`。只有当需要外部库、opaque kernel、融合性能或后端不认识的新语义时，才引入 custom op。

## 2. 从公式到可验证 backward

令 $s=\sigma(\beta x)$，则：

$$
\frac{\partial y}{\partial x}
=s+\beta x s(1-s),
\qquad
\frac{\partial L}{\partial b}
=\sum_i\frac{\partial L}{\partial y_i}.
$$

偏置是标量，因此其梯度必须把所有 broadcast 维度归约掉。`backward` 返回值数量和顺序必须与 `forward` 输入一致；`beta` 是 Python `float`，不可训练，所以返回 `None`。

[`assets/autograd_function.py`](assets/autograd_function.py)给出了最小 `autograd.Function`。它适合学习 tape 与 `ctx.save_for_backward`，但工程集成不应止步于此：compiler 看不到独立的 operator schema，也无法仅凭这个类获得其他 runtime 的实现。

### 2.1 两类测试不能互相替代

- `torch.autograd.gradcheck`：用有限差分检查梯度数学是否正确；通常使用 `float64` 和小输入。
- `torch.library.opcheck`：检查 schema、autograd 注册、FakeTensor 与 AOT dispatch 等注册契约；它不证明导数公式正确。

两者都通过才覆盖“数学正确”和“框架注册正确”。

## 3. `torch.library`：把算子变成 dispatcher 的一等公民

可运行实现见 [`assets/custom_swish.py`](assets/custom_swish.py)。它由四部分组成：

1. `custom_op("learn_ops::swish_bias", mutates_args=())` 定义命名空间、schema 和无 mutation 契约；
2. Python implementation 提供 eager CPU/CUDA 行为；
3. `register_fake` 只根据 metadata 构造输出，不访问真实 data；
4. `register_autograd` 的 `setup_context` 与 backward 提供训练语义。

### 3.1 Schema 是承诺，不只是类型提示

框架会利用 alias/mutation 信息做 functionalization、重排和内存规划。若声明无 mutation，却原地修改输入，eager 可能暂时“能跑”，compiled graph 可能错误。输出若是输入 view，也必须按 schema 表达 aliasing。

### 3.2 Dispatcher 做什么

调用 `torch.ops.learn_ops.swish_bias` 时，dispatcher 根据 tensor 的 device、autograd、vmap 等 dispatch key 选择实现。典型 native 扩展会分别注册：

```cpp
TORCH_LIBRARY(learn_ops, m) {
  m.def("swish_bias(Tensor x, Tensor bias, float beta) -> Tensor");
}

TORCH_LIBRARY_IMPL(learn_ops, CPU, m) {
  m.impl("swish_bias", &swish_bias_cpu);
}

TORCH_LIBRARY_IMPL(learn_ops, CUDA, m) {
  m.impl("swish_bias", &swish_bias_cuda);
}
```

定义 schema 和实现 kernel 是两个阶段。漏掉 CUDA 注册时，CPU 测试通过并不意味着 GPU 可用。

### 3.3 FakeTensor kernel 为什么重要

`torch.compile`/`torch.export` 在不执行真实 kernel 的情况下传播 shape、dtype 和 device。Fake kernel 应只做 metadata 推导：

- 不读取 data pointer；
- 不执行 `.item()`；
- 不依赖真实 CUDA kernel；
- 保持真实实现的 shape、stride、dtype 与 alias 语义。

数据依赖输出 shape 比“输出与输入同 shape”难得多，通常需要动态 shape API，并会限制后端优化。

## 4. 从 Python reference 到 C++/CUDA Extension

建议按以下顺序推进：

1. 用原生 PyTorch 组合写 reference；
2. 推导 backward，并通过 `gradcheck`；
3. 注册 custom op、Fake 与 autograd，运行 `opcheck`；
4. 实现 CPU kernel，逐元素与 reference 对齐；
5. 实现 CUDA kernel，覆盖 dtype、非连续输入、空 tensor 和极端值；
6. benchmark 证明扩展确有价值，再决定是否长期维护。

### 4.1 CUDA kernel 的最低检查项

- 用当前 CUDA stream，而非隐式创建或错误同步；
- launch 后检查错误；
- 明确 contiguous/layout 约束，或正确处理 stride；
- 使用合理的 index width，防止大 tensor 溢出；
- 对 FP16/BF16 明确 accumulation dtype；
- 支持零元素 tensor；
- 不把 forward kernel 误当成 backward kernel。

对于逐元素 Swish，手写 kernel 未必快过 `torch.compile` 的 fusion。性能测试必须包含编译后 reference，否则比较对象不公平。

### 4.2 构建与 ABI

`torch.utils.cpp_extension` 可用于开发期 JIT 或 setuptools wheel。PyTorch 2.10+ 提供 stable ABI 路线，但只有使用其支持 API 才能跨多个 PyTorch 版本复用 wheel。生产发布仍需记录：

```text
PyTorch ABI × Python ABI × compiler/libstdc++ × CUDA toolkit × SM architecture
```

## 5. `torch.compile` 与组合性门禁

运行：

```bash
cd knowledge/pytorch-custom-operators-to-tensorrt/assets
python test_custom_swish.py
```

脚本依次验证 eager forward、`gradcheck`、`opcheck`、`torch.compile(fullgraph=True)`；若有 GPU，还验证 CUDA dispatch/autograd。预期输出：

```text
forward: PASS
gradcheck: PASS
opcheck: PASS
torch.compile: PASS
```

常见失败：

| 症状 | 常见原因 | 优先动作 |
| --- | --- | --- |
| eager 正确，`opcheck` 失败 | schema 与 mutation/view 行为不一致 | 修正 schema 或实现 |
| backward 数值错 | broadcast reduction、链式法则或返回顺序错误 | 用 `float64` 小输入逐项 `gradcheck` |
| compile graph break | 缺 Fake kernel、Python 数据依赖或副作用 | 查看 graph break 日志，补 metadata 实现 |
| CPU 通过、CUDA not implemented | 没注册 CUDA dispatch key | 检查共享库加载和 `TORCH_LIBRARY_IMPL` |
| 第一次 benchmark 极慢 | 包含编译/JIT 开销 | 分离 warmup、compile 与 steady-state |

## 6. ONNX：优先分解，必要时保留自定义节点

有两条路线：

### 6.1 路线 A：翻译为标准 ONNX 子图

SwishBias 可以翻译成 `Mul → Sigmoid → Mul → Add`。现代 `torch.onnx.export(dynamo=True)` 可用 `custom_translation_table` 把 FX node target 映射到 ONNX Script builder function。

优点：ORT/TensorRT 等 runtime 无需专用插件；图优化器可能继续融合。缺点：无法表达外部库调用或真正 opaque 的新语义，也未必保留手写融合 kernel 的性能。

可运行示例 [`assets/export_standard_onnx.py`](assets/export_standard_onnx.py) 使用 `custom_translation_table` 和 ONNX Script 将 `learn_ops::swish_bias` 翻译成标准节点，并用 ORT 验证动态 batch 1/8/16：

```bash
python -m pip install onnx onnxruntime onnxscript
cd knowledge/pytorch-custom-operators-to-tensorrt/assets
python export_standard_onnx.py --output artifacts/swish_bias.onnx
```

这个文件可以直接被普通 ORT 执行，也能交给 TensorRT parser；代价是 ONNX 图里不再保留 `learn_ops::swish_bias` 这个融合边界。

### 6.2 路线 B：保留 `com.example::SwishBias` 自定义节点

当标准 ONNX 无法表达语义，或必须命中特定 TensorRT Plugin 时，导出自定义 domain、op type 和属性。此时必须同时维护：

- domain/op/version 与 attribute schema；
- shape/type inference；
- reference runtime 或测试 oracle；
- TensorRT parser 到 plugin creator 的映射；
- 插件 shared library 的加载和版本兼容。

不要只看“ONNX 文件生成成功”。至少用 graph inspection 确认自定义 node 名称和属性，再在目标 parser 上构建。

## 7. TensorRT Plugin V3

Plugin 是 TensorRT 的推理 layer，不包含训练 backward。训练侧 custom op 与部署 Plugin 通过共同的数学、shape、dtype 和 attribute contract 对齐，而不是复用同一段框架代码。

### 7.1 三类 capability

`IPluginV3` 通过 capability interface 划分职责：

- Core：plugin 名称、版本、namespace、序列化相关身份；
- Build：输出 shape/type、format combination、workspace、tactic；
- Runtime：`enqueue`、runtime 资源与执行。

Plugin creator 负责解析字段、创建 plugin 并注册到 registry。新插件应采用 `IPluginV3` 与 `IPluginCreatorV3One`；不要为新项目照抄旧 `IPluginV2DynamicExt` 教程。

### 7.2 从 ONNX parser 命中 Plugin

链路可概括为：

```text
ONNX custom node
  domain + op_type + attributes
             │ parser lookup
             ▼
IPluginCreatorV3One in plugin registry
             │ createPlugin(fields, phase)
             ▼
IPluginV3 build capability → engine serialization
IPluginV3 runtime capability → enqueue CUDA kernel
```

库必须在 parse/deserialization 前加载并注册。engine 能反序列化还要求运行环境找到兼容 plugin creator/library；因此 plan 与 plugin `.so`、TensorRT 版本、CUDA 和 GPU 环境应作为一个发布单元管理。

### 7.3 Plugin 验证矩阵

| 维度 | 最少覆盖 |
| --- | --- |
| shape | min/opt/max、空 batch 是否允许、非典型维度 |
| dtype | FP32、FP16；量化后再扩展 INT8 |
| format | 仅声明 kernel 真正支持的 format |
| 数值 | PyTorch reference → custom op → ONNX reference → TRT Plugin |
| stream | 非默认 stream、连续异步调用、无隐藏同步 |
| lifecycle | build、serialize、deserialize、重复 context |
| 性能 | warmup 后 P50/P95、吞吐、workspace、与 compiled reference 比较 |

## 8. 逐级调试决策树

```text
forward 与 reference 不一致
  └─ 先查公式、broadcast、dtype、contiguous 与边界值

forward 正确，gradcheck 失败
  └─ 查链式法则、归约轴、不可导点和数值差分精度

eager 正确，opcheck/compile 失败
  └─ 查 schema、alias/mutation、FakeTensor 与 Python 副作用

PyTorch 正确，ONNX export 失败
  └─ 选择标准分解或 custom translation；检查 FX target

ONNX 正确，TensorRT parser 找不到 Plugin
  └─ 查 domain/op/version、creator 名称/namespace、库加载顺序

engine 构建成功但数值错
  └─ 查 format/dtype、attribute 序列化、shape、stream 与 kernel 越界

数值正确但更慢
  └─ 与 compiled reference 公平比较；查 launch、融合、访存和 format conversion
```

## 9. 工程目录与发布门禁

```text
custom-op/
├── reference.py
├── registration.py
├── csrc/
│   ├── op.cpp
│   ├── op_cpu.cpp
│   └── op_cuda.cu
├── onnx/
│   ├── translation.py
│   └── shape_inference.py
├── tensorrt/
│   ├── plugin.cpp
│   ├── plugin_kernel.cu
│   └── CMakeLists.txt
└── tests/
    ├── test_grad.py
    ├── test_opcheck.py
    ├── test_export.py
    └── test_plugin.py
```

发布门禁至少要求：

1. reference 单元测试和边界值通过；
2. `gradcheck` 与 `opcheck` 通过；
3. CPU/CUDA、eager/compiled 输出和梯度对齐；
4. ONNX checker、shape inference 和 runtime/reference 对齐；
5. TensorRT min/opt/max 与 serialize/deserialize 通过；
6. sanitizer 或 CUDA memory checker 无越界；
7. 性能相对 compiled reference 有可复现收益；
8. wheel/plugin library/engine 的版本 manifest 完整。

## 10. 自测与练习

### 概念自测

1. 为什么 `autograd.Function` 能训练，却不等于注册了可供 dispatcher 和 ONNX 识别的新算子？
2. `gradcheck` 与 `opcheck` 分别能发现什么问题？
3. FakeTensor kernel 为什么不能读取 tensor data？
4. 为什么标量 bias 的梯度需要 `sum`？
5. 什么情况下应将 custom op 分解成标准 ONNX，而不是写 Plugin？
6. TensorRT Plugin 为什么不需要 backward？
7. 为什么 custom CUDA kernel 应与 `torch.compile` 后的 reference 比性能？

### 动手练习

1. 将 bias 从标量改为最后一维 `[C]`，修正 backward 的归约维度并重新 `gradcheck`；
2. 故意在实现中对 `x` 原地写入，观察 `opcheck` 如何报告 schema 问题；
3. 删除 Fake kernel，记录 `torch.compile(fullgraph=True)` 的失败位置；
4. 为输入加入非连续 view，决定 kernel 是支持 stride 还是显式要求 contiguous；
5. 把 SwishBias 分解成标准 ONNX 子图，并与 PyTorch 比较 batch 1/8/16；
6. 设计 custom node 的 domain、op version 与 attributes，并解释升级规则；
7. 在具备 TensorRT SDK 的环境实现 Plugin V3，完成 serialize/deserialize 和非默认 stream 测试。

## 11. 相关知识与官方资料

- 仓库内：[PyTorch 到 ONNX 与 TensorRT 部署](../pytorch-onnx-tensorrt-deployment/README.md)。
- PyTorch 官方：[Extending PyTorch](https://docs.pytorch.org/docs/stable/notes/extending.html)、[`torch.library`](https://docs.pytorch.org/docs/stable/library.html)、[Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)、[ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)。
- NVIDIA 官方：[Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)与[Plugin API Description](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html)。

## 下一步

进入[模型量化专题](../pytorch-onnx-tensorrt-quantization/README.md)：从数值表示与 observer 开始，连接 PyTorch PTQ/QAT、ONNX Q/DQ 与 TensorRT 强类型量化，并建立逐层误差定位与性能门禁。
