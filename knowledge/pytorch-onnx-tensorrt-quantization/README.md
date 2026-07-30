---
title: PyTorch、ONNX 与 TensorRT 模型量化
topic: pytorch-onnx-tensorrt-quantization
topics: [pytorch-onnx-tensorrt-quantization, pytorch-onnx-tensorrt-deployment]
framework: PyTorch
status: learning
updated: 2026-07-30
---

# PyTorch、ONNX 与 TensorRT 模型量化

## 学习目标

完成本文后，应当能够：

- 从 affine mapping 推导 scale、zero-point、舍入、饱和与反量化；
- 比较 symmetric/asymmetric、per-tensor/per-channel、static/dynamic/weight-only；
- 设计具有代表性的 calibration dataset，并识别 outlier 与分布漂移；
- 解释 PTQ 与 QAT 的数据流、FakeQuantize 和 STE；
- 阅读 ONNX `QuantizeLinear`/`DequantizeLinear`（Q/DQ）图；
- 按 TensorRT 11 强类型与显式量化工作流构建 engine；
- 用逐层误差、任务指标和性能门禁决定敏感层的精度策略。

本文先聚焦 INT8 affine quantization；INT4/FP8/FP4 的 group size、block scaling 与硬件约束可在掌握这条主线后扩展。

## 前置知识与环境边界

- 熟悉 tensor dtype、数值范围、训练与推理；
- 已阅读[部署主链路](../pytorch-onnx-tensorrt-deployment/README.md)；自定义量化算子可参考[算子扩展](../pytorch-custom-operators-to-tensorrt/README.md)；
- CPU 实验只模拟 Q/DQ 数值，不声称具有 INT8 kernel 加速；
- TensorRT 10.x 与 11.x 的量化接口不同，生产项目必须先确认版本。

## 核心心智模型：量化是带误差预算的数值契约

```text
浮点模型与任务指标
        │ 选择数值格式、granularity、range estimator
        ▼
scale / zero-point + rounding + clamp
        │ PTQ calibration 或 QAT fake quantization
        ▼
显式 Q/DQ 图与逐层精度边界
        │ backend lowering / kernel selection
        ▼
量化 engine：精度、延迟、吞吐、显存共同验收
```

“模型变成 INT8 文件”不是目标。目标是在任务精度预算内，让目标硬件真正执行更高效的 kernel。

## 1. Affine quantization 数学

浮点值 $x$ 映射到整数 $q$：

$$
q=\operatorname{clamp}\left(\operatorname{round}\left(\frac{x}{s}\right)+z,q_{\min},q_{\max}\right),
$$

反量化为：

$$
\hat{x}=s(q-z).
$$

$s>0$ 是 scale，$z$ 是 zero-point。误差来自舍入和饱和：范围内理想舍入误差约不超过 $s/2$；超出 representable range 的值被 clamp，误差可能很大。

### 1.1 Symmetric 与 asymmetric

- symmetric：通常令 $z=0$，范围以零为中心，权重量化常用；
- asymmetric：允许非零 $z$，能更充分覆盖偏离零的 activation 范围，但 kernel 与 metadata 更复杂。

INT8 symmetric 常用 $q\in[-127,127]$，避免正负范围不对称带来的细节问题；实际取值必须与导出工具和 backend contract 一致。

### 1.2 Granularity

| 粒度 | scale 数量 | 精度与代价 |
| --- | --- | --- |
| per-tensor | 整个 tensor 一个 | 简单，但一个 outlier 会压缩其他值 |
| per-channel | 每个输出 channel 一个 | 权重精度通常明显更好，metadata 更多 |
| per-group/block | 每组一个 | INT4/LLM 常用，在精度与开销间折中 |
| per-token/row | 随输入动态变化 | 适合部分 transformer activation，运行成本更高 |

对 Linear 权重 `[out_features, in_features]`，常见 per-channel axis 是 `out_features`。axis 选错可能 shape 合法但数值错误。

## 2. 可运行数值实验

运行：

```bash
python knowledge/pytorch-onnx-tensorrt-quantization/assets/quantization_lab.py
```

[`quantization_lab.py`](assets/quantization_lab.py)展示：

1. calibration range 没覆盖 evaluation outlier 时发生 saturation；
2. 用 evaluation 自身计算的 oracle scale 不是合法线上方案，只用于诊断；
3. 某个权重 channel 存在大 outlier 时，per-channel 通常优于 per-tensor；
4. 权重量化误差如何传播到 Linear 输出。

不要从一次随机实验推导固定结论。修改 outlier 比例、channel 数和分布，观察 quantization noise 与 saturation 的权衡。

## 3. Calibration 与 Observer

PTQ static activation quantization 需要用代表性数据估计范围。calibration dataset 不需要标签来收集统计量，但最终任务评估需要标签或可靠指标。

### 3.1 数据集应覆盖什么

- 真实预处理、dtype、layout 与 shape 分布；
- 白天/夜间、近场/远场、空场景/拥挤场景等业务模式；
- 合理极值，而不是只取“平均样本”；
- 动态 shape 的主流 bucket；
- 与生产一致的 batch 方式，避免 padding 改变分布。

### 3.2 Range estimator

- MinMax：覆盖所有观察值，outlier 会增大 scale；
- moving average：适合训练期间平滑统计，但顺序和 momentum 会影响结果；
- histogram/percentile/entropy：允许裁掉少量 outlier，减少主体 rounding error；
- MSE-based：搜索使重建误差较小的 clipping range。

校准目标不等于最终任务目标。activation MSE 更小不保证检测 AP 或语言模型 perplexity 更好。

### 3.3 常见 calibration 失败

| 症状 | 原因 | 检查 |
| --- | --- | --- |
| 离线通过、线上偶发崩坏 | calibration 未覆盖长尾 | 记录 saturation ratio 与线上分位数 |
| batch/分辨率变化后掉点 | shape 改变 activation 分布 | 分 bucket 校准和评估 |
| 个别层误差巨大 | outlier、Softmax/归一化敏感 | 逐层比较，保留高精度验证 |
| 换预处理后整体掉点 | 输入契约漂移 | 比较量化前 FP32 基线也是否下降 |

## 4. PTQ、Dynamic 与 Weight-only

### 4.1 Static PTQ

权重离线量化，activation scale 由 calibration 固定。适合 CNN 和稳定输入分布；能获得完整 INT8 data path，但对 calibration 依赖更强。

### 4.2 Dynamic quantization

权重预量化，activation qparams 在运行时计算。对部分 Linear/RNN 场景方便，但动态统计有开销，且 backend 支持因硬件而异。

### 4.3 Weight-only

只压缩权重，activation 保持较高精度。LLM 常见 INT8/INT4 weight-only；收益主要是权重带宽与容量，不等同于所有计算都使用低位整数。

选择量化方案必须从目标 kernel 支持反推，不能只看模型文件大小。

## 5. QAT 与 FakeQuantize

QAT 在训练 forward 中模拟：

$$
x_{fq}=\operatorname{dequantize}(\operatorname{quantize}(x)),
$$

tensor 通常仍以浮点存储和计算，让模型参数适应 rounding/clipping noise。`round` 几乎处处梯度为零，因此常用 Straight-Through Estimator（STE）近似 backward。

典型流程：

```text
float checkpoint
  → prepare：插入 observer/fake quant
  → 短程 fine-tune：先更新统计，再冻结 observer
  → convert：形成目标量化表达
  → export Q/DQ ONNX
  → backend build + 任务指标验证
```

PyTorch 新项目应优先评估 `torchao.quantize_` 与 `QATConfig`；旧 `torch.ao.quantization` 示例仍有学习价值，但不要无条件作为新生产栈。QAT 不是“必然恢复全部精度”，学习率、训练数据、observer 冻结时机、BN 状态和目标 backend scheme 都会影响结果。

## 6. ONNX 显式 Q/DQ 图

显式量化常以：

```text
float tensor → QuantizeLinear → integer domain → DequantizeLinear → float graph value
```

表达 scale、zero-point、axis 和量化边界。backend 可以识别 Q/DQ pattern 并 lower 到量化 kernel；图中出现 Q/DQ 不等于每个中间 tensor 都发生真实显存 round-trip。

检查 ONNX 时重点确认：

- `QuantizeLinear`/`DequantizeLinear` 是否成对且位置合理；
- scale/zero-point dtype、shape 和 axis；
- weight 是 per-channel 还是意外退化为 per-tensor；
- 敏感算子前后是否有预期的高精度边界；
- graph optimization 是否移动或删除了量化边界；
- ORT/reference 的任务指标，而非只做 checker。

QOperator（如 `QLinearConv`）与 Q/DQ 是不同 ONNX 表达风格。TensorRT 现代主线是显式 Q/DQ；导出前确认目标 parser 支持的 opset 和 scheme。

## 7. TensorRT 10.x 与 11.x

### 7.1 TensorRT 11.x：强类型 + 显式量化

TensorRT 11 已移除：

- `BuilderFlag.INT8`；
- `trtexec --int8`；
- `IInt8Calibrator` 及其子类；
- implicit quantization 和 dynamic range API。

现代流程是：

```text
FP32 ONNX + representative calibration data
       │ NVIDIA ModelOpt 或手动 Q/DQ
       ▼
explicitly quantized ONNX
       │ TensorRT parser/build（不再打开 INT8 flag）
       ▼
strongly typed engine
```

示意命令应以安装版本 `--help` 为准：

```bash
python -m modelopt.onnx.quantization \
  --onnx_path=model.onnx \
  --calibration_data=calibration.npz

trtexec \
  --onnx=model_quantized.onnx \
  --saveEngine=model_int8.plan
```

### 7.2 TensorRT 10.x legacy

旧项目可能使用 calibrator、`BuilderFlag.INT8` 或 `trtexec --int8`。理解它们有助于迁移，但不要把旧教程代码写进 11.x 新项目。迁移时先生成显式 Q/DQ ONNX，再删除 builder calibration 逻辑，并重新做数值与性能基线。

## 8. 精度定位：不要只比较最终输出

建议建立阶梯：

```text
PyTorch FP32
  ≈ ONNX/ORT FP32
  ≈ FakeQuant/QAT reference
  ≈ ONNX Q/DQ reference
  ≈ TensorRT quantized engine
```

找到首次明显偏离的阶段，再暴露中间 tensor。逐层记录：

$$
\operatorname{MSE},\quad
\max|e|,\quad
\operatorname{cosine}(x,\hat{x}),\quad
\text{saturation ratio},\quad
\text{SQNR}=10\log_{10}\frac{\mathbb{E}[x^2]}{\mathbb{E}[(x-\hat{x})^2]}.
$$

高风险位置包括输入/输出、Softmax、LayerNorm、很小的 residual、回归 head、检测阈值附近以及 heavy-tailed activation。发现敏感层后，用“该层恢复 FP16/FP32是否修复任务指标”做因果验证。

## 9. 性能门禁

量化可能不加速，常见原因：

- 目标 GPU/shape 没有合适量化 kernel；
- Q/DQ、reformat 或 Cast 过多；
- 模型受 launch、CPU 或预后处理限制；
- batch 太小；
- 部分层回退高精度导致频繁边界转换；
- 权重更小，但 activation/compute 仍是高精度。

报告必须同时记录任务指标、P50/P95/P99 latency、throughput、峰值显存/engine size、功耗条件、shape/batch 和 layer precision。只报告“INT8 理论算力”没有工程意义。

## 10. 发布门禁与排错树

发布门禁：

1. FP32 checkpoint、预处理和 baseline 指标冻结；
2. calibration manifest 可复现，样本无训练/验证泄漏；
3. Q/DQ graph schema、axis、scale dtype 检查通过；
4. min/opt/max shape 与各业务 bucket 通过；
5. 逐层误差和 saturation 无异常；
6. 任务指标下降不超过预算；
7. 性能相对同版本 FP16/FP32 有稳定收益；
8. engine、ONNX、calibration 数据摘要和环境版本可追溯。

```text
PTQ 掉点
  ├─ FP32 也掉点 → 先查导出与输入契约
  ├─ FakeQuant 已掉点 → 查 range、granularity、敏感层；考虑 QAT
  ├─ Q/DQ ONNX 才掉点 → 查 axis/opset/graph rewrite
  └─ TensorRT 才掉点 → 查 parser lowering、format、plugin 与版本

精度正确但不加速
  └─ 查 layer precision、Q/DQ/reformat、kernel 支持和端到端瓶颈
```

## 11. 自测与练习

1. scale 变大为什么会减少 saturation，却增大主体 rounding error？
2. 为什么权重常用 per-output-channel，而 activation 更常见 per-tensor？
3. calibration 为什么通常不需要标签，最终验收却需要任务指标？
4. FakeQuant 的 tensor 为什么仍是浮点？STE 解决什么问题？
5. ONNX 图有 Q/DQ 为什么不保证 TensorRT 一定使用 INT8 kernel？
6. TensorRT 11 为什么不能继续使用旧 calibrator 代码？
7. 修改实验，让 calibration 也包含 outlier，比较误差和 saturation。
8. 给每个权重 channel 注入不同量级，比较 per-tensor/per-channel SQNR。
9. 在真实模型上逐层恢复 FP16，寻找最小高精度集合。

## 12. 相关知识与官方资料

- 仓库内：[部署主链路](../pytorch-onnx-tensorrt-deployment/README.md)、[自定义算子到 TensorRT Plugin](../pytorch-custom-operators-to-tensorrt/README.md)。
- PyTorch 官方：[torchao QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)、[Quantization API reference](https://docs.pytorch.org/docs/stable/quantization-support)。
- NVIDIA 官方：[Quantization Workflows](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html)、[TensorRT 10.x 到 11.x 迁移](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html)。

## 下一步

选择一个真实模型，把三篇知识串成可复现工程：冻结 FP32 baseline，导出 ONNX，建立 PTQ/QAT 分支，生成显式 Q/DQ 图，在目标 GPU 构建 engine，并用统一 manifest 对齐数值、任务指标和性能。
