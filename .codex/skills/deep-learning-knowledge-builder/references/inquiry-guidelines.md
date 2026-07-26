# Opening Inquiry Guidelines

## Goal

Convert an initial topic into a focused learning contract without making the user design the curriculum alone.

## Method

1. Briefly restate the likely topic boundary.
2. Offer 4–7 topic-specific aspects, ordered from foundations to advanced or engineering concerns.
3. Ask the user to select, rank, remove, or add aspects.
4. Ask at most two additional questions only when their answers materially alter the artifact.

Infer defaults whenever safe:

- Language: Chinese.
- Code framework: PyTorch.
- Output: one canonical document.
- Experience: infer from the request and existing repository; ask only if depth cannot be chosen responsibly.

## Example Pattern

For “我想学习训练循环”, offer choices such as:

- data batch 到 loss 的完整数据流
- `zero_grad`、`backward`、`step` 的作用与顺序
- train/eval 模式与验证循环
- 梯度累积、裁剪和混合精度
- 日志、checkpoint 与恢复训练
- 常见 bug 的定位方法

Then ask which parts to emphasize and whether the learner wants a minimal implementation or an engineering-oriented loop. Do not reuse these choices mechanically for unrelated topics.

## Updating a Topic

For an existing document, propose uncovered or weak areas based on inspection. Ask whether the user wants correction, deeper explanation, more experiments, or engineering extension.
