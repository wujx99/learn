# Repository Conventions

## Content Lines

- Keep paper-specific close reading under `papers/`.
- Keep reusable conceptual and implementation knowledge under `knowledge/`.
- Share topic slugs through `_index/topics.md` so both content lines can be discovered together.

## Knowledge Paths

Use one directory and one canonical document for each narrowly scoped topic:

```text
knowledge/<topic-slug>/README.md
knowledge/<topic-slug>/<topic-slug>.ipynb
knowledge/<topic-slug>/assets/
```

Choose either Markdown or Notebook as the canonical document, not both. Supplementary assets do not count as separate learning documents.

Use English kebab-case for directory and notebook names. Prefer stable concept names such as `training-loop`, `gradient-accumulation`, or `batch-normalization`.

## Knowledge Index

Maintain `_index/knowledge.md` with this schema:

```markdown
| Topic | Format | Framework | Status | Related Topics | Document |
| --- | --- | --- | --- | --- | --- |
```

- `Format`: `markdown` or `notebook`.
- `Framework`: use `PyTorch`, another named framework, or `framework-agnostic`.
- `Status`: normally `learning`, `review`, or `stable`.
- `Related Topics`: comma-separated topic slugs or `—`.

Add missing reusable slugs to `_index/topics.md`. Do not add every incidental term as a topic.

## Cross-links

- Use repository-relative Markdown links.
- Add a “相关知识与论文” section when useful.
- Link to the canonical artifact, not merely its directory.
- Avoid reciprocal links that do not help navigation or understanding.
