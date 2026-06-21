# Archive Conventions

## Directory Rules

- `_inbox/` is temporary local input. Keep PDFs, arXiv source archives, extracted TeX trees, and unprocessed figures there.
- `papers/` is the durable library. Store Markdown notes and only the figures referenced by those notes.
- Archive by topic first: `papers/<topic-slug>/<year>-<paper-slug>/README.md`.
- Do not archive primarily by year or paper series.
- Track series in YAML frontmatter and index columns.

## Naming Rules

- Use English kebab-case for slugs.
- Keep topic slugs broad and reusable, for example `robotics-vla`, `offline-rl`, `representation-learning`.
- Use `_uncategorized` only when the paper's topic cannot be inferred confidently from the source.

## Index Rules

`_index/papers.md` columns:

```markdown
| Paper | Year | Topics | Series | Status | Note |
| --- | --- | --- | --- | --- | --- |
```

`_index/topics.md` columns:

```markdown
| Topic Slug | Display Name | Description |
| --- | --- | --- |
```

`_index/reading-log.md` columns:

```markdown
| Date | Paper | Action | Notes |
| --- | --- | --- | --- |
```

## Raw Input Policy

- Do not delete raw input unless the user explicitly asks.
- Do not commit `_inbox` paper inputs.
- Do not move paper sources out of `_inbox` unless the user requests inbox cleanup.
- If the user wants cleanup, move processed raw materials to `_inbox/_processed/` rather than into `papers/`.
