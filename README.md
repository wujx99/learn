# Learn

This repository is a personal paper-learning library. Raw papers, TeX source trees,
and PDFs enter through `_inbox/`; durable knowledge lives in Markdown notes,
indexes, templates, and selected note assets.

## Repository Layout

```text
_inbox/             Local input area for papers and TeX source trees.
_index/             Library-wide indexes and reading log.
_templates/         Reusable note templates.
papers/             Topic-based paper notes.
```

## Workflow

1. Put a paper PDF or extracted TeX directory under `_inbox/`.
2. Read the TeX source when available; use the PDF only as a fallback or visual reference.
3. Create a Chinese close-reading note under `papers/<topic-slug>/<year>-<paper-slug>/README.md`.
4. Copy only images that are directly referenced by the note into that paper's `assets/` directory.
5. Update `_index/papers.md`, `_index/topics.md`, and `_index/reading-log.md`.

Raw source bundles and PDFs are intentionally ignored by Git. Keep them in
`_inbox/` while processing, then move them to `_inbox/_processed/` or delete them
outside the formal library when no longer needed.
