# Learn

This repository is a personal deep-learning knowledge base. It combines
paper-specific close reading with focused, reusable learning documents and
executable notebooks.

## Repository Layout

```text
_inbox/             Local input area for papers and TeX source trees.
_index/             Shared topic, paper, knowledge, and information-source indexes.
_templates/         Reusable note templates.
papers/             Topic-based paper notes.
knowledge/          Focused conceptual notes and executable learning notebooks.
```

## Paper Workflow

1. Put a paper PDF or extracted TeX directory under `_inbox/`.
2. Read the TeX source when available; use the PDF only as a fallback or visual reference.
3. Create a Chinese close-reading note under `papers/<topic-slug>/<year>-<paper-slug>/README.md`.
4. Copy only images that are directly referenced by the note into that paper's `assets/` directory.
5. Update `_index/papers.md`, `_index/topics.md`, and `_index/reading-log.md`.

Raw source bundles and PDFs are intentionally ignored by Git. Keep them in
`_inbox/` while processing, then move them to `_inbox/_processed/` or delete them
outside the formal library when no longer needed.

## Knowledge Workflow

1. Start with a narrowly scoped deep-learning topic.
2. Select the aspects to study from a short, topic-specific set of suggestions.
3. Use Markdown for concept-led material or a PyTorch notebook when running and
   modifying code is central to learning.
4. Keep one canonical, continuously improved document under
   `knowledge/<topic-slug>/`.
5. Update `_index/knowledge.md` and reuse topic slugs from `_index/topics.md`.
6. Link reusable knowledge to relevant paper notes without nesting general
   tutorials inside a paper directory.

## Information Sources

Keep recurring paper, technical, course, and engineering sources in
[`_index/sources.md`](_index/sources.md). Add one row per source and describe
what it is useful for instead of storing an unexplained URL list.
