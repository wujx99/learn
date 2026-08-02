---
name: paper-library-archiver
description: Analyze academic papers from local inbox inputs and archive them as structured Markdown learning notes. Use when Codex needs to process a paper TeX source tree, arXiv source bundle, or PDF under `_inbox`; extract the paper's metadata, method, formulas, experiments, limitations, and relationships; create a Chinese close-reading note; copy only necessary figures; and update a topic-based paper library index.
---

# Paper Library Archiver

## Overview

Turn paper inputs under `_inbox/` into durable, topic-based Markdown notes. Prefer TeX source over PDF because TeX preserves structure, formulas, figure captions, labels, and bibliography.

## Workflow

1. Inspect the workspace before editing:
   - Check `git status --short --branch`.
   - Use `rg --files _inbox` to find candidate paper inputs.
   - Prefer `00README.json`, `main.tex`, root `.tex` files, and `references.bib`.
2. Identify the main paper source:
   - Use `00README.json` if it declares the top-level TeX file.
   - Otherwise inspect `\title`, `\author`, `\begin{abstract}`, `\section`, `\caption`, and `\label`.
   - If only a PDF exists, ask before doing a PDF-only interpretation unless the user explicitly requested it.
3. Extract the paper substance:
   - Metadata: title, authors, year, venue, project URL, arXiv ID if available.
   - Structure: abstract, introduction, related work, method, implementation, experiments, discussion, appendix details.
   - Technical content: definitions, objectives, algorithms, formulas, training data, evaluation metrics, baselines, limitations.
   - Figures: select only figures directly useful for the note.
4. Choose the archive path:
   - Use pure topic folders, not year-first or series-first folders.
   - Path format: `papers/<topic-slug>/<year>-<paper-slug>/README.md`.
   - Put uncertain papers under `papers/_uncategorized/<year>-<paper-slug>/README.md`.
   - Use English kebab-case for slugs.
5. Write the note:
   - Use `assets/paper-note-template.md` as the structure.
   - Write the note in Chinese unless the user asks otherwise.
   - When a visual overview improves comprehension, add a Mermaid diagram under the background, overall method, or system section to show the problem landscape, component relationships, data flow, training/inference stages, or interaction sequence.
   - Use Mermaid for explanatory abstractions of key concepts when structure, sequence, hierarchy, state transitions, or interactions are clearer visually. Introduce what the reader should notice and state the takeaway after the diagram; do not turn an adjacent bullet list into a redundant diagram.
   - Use selected source figures instead of Mermaid for exact model layouts that must preserve paper-specific detail, quantitative plots, tables, qualitative results, and other empirical evidence. Clearly distinguish an explanatory Mermaid synthesis from a figure claimed by the paper.
   - Use LaTeX for all mathematical notation: inline `$...$`, display `$$...$$`.
   - Do not write formulas as plain-text code blocks.
   - Keep model-input strings, paths, commands, and literal tokens in backticks.
6. Handle assets:
   - Create `assets/` under the paper directory.
   - Copy only figures referenced by the note.
   - Use relative image links such as `![caption](assets/model_architecture.png)`.
   - Do not copy the full TeX source tree, source archive, or paper PDF into `papers/`.
7. Update indexes:
   - `_index/papers.md`: one row per paper.
   - `_index/topics.md`: add missing topic slugs.
   - `_index/reading-log.md`: record the archive date and source input.
   - Topic index, for example `papers/<topic-slug>/README.md`, should list papers in that topic.
8. Validate before finishing:
   - `rg --files` should show only Markdown and selected note assets in the durable library.
   - Raw inputs in `_inbox/` should remain untouched and ignored by Git.
   - Search the generated note for formula-like plain text such as `min_phi`, `proportional`, `pi_ref`, `epsilon_language`, and convert it to LaTeX.
   - Check every Mermaid block for valid syntax, readable labels, and agreement with the paper. Prefer broadly supported `flowchart`, `sequenceDiagram`, `stateDiagram-v2`, or `classDiagram`; keep LaTeX, citations, and long prose outside node labels.
   - On Windows, read Markdown with `Get-Content -Encoding UTF8` when checking Chinese text.

## Repository Conventions

If the workspace does not already have the library structure, create the minimal structure:

```text
_inbox/
  README.md
  _processed/
_index/
  papers.md
  topics.md
  reading-log.md
_templates/
  paper-note.md
papers/
  README.md
```

Use `.gitignore` to keep `_inbox` raw inputs local while allowing `_inbox/README.md` and `_inbox/_processed/.gitkeep` to be tracked.

## Output Quality Bar

The note should help the user study the paper without rereading the full source. It must include the problem framing, core contribution, method mechanics, key formulas, experimental evidence, limitations, and follow-up reading questions. Use Mermaid where it materially clarifies the overview or a key concept, but do not require a diagram when prose or a source figure communicates the point better. Prefer accurate extraction from TeX over broad background knowledge.
