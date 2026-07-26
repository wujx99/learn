---
name: deep-learning-knowledge-builder
description: Build and maintain focused deep-learning learning documents through an interactive topic-definition workflow. Use when the user says they want to learn, understand, review, or systematically study a deep-learning concept, implementation pattern, or engineering practice; asks to turn a discussion into durable repository knowledge; or wants to extend an existing knowledge topic. Clarify desired aspects with suggested choices, default executable code content to PyTorch notebooks, create or update one canonical Chinese document per narrowly scoped topic, and integrate it with the repository's shared topics, paper notes, and knowledge index.
---

# Deep Learning Knowledge Builder

Turn a focused learning request into one durable, evolving knowledge document. Treat the conversation as guided study, not merely a document-generation command.

## Workflow

1. Inspect before asking or editing:
   - Check `git status --short --branch` and preserve unrelated changes.
   - Read `README.md`, `_index/topics.md`, `_index/knowledge.md`, and relevant files under `knowledge/` and `papers/` when present.
   - Search by topic names, aliases, and related terms to avoid duplicate documents.
2. Define the learning scope before writing:
   - Infer what is already clear from the user's request and repository history.
   - Offer 4–7 concrete aspects relevant to the topic and allow free-form additions.
   - Ask only the questions that materially affect the result: current level, intended depth, emphasis, framework, or desired exercises.
   - Default to PyTorch when code is involved. Do not ask about framework unless another choice is plausible or consequential.
   - Keep the topic narrow enough for one coherent document. Suggest splitting only when the requested scope contains independently useful subjects.
   - Do not create the learning artifact until the user answers, unless the initial request already specifies enough scope or explicitly asks to skip clarification.
3. Choose the canonical artifact:
   - Use `knowledge/<topic-slug>/README.md` for conceptual material where code is illustrative rather than something the learner should run and modify.
   - Use `knowledge/<topic-slug>/<topic-slug>.ipynb` when executing, inspecting, or changing code is central to learning.
   - Use exactly one canonical learning document per scoped topic. Prefer expanding it on later requests instead of creating fragments.
   - Add `assets/` only for files directly used by the canonical document.
4. Build the material around the learner's choices:
   - Start with learning goals and prerequisites.
   - Establish a mental model before details.
   - Progress from a minimal example to realistic usage.
   - Explain mechanisms, tradeoffs, common mistakes, and debugging signals.
   - Include self-check questions or small exercises and a concise next-step path.
   - Link related knowledge documents and paper notes when the relationship helps learning.
   - Write in Chinese by default; retain English API names and standard technical terms when clearer.
   - Use LaTeX for mathematics.
5. Integrate the result:
   - Use a stable English kebab-case topic slug shared with `_index/topics.md` when appropriate.
   - Add or update one row in `_index/knowledge.md`.
   - Add a missing topic to `_index/topics.md`; do not create near-duplicate topic slugs.
   - Add reciprocal links when a paper note directly motivates or applies the knowledge topic and editing that note is in scope.
6. Validate:
   - Read `references/quality-checklist.md` and apply it before finishing.
   - Verify all relative links and ensure the canonical artifact is indexed.
   - For notebooks, execute all cells in order when dependencies and compute permit. If execution is not possible, state exactly what remains unverified.

## Repository Rules

Read `references/repository-conventions.md` before choosing or changing paths and indexes. Read `references/inquiry-guidelines.md` before conducting the opening clarification.

Use `assets/knowledge-note-template.md` or `assets/learning-notebook-template.ipynb` only when the repository does not provide the corresponding `_templates/` file. Adapt the structure to the topic rather than preserving empty or irrelevant sections.

## Updating Existing Topics

When the user revisits a topic, first summarize what the canonical document already covers and offer gaps or extensions as the candidate aspects. Preserve sound existing material, integrate additions at the right conceptual location, and revise stale explanations rather than appending a disconnected addendum.

## Boundary With Paper Archiving

Use the paper archiver for extracting and documenting a specific paper. Use this skill for reusable understanding that should stand independently of any one paper. When a learning request originates from a paper, create the reusable explanation under `knowledge/` and link it to the paper note instead of nesting a general tutorial inside the paper directory.
