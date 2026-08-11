---
name: document-qa-curator
description: Answer questions about a specific paper note or knowledge document while maintaining a deduplicated `## QA` section in that document. Use when the user asks a substantive question about a paper or knowledge note in this repository and the target document is explicit or clear from the active context. Before answering, inspect prior QA entries; reuse a sound existing answer without editing the file, revise a matching but incomplete or stale entry in place, and append only genuinely new questions. Do not use for generic questions with no identifiable target document.
---

# Document QA Curator

Turn paper and knowledge discussions into a durable, non-repetitive QA record in the canonical Markdown document.

## Workflow

1. Resolve the target document.
   - Prefer the file or paper/knowledge note explicitly named, linked, attached, or currently being discussed.
   - Otherwise infer the target only when recent context and repository structure identify one canonical Markdown note unambiguously.
   - Search under `papers/` and `knowledge/` by title, slug, acronym, and aliases when needed.
   - Ask for the target file only when multiple plausible documents remain; do not risk writing to the wrong note.
   - If the target is a PDF, TeX tree, or other read-only source with no canonical Markdown note, ask the user to identify a note or invoke the appropriate note-building workflow. Do not add QA to raw paper inputs.
2. Inspect before answering or editing.
   - Check `git status --short` and preserve unrelated changes.
   - Find a level-two heading named `QA`, `Q&A`, or `问答`. Treat it as the document's single QA section.
   - Read the entire existing QA section before composing an answer. If the section is absent, defer creating it until a new answer is ready to persist.
   - Read the relevant parts of the target document and locally available primary paper/source material needed to verify the answer.
3. Compare by meaning, not wording.
   - Treat paraphrases, narrower restatements, acronym expansions, and follow-up wording with the same underlying information need as potential matches.
   - Consider an existing answer reusable only if it directly answers the current question, remains correct given the source, includes material conditions or caveats, and is understandable without missing context.
   - Classify the result as `reusable`, `needs revision`, or `new`.
4. Apply exactly one action.
   - `reusable`: answer the user from the existing QA entry and make no file change. Do not touch formatting, metadata, indexes, or timestamps merely to record that it was reused.
   - `needs revision`: improve the matching QA entry in place, preserving one canonical question. Incorporate the current question's missing angle without adding a duplicate entry.
   - `new`: produce a source-grounded answer and append one entry to the existing QA section. If no QA section exists, append `## QA` and the entry at the end of the document.
5. Verify the result.
   - Re-read the final QA section after any edit.
   - Confirm that only one entry covers the underlying question, the Markdown hierarchy is valid, and no unrelated content changed.
   - Tell the user whether the answer was reused, revised, or newly recorded, and link the target file.

## QA Entry Format

Use this structure consistently:

```markdown
### Q：为什么……？

A：……
```

- Keep the user's question intact when it is already self-contained. Lightly rewrite pronouns such as “这个” or “它” only to make the archived question understandable later.
- Write the answer as a durable explanation, not a transcript. Omit conversational filler, promises, and remarks about editing the file.
- Write in the document's primary language; default to Chinese. Preserve standard English technical terms when clearer.
- Use lists, formulas, code blocks, citations, or fourth-level headings inside an answer when they materially improve it.
- Do not add dates, sequence numbers, or a second QA section unless the document already follows such a convention.

## Evidence Rules

- Base claims about a paper on the paper or its local note whenever possible, and distinguish the authors' claims from interpretation.
- Base conceptual answers on the target knowledge document and reliable primary material when the document is insufficient.
- State uncertainty or evidence limits instead of inventing details.
- Keep links and citations useful after the conversation ends; avoid references such as “as mentioned above” or “in your question.”

## Scope Boundaries

- Maintain QA only in the canonical paper or knowledge Markdown note; do not update repository indexes for QA-only changes.
- Do not create a new paper archive or broad knowledge article through this skill.
- Do not append greetings, requests about repository maintenance, or other non-substantive conversation to QA.
