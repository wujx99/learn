---
name: download-arxiv-papers
description: Resolve paper titles or acronyms to exact arXiv records, deduplicate a requested batch, download each paper's latest TeX source bundle into its `_inbox/papers/` folder, safely extract it in place, and save the corresponding PDF under `_inbox/pdfs/`. Use when Codex needs to populate this repository's raw paper inbox from paper names, acronyms, titles, arXiv links, or arXiv IDs without creating reading notes or durable paper-library entries.
---

# Download arXiv Papers

Acquire raw arXiv sources and PDFs for later reading or archival. Keep this workflow separate from `paper-library-archiver`, which analyzes already-downloaded inputs and creates durable notes.

## Workflow

1. Inspect the repository before writing:
   - Run `git status --short --branch`.
   - Confirm the repository root and inspect `_inbox/papers/` and `_inbox/pdfs/` for existing targets.
   - Preserve unrelated changes and existing paper inputs.
2. Resolve every requested name against the official arXiv record:
   - Search the web when the user supplies titles or acronyms rather than IDs.
   - Prefer the official `https://arxiv.org/abs/<id>` page over mirrors or secondary indexes.
   - Record the requested folder label, exact title, and arXiv ID.
   - Treat an unversioned ID as a request for the latest available revision. Preserve an explicit `vN` suffix when the user requests a specific revision.
   - Deduplicate repeated names and repeated IDs. If one acronym maps plausibly to multiple papers, ask before downloading it.
3. Choose folder labels:
   - Preserve concise labels supplied by the user, including capitalization and hyphens.
   - Do not allow path separators, `.`/`..`, hidden names, or control characters.
   - Use one source folder and one PDF filename per unique paper.
4. Run the bundled script with one `--paper '<folder>=<arXiv-id>'` argument per paper:

   ```bash
   bash .codex/skills/download-arxiv-papers/scripts/download_arxiv_papers.sh \
     --repo-root /absolute/path/to/repository \
     --jobs 3 \
     --paper 'PETR=2203.05625' \
     --paper 'PETRv2=2206.01256'
   ```

   The script downloads from arXiv, retains `arXiv-<id>-source.tar.gz`, rejects unsafe archive members and links, extracts the source, checks that at least one `.tex` file exists, validates the PDF type, and refuses to overwrite incomplete or conflicting targets. A fully valid existing target is skipped, making retries safe.
5. Verify the result:
   - Confirm one directory under `_inbox/papers/` and one PDF under `_inbox/pdfs/` for every unique request.
   - Count `.tex` files and inspect `\title{...}` where practical to catch a wrong acronym resolution.
   - Confirm raw inbox files remain ignored by Git and that no tracked files changed unexpectedly.
6. Report:
   - State the number of unique papers and identify duplicates that were merged.
   - List the folder-to-arXiv mapping with links to official abstract pages.
   - Report download, extraction, and validation failures explicitly; do not describe a partial result as complete.

## Safety and Scope

- Default only to `_inbox/papers/` and `_inbox/pdfs/`; do not update `_index/` or create durable notes unless separately requested.
- Keep source archives after extraction.
- Never overwrite an existing partial folder or PDF automatically. Inspect and resolve it with the user if it is not a fully valid prior download.
- Limit concurrent arXiv downloads to four or fewer; the script defaults to three.
- Do not infer a materially different paper when an acronym remains ambiguous after searching.

## Bundled Script

Use `scripts/download_arxiv_papers.sh` for deterministic downloading, safe extraction, idempotency, and validation. Run it directly; patch it only when repository conventions genuinely differ.
