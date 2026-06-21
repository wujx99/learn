# Inbox

Use this directory for temporary paper inputs:

- PDF files
- arXiv source archives
- extracted TeX source trees
- figures needed to understand the source paper

These files are local working material and are ignored by Git. The durable output
belongs in `papers/` as Markdown notes plus only the assets required by those
notes.

Recommended input shape:

```text
_inbox/
  papers/
    <paper>.pdf
  <series-or-topic>/
    <paper-slug>/
      main.tex
      references.bib
      figures/
```

After processing, move source material to `_inbox/_processed/` if you want to keep
a local copy without making it part of the formal knowledge base.
