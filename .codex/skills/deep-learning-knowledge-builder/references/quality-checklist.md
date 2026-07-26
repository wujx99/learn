# Quality Checklist

## Every Artifact

- The scope matches the aspects selected by the user.
- Learning goals and prerequisites are explicit.
- A concise mental model appears before detailed mechanics.
- Explanations state why, not only what or how.
- Terms, tensor shapes, units, and mathematical symbols are unambiguous where relevant.
- Common mistakes include observable symptoms and fixes.
- Exercises test the stated learning goals.
- Related repository knowledge and papers are linked when genuinely relevant.
- The document contains no empty template sections.

## Markdown

- Code is illustrative; if running and modifying it is central, use a notebook instead.
- Mathematical notation uses LaTeX.
- Headings form a coherent learning progression rather than a glossary dump.

## Notebook

- Markdown cells explain what to observe before important code cells.
- Imports and setup are explicit; no important hidden state is assumed.
- Examples default to small synthetic or built-in data and avoid large downloads.
- Random seeds are fixed when randomness affects the lesson.
- Tensor shapes and expected outputs are explained.
- Cells execute from top to bottom in a clean kernel when practical.
- Expensive, optional, or hardware-specific cells are clearly marked.
- Exercises are separated from reference solutions or provide useful hints.
