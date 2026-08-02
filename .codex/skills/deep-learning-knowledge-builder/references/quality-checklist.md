# Quality Checklist

## Every Artifact

- The scope matches the aspects selected by the user.
- Learning goals and prerequisites are explicit.
- A concise mental model appears before detailed mechanics.
- When relationships, flow, hierarchy, states, or interactions are central to the mental model or a key concept, a focused Mermaid diagram is used; prose-only treatment is acceptable when a diagram would not improve understanding.
- Every diagram is introduced and followed by its teaching takeaway, adds information beyond nearby prose, and stays consistent with the terminology and direction used in the document.
- Explanations state why, not only what or how.
- Terms, tensor shapes, units, and mathematical symbols are unambiguous where relevant.
- Common mistakes include observable symptoms and fixes.
- Exercises test the stated learning goals.
- Related repository knowledge and papers are linked when genuinely relevant.
- The document contains no empty template sections.

## Markdown

- Code is illustrative; if running and modifying it is central, use a notebook instead.
- Mathematical notation uses LaTeX.
- Mermaid blocks use valid, broadly supported syntax; labels are concise and do not contain LaTeX or fragile punctuation.
- Headings form a coherent learning progression rather than a glossary dump.

## Notebook

- Markdown cells explain what to observe before important code cells.
- Mermaid diagrams may appear in Markdown cells only when the target notebook renderer supports them; otherwise use a concise textual fallback rather than adding generated image files solely for the diagram.
- Imports and setup are explicit; no important hidden state is assumed.
- Examples default to small synthetic or built-in data and avoid large downloads.
- Random seeds are fixed when randomness affects the lesson.
- Tensor shapes and expected outputs are explained.
- Cells execute from top to bottom in a clean kernel when practical.
- Expensive, optional, or hardware-specific cells are clearly marked.
- Exercises are separated from reference solutions or provide useful hints.
