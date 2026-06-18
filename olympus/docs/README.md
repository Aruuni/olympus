# Architecture diagrams

The figures here are TikZ sources compiled to PDF, plus a PNG preview for quick
viewing / embedding.

| Source | PDF | Preview | Shows |
| --- | --- | --- | --- |
| `architecture.tex`           | `architecture.pdf`           | `preview-1.png`           | Single-agent: `×N` parallel envs, one worker each |
| `architecture_multiflow.tex` | `architecture_multiflow.pdf` | `preview_multiflow-1.png` | Single-agent multi-flow: one env, one **live** worker trains + lagged self-play background workers |
| `architecture_marl.tex`      | `architecture_marl.pdf`      | `preview_marl-1.png`      | Multi-agent (MARL): one env, `×N` agents jointly trained (CTDE) |

## Requirements

- `pdflatex` (TeX Live) — uses only `tikz`, `helvet`, and standard libraries.
- `pdftoppm` (from `poppler-utils`) — for the PNG preview.

## Regenerate

Run from this `olympus/docs/` directory. To edit, change the `.tex`, then:

```bash
# 1. compile the figure (standalone -> tight-cropped PDF)
pdflatex -interaction=nonstopmode -halt-on-error architecture.tex

# 2. refresh the PNG preview at 150 dpi (writes preview-1.png)
rm -f preview-1.png
pdftoppm -png -r 150 architecture.pdf preview

# 3. drop the LaTeX scratch files
rm -f architecture.aux architecture.log
```

For the other figures, swap `architecture` → `architecture_multiflow` /
`architecture_marl` and `preview` → `preview_multiflow` / `preview_marl` in the
commands above.

> `pdftoppm` appends `-1` to the output name (one file per page), so
> `... preview` produces `preview-1.png`. Don't add `-singlefile`, which would
> drop the `-1` suffix and break the table above.

## Notes

- The sources are `standalone` documents — `pdflatex` crops to the drawing.
  To drop a figure into a paper instead, `\input{}` the body and remove the
  standalone preamble (see the comment at the top of each `.tex`).
- Styles to know when editing: `proc` (component box), `agent` (the Worker —
  the RL loop), `bridge` (the de-emphasised `oc_listener` hand-off), `card`
  (the faint ×N rollout stack), and the edge styles `ctrl` / `data` / `tap` /
  `ipc` / `cfg`.
