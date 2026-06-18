# Architecture diagrams
Olympus architecture diagram explaining how all parts fit together. 

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
  the RL loop), `bridge` (the de-emphasised `oc_bridge` hand-off), `card`
  (the faint ×N rollout stack), and the edge styles `ctrl` / `data` / `tap` /
  `ipc` / `cfg`.
