# Project Phase 1 Report
## Historical Weather Data Digitization and Transfer Learning for Data-Efficient Local Weather Forecasting

Built on the department project-report LaTeX template (`dmathesis` class, `format.tex`, `frontpage.tex`),
with 1.5 line spacing (from `format.tex`).

### Build

    pdflatex thesis && makeindex thesis.nlo -s nomencl.ist -o thesis.nls && pdflatex thesis && pdflatex thesis

### Layout
- Chapter sources: `chapter1.tex` … `chapter13.tex`.
- Front matter (title, declaration, certificate, acknowledgement, abstract, lists): `frontpage.tex`.
- Captions used in the figures/tables: `captions.tex`.
- List-of-Abbreviations entries: `nomen.tex` (fed to `\printnomenclature`).
- Conclusion + references: `ref.tex` (conclusion body in `conclusion_body.tex`).
- Vision/Mission: `vm.tex`.

### Individual copies
`tools/build_individual.sh` builds one PDF per student in `individual/`; only the declaration page differs.

### Charts
`tools/make_charts.py` redraws the result charts from `data/metrics.json`.
