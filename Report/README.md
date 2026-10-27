# Project Phase 1 Report

**Historical Weather Data Digitization and Transfer Learning for Data-Efficient Local Weather Forecasting**

The built report is [thesis.pdf](thesis.pdf) (47 pages). It uses the department LaTeX template
(`dmathesis` class with `format.tex`): 12 pt, 1.5 line spacing, chapter headers and the department footer.

## Contents

| Chapter | File | Sections |
|---|---|---|
| Front matter | `frontpage.tex` | Title, declaration, certificate, acknowledgement, abstract, contents, abbreviations, lists of figures and tables |
| 1 Introduction | `chapter1.tex` | Overview, background, proposed approach |
| 2 Literature Review | `chapter2.tex` | Twelve papers: mechanism, architecture, technical impact |
| 3 Proposed Methodology | `chapter3.tex` | Workflow, data sources, digitization, model, staged transfer learning, evaluation |
| 4 Results and Discussion | `chapter4.tex` | Data, training, forecast accuracy, baselines, data efficiency, live and seasonal tests, app interface, discussion |
| Conclusion and References | `ref.tex`, `conclusion_body.tex` | One-page conclusion, IEEE references |
| Appendices | `appendix.tex` | A: SDG 13 Climate Action; B: project cost estimation (WBS, PERT, role rates) |
| Vision and Mission | `vm.tex` | Department vision and mission |

Supporting files: `captions.tex` (figure and table captions), `nomen.tex` (list of abbreviations),
`figures/` (charts and app screenshots), `data/metrics.json` (the evaluation output the charts are drawn from).

## Build

```bash
latexmk -pdf thesis.tex
```

or, without latexmk:

```bash
pdflatex thesis && makeindex thesis.nlo -s nomencl.ist -o thesis.nls && pdflatex thesis && pdflatex thesis
```

## Tools

- `tools/make_charts.py` redraws the result charts in `figures/` from `data/metrics.json`.
- `tools/build_individual.sh` builds one PDF per student in `individual/`; only the declaration page differs.
