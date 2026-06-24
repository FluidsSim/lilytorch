# LilyTorch RA-L Paper

LaTeX sources for the manuscript describing the **LilyTorch**
GPU-accelerated immersed-boundary CFD solver and its coupling with
MuJoCo through the FARMS framework.

The paper is written in the **IEEE Robotics and Automation Letters
(RA-L)** format, i.e. the `IEEEtran` document class with the
`journal` option.

## Files

| File | Description |
| --- | --- |
| `main.tex` | Main manuscript (abstract, introduction, methods, discussion). |
| `supplementary.tex` | Supplementary Information: detailed formulations of the advection schemes, boundary conditions, pressure/Poisson solvers, and the strong (implicit) FSI coupling. Standalone document; shares `references.bib`. |
| `references.bib` | BibTeX database (BDIM, immersed boundary methods, projection methods, advection schemes, LES, MuJoCo/FARMS, biological swimming, validation benchmarks, ML-for-CFD, partitioned/implicit coupling, software). |
| `figures/` | Figures directory (empty placeholder — figures are added in a separate pass). |

The sections currently drafted are **Introduction**, **Methods**, and
**Discussion**, as requested.  A Results / validation section is left
for a later pass once all numerical experiments are finalised.

## Building

You need a standard TeX Live distribution with `IEEEtran.cls`
(packaged as `texlive-publishers` on Debian/Ubuntu) and BibTeX.

```bash
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

or, equivalently, via `latexmk`:

```bash
cd paper
latexmk -pdf main.tex
```

## Style conventions

- The manuscript uses the RA-L-mandated `IEEEtran` journal layout
  (two-column, 10 pt).
- Equations are referenced with `\eqref{}` and sections with
  `\ref{}`.
- Macros `\lilytorch`, `\farms`, `\mujoco`, `\pytorch`, `\waterlily`,
  and `\bdim` provide consistent typographic formatting.
- Citations follow the `IEEEtran` bibliography style
  (`\bibliographystyle{IEEEtran}`); numeric in-text citations are
  used throughout.

## Scope

Per the task instructions, only files under `/paper` are modified.
The rest of the repository is left untouched.
