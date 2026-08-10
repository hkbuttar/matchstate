# Research Notebook

`research.ipynb` is a pre-executed narrative walkthrough of the full project, reusing the already-fitted artifacts under `data/processed/` where possible and refitting live (Dixon-Coles, the Bayesian model, GBM) where a real, visible computation adds value. It's a companion to the modular codebase, not a replacement for it -- the production code and full documentation live in each module (`baseline/`, `bayesian/`, `possession_value/`, `features/`, `models/`, `calibration/`, `market/`, `backtest/`, `tests/`), each with its own README.

Includes every substantive finding from the project, not just the headline numbers: both bugs caught and fixed during development (`bayesian/`'s `arviz` rounding default, `market/`'s DataFrame row-alignment issue), the formation-leakage finding from `models/`, the player-quality validation (Özil) from `features/`, and `tests/`'s synthetic test suite -- including the two tests that failed instructively on first attempt (Dixon-Coles parameter recovery at realistic sample sizes, and the xT model's flat-grid result from a synthetic construction with no turnover risk) before being fixed and turned into documented findings rather than just passing quietly.

Re-run with:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/research.ipynb
```

(Run from the repo root, or from anywhere -- the first cell locates the repo root automatically regardless of the kernel's working directory.)
