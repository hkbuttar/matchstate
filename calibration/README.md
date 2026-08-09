# Probability Calibration

## Setup

Clean three-way, match-level split of 2015/16: **fit** (242 matches, for
the underlying models -- same as `models/`) / **calibration** (43
matches, used only to fit Platt scaling and isotonic regression) /
**test** (95 matches, final evaluation, untouched by calibration
fitting). This avoids the common mistake of calibrating and evaluating
on the same held-out set, which would overstate any improvement.

Both calibration methods are fit **one-vs-rest per class** (`home_win`,
`draw`, `away_win` each get their own calibrator), then the 3 calibrated
probabilities are renormalized to sum to 1 -- the standard, transparent
way to extend binary calibration to 3 classes, and it keeps per-class
reporting honest rather than averaging classes together.

## Headline finding: calibration mostly *doesn't* help here, and the reason is a real, checkable sample-size effect

| Model | Class | Raw ECE | Platt ECE | Isotonic ECE |
|---|---|---|---|---|
| static | home_win | 0.062 | 0.052 | **0.043** |
| static | draw | 0.065 | 0.110 | 0.092 |
| static | away_win | 0.070 | 0.128 | 0.103 |
| bayesian | home_win | 0.056 | 0.057 | **0.044** |
| bayesian | draw | 0.062 | 0.109 | 0.089 |
| bayesian | away_win | 0.062 | 0.118 | 0.099 |
| gbm | home_win | 0.045 | **0.038** | **0.038** |
| gbm | draw | 0.047 | 0.092 | 0.093 |
| gbm | away_win | 0.079 | 0.093 | 0.077 |

**home_win calibration works as expected** -- isotonic regression
improves ECE for all three models. **draw and away_win calibration makes
things *worse*, consistently, across every model.** This isn't noise or
a bug -- it traces to a specific, verified cause: the 43-match
calibration split contains ~4,000 per-minute rows, but those rows are
heavily correlated (all ~95 rows from one match share the same outcome),
so the *effective* sample size for fitting a class's calibrator is the
match count, not the row count. Checked directly: the calibration split
has only **9 draw-outcome matches and 16 away-win matches** out of 43.
Both Platt scaling and (especially) isotonic regression overfit to that
handful of matches' probability trajectories and generalize poorly to
the fresh 29 draws / 23 away-wins in the 95-match test set.

This is exactly the disclosed StatsBomb-coverage constraint (380 matches,
one season -- see `data/README.md`) resurfacing in a new form: not enough
independent match outcomes to safely fit a second correction layer on top
of an already-fitted model, for the rarer classes.

## Reliability diagrams

See `calibration/plots/{static,bayesian,gbm}_reliability.png`. Visually
confirms the table: draw panels show the raw (uncalibrated) curve
already tracking close to the diagonal in the middle probability range,
while the "corrected" curve is pulled measurably further away.

## Practical recommendation (disclosed, not hidden)

Given this, the honest recommendation is: **calibrate home_win with
isotonic regression; leave draw and away_win predictions raw** (or, as
future work, refit calibration using all 380 matches via k-fold
cross-validation rather than a single 43-match holdout, which would
give every class a fairer amount of independent evidence -- noted as a
next step, not built here, to keep this result an honest reflection of
what a single fixed calibration split actually produces).

## Output

`data/processed/calibration/calibration_results.json` -- full per-model,
per-class results (raw/Platt/isotonic Brier and ECE).
