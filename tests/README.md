# Testing & Validation

21 tests, ~13 seconds total. Two kinds of test here, worth distinguishing:

- **Regression tests**: formalize sanity checks already done ad hoc
  elsewhere (`baseline/sanity_check.py`, `possession_value/fit_and_check.py`,
  `bayesian/run_all_seasons.py`'s convergence report) so they run
  permanently rather than being one-off scripts.
- **New validation, not done anywhere earlier in the project**: parameter
  recovery on synthetic data, posterior predictive checks, calibration
  procedure validation on deliberately-miscalibrated synthetic data, and
  bootstrap procedure validation. These test the *methods themselves*,
  independent of whether real football data happens to cooperate.

## `test_dixon_coles.py`

Generates match results from a Dixon-Coles process with **known** true
attack/defense/home_adv/rho, fits the model, and checks the fit recovers
them. Caught something real in the process: with a realistic
single-season sample size (380 matches), defense correlation and rho's
sign are genuinely *not* reliably recoverable (verified directly, not
assumed) -- confirming that individual seasons' rho sign instability,
observed on real data in `baseline/`, is expected estimator behavior at
this sample size, not a red flag. The fixture uses 3x the data (1,140
matches) for a stable, non-flaky test; a dedicated test documents the
small-sample finding explicitly rather than hiding it.

## `test_bayesian.py`

Hard-assertion convergence diagnostics (max r_hat < 1.02, min ESS > 200,
divergence rate < 1%) using `round_to="none"` -- the exact rounding bug
caught in `bayesian/` that silently rounds r_hat to 2dp by default. Plus a
genuine posterior predictive check: simulate full seasons of goals from
200 random posterior draws (not just the posterior mean) and verify the
*real* season's mean goals/match and draw rate fall within the simulated
95% range.

## `test_possession_value.py`

A synthetic value-iteration test surfaced a real, instructive modeling
property on the first attempt: a toy pitch with deterministic,
always-succeeding transitions toward goal produced a perfectly *flat* xT
grid (every zone valued identically), not an increasing one. That's
correct behavior, not a bug -- with a guaranteed eventual shot and no
discounting, position doesn't matter; it's specifically the *risk of a
turnover*, and that risk shrinking as you approach goal, that produces
the monotonic gradient real football data shows. Fixed by adding
explicit turnover risk to the synthetic construction. Also includes the
real-data monotonicity regression test from `possession_value/`.

## `test_calibration.py`

Constructs synthetic 3-class data with a known, deliberate overconfidence
distortion (probabilities cubed and renormalized) and verifies both
Platt scaling and isotonic regression cut the resulting calibration error
by at least 40% on held-out data. Also checks calibration doesn't
meaningfully harm already-well-calibrated input.

## `test_bootstrap.py`

Validates the block-bootstrap machinery itself: (1) on data constructed
with strong within-match correlation, match-block resampling produces
meaningfully wider (more honest) confidence intervals than naive
row-level resampling would -- the actual justification for blocking by
match throughout `calibration/` and `backtest/`; (2) a planted, real difference between two
synthetic models is correctly flagged significant, with correct sign;
(3) two identical models correctly do NOT get flagged as different
(false-positive check); (4) approximate 95% CI coverage across 150
repeated synthetic experiments with a known true value.

## Running

```
pytest tests/ -v
```
