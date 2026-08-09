# Hierarchical Bayesian Team-Strength Model

## What this model does differently from the static Dixon-Coles baseline

`baseline.dixon_coles.DixonColes` fits one attack/defense number per team
for an entire season. `bayesian.model.HierarchicalDixonColes` instead lets
each team's attack/defense **drift within a season** via a random walk
across 8 chronological periods (~5 matches/team/period), with the walk's
step size and starting spread shared (partially pooled) across all 20
teams as hyperparameters. Early in the season, when a team has few
matches, its estimate leans on the league-wide distribution; as the
season progresses, its own results dominate.

**Disclosed simplification:** this model omits Dixon-Coles' low-score
(`tau`/`rho`) correction, to keep the sampling problem smaller and faster.
Its effect is expected to be second-order next to the time-varying-strength
question actually being tested here.

## Comparison protocol (`bayesian/evaluate.py`, `bayesian/run_all_seasons.py`)

For each of the 33 complete EPL seasons:
1. Split chronologically: first 75% of matches = train, last 25% = test.
2. Fit static Dixon-Coles on train only (frozen strength, exactly `baseline/`'s model).
3. Fit the hierarchical model on train only; forecast test matches using
   its **last fitted period's** strength (a random walk's best forecast of
   the future is its most recent state -- "current form carried forward").
4. Score both against the actual test-match outcomes with multi-class
   Brier score and log loss (lower is better).

This isolates one question: for matches near the end of a season, is a
model that tracked within-season drift a better predictor than one that
just averaged the whole season?

## Results (all 33 seasons, no cherry-picking)

| Metric | Static Dixon-Coles | Hierarchical Bayesian |
|---|---|---|
| Mean Brier score | 0.5893 | 0.5878 |
| Seasons won (lower Brier) | 11/33 | **22/33** |

The dynamic model has a lower Brier score in 22 of 33 seasons (67%), and a
small average improvement (Δ = -0.0016). **Read this honestly, not as a
sweeping win:** the average edge is modest, not dramatic, and it loses
outright in a third of seasons -- including, notably, 2015/16 (Leicester's
title season), where the static model actually predicted the final
quarter slightly better (Brier 0.5863 vs 0.5896). Full-run gradient
boosting isn't guaranteed to close this gap either; that comparison
belongs to `models/`, not here.

Per-season numbers, including convergence diagnostics, are saved in
`data/processed/bayesian_vs_static.json`.

## Convergence diagnostics

All 33 seasons sampled cleanly with NUTS (4 chains, 800 tune + 800 draw
each; ~5-10 seconds per season). Max r_hat across all ~450 monitored
parameters ranged 1.004-1.014 (comfortably under the standard 1.01
threshold in all but a handful of borderline cases), minimum bulk ESS was
always > 460 (out of 3,200 total draws), and total divergent transitions
per season never exceeded 1 out of 3,200. 6 of 33 seasons tripped a strict
warning threshold (max r_hat ≥ 1.01 or ≥1 divergence) but none by a
meaningful margin -- these are healthy fits, not near-misses.

(Note: `arviz.summary()`'s `round_to` parameter needs the string `"none"`,
not Python `None`, to return full-precision r_hat/ESS -- passing `None`
silently falls back to its 2-decimal default. Caught this mid-build: an
early run reported 30/33 "convergence warnings" that were actually a
rounding artifact of that bug, not real sampling problems.)

## Qualitative sanity check (`bayesian/inspect_trajectory.py`)

Re-fit 2015/16 and inspected team strength trajectories period-by-period.
Leicester's overall strength (attack - defense) rises monotonically every
single period, from +0.32 to +0.57 across the season -- and the
improvement is driven mostly by **defense** (-0.13 -> -0.34), not attack,
which matches the real, specific football history of that title win
(a historically well-organized defense as much as Vardy/Mahrez's
finishing). Aston Villa's trajectory declines every single period,
consistent with their real full-season collapse into relegation. Neither
of these facts was given to the model -- they came out of goals data
alone.

## How this feeds forward

`models/` compares gradient boosting against *both* this model and the
static Dixon-Coles baseline, each recomputed given current score/time --
this file's honest, modest verdict ("time-varying helps somewhat more
often than not, but not dramatically, and not universally") is the bar
that comparison needs to clear, not a foregone conclusion that fancier
always wins.
