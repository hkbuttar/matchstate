# Results & Honest Comparison

This is the synthesis of every model built in this project: one place
with the full comparison table and direct answers to the four questions
it set out to answer, stated plainly rather than spun toward a
predetermined conclusion. Every number here is a held-out or
cross-validated result with a citation back to the module that produced
it -- nothing in this file is a new claim, only an assembly of what was
already found.

## Full comparison table: model x outcome class x market, with 95% CIs

Pre-match (95 test matches) and in-game (9,020 match-minute rows, same
95 matches), Brier score per class (lower is better), block-bootstrapped
by match. From `backtest/final_comparison.py`.

### Pre-match

| Model | Home win | Draw | Away win |
|---|---|---|---|
| Market (Pinnacle, de-vigged) | 0.204 [0.175, 0.236] | 0.216 [0.168, 0.263] | 0.165 [0.132, 0.201] |
| Static Dixon-Coles | 0.209 [0.179, 0.242] | 0.216 [0.171, 0.259] | 0.168 [0.134, 0.206] |
| Hierarchical Bayesian | 0.215 [0.194, 0.238] | 0.214 [0.169, 0.258] | 0.167 [0.134, 0.203] |
| Gradient Boosting | 0.252 [0.226, 0.279] | 0.212 [0.177, 0.247] | 0.188 [0.158, 0.220] |

### In-game (all match-minutes)

| Model | Home win | Draw | Away win |
|---|---|---|---|
| Market + naive adjustment | 0.148 [0.116, 0.183] | 0.172 [0.135, 0.211] | 0.112 [0.086, 0.140] |
| Static Dixon-Coles | 0.152 [0.120, 0.186] | 0.174 [0.136, 0.213] | 0.113 [0.086, 0.143] |
| Hierarchical Bayesian | 0.153 [0.123, 0.186] | 0.171 [0.134, 0.209] | 0.111 [0.085, 0.139] |
| Gradient Boosting | 0.161 [0.133, 0.190] | 0.174 [0.138, 0.211] | 0.126 [0.101, 0.153] |

Every confidence interval here overlaps substantially with every other
model's in its column -- consistent with `backtest/`'s finding that 95
matches mostly can't distinguish these models from each other at the
per-class level. The one exception with real statistical teeth (from
`backtest/`'s combined 3-class analysis): GBM is significantly worse than
the Bayesian model in-game, and the market is significantly better than
GBM in-game.

---

## The four central questions, answered directly

### 1. Does the ML model (gradient boosting) beat the statistical baselines?

**No, not overall -- but with a real, specific exception.** Aggregate
Brier score: GBM 0.4613 vs. static Dixon-Coles 0.4384 vs. Bayesian 0.4346
(`models/`). Bootstrapped (`backtest/`): GBM is *significantly* worse than the
Bayesian model (diff +0.0267, CI [+0.005, +0.048]); not significantly
different from static.

But broken down by match phase (`models/`), GBM **wins clearly in the first
30 minutes** (Brier 0.549 vs. 0.578/0.574 for static/Bayesian) -- exactly
when score/time alone carry the least signal and momentum/xG/cards add
real information the Poisson-family models structurally can't see
(big-moment detection in `models/` confirmed this concretely: GBM reacts
to red cards, static/Bayesian essentially don't, 0.157 vs. 0.005 mean
swing at the red-card minute across 16 real red cards). It loses ground
steadily as the match progresses, ending up worse in the final phase,
traced to a specific, demonstrated mechanism (`models/`): with only 285
training matches, GBM is systematically underconfident on rare, extreme
late-game states (a 2+ goal lead at minute 80 was a 100%-of-58-cases
certain win in the test set; GBM assigned it 79-95%, static Dixon-Coles
correctly assigned 97-99.9%).

**Honest verdict: gradient boosting's real, demonstrated value in this
project is early-match state-awareness specifically, not a wholesale
replacement for the simpler statistical models** -- exactly the kind of
data-volume-limited outcome the disclosed 380-match StatsBomb ceiling
(`data/README.md`) predicted was possible.

### 2. Does it beat the betting market pre-match, or only in-game?

**Neither, at the confidence this dataset can support -- but the gap
narrows substantially in-game.** Pre-match, the market beats every model
(`market/`): Brier 0.585 vs. our best (static) at 0.593, a small (~1.4%
relative) and expected gap -- stated in the plan itself as the normal,
unsurprising result. In-game, a market-plus-naive-score-adjustment
baseline still edges out our best model (0.432 vs. Bayesian's 0.435,
~0.6% relative), and that gap is *consistent across every match phase*
(`market/`'s phase breakdown), but when bootstrapped (`backtest/`) it is **not
statistically significant** (CI [-0.017, +0.012] for market vs.
Bayesian). The market is, however, significantly better than GBM
specifically in-game.

**Honest verdict:** the market is not proven to beat our best in-game
model at 95% confidence, but it isn't proven to lose either -- the point
estimate favors the market throughout, and the honest reading is "our
in-game updating closes most, but not conclusively all, of the gap that
exists before kickoff." This is a meaningfully more encouraging result
than pre-match alone would suggest, without overclaiming a win the
statistics don't support.

### 3. Is draw prediction meaningfully worse than win/loss, across every model tested?

**Yes, but not in the way the per-class Brier/ECE numbers alone would
suggest -- and this is a real, nuanced correction worth stating
explicitly.** By raw Brier score and calibration error (`calibration/`), draw is
*not* uniformly the worst-calibrated class: away-win calibration is
sometimes worse (e.g. GBM's away-win ECE 0.079 vs. its draw ECE 0.047).
Reading only that table would understate the real problem with draws.

The real, well-known phenomenon shows up in a different, more
fundamental measure: **discrimination.** Across all 95 pre-match test
predictions, from every one of the three models, **draw is never once
the predicted (highest-probability) outcome** -- despite actually
occurring in 30.5% of those matches. The highest pre-match draw
probability any model ever assigned, across the entire test set, was
0.35 (static Dixon-Coles) -- never approaching a plurality call. This is
the concrete, project-specific version of a well-documented fact in
football analytics: draws are structurally difficult to call with
confidence because they aren't really "a third kind of result" so much
as "both teams turned out close enough that a win didn't happen" --
there's no direct positive signal for "this will be a draw" the way
there's a signal for "the home team is much stronger."

**Honest verdict: draw prediction is meaningfully worse, but specifically
in the sense of discrimination (a draw is essentially never anyone's top
pick), not uniformly in calibration error** -- both facts are true and
worth keeping separate rather than collapsed into one soundbite.

### 4. Does hierarchical Bayesian updating actually help over static per-season strength?

**Probably, on average, but the effect is small and doesn't clear
statistical significance on the data available.** Across all 33 EPL
seasons (`bayesian/`), the dynamic model has lower Brier score than static
Dixon-Coles in 22/33 seasons (67%), with a small average improvement
(0.5878 vs. 0.5893). Re-tested with proper uncertainty quantification on
the single-season in-game test set (`backtest/`): the gap (Bayesian - static
= -0.0038) has a confidence interval of [-0.0129, +0.0057] -- comfortably
including zero, not statistically significant.

These two results aren't contradictory -- they're different questions at
different statistical power. 33 seasons of directional evidence
(`bayesian/`) is suggestive; 95 matches of rigorous bootstrap evidence
(`backtest/`) is not enough to confirm it at 95% confidence. Where this
project *does* have strong, tightly-bounded evidence on a closely related
question (`backtest/`'s cross-season walk-forward, 8,808 matches): using
**last season's** strength to predict **this season** is significantly
worse than fitting on the season itself (+0.0336 Brier, CI [+0.029,
+0.038], unambiguous). That's not quite the same claim as "within-season
updating helps," but it's a rigorous confirmation of the underlying
premise `bayesian/` was built on -- team strength genuinely does shift in
ways a single fixed estimate misses, at least across season boundaries.
The within-season version of that claim is directionally supported but
not proven at the confidence this project's single-season event dataset
can deliver.

**Honest verdict: yes, probably, but "probably" is doing real work in
that sentence** -- this is a case where more data (more StatsBomb-covered
seasons) would very plausibly settle the question either way; the
directional evidence across 33 seasons is real and shouldn't be
dismissed, but it doesn't meet the higher bar the rigorous bootstrap
analysis rightly holds it to on the smaller, event-level-matched sample.

---

## What ties all four answers together

Every "no, not proven" in this document traces to the same root cause,
disclosed from the very start (`data/README.md`): **StatsBomb's free open data covers
exactly one general-purpose EPL season (380 matches)**, which caps every
event-level comparison (GBM, in-game Bayesian-vs-static, in-game
market-vs-models) at a 95-match held-out test set -- not enough
statistical power to cleanly separate models whose true performance
differs by a few percent. Where this project could use much larger,
purely goals-based data (the 33-season Dixon-Coles walk-forward), the
same kinds of questions get answered with real, tight confidence. That
contrast is itself the project's most important methodological finding,
not a caveat to it.
