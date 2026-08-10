# Gradient Boosting Win Probability Model

## What this builds

An XGBoost 3-class classifier (`multi:softprob`) predicting home_win / draw / away_win from `features/`'s in-game state features: score differential, time, running xG differential, `possession_value/`'s possession-value momentum, red cards, substitution count, and lineup-aware substitution quality. Team identity is deliberately excluded (see below).

## Comparison protocol (`models/compare.py`)

Same chronological 75/25 match-level split as `bayesian/evaluate.py` (285 train / 95 test matches of 2015/16), so all three models are directly comparable:

1. **Static Dixon-Coles** (`baseline/`) and **Hierarchical Bayesian** (`bayesian/`) fit on the 285 train matches' results only, each recomputed at every test row's exact (minute, score) via their `in_game_probabilities()` methods (the Bayesian model's was added here, mirroring the static model's time-scaling approach).
2. **Gradient boosting** trained on the 285 train matches' ~27,000 per-minute rows (with an internal 85/15 chronological split for early stopping), scored on the same 95 held-out matches' ~9,000 rows.

All three scored with the same multi-class Brier score and log loss used in `bayesian/`, on the exact same test rows.

## Formation was tried and dropped -- a measured, not assumed, decision

Formation initially looked informative (~15% combined feature importance), but checking *why* revealed a real problem: several clubs used one formation almost exclusively that season (Arsenal 4-2-3-1 in 19/19 home matches, Chelsea 18/19, Leicester 4-4-2 in 16/19) -- with only 20 teams appearing in both train and test (the StatsBomb single-season coverage constraint, see `data/README.md`, means there's no "unseen team" holdout), formation functions as a near-team-identity fingerprint rather than a dynamic in-game signal. Confirmed empirically, not just by suspicion: held-out Brier score was **worse with formation included** (0.3441) than without (0.3110) -- a measured overfitting effect. Dropped from the shipped feature set.

## Results (95 held-out matches, ~9,020 match-minute snapshots)

| Model | Brier score | Log loss |
|---|---|---|
| Static Dixon-Coles | 0.2978 | 0.5137 |
| Hierarchical Bayesian | **0.2923** | 0.5141 |
| Gradient Boosting | 0.3110 | 0.5716 |

**Read plainly: gradient boosting does not beat either statistical baseline overall**, on only 285 training matches. That's a real result, not a failure to tune -- and the phase breakdown shows precisely why:

| Phase | Static | Bayesian | GBM |
|---|---|---|---|
| 0-30' | 0.6256 | 0.6121 | **0.5492** |
| 30-60' | 0.2615 | 0.2578 | 0.3103 |
| 60'+ | 0.0477 | 0.0473 | 0.1072 |

**Gradient boosting wins clearly in the first 30 minutes** (12-10% better Brier than the baselines) -- exactly when score/time alone carry the least information and momentum/xG/card state add real signal the Poisson models structurally can't see. It then **loses ground steadily as the match progresses**, ending up roughly 2x worse than the baselines in the final phase.

**Concrete mechanism, not just aggregate numbers:** every one of the 58 held-out snapshots with a 2+ goal lead around the 80th minute ended in a win for the leading team. Static Dixon-Coles correctly assigns near-certainty there (0.97-0.999 win probability, exactly what a well-specified Poisson process implies from an insurmountable-in-the- time-remaining lead). Gradient boosting is systematically underconfident in the same spots (0.79-0.95) -- with only 285 matches, it hasn't seen enough of the (rare, extreme) large-lead-late-in-match states to learn how close to certain they really are, where the analytic baseline gets this right "for free" from its parametric assumptions.

## Honest takeaway

This is exactly the outcome the disclosed StatsBomb-coverage constraint (380 matches, one season -- see `data/README.md`) predicted was possible: a flexible model needs more data than a well-specified simple parametric one to match it in the regimes that parametric model was built to handle well. Gradient boosting's real, demonstrated value here is specifically in early-match state-awareness, not as a wholesale replacement for the Dixon-Coles family. A natural, not-yet-built refinement: blend the two (e.g. GBM early, baseline-dominated late, or an explicit ensemble) -- noted as future work rather than implemented here, to keep this result honestly a straight three-way comparison rather than a post-hoc-tuned "win."

## Output

`data/processed/gbm_vs_baselines.json` -- summary metrics. Full per-model prediction code is in `models/gbm.py` (feature prep, training) and `models/compare.py` (the three-way evaluation).

---

# Big-Moment Detection (`models/big_moments.py`)

Flags the largest win-probability swings per match (total variation distance between consecutive minutes' [home/draw/away] vectors) and cross-references them against real goals/red cards/subs -- both a demo-able feature and a genuine sanity check on whether the models react sensibly to known-impactful events.

## A testable hypothesis, not an assumption

Dixon-Coles and the Bayesian model only ever see score and time -- they have no red-card input at all. The prediction: they should show almost no reaction to a sending-off unless/until a goal follows, while gradient boosting (which has `red_cards_diff` as a feature) should react directly. Checked this explicitly rather than asserting it.

**Concrete example** (Crystal Palace 0-0 Everton, test set): Everton's James McCarthy picks up a second yellow at minute 51.
- GBM swing = **0.505** (P(home win) jumps 0.30 -> 0.81 -- correctly reflecting that Crystal Palace, now facing 10 men, are much likelier to win)
- Static Dixon-Coles swing = **0.006** (essentially flat -- it has no way to know a man went off)

**Aggregate, not just one anecdote:** across all 16 red cards in the 95 held-out test matches, mean swing at the red-card minute was **0.157** for GBM vs. **0.0049** for static Dixon-Coles -- confirming this is a systematic, structural blind spot in the Poisson-family models, not a one-off. This makes the gradient boosting comparison above concrete: even though gradient boosting loses on aggregate Brier score, it captures information (red cards, momentum) the baselines are structurally incapable of using at all.

**Goals are a positive control, and both models pass it:** e.g. Watford's 89th-minute goal against Aston Villa produces a swing of 0.750 (GBM) and 0.962 (static DC) -- both react strongly, as expected, since score is an input to every model in this project.
