# Step 10: Walk-Forward Validation & Statistical Rigor

Every comparison in Steps 3, 6, 8, and 9 reported point estimates on
held-out data. This step asks the question those steps couldn't answer
on their own: **are these differences real, or within noise?** Two
genuinely different analyses, both using block bootstrap (blocked by
match, never by row -- Step 8 already showed row-level correlation
within a match understates uncertainty if ignored).

## Part 1: In-game model comparison, with confidence intervals (`backtest/ingame_bootstrap.py`)

Re-scores Steps 6/9's static Dixon-Coles / hierarchical Bayesian / GBM /
market-plus-naive-adjustment comparison with 2,000 match-block bootstrap
resamples on the same 95 test matches.

| Model | Brier | 95% CI |
|---|---|---|
| Market + naive adjustment | 0.4320 | [0.364, 0.503] |
| Hierarchical Bayesian | 0.4346 | [0.370, 0.504] |
| Static Dixon-Coles | 0.4384 | [0.371, 0.512] |
| Gradient Boosting | 0.4613 | [0.405, 0.523] |

| Comparison | Diff | 95% CI | Result |
|---|---|---|---|
| Bayesian - Static | -0.0038 | [-0.013, +0.006] | not significant |
| GBM - Static | +0.0229 | [-0.004, +0.051] | not significant |
| **GBM - Bayesian** | **+0.0267** | **[+0.005, +0.048]** | **significant** |
| Market - Bayesian | -0.0026 | [-0.017, +0.012] | not significant |
| Market - Static | -0.0064 | [-0.021, +0.009] | not significant |
| **Market - GBM** | **-0.0293** | **[-0.056, -0.002]** | **significant** |

**Honest, humbling headline: almost none of the point-estimate
differences discussed in Steps 3, 6, and 9 survive proper uncertainty
quantification on this single 95-match test set.** Step 3's "Bayesian
beats static in 22/33 seasons" and Step 9's "market edges out our best
model in-game" are both directionally consistent with what the bootstrap
shows, but neither difference clears statistical significance here --
the confidence intervals comfortably include zero. Only two things are
statistically solid: GBM is measurably worse than the Bayesian model, and
the market is measurably better than GBM. Everything else discussed
earlier in this project should be read as "the best available point
estimate, with real, non-trivial uncertainty" rather than a settled
ranking -- exactly the discipline this step exists to enforce.

## Part 2: Cross-season Dixon-Coles walk-forward (`backtest/season_walkforward.py`)

The Bayesian/GBM/market comparisons are necessarily confined to the
single StatsBomb season (Step 1's disclosed coverage constraint) -- but
Dixon-Coles only needs goal-level results, available for all 33 seasons.
This is the one model in the project that can be tested with genuine
cross-season walk-forward validation: fit on season N, predict season
N+1, zero lookahead.

**Scope, disclosed:** ~3 clubs are promoted/relegated every season and
have no prior-season fit at all -- rather than guess, matches involving
such a team are excluded (3,434 of 12,242 candidate matches, 28%). The
result below is specifically "how well does last season's strength
predict this season, among clubs that stayed in the league" -- a real,
disclosed narrowing, not the harder problem of predicting newly-promoted
teams.

Compared against the SAME matches' Step 2 in-sample fit (same-season
parameters -- the honest ceiling of "what if you got to fit on the
season you're predicting"), across 8,808 matches spanning 32
season-transitions (1994/95-2025/26):

| | Brier | 95% CI |
|---|---|---|
| Walk-forward (last season's params) | 0.6033 | [0.596, 0.610] |
| In-sample (same-season params) | 0.5697 | [0.563, 0.577] |
| **Difference** | **+0.0336** | **[+0.029, +0.038] -- significant** |

**This is the clean, well-powered result the 95-match test set couldn't
give.** With 8,808 matches instead of 95, confidence intervals are an
order of magnitude tighter (~0.01 wide vs. ~0.14 wide), and this gap is
unambiguous: predicting purely from last season's fitted strength really
does cost real accuracy compared to fitting on the season itself. This is
a direct, rigorously quantified confirmation of the exact problem that
motivated Step 3's partial-pooling design (teams are hardest to predict
early in a season, before their own results accumulate) -- now backed by
30x the data of any single-season comparison in this project.

## The methodological lesson, stated directly

The contrast between these two analyses is itself a finding: **95
matches is not enough data to reliably distinguish close models from
each other**, even when their point estimates look meaningfully
different. 8,808 matches is enough to detect a real, moderate-sized
effect cleanly. Every claim earlier in this project drawn from the
single StatsBomb season should be read with that asymmetry in mind --
it's the direct, quantified consequence of Step 1's disclosed 380-match
ceiling on event-level data.

## Output

`data/processed/step10_ingame_bootstrap.json`,
`data/processed/step10_season_walkforward.json`.
