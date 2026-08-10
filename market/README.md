# Betting-Market Benchmark

## Data and method

**Pinnacle's closing odds** (`PSCH/D/A` in football-data.co.uk's columns), full coverage across all 380 matches of 2015/16. Pinnacle is used deliberately rather than a recreational bookmaker like Bet365: it's widely regarded as the sharpest, lowest-margin market in the industry (it accepts rather than limits winning bettors, which forces its lines toward true probability), making it the most credible test available in this dataset. Confirms this in the data itself: mean overround **2.04%**, notably tighter than a typical recreational book (~5-8%).

**De-vig method (disclosed judgment call):** simple multiplicative normalization -- convert odds to raw implied probabilities (1/odds), then divide by their sum so they total 1 exactly. More sophisticated methods (e.g. Shin's method, modeling a specific insider-trading mechanism) exist but presume a particular theory of *why* the margin is shaped as it is; simple normalization is transparent and doesn't make that assumption.

## Part 1: Pre-match (95 held-out test matches)

| Model | Brier | Log loss |
|---|---|---|
| **Market (Pinnacle, de-vigged)** | **0.5850** | **0.9762** |
| Static Dixon-Coles | 0.5933 | 0.9871 |
| Hierarchical Bayesian | 0.5960 | 0.9964 |
| Gradient Boosting | 0.6516 | 1.0770 |

**The market wins, as expected -- stated plainly, this is normal, not a failure.** The gap to our best model (static Dixon-Coles) is small (~1.4% relative Brier), consistent with a simple, publicly-documented statistical model getting reasonably close to a professional market without beating it.

**Why GBM does noticeably worse here, specifically:** this isn't really a fair test of GBM's value proposition. `models/` deliberately excluded team identity from its features, so at minute 0 every match's score/xG/ cards/subs are all ~zero -- GBM has essentially no way to differentiate matches before kickoff. Its whole value (see `models/README.md`) is in-game state-awareness; including it in a pre-match comparison mostly measures the absence of a feature it was never designed to have.

## Part 2: In-game -- the isolated question that matters most here

Does the models' real-time (score/time-conditioned) update add genuine value beyond just taking the market's pre-match view and adjusting it naively for the current score? Built a "market + naive adjustment" baseline that inverts the market's pre-match probabilities to an implied (lambda, mu) goal-rate pair, then applies the *exact same* time-scaling mechanism our own baseline uses -- isolating one variable (whose pre-match prior is better) while holding the update mechanism identical.

| Model | Brier | Log loss |
|---|---|---|
| **Market + naive score/time adjustment** | **0.4320** | 0.7730 |
| Hierarchical Bayesian | 0.4346 | 0.7739 |
| Static Dixon-Coles | 0.4384 | 0.7816 |
| Gradient Boosting | 0.4613 | 0.7960 |

**Still don't clearly beat it -- but the gap shrinks substantially.** Pre-match, our best model trailed the market by ~1.4-1.9% relative Brier; in-game, the Bayesian model trails the market-plus-naive-adjustment baseline by only ~0.6%. That's a genuinely interesting, honest finding: our more sophisticated in-game updating (real fitted attack/defense strength evolution vs. a naive fixed-rate rescaling of the market's prior) closes most, but not all, of the gap that existed before kickoff. With only 95 test matches, a 0.6% gap is small enough that it's plausibly within noise -- `backtest/`'s bootstrap confidence intervals assess this properly rather than reading a point estimate as definitive.

| Phase | Market+naive | Static | Bayesian | GBM |
|---|---|---|---|---|
| 0-30' | 0.5706 | 0.5782 | 0.5745 | 0.6125 |
| 30-60' | 0.4894 | 0.4964 | 0.4912 | 0.5033 |
| 60'+ | 0.2638 | 0.2686 | 0.2659 | 0.2954 |

The ranking (market-naive slightly ahead, Bayesian closest of ours, static close behind, GBM trailing throughout) is consistent across every phase -- not just an average masking a reversal somewhere.

## A real bug caught and fixed here, worth recording

The first run of this comparison showed the market performing *catastrophically* (Brier 0.70+ pre-match, 0.86 in the 60'+ phase -- worse than random guessing), which is not a plausible result for a professional market and was correctly treated as a red flag rather than reported. Root cause: `market/compare.py` re-sorted its own copy of the test DataFrame by `(match_id, minute)` after `calibration.data`'s predictions had already been computed from an *unsorted* copy -- same row count, silently misaligned rows. Fixed by having `build_splits_and_predictions()` return the exact DataFrames its predictions were computed from, rather than trusting two independent calls to reproduce identical row order. Numbers above are post-fix.

## Output

`data/processed/market_comparison.json`.
