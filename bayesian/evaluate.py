"""
This module's core deliverable: an honest, direct comparison of the
static per-season Dixon-Coles baseline (`baseline/`) against this
hierarchical, within-season time-varying Bayesian model, on genuinely
held-out matches.

Protocol per season:
  1. Split chronologically: first 75% of matches = train, last 25% = test.
  2. Fit static Dixon-Coles on train only (frozen -- same strength used
     for every test match, exactly `baseline/`'s model).
  3. Fit the hierarchical Bayesian model on train only, then forecast test
     matches using its LAST fitted period's strength estimate (a random
     walk's best forecast of the future is its most recent state -- i.e.
     "current form carried forward").
  4. Score both on the test matches with multi-class Brier score and log
     loss. Lower is better for both.

This isolates the question this module asks: does letting strength drift
within a season, rather than averaging it over the whole season, produce
better predictions for matches near the end of that season? Reported
honestly either way -- no cherry-picking of only-favorable seasons.
"""

import numpy as np
import pandas as pd

from baseline.data import load_results
from baseline.dixon_coles import DixonColes
from bayesian.model import HierarchicalDixonColes


def outcome_vector(row) -> np.ndarray:
    if row["FTHG"] > row["FTAG"]:
        return np.array([1.0, 0.0, 0.0])
    elif row["FTHG"] == row["FTAG"]:
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def brier_and_logloss(probs: list[dict], test_df: pd.DataFrame) -> tuple[float, float]:
    briers, logs = [], []
    for p, (_, row) in zip(probs, test_df.iterrows()):
        pred = np.array([p["home_win"], p["draw"], p["away_win"]])
        pred = np.clip(pred, 1e-10, 1.0)
        actual = outcome_vector(row)
        briers.append(np.sum((pred - actual) ** 2))
        logs.append(-np.log(pred[actual.argmax()]))
    return float(np.mean(briers)), float(np.mean(logs))


def evaluate_season(season: str, n_periods: int = 8, train_frac: float = 0.75, **sample_kwargs) -> dict:
    results = load_results()
    season_df = results[results["Season"] == season].sort_values("Date").reset_index(drop=True)
    n_train = int(len(season_df) * train_frac)
    train, test = season_df.iloc[:n_train], season_df.iloc[n_train:]
    print(f"\n=== {season}: {len(train)} train matches, {len(test)} test matches ===")

    static_model = DixonColes().fit(train)
    static_probs = [static_model.match_probabilities(r.HomeTeam, r.AwayTeam) for r in test.itertuples()]
    static_brier, static_logloss = brier_and_logloss(static_probs, test)

    dyn_model = HierarchicalDixonColes(n_periods=n_periods).fit(train, **sample_kwargs)
    dyn_probs = [dyn_model.match_probabilities(r.HomeTeam, r.AwayTeam) for r in test.itertuples()]
    dyn_brier, dyn_logloss = brier_and_logloss(dyn_probs, test)

    print(f"  Static Dixon-Coles   : Brier={static_brier:.4f}  LogLoss={static_logloss:.4f}")
    print(f"  Hierarchical Bayesian: Brier={dyn_brier:.4f}  LogLoss={dyn_logloss:.4f}")
    winner = "Bayesian" if dyn_brier < static_brier else "Static"
    print(f"  -> lower Brier score: {winner}")

    return {
        "season": season,
        "n_train": len(train),
        "n_test": len(test),
        "static_brier": static_brier,
        "static_logloss": static_logloss,
        "dynamic_brier": dyn_brier,
        "dynamic_logloss": dyn_logloss,
        "dyn_model": dyn_model,
    }


if __name__ == "__main__":
    import sys

    season = sys.argv[1] if len(sys.argv) > 1 else "2015/16"
    evaluate_season(season, draws=800, tune=800, chains=4)
