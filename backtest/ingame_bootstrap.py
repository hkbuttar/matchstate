"""
Wraps the in-game model comparison from `models/` and `market/` (static
Dixon-Coles, hierarchical Bayesian, gradient boosting,
market+naive-adjustment) with match-block bootstrap confidence intervals
-- are the small gaps found there real, or within noise given only 95
test matches?
"""

import json
from pathlib import Path

import numpy as np

from backtest.block_bootstrap import block_bootstrap_brier, summarize_ci, summarize_diff
from calibration.data import CLASS_ORDER, build_splits_and_predictions
from data.team_names import to_football_data
from market.implied_strength import implied_lambda_mu, market_in_game_probabilities
from market.odds import load_market_probabilities

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def per_row_squared_error(probs: np.ndarray, actual_class: np.ndarray) -> np.ndarray:
    n = len(actual_class)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual_class] = 1.0
    return np.sum((probs - onehot) ** 2, axis=1)


def main(n_boot: int = 2000):
    predictions, splits = build_splits_and_predictions()
    test_df = splits["test"]
    test_actual = predictions["test"]["actual"]

    odds = load_market_probabilities()
    odds_lookup = {(r.HomeTeam, r.AwayTeam): r for r in odds.itertuples()}
    test_matches = test_df[["match_id", "home_team", "away_team"]].drop_duplicates()
    market_lam_mu = {}
    for r in test_matches.itertuples():
        key = (to_football_data(r.home_team), to_football_data(r.away_team))
        if key in odds_lookup:
            row = odds_lookup[key]
            market_lam_mu[r.match_id] = implied_lambda_mu(row.market_home_win, row.market_draw, row.market_away_win)

    has_market_row = test_df["match_id"].isin(market_lam_mu.keys()).to_numpy()
    sub_df = test_df[has_market_row].reset_index(drop=True)
    sub_actual = test_actual[has_market_row]
    match_ids = sub_df["match_id"].to_numpy()

    market_probs = []
    for r in sub_df.itertuples():
        lam, mu = market_lam_mu[r.match_id]
        p = market_in_game_probabilities(lam, mu, r.minute, r.home_goals, r.away_goals)
        market_probs.append([p[c] for c in CLASS_ORDER])
    market_probs = np.array(market_probs)

    sq_err = {"market": per_row_squared_error(market_probs, sub_actual)}
    for name in ["static", "bayesian", "gbm"]:
        probs = predictions["test"][name][has_market_row]
        sq_err[name] = per_row_squared_error(probs, sub_actual)

    print(f"Block bootstrap ({n_boot} draws, blocked by match, n={len(np.unique(match_ids))} matches) ...")
    boot = block_bootstrap_brier(match_ids, sq_err, n_boot=n_boot)

    print("\nPer-model Brier score, 95% CI:")
    cis = {}
    for name, vals in sq_err.items():
        ci = summarize_ci(boot[name], vals.mean())
        cis[name] = ci
        print(f"  {name:10s}: {ci['point']:.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    print("\nPairwise differences (negative = first model better), 95% CI:")
    pairs = [
        ("bayesian", "static"),
        ("gbm", "static"),
        ("gbm", "bayesian"),
        ("market", "bayesian"),
        ("market", "static"),
        ("market", "gbm"),
    ]
    diffs = {}
    for a, b in pairs:
        d = summarize_diff(boot[a], boot[b], sq_err[a].mean(), sq_err[b].mean())
        diffs[f"{a}_minus_{b}"] = d
        sig = "SIGNIFICANT" if d["significant_at_95"] else "not significant"
        print(f"  {a:10s} - {b:10s}: {d['point_diff']:+.4f}  [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  ({sig})")

    out = {"per_model_ci": cis, "pairwise_diffs": diffs, "n_boot": n_boot, "n_matches": int(len(np.unique(match_ids)))}
    (PROCESSED_DIR / "ingame_bootstrap.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {PROCESSED_DIR / 'ingame_bootstrap.json'}")


if __name__ == "__main__":
    main()
