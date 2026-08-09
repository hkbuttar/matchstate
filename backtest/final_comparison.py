"""
Step 11: the full model x outcome-class x market comparison table, with
confidence intervals -- extends Step 10's bootstrap to per-class
granularity (Step 10 only bootstrapped the combined 3-class Brier score).
Reuses the exact same fitted models, splits, and market data as Steps
6/9/10 -- no refitting, just finer-grained reporting on results already
established.
"""

import json
from pathlib import Path

import numpy as np

from backtest.block_bootstrap import block_bootstrap_brier, summarize_ci
from calibration.data import CLASS_ORDER, build_splits_and_predictions
from data.team_names import to_football_data
from market.implied_strength import implied_lambda_mu, market_in_game_probabilities
from market.odds import load_market_probabilities

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def per_class_squared_errors(probs: np.ndarray, actual_class: np.ndarray) -> dict[str, np.ndarray]:
    n = len(actual_class)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual_class] = 1.0
    return {CLASS_ORDER[c]: (probs[:, c] - onehot[:, c]) ** 2 for c in range(3)}


def build_market_predictions(test_df, minute0_only: bool):
    odds = load_market_probabilities()
    odds_lookup = {(r.HomeTeam, r.AwayTeam): r for r in odds.itertuples()}
    test_matches = test_df[["match_id", "home_team", "away_team"]].drop_duplicates()

    market_lam_mu, market_prematch = {}, {}
    for r in test_matches.itertuples():
        key = (to_football_data(r.home_team), to_football_data(r.away_team))
        if key not in odds_lookup:
            continue
        row = odds_lookup[key]
        p = (row.market_home_win, row.market_draw, row.market_away_win)
        market_prematch[r.match_id] = p
        market_lam_mu[r.match_id] = implied_lambda_mu(*p)

    has_market = test_df["match_id"].isin(market_lam_mu.keys()).to_numpy()
    sub_df = test_df[has_market].reset_index(drop=True)

    if minute0_only:
        sub_df = sub_df[sub_df["minute"] == 0].reset_index(drop=True)
        probs = np.array([market_prematch[mid] for mid in sub_df["match_id"]])
    else:
        probs = []
        for r in sub_df.itertuples():
            lam, mu = market_lam_mu[r.match_id]
            p = market_in_game_probabilities(lam, mu, r.minute, r.home_goals, r.away_goals)
            probs.append([p[c] for c in CLASS_ORDER])
        probs = np.array(probs)

    return sub_df, probs, has_market


def run_comparison(label: str, n_boot: int = 2000) -> dict:
    predictions, splits = build_splits_and_predictions()
    test_df = splits["test"]
    test_actual = predictions["test"]["actual"]
    minute0_only = label == "prematch"

    market_df, market_probs, has_market = build_market_predictions(test_df, minute0_only)
    match_ids = market_df["match_id"].to_numpy()

    if minute0_only:
        idx = test_df.index[(test_df["minute"] == 0) & has_market].to_numpy()
    else:
        idx = test_df.index[has_market].to_numpy()
    actual = test_actual[idx]

    all_sq_err: dict[str, np.ndarray] = {}
    for cls, arr in per_class_squared_errors(market_probs, actual).items():
        all_sq_err[f"market__{cls}"] = arr
    for name in ["static", "bayesian", "gbm"]:
        probs = predictions["test"][name][idx]
        for cls, arr in per_class_squared_errors(probs, actual).items():
            all_sq_err[f"{name}__{cls}"] = arr

    print(f"\n=== {label} ({len(np.unique(match_ids))} matches, {len(match_ids)} rows) ===")
    boot = block_bootstrap_brier(match_ids, all_sq_err, n_boot=n_boot)

    table = {}
    for key, vals in all_sq_err.items():
        model, cls = key.split("__")
        ci = summarize_ci(boot[key], vals.mean())
        table.setdefault(model, {})[cls] = ci
        print(f"  {model:10s} {cls:10s}: {ci['point']:.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    return table


def main():
    prematch_table = run_comparison("prematch")
    ingame_table = run_comparison("ingame")

    out = {"prematch": prematch_table, "ingame": ingame_table}
    (PROCESSED_DIR / "step11_final_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {PROCESSED_DIR / 'step11_final_comparison.json'}")


if __name__ == "__main__":
    main()
