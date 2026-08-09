"""
Step 6's core deliverable: compare the gradient boosting model against
BOTH baselines -- static Dixon-Coles (Step 2) and the hierarchical
Bayesian model (Step 3) -- each recomputed given current score/time, on
the exact same held-out match-minutes.

Protocol: chronological 75/25 match-level split of 2015/16 (the same
season/split convention bayesian/evaluate.py used, for direct
comparability with that step's numbers). All three models are fit/trained
on the same 285 train matches and scored on the same 95 test matches'
per-minute rows -- roughly 9,000 held-out (match, minute) snapshots.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.data import load_results
from baseline.dixon_coles import DixonColes
from bayesian.model import HierarchicalDixonColes
from data.team_names import to_football_data
from models.gbm import match_level_split, predict_proba_dicts, train_gbm

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "ingame_features_2015_16.parquet"
MATCHES_FILE = Path(__file__).parent.parent / "data" / "raw" / "statsbomb" / "matches" / "2_27.json"


def _match_dates() -> dict[int, pd.Timestamp]:
    matches = json.load(open(MATCHES_FILE))
    return {m["match_id"]: pd.Timestamp(m["match_date"]) for m in matches}


def outcome_vector(row) -> np.ndarray:
    if row["home_goals"] > row["away_goals"]:
        return np.array([1.0, 0.0, 0.0])
    elif row["home_goals"] == row["away_goals"]:
        return np.array([0.0, 1.0, 0.0])
    return np.array([0.0, 0.0, 1.0])


def score(probs: list[dict], rows: pd.DataFrame) -> tuple[float, float]:
    briers, logs = [], []
    for p, (_, row) in zip(probs, rows.iterrows()):
        pred = np.clip(np.array([p["home_win"], p["draw"], p["away_win"]]), 1e-10, 1.0)
        actual = outcome_vector(row)
        briers.append(np.sum((pred - actual) ** 2))
        logs.append(-np.log(pred[actual.argmax()]))
    return float(np.mean(briers)), float(np.mean(logs))


def main():
    features = pd.read_parquet(FEATURES_PATH)
    match_dates = _match_dates()
    train_df, test_df = match_level_split(features, match_dates, train_frac=0.75)
    print(f"Train: {train_df['match_id'].nunique()} matches ({len(train_df):,} rows) | "
          f"Test: {test_df['match_id'].nunique()} matches ({len(test_df):,} rows)")

    # --- Fit the two baselines on the SAME train matches (goal results only) ---
    results = load_results()
    results_2015_16 = results[results["Season"] == "2015/16"]
    train_home_fd = {to_football_data(t) for t in train_df["home_team"].unique()}
    # restrict by matching (HomeTeam, AwayTeam) pairs actually in the train split
    train_pairs = {
        (to_football_data(h), to_football_data(a))
        for h, a in train_df[["home_team", "away_team"]].drop_duplicates().itertuples(index=False)
    }
    train_results = results_2015_16[
        results_2015_16.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in train_pairs, axis=1)
    ]
    assert len(train_results) == train_df["match_id"].nunique(), "train match count mismatch vs football-data results"

    static_model = DixonColes().fit(train_results)
    dynamic_model = HierarchicalDixonColes(n_periods=8).fit(train_results, draws=800, tune=800, chains=4)

    # --- Train the GBM on an inner chronological split of train (for early stopping) ---
    train_match_dates = {mid: d for mid, d in match_dates.items() if mid in set(train_df["match_id"])}
    gbm_fit_df, gbm_val_df = match_level_split(train_df, train_match_dates, train_frac=0.85)
    gbm = train_gbm(gbm_fit_df, gbm_val_df)

    # --- Score all three on the same held-out test rows ---
    static_probs = [
        static_model.in_game_probabilities(to_football_data(r.home_team), to_football_data(r.away_team), r.minute, r.home_goals, r.away_goals)
        for r in test_df.itertuples()
    ]
    dynamic_probs = [
        dynamic_model.in_game_probabilities(to_football_data(r.home_team), to_football_data(r.away_team), r.minute, r.home_goals, r.away_goals)
        for r in test_df.itertuples()
    ]
    gbm_probs = predict_proba_dicts(gbm, test_df)

    results_table = []
    for name, probs in [("Static Dixon-Coles", static_probs), ("Hierarchical Bayesian", dynamic_probs), ("Gradient Boosting", gbm_probs)]:
        brier, logloss = score(probs, test_df)
        results_table.append({"model": name, "brier": brier, "logloss": logloss})
        print(f"{name:22s}: Brier={brier:.4f}  LogLoss={logloss:.4f}")

    # --- Breakdown by match phase ---
    print("\nBrier score by match phase:")
    phase_bins = [(0, 30, "0-30'"), (30, 60, "30-60'"), (60, 200, "60'+")]
    for lo, hi, label in phase_bins:
        mask = (test_df["minute"] >= lo) & (test_df["minute"] < hi)
        phase_rows = test_df[mask]
        idx = np.where(mask.to_numpy())[0]
        row_line = f"  {label:8s}"
        for name, probs in [("Static", static_probs), ("Bayesian", dynamic_probs), ("GBM", gbm_probs)]:
            sel = [probs[i] for i in idx]
            brier, _ = score(sel, phase_rows)
            row_line += f"  {name}={brier:.4f}"
        print(row_line)

    out_path = PROCESSED_DIR / "step6_comparison.json"
    out_path.write_text(json.dumps(results_table, indent=2))
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
