"""
Generates raw (uncalibrated) predictions from all three Step 6 models on
a clean three-way split: fit (242 matches) / calibration (43 matches) /
test (95 matches). The calibration set is disjoint from both model
fitting and the final test set, so calibrators fit here and evaluated on
the test set aren't leaking test information into their own correction.
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
from possession_value.data import RAW_DIR

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "ingame_features_2015_16.parquet"
MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"

CLASS_ORDER = ["home_win", "draw", "away_win"]
RESULT_TO_CLASS = {"H": 0, "D": 1, "A": 2}


def _match_dates() -> dict[int, pd.Timestamp]:
    matches = json.load(open(MATCHES_FILE))
    return {m["match_id"]: pd.Timestamp(m["match_date"]) for m in matches}


def _results_for(df: pd.DataFrame, results_2015_16: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        (to_football_data(h), to_football_data(a))
        for h, a in df[["home_team", "away_team"]].drop_duplicates().itertuples(index=False)
    }
    return results_2015_16[results_2015_16.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in pairs, axis=1)]


def probs_to_matrix(probs: list[dict]) -> np.ndarray:
    return np.array([[p[c] for c in CLASS_ORDER] for p in probs])


def build_splits_and_predictions():
    features = pd.read_parquet(FEATURES_PATH)
    match_dates = _match_dates()
    train_df, test_df = match_level_split(features, match_dates, train_frac=0.75)
    train_match_dates = {mid: d for mid, d in match_dates.items() if mid in set(train_df["match_id"])}
    fit_df, cal_df = match_level_split(train_df, train_match_dates, train_frac=0.85)

    print(f"fit: {fit_df['match_id'].nunique()} matches | calibration: {cal_df['match_id'].nunique()} matches | "
          f"test: {test_df['match_id'].nunique()} matches")

    results = load_results()
    results_2015_16 = results[results["Season"] == "2015/16"]
    fit_results = _results_for(fit_df, results_2015_16)

    static_model = DixonColes().fit(fit_results)
    dynamic_model = HierarchicalDixonColes(n_periods=8).fit(fit_results, draws=800, tune=800, chains=4)
    gbm = train_gbm(fit_df, cal_df)

    def dc_probs(model, df):
        return [
            model.in_game_probabilities(to_football_data(r.home_team), to_football_data(r.away_team), r.minute, r.home_goals, r.away_goals)
            for r in df.itertuples()
        ]

    predictions = {}
    for split_name, df in [("cal", cal_df), ("test", test_df)]:
        predictions[split_name] = {
            "static": probs_to_matrix(dc_probs(static_model, df)),
            "bayesian": probs_to_matrix(dc_probs(dynamic_model, df)),
            "gbm": probs_to_matrix(predict_proba_dicts(gbm, df)),
            "actual": df["final_result"].map(RESULT_TO_CLASS).to_numpy(),
        }

    return predictions
