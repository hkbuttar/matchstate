"""
Application state: models fit once at API startup and kept in memory,
plus lazily-loaded static data artifacts from earlier steps.

Distinction worth being explicit about: these are "production" models,
fit on the FULL 380-match 2015/16 season -- different from the 285/25
train/test split used throughout Steps 6-11 for honest evaluation. The
comparison/calibration/market endpoints serve those steps' already-computed,
rigorously held-out-evaluated JSON artifacts unchanged; the trajectory
endpoints use these full-season models to give the best available live
prediction for any of the 380 matches, not just the 95 held out for
evaluation. Mixing the two would misrepresent the evaluation results, so
they're kept in genuinely separate code paths, not just separate variables.
"""

import json
from pathlib import Path

import pandas as pd

from baseline.data import load_results
from baseline.dixon_coles import DixonColes
from bayesian.model import HierarchicalDixonColes
from features.lineups import parse_match
from models.gbm import prepare_xy, train_gbm
from possession_value.data import RAW_DIR

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"


class AppState:
    static_model: DixonColes
    bayesian_model: HierarchicalDixonColes
    gbm_model: object
    features_df: pd.DataFrame
    match_meta: dict[int, dict]

    def load(self):
        print("Fitting production models on full 2015/16 season ...")
        results = load_results()
        season_results = results[results["Season"] == "2015/16"]

        self.static_model = DixonColes().fit(season_results)
        self.bayesian_model = HierarchicalDixonColes(n_periods=8).fit(
            season_results, draws=800, tune=800, chains=4, random_seed=42
        )

        self.features_df = pd.read_parquet(PROCESSED_DIR / "ingame_features_2015_16.parquet")
        n = len(self.features_df)
        fit_n = int(n * 0.85)
        # match-level split for the internal validation set, consistent
        # with models/gbm.py's convention elsewhere in the project
        match_order = self.features_df.drop_duplicates("match_id")["match_id"].tolist()
        fit_matches = set(match_order[: int(len(match_order) * 0.85)])
        fit_df = self.features_df[self.features_df["match_id"].isin(fit_matches)]
        val_df = self.features_df[~self.features_df["match_id"].isin(fit_matches)]
        self.gbm_model = train_gbm(fit_df, val_df)

        matches = json.load(open(MATCHES_FILE))
        self.match_meta = {m["match_id"]: m for m in matches}

        print(f"Ready: {self.features_df['match_id'].nunique()} matches available.")
        return self


state = AppState()
