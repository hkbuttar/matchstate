"""
Gradient boosting win-probability model (Step 6): predicts 3-class
(home_win / draw / away_win) from in-game state snapshots built in
features/state.py.

Class order is fixed as [home_win, draw, away_win] = [0, 1, 2] throughout
this module and models/compare.py, matching the baseline models' output
dict order, so predictions line up column-for-column across all three
models being compared.

Deliberately excludes team identity as a feature (no team name/ID, and no
formation). The model can only learn from the *state* -- score, time,
momentum, cards, subs -- not "this is Man City". With only 285 training
matches (the same 20 teams appearing in both train and test -- see
data/README.md's StatsBomb coverage constraint), letting the model key
off team identity would just memorize which teams won that season rather
than learn transferable in-game dynamics.

Formation was tried as a feature and dropped: several clubs used one
formation almost every match that season (Arsenal: 4-2-3-1 in 19/19 home
matches; Chelsea 18/19; Leicester 4-4-2 in 16/19), so it acts as a near
team-identity proxy rather than a dynamic in-game signal. Empirically, it
also made held-out performance worse (Brier 0.3441 with formation vs.
0.3110 without) -- a real, measured overfitting effect, not just a
principled guess, so it's excluded from the shipped feature set. See
models/README.md.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLS = [
    "minute",
    "minutes_remaining",
    "score_diff",
    "xg_diff",
    "momentum_5min_diff",
    "momentum_10min_diff",
    "red_cards_diff",
    "subs_diff",
    "sub_quality_diff",
]
CATEGORICAL_COLS: list[str] = []
RESULT_TO_CLASS = {"H": 0, "D": 1, "A": 2}


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    y = df["final_result"].map(RESULT_TO_CLASS).to_numpy()
    return X, y


def match_level_split(features: pd.DataFrame, match_dates: dict[int, pd.Timestamp], train_frac: float = 0.75):
    """Chronological match-level split -- mirrors bayesian/evaluate.py's
    protocol so all three models (static, dynamic Bayesian, GBM) are
    compared on the exact same held-out matches."""
    ordered_matches = sorted(match_dates, key=lambda m: match_dates[m])
    n_train = int(len(ordered_matches) * train_frac)
    train_ids = set(ordered_matches[:n_train])
    test_ids = set(ordered_matches[n_train:])
    return features[features["match_id"].isin(train_ids)], features[features["match_id"].isin(test_ids)]


def train_gbm(train_df: pd.DataFrame, val_df: pd.DataFrame | None = None, random_state: int = 42) -> xgb.XGBClassifier:
    X_train, y_train = prepare_xy(train_df)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        enable_categorical=True,
        tree_method="hist",
        random_state=random_state,
        eval_metric="mlogloss",
        early_stopping_rounds=30 if val_df is not None else None,
    )
    if val_df is not None:
        X_val, y_val = prepare_xy(val_df)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba_dicts(model: xgb.XGBClassifier, df: pd.DataFrame) -> list[dict]:
    X, _ = prepare_xy(df)
    probs = model.predict_proba(X)
    return [{"home_win": float(p[0]), "draw": float(p[1]), "away_win": float(p[2])} for p in probs]
