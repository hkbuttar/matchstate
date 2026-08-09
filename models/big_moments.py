"""
Step 7: automatically flag the largest win-probability swings per match,
and use them as a live sanity check -- do the model's probabilities
actually respond to known-impactful events (goals, red cards)?

Swing magnitude uses total variation distance (TVD) between consecutive
minutes' [home_win, draw, away_win] vectors -- 0.5 * sum(|p[t]-p[t-1]|) --
rather than tracking home_win alone, so a shift into/out of a draw counts
too, not just swings toward/away from a home win.

Reuses the exact fitted models from models/compare.py (same train/test
split), run on the 95 held-out test matches -- consistent with Step 6,
and avoids conflating "big swing" with "model overfit to a match it was
trained on."

Honest hypothesis worth testing explicitly: Dixon-Coles and the
hierarchical Bayesian model only ever see score and time -- NOT red
cards. They should show near-zero reaction to a sending-off unless a
goal follows soon after, while gradient boosting (which has
red_cards_diff as an input feature) should react directly. This is
checked, not assumed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.data import load_results
from baseline.dixon_coles import DixonColes
from data.team_names import to_football_data
from features.lineups import parse_match
from models.gbm import match_level_split, predict_proba_dicts, train_gbm
from possession_value.data import RAW_DIR

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "ingame_features_2015_16.parquet"
MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"


def _match_dates() -> dict[int, pd.Timestamp]:
    matches = json.load(open(MATCHES_FILE))
    return {m["match_id"]: pd.Timestamp(m["match_date"]) for m in matches}


def _goal_events(match_id: int) -> list[dict]:
    events = json.load(open(RAW_DIR / "events" / f"{match_id}.json"))
    goals = []
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"].get("outcome", {}).get("name") == "Goal":
            goals.append({"minute": e["minute"], "team": e["team"]["name"], "kind": "goal"})
        elif e["type"]["name"] == "Own Goal For":
            goals.append({"minute": e["minute"], "team": e["team"]["name"], "kind": "own goal (for)"})
    return goals


def fit_models():
    features = pd.read_parquet(FEATURES_PATH)
    match_dates = _match_dates()
    train_df, test_df = match_level_split(features, match_dates, train_frac=0.75)

    results = load_results()
    results_2015_16 = results[results["Season"] == "2015/16"]
    train_pairs = {
        (to_football_data(h), to_football_data(a))
        for h, a in train_df[["home_team", "away_team"]].drop_duplicates().itertuples(index=False)
    }
    train_results = results_2015_16[results_2015_16.apply(lambda r: (r["HomeTeam"], r["AwayTeam"]) in train_pairs, axis=1)]

    static_model = DixonColes().fit(train_results)
    train_match_dates = {mid: d for mid, d in match_dates.items() if mid in set(train_df["match_id"])}
    gbm_fit_df, gbm_val_df = match_level_split(train_df, train_match_dates, train_frac=0.85)
    gbm = train_gbm(gbm_fit_df, gbm_val_df)

    return static_model, gbm, train_df, test_df


def trajectory_for_match(match_id: int, test_df: pd.DataFrame, static_model: DixonColes, gbm) -> pd.DataFrame:
    match_rows = test_df[test_df["match_id"] == match_id].sort_values("minute").reset_index(drop=True)
    home, away = match_rows["home_team"].iloc[0], match_rows["away_team"].iloc[0]
    home_fd, away_fd = to_football_data(home), to_football_data(away)

    gbm_probs = predict_proba_dicts(gbm, match_rows)
    static_probs = [
        static_model.in_game_probabilities(home_fd, away_fd, r.minute, r.home_goals, r.away_goals)
        for r in match_rows.itertuples()
    ]

    def tvd_series(probs: list[dict]) -> np.ndarray:
        arr = np.array([[p["home_win"], p["draw"], p["away_win"]] for p in probs])
        diffs = np.abs(np.diff(arr, axis=0)).sum(axis=1) * 0.5
        return np.concatenate([[0.0], diffs])

    match_rows["gbm_home_win"] = [p["home_win"] for p in gbm_probs]
    match_rows["static_home_win"] = [p["home_win"] for p in static_probs]
    match_rows["gbm_swing"] = tvd_series(gbm_probs)
    match_rows["static_swing"] = tvd_series(static_probs)
    return match_rows


def annotate_events(match_id: int, minute: int, home: str, away: str) -> str:
    parsed = parse_match(match_id)
    tags = []
    for g in _goal_events(match_id):
        if g["minute"] == minute:
            side = "home" if g["team"] == home else "away"
            tags.append(f"GOAL ({side}, {g['team']}, {g['kind']})")
    for r in parsed["red_cards"]:
        if r["minute"] == minute:
            side = "home" if r["team"] == home else "away"
            tags.append(f"RED CARD ({side}, {r['team']}, {r['player_name']}, {r['card_type']})")
    for s in parsed["substitutions"]:
        if s["minute"] == minute:
            side = "home" if s["team"] == home else "away"
            tags.append(f"SUB ({side}, {s['team']}: {s['player_off_name']} -> {s['player_on_name']})")
    return "; ".join(tags) if tags else ""


def top_swings_report(match_id: int, test_df: pd.DataFrame, static_model: DixonColes, gbm, n: int = 5):
    traj = trajectory_for_match(match_id, test_df, static_model, gbm)
    home, away = traj["home_team"].iloc[0], traj["away_team"].iloc[0]
    final = traj.iloc[-1]
    print(f"\n=== {home} {final.final_home_goals}-{final.final_away_goals} {away} (match_id={match_id}) ===")

    top = traj.nlargest(n, "gbm_swing")
    for _, row in top.sort_values("minute").iterrows():
        events = annotate_events(match_id, int(row.minute), home, away)
        print(f"  minute {int(row.minute):3d}: GBM swing={row.gbm_swing:.3f} (P_home {row.gbm_home_win:.3f})  "
              f"static swing={row.static_swing:.3f} (P_home {row.static_home_win:.3f})  {events}")


def red_card_blind_spot_check(test_df: pd.DataFrame, static_model: DixonColes, gbm):
    """Aggregate check across every red card in the test set: does each
    model's probability actually move at that minute?"""
    match_ids = test_df["match_id"].unique()
    gbm_swings, static_swings = [], []
    for mid in match_ids:
        parsed = parse_match(mid)
        if not parsed["red_cards"]:
            continue
        traj = trajectory_for_match(mid, test_df, static_model, gbm)
        for r in parsed["red_cards"]:
            row = traj[traj["minute"] == r["minute"]]
            if len(row):
                gbm_swings.append(float(row["gbm_swing"].iloc[0]))
                static_swings.append(float(row["static_swing"].iloc[0]))

    print(f"\n=== Red-card blind-spot check ({len(gbm_swings)} red cards in test set) ===")
    print(f"Mean GBM swing at red-card minute:    {np.mean(gbm_swings):.4f}")
    print(f"Mean static-DC swing at red-card minute: {np.mean(static_swings):.4f}")
    print(f"(Baseline typical minute-to-minute swing, for reference, is much smaller than a goal's -- "
          f"see per-match reports above.)")


def main():
    static_model, gbm, train_df, test_df = fit_models()

    # A few illustrative matches: pick ones with goals AND red cards for a rich demo.
    interesting = []
    for mid in test_df["match_id"].unique():
        parsed = parse_match(mid)
        if parsed["red_cards"]:
            interesting.append(mid)
    interesting = interesting[:3] if interesting else list(test_df["match_id"].unique()[:3])

    for mid in interesting:
        top_swings_report(mid, test_df, static_model, gbm)

    red_card_blind_spot_check(test_df, static_model, gbm)


if __name__ == "__main__":
    main()
