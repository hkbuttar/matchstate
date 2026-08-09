"""
Build the per-match, per-minute in-game state feature table:
score, time, running xG differential, possession-value momentum
(possession_value/), red cards, and lineup-aware substitution quality.

Disclosed simplification: formation is taken from each team's starting
XI only. StatsBomb does expose mid-match tactical shifts, but tracking
those adds real complexity for what's expected to be a second-order
effect next to score/time/momentum -- left as a documented possible
extension, not built here.
"""

import json
from pathlib import Path

import pandas as pd

from data.team_names import to_football_data
from features.lineups import parse_match
from possession_value.data import RAW_DIR, season_match_ids
from possession_value.momentum import compute_momentum, momentum_differential
from possession_value.xt_model import ExpectedThreat

MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _match_meta() -> dict[int, dict]:
    matches = json.load(open(MATCHES_FILE))
    return {
        m["match_id"]: {
            "home_team": m["home_team"]["home_team_name"],
            "away_team": m["away_team"]["away_team_name"],
        }
        for m in matches
    }


def _goal_events(match_id: int) -> list[dict]:
    """Regular shot-goals plus own goals. StatsBomb logs an own goal as
    both 'Own Goal Against' (on the team that conceded) and 'Own Goal For'
    (on the team credited with the goal) -- only the latter is counted
    here to avoid double-counting."""
    events = json.load(open(RAW_DIR / "events" / f"{match_id}.json"))
    goals = []
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"].get("outcome", {}).get("name") == "Goal":
            goals.append({"minute": e["minute"], "team": e["team"]["name"]})
        elif e["type"]["name"] == "Own Goal For":
            goals.append({"minute": e["minute"], "team": e["team"]["name"]})
    return goals


def build_match_features(
    match_id: int,
    home_team: str,
    away_team: str,
    actions: pd.DataFrame,
    xt_model: ExpectedThreat,
    quality_lookup: dict[str, float],
    default_quality: float,
) -> pd.DataFrame:
    match_actions = actions[actions["match_id"] == match_id]
    parsed = parse_match(match_id)
    end_minute = parsed["match_end_minute"]

    goals = _goal_events(match_id)
    momentum = compute_momentum(match_actions, xt_model, match_id)
    mom_diff = momentum_differential(momentum, home_team, away_team).set_index("minute")

    shots = match_actions[match_actions["event_type"] == "Shot"]
    xg_by_min_team = shots.groupby(["minute", "team"])["shot_xg"].sum().unstack("team", fill_value=0.0)
    xg_by_min_team = xg_by_min_team.reindex(range(0, end_minute + 1), fill_value=0.0)
    for team in (home_team, away_team):
        if team not in xg_by_min_team.columns:
            xg_by_min_team[team] = 0.0
    cum_xg_home = xg_by_min_team[home_team].cumsum()
    cum_xg_away = xg_by_min_team[away_team].cumsum()

    rows = []
    home_goals = away_goals = 0
    home_red = away_red = 0
    home_subs = away_subs = 0
    home_sub_quality = away_sub_quality = 0.0

    goals_by_minute: dict[int, list[str]] = {}
    for g in goals:
        goals_by_minute.setdefault(g["minute"], []).append(g["team"])
    reds_by_minute: dict[int, list[str]] = {}
    for r in parsed["red_cards"]:
        reds_by_minute.setdefault(r["minute"], []).append(r["team"])
    subs_by_minute: dict[int, list[dict]] = {}
    for s in parsed["substitutions"]:
        subs_by_minute.setdefault(s["minute"], []).append(s)

    for minute in range(0, end_minute + 1):
        for team in goals_by_minute.get(minute, []):
            if team == home_team:
                home_goals += 1
            elif team == away_team:
                away_goals += 1
        for team in reds_by_minute.get(minute, []):
            if team == home_team:
                home_red += 1
            elif team == away_team:
                away_red += 1
        for s in subs_by_minute.get(minute, []):
            q_off = quality_lookup.get(s["player_off_name"], default_quality)
            q_on = quality_lookup.get(s["player_on_name"], default_quality)
            delta = q_on - q_off
            if s["team"] == home_team:
                home_subs += 1
                home_sub_quality += delta
            elif s["team"] == away_team:
                away_subs += 1
                away_sub_quality += delta

        rows.append(
            {
                "match_id": match_id,
                "minute": minute,
                "minutes_remaining": max(0, 90 - minute),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "score_diff": home_goals - away_goals,
                "xg_diff": float(cum_xg_home.get(minute, cum_xg_home.iloc[-1] if len(cum_xg_home) else 0.0) -
                                  cum_xg_away.get(minute, cum_xg_away.iloc[-1] if len(cum_xg_away) else 0.0)),
                "momentum_5min_diff": float(mom_diff["momentum_5min_diff"].get(minute, 0.0)),
                "momentum_10min_diff": float(mom_diff["momentum_10min_diff"].get(minute, 0.0)),
                "red_cards_diff": home_red - away_red,
                "subs_diff": home_subs - away_subs,
                "sub_quality_diff": home_sub_quality - away_sub_quality,
                "home_formation": parsed["formations"].get(home_team),
                "away_formation": parsed["formations"].get(away_team),
            }
        )

    return pd.DataFrame(rows)


def build_all_features(match_ids: list[int] | None = None) -> pd.DataFrame:
    if match_ids is None:
        match_ids = season_match_ids()

    meta = _match_meta()
    actions = pd.read_parquet(PROCESSED_DIR / "possession_actions_2015_16.parquet")
    xt_model = ExpectedThreat.load(PROCESSED_DIR / "xt_grid_2015_16.npz")
    quality = pd.read_parquet(PROCESSED_DIR / "player_quality_2015_16.parquet")
    quality_lookup = dict(zip(quality["player_name"], quality["quality_per90"]))
    default_quality = float(quality.loc[quality["reliable"], "quality_per90"].mean())

    frames = []
    for mid in match_ids:
        home, away = meta[mid]["home_team"], meta[mid]["away_team"]
        frames.append(build_match_features(mid, home, away, actions, xt_model, quality_lookup, default_quality))
    features = pd.concat(frames, ignore_index=True)

    # attach final outcome label from football-data.co.uk
    from baseline.data import load_results

    results = load_results()
    results_2015_16 = results[results["Season"] == "2015/16"]
    label_lookup = {}
    for mid in match_ids:
        home_fd = to_football_data(meta[mid]["home_team"])
        away_fd = to_football_data(meta[mid]["away_team"])
        match_row = results_2015_16[(results_2015_16["HomeTeam"] == home_fd) & (results_2015_16["AwayTeam"] == away_fd)]
        if len(match_row) != 1:
            raise ValueError(f"Expected exactly one football-data match for {home_fd} vs {away_fd}, found {len(match_row)}")
        row = match_row.iloc[0]
        label_lookup[mid] = {"final_home_goals": int(row["FTHG"]), "final_away_goals": int(row["FTAG"]), "final_result": row["FTR"]}

    features["final_home_goals"] = features["match_id"].map(lambda m: label_lookup[m]["final_home_goals"])
    features["final_away_goals"] = features["match_id"].map(lambda m: label_lookup[m]["final_away_goals"])
    features["final_result"] = features["match_id"].map(lambda m: label_lookup[m]["final_result"])

    return features


if __name__ == "__main__":
    df = build_all_features()
    out_path = PROCESSED_DIR / "ingame_features_2015_16.parquet"
    df.to_parquet(out_path)
    print(f"{len(df):,} rows ({df['match_id'].nunique()} matches) saved to {out_path}")
    print(df.head(10).to_string())
