"""
Demo/sanity check: does the momentum feature actually rise before goals?
Picks a real 2015/16 match, finds when goals were scored, and prints the
scoring team's momentum in the minutes leading up to each goal.
"""

import json
from pathlib import Path

import pandas as pd

from possession_value.data import RAW_DIR
from possession_value.momentum import compute_momentum
from possession_value.xt_model import ExpectedThreat

ACTIONS_PATH = Path(__file__).parent.parent / "data" / "processed" / "possession_actions_2015_16.parquet"
MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "xt_grid_2015_16.npz"


def find_goals(match_id: int) -> list[dict]:
    events = json.load(open(RAW_DIR / "events" / f"{match_id}.json"))
    goals = []
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"].get("outcome", {}).get("name") == "Goal":
            goals.append({"minute": e["minute"], "second": e["second"], "team": e["team"]["name"], "player": e["player"]["name"]})
    return goals


def main():
    actions = pd.read_parquet(ACTIONS_PATH)
    xt_model = ExpectedThreat.load(MODEL_PATH)

    # pick a high-scoring match (within a scanned slice, for speed) for a clearer demo
    match_id = actions["match_id"].unique()[0]
    best_n_goals = -1
    for mid in actions["match_id"].unique()[:40]:  # scan a slice for speed
        goals = find_goals(mid)
        if len(goals) > best_n_goals:
            best_n_goals = len(goals)
            match_id = mid

    goals = find_goals(match_id)
    home, away = actions[actions["match_id"] == match_id]["team"].unique()[:2]
    print(f"Demo match_id={match_id} ({home} vs {away}), {len(goals)} goals")

    momentum = compute_momentum(actions, xt_model, match_id)

    for g in goals:
        team = g["team"]
        goal_minute = g["minute"]
        print(f"\nGoal: {team}'s {g['player']} at {goal_minute}'{g['second']:02d}")
        window = momentum[(momentum["team"] == team) & (momentum["minute"].between(max(0, goal_minute - 10), goal_minute))]
        for _, row in window.iterrows():
            print(f"  minute {int(row['minute']):3d}: threat_this_min={row['threat_this_minute']*1000:6.2f} "
                  f"momentum_5min={row['momentum_5min']*1000:6.2f} momentum_10min={row['momentum_10min']*1000:6.2f}")


if __name__ == "__main__":
    main()
