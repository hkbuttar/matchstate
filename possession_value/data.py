"""
Load StatsBomb event data and extract a tidy per-action table (passes,
carries, shots) for building the possession-value (xT) model.

Scope: 2015/16 season only (StatsBomb open-data match_ids from
data/raw/statsbomb/matches/2_27.json). Per data/README.md, the other
available EPL set (2003/04) covers only Arsenal's matches and is excluded
here to avoid biasing the model toward one team's tactics/personnel.

Coordinate normalization (disclosed judgment call): StatsBomb pitch
coordinates (x in [0,120], y in [0,80]) are NOT normalized for attacking
direction -- a team's x increases toward whichever goal it happens to be
attacking in that period, and teams switch ends at half-time. To build a
single, universal "threat" grid, every team-period's events are flipped
(x -> 120-x, y -> 80-y) as needed so that x always increases toward the
attacking team's target goal. Direction is inferred from that team's own
shot locations in that period (mean shot x > 60 => already attacking
toward x=120); if a team took no shots in a period, its opponent's shots
in the same period are used instead (they attack the opposite end by
definition); if neither team shot in that period at all (rare), that
match-period's events are dropped from the possession-value training set
rather than guessed.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "statsbomb"
MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"  # 2015/16, full season


def season_match_ids() -> list[int]:
    matches = json.load(open(MATCHES_FILE))
    return [m["match_id"] for m in matches]


def _period_directions(events: list[dict]) -> dict[tuple[int, str], int]:
    """Returns {(period, team_name): +1 or -1} where +1 means events are
    already oriented with x increasing toward the attacking goal, -1
    means they need to be flipped. Teams with no shots in a period borrow
    their opponent's (flipped)."""
    shots_by_period_team: dict[tuple[int, str], list[float]] = {}
    for e in events:
        if e["type"]["name"] == "Shot":
            key = (e["period"], e["team"]["name"])
            shots_by_period_team.setdefault(key, []).append(e["location"][0])

    periods = sorted(set(e["period"] for e in events))
    teams = sorted(set(e["team"]["name"] for e in events if "team" in e))
    directions: dict[tuple[int, str], int] = {}
    for period in periods:
        period_teams = [t for t in teams if any(e["period"] == period and e.get("team", {}).get("name") == t for e in events)]
        for team in period_teams:
            key = (period, team)
            if key in shots_by_period_team:
                mean_x = np.mean(shots_by_period_team[key])
                directions[key] = 1 if mean_x > 60 else -1
        # fill in teams with no shots that period from their opponent
        for team in period_teams:
            key = (period, team)
            if key not in directions:
                others = [directions[(period, t)] for t in period_teams if t != team and (period, t) in directions]
                if others:
                    directions[key] = -others[0]
    return directions


def extract_actions(match_id: int) -> pd.DataFrame:
    events = json.load(open(RAW_DIR / "events" / f"{match_id}.json"))
    directions = _period_directions(events)

    rows = []
    for e in events:
        t = e["type"]["name"]
        if t not in ("Pass", "Carry", "Shot"):
            continue
        if "team" not in e or "location" not in e:
            continue
        period, team = e["period"], e["team"]["name"]
        direction = directions.get((period, team))
        if direction is None:
            continue  # no shot evidence at all for either team this period -- drop

        start_x, start_y = e["location"][0], e["location"][1]
        end_x = end_y = np.nan
        success = True
        shot_xg = np.nan

        if t == "Pass":
            p = e["pass"]
            end_x, end_y = p["end_location"][0], p["end_location"][1]
            success = "outcome" not in p
        elif t == "Carry":
            c = e["carry"]
            end_x, end_y = c["end_location"][0], c["end_location"][1]
            success = True
        elif t == "Shot":
            s = e["shot"]
            shot_xg = s["statsbomb_xg"]
            end_x, end_y = start_x, start_y  # shots don't move the ball for xT purposes

        if direction == -1:
            start_x, start_y = 120 - start_x, 80 - start_y
            end_x, end_y = 120 - end_x, 80 - end_y

        rows.append(
            {
                "match_id": match_id,
                "team": team,
                "period": period,
                "minute": e["minute"],
                "second": e["second"],
                "clock_seconds": e["minute"] * 60 + e["second"],
                "event_type": t,
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "success": success,
                "shot_xg": shot_xg,
            }
        )
    return pd.DataFrame(rows)


def load_all_actions(match_ids: list[int] | None = None) -> pd.DataFrame:
    if match_ids is None:
        match_ids = season_match_ids()
    frames = [extract_actions(mid) for mid in match_ids]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ids = season_match_ids()
    print(f"{len(ids)} matches in 2015/16")
    df = extract_actions(ids[0])
    print(df.head(10))
    print(df["event_type"].value_counts())
