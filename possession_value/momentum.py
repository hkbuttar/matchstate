"""
Aggregate per-action xT (possession) values into a rolling "momentum"
feature per team: how much threat has this team generated in the last 5
/ 10 minutes? This is richer than a raw shot/xG count because it credits
sustained good positional play even in spells with no shots at all --
exactly the signal Step 6's win-probability model needs beyond what
Steps 2-3's score-based models can see.
"""

import pandas as pd

from possession_value.xt_model import ExpectedThreat


def compute_momentum(actions: pd.DataFrame, xt_model: ExpectedThreat, match_id: int) -> pd.DataFrame:
    """Returns a tidy DataFrame: one row per (minute, team), with threat
    generated that minute and rolling 5-/10-minute trailing sums."""
    m = actions[
        (actions["match_id"] == match_id)
        & actions["event_type"].isin(["Pass", "Carry"])
        & actions["success"]
    ].copy()
    if m.empty:
        return pd.DataFrame(columns=["match_id", "minute", "team", "threat_this_minute", "momentum_5min", "momentum_10min"])

    m["value"] = xt_model.action_value(m)
    m["minute_bucket"] = (m["clock_seconds"] // 60).astype(int)

    teams = sorted(m["team"].unique())
    max_minute = int(m["minute_bucket"].max())
    per_minute = (
        m.groupby(["minute_bucket", "team"])["value"].sum().unstack("team", fill_value=0.0).reindex(
            range(0, max_minute + 1), fill_value=0.0
        )
    )
    for team in teams:
        if team not in per_minute.columns:
            per_minute[team] = 0.0

    momentum_5 = per_minute.rolling(5, min_periods=1).sum()
    momentum_10 = per_minute.rolling(10, min_periods=1).sum()

    rows = []
    for minute in per_minute.index:
        for team in teams:
            rows.append(
                {
                    "match_id": match_id,
                    "minute": minute,
                    "team": team,
                    "threat_this_minute": float(per_minute.loc[minute, team]),
                    "momentum_5min": float(momentum_5.loc[minute, team]),
                    "momentum_10min": float(momentum_10.loc[minute, team]),
                }
            )
    return pd.DataFrame(rows)


def momentum_differential(momentum_df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    """Collapse to one row per minute: home team's momentum minus away
    team's -- the actual feature Step 5/6 will consume (positive = home
    team currently dominating possession-value)."""
    wide = momentum_df.pivot(index="minute", columns="team", values=["momentum_5min", "momentum_10min"])
    out = pd.DataFrame(index=wide.index)
    out["momentum_5min_diff"] = wide["momentum_5min"].get(home_team, 0.0) - wide["momentum_5min"].get(away_team, 0.0)
    out["momentum_10min_diff"] = wide["momentum_10min"].get(home_team, 0.0) - wide["momentum_10min"].get(away_team, 0.0)
    return out.reset_index()
