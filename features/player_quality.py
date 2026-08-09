"""
Season-long player "quality" proxy: total xT (possession-value) generated
per 90 minutes played, across all 2015/16 matches. This is what makes
substitutions in Step 5's feature table non-interchangeable -- swapping
off a high-xT90 player reads differently than swapping off a bench-level
one.

Disclosed limitation: this only credits ball-progression actions (passes
and carries) valued by the xT grid -- it does not capture defensive
quality, aerial duels, goalkeeping, or off-ball movement, so it's a
genuinely partial "quality" signal, biased toward attacking/creative
players. It's used only as a differencing feature (quality of player
coming on minus quality of player going off), which partially cancels
this bias when comparing broadly similar positions, but not when a
defensive player is swapped for an attacking one.

Players with under MIN_MINUTES total minutes get a shrinkage-adjusted
estimate pulled toward the league-average rate, rather than a noisy raw
per-90 figure from a handful of minutes.
"""

from pathlib import Path

import pandas as pd

from features.lineups import parse_match, player_minutes
from possession_value.data import season_match_ids
from possession_value.xt_model import ExpectedThreat

MIN_MINUTES = 180  # ~2 full matches' worth, before trusting the raw rate
SHRINKAGE_MINUTES = 180  # pseudo-minutes of league-average pulled into every estimate


def build_player_minutes_table(match_ids: list[int] | None = None) -> pd.DataFrame:
    if match_ids is None:
        match_ids = season_match_ids()
    rows = []
    for mid in match_ids:
        parsed = parse_match(mid)
        for pid, info in player_minutes(parsed).items():
            rows.append({"match_id": mid, "player_id": pid, "player_name": info["name"], "team": info["team"], "minutes": info["minutes"]})
    return pd.DataFrame(rows)


def build_player_quality(
    actions: pd.DataFrame, xt_model: ExpectedThreat, minutes_table: pd.DataFrame
) -> pd.DataFrame:
    moves = actions[actions["event_type"].isin(["Pass", "Carry"]) & actions["success"]].copy()
    moves["xt_value"] = xt_model.action_value(moves)

    # possession_value actions are keyed by player *name* (see possession_value/data.py);
    # aggregate minutes by name too so the two tables join cleanly.
    total_minutes = minutes_table.groupby("player_name")["minutes"].sum()
    total_xt = moves.groupby("player")["xt_value"].sum()

    league_total_minutes = total_minutes.sum()
    league_total_xt = total_xt.reindex(total_minutes.index).fillna(0.0).sum()
    league_rate_per90 = league_total_xt / league_total_minutes * 90

    df = pd.DataFrame({"minutes": total_minutes}).join(total_xt.rename("total_xt"), how="left")
    df["total_xt"] = df["total_xt"].fillna(0.0)
    df["raw_rate_per90"] = (df["total_xt"] / df["minutes"] * 90).where(df["minutes"] > 0, 0.0)

    # shrinkage toward league rate: blend raw rate with the league average,
    # weighted by how many minutes of real evidence we have
    df["quality_per90"] = (
        df["total_xt"] + SHRINKAGE_MINUTES * league_rate_per90 / 90
    ) / (df["minutes"] + SHRINKAGE_MINUTES) * 90

    df["reliable"] = df["minutes"] >= MIN_MINUTES
    df = df.reset_index().rename(columns={"index": "player_name"})
    return df.sort_values("quality_per90", ascending=False)


def main():
    actions = pd.read_parquet(Path(__file__).parent.parent / "data" / "processed" / "possession_actions_2015_16.parquet")
    xt_model = ExpectedThreat.load(Path(__file__).parent.parent / "data" / "processed" / "xt_grid_2015_16.npz")

    minutes_table = build_player_minutes_table()
    minutes_table.to_parquet(Path(__file__).parent.parent / "data" / "processed" / "player_minutes_2015_16.parquet")

    quality = build_player_quality(actions, xt_model, minutes_table)
    quality.to_parquet(Path(__file__).parent.parent / "data" / "processed" / "player_quality_2015_16.parquet")

    print("Top 15 by quality_per90 (min 900 minutes played):")
    print(quality[quality["minutes"] >= 900].head(15).to_string(index=False))
    print("\nBottom 10 by quality_per90 (min 900 minutes played):")
    print(quality[quality["minutes"] >= 900].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
