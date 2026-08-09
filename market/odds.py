"""
Load and de-vig closing betting odds for 2015/16, to benchmark model
probabilities against the real market.

Uses Pinnacle's closing odds (PSCH/D/A in football-data.co.uk's columns)
rather than a recreational bookmaker like Bet365 -- Pinnacle is widely
regarded in the betting industry as the sharpest, lowest-margin market
(it accepts, rather than limits, winning bettors, which forces its lines
toward true probability), making it the most credible "how good is the
market really" benchmark available in this dataset. Coverage confirmed:
all 380 matches of 2015/16 have non-null PSCH/D/A.

De-vig method (disclosed judgment call): simple multiplicative
normalization -- convert each odds price to a raw implied probability
(1/odds), then divide by the sum of the three raw probabilities so they
sum to 1 exactly, removing the bookmaker's overround. This is the
standard, simplest de-vig method. More sophisticated approaches (e.g.
Shin's method, which models a specific insider-trading mechanism behind
the overround) exist and would be a reasonable extension, but simple
normalization is transparent and doesn't presume a particular model of
*why* the margin is shaped the way it is.
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "football_data" / "E0_1516.csv"


def load_market_probabilities() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "PSCH", "PSCD", "PSCA"]).copy()

    raw_home = 1.0 / df["PSCH"]
    raw_draw = 1.0 / df["PSCD"]
    raw_away = 1.0 / df["PSCA"]
    overround = raw_home + raw_draw + raw_away

    df["market_home_win"] = raw_home / overround
    df["market_draw"] = raw_draw / overround
    df["market_away_win"] = raw_away / overround
    df["overround_pct"] = (overround - 1.0) * 100

    return df[["HomeTeam", "AwayTeam", "market_home_win", "market_draw", "market_away_win", "overround_pct"]]


if __name__ == "__main__":
    odds = load_market_probabilities()
    print(f"{len(odds)} matches with Pinnacle closing odds")
    print(f"Mean overround: {odds['overround_pct'].mean():.2f}%")
    print(odds.head())
