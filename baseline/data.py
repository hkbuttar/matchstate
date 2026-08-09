"""
Load and clean football-data.co.uk EPL results into a single, tidy
DataFrame. Raw season CSVs (see data/download_football_data.py) have
trailing blank rows in older seasons and a growing set of odds columns
over time (see data/README.md) -- this loader drops the blank rows and
keeps the core result columns needed for Dixon-Coles fitting.
"""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "football_data"

CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def season_label(code: str) -> str:
    """'2324' -> '2023/24', '9899' -> '1998/99'."""
    a, b = code[:2], code[2:]
    century = "20" if int(a) < 50 else "19"
    return f"{century}{a}/{b}"


def _read_ragged_csv(path: Path) -> pd.DataFrame:
    """
    Some football-data.co.uk season files have stray trailing commas on a
    subset of rows (extra empty fields beyond the header count -- verified
    always empty, never real data), which trips pandas' C parser. Truncate
    every row to the header's field count before handing it to pandas.
    """
    with open(path, encoding="latin1", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    n_cols = len(rows[0])
    fixed_rows = [r[:n_cols] for r in rows]
    buf = StringIO()
    csv.writer(buf).writerows(fixed_rows)
    buf.seek(0)
    return pd.read_csv(buf)


def load_results(min_season: str | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(RAW_DIR.glob("E0_*.csv")):
        code = path.stem.split("_")[1]
        df = _read_ragged_csv(path)
        missing = [c for c in CORE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing expected columns: {missing}")
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        df = df[CORE_COLS].copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df["Season"] = season_label(code)
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    if min_season:
        all_df = all_df[all_df["Season"] >= min_season]
    return all_df


if __name__ == "__main__":
    df = load_results()
    print(f"{len(df)} matches loaded, seasons {df['Season'].min()}-{df['Season'].max()}")
    print(df.head())
    print(df.tail())
