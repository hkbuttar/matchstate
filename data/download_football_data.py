"""
Download historical EPL results + closing odds from football-data.co.uk.

football-data.co.uk publishes one CSV per league per season at a
predictable URL: https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
where `div` = "E0" for the English Premier League and `season` is the two
two-digit year pair, e.g. "2324" for the 2023-24 season.

Column coverage varies by era:
  - Full match results (FTHG/FTAG/FTR etc.) go back to the 1993-94 season.
  - Betting odds columns (B365H/D/A, and others) are only populated
    consistently from around the 2000-01 season onward, and the *set* of
    bookmakers included grows over time (fewer bookmakers in early seasons).
This matters for Step 9 (betting-market benchmark): earlier seasons may
have partial or missing odds and should not be silently treated as
"market says X" when the market column is actually NaN.
"""

import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "raw" / "football_data"
DIV = "E0"  # English Premier League
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# Seasons from 1993-94 through 2025-26 (in-progress), coded as football-data
# does: "9394" for 1993-94, ..., "9900" for 1999-2000, "0001" for 2000-01, ...
def season_codes(start_year: int, end_year: int) -> list[str]:
    codes = []
    for y in range(start_year, end_year):
        a, b = y % 100, (y + 1) % 100
        codes.append(f"{a:02d}{b:02d}")
    return codes


def download_season(session: requests.Session, season: str) -> Path | None:
    url = BASE_URL.format(season=season, div=DIV)
    dest = RAW_DIR / f"E0_{season}.csv"
    if dest.exists():
        print(f"  [skip] {season} already downloaded")
        return dest

    resp = session.get(url, timeout=30)
    if resp.status_code != 200 or not resp.content:
        print(f"  [miss] {season}: HTTP {resp.status_code}")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"  [ok]   {season}: {len(resp.content):,} bytes -> {dest.name}")
    return dest


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    codes = season_codes(1993, 2026)
    session = requests.Session()
    session.headers.update({"User-Agent": "matchstate-research/0.1 (data acquisition script)"})

    print(f"Downloading {len(codes)} EPL seasons from football-data.co.uk ...")
    ok, miss = 0, 0
    for season in codes:
        result = download_season(session, season)
        if result is not None:
            ok += 1
        else:
            miss += 1
        time.sleep(0.3)  # be polite to a free, unauthenticated public host

    print(f"\nDone. {ok} seasons downloaded/present, {miss} missing.")
    if miss:
        print("Missing seasons are expected at the ends of the range "
              "(current season incomplete, or code before site coverage starts).")


if __name__ == "__main__":
    sys.exit(main())
