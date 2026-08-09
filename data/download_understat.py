"""
Download match-level xG data for the EPL from Understat.

Understat no longer embeds its `datesData`/`teamsData`/`playersData` blobs
directly in the league page HTML (the old, commonly-documented scraping
trick of regex-extracting a `JSON.parse('...')` literal from the page no
longer works -- verified 2026-08-09, the page ships without that inline
script and fetches data client-side instead).

The page's own JS (js/league.min.js) calls a JSON endpoint:
    GET https://understat.com/getLeagueData/{league}/{season}
    -> {"teams": {...}, "players": {...}, "dates": [...]}
This script hits that endpoint directly and saves the "dates" array (one
entry per match, with Understat's match-level xG and pre-match forecast
probabilities) per season.

Understat's EPL coverage starts with the 2014/15 season (its earliest
tracked season for every league) through the current season.
"""

import json
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "raw" / "understat"
LEAGUE = "EPL"
SEASONS = list(range(2014, 2026))  # 2014 -> 2014/15 season, ... 2025 -> 2025/26

HEADERS = {
    "User-Agent": "Mozilla/5.0 (matchstate-research/0.1 data acquisition script)",
    "X-Requested-With": "XMLHttpRequest",
}


def download_season(session: requests.Session, season: int) -> Path | None:
    dest = RAW_DIR / f"understat_EPL_{season}.json"
    if dest.exists():
        print(f"  [skip] {season} already downloaded")
        return dest

    url = f"https://understat.com/getLeagueData/{LEAGUE}/{season}"
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"  [miss] {season}: HTTP {resp.status_code}")
        return None

    try:
        payload = resp.json()
    except json.JSONDecodeError:
        print(f"  [miss] {season}: non-JSON response (likely blocked/rate-limited)")
        return None

    dates = payload.get("dates", [])
    if not dates:
        print(f"  [miss] {season}: no match data returned")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(dates))
    n_played = sum(1 for d in dates if d.get("isResult"))
    print(f"  [ok]   {season}: {len(dates)} scheduled, {n_played} played -> {dest.name}")
    return dest


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    print(f"Downloading Understat EPL match xG for {len(SEASONS)} seasons ...")
    ok, miss = 0, 0
    for season in SEASONS:
        result = download_season(session, season)
        if result is not None:
            ok += 1
        else:
            miss += 1
        time.sleep(0.5)

    print(f"\nDone. {ok} seasons downloaded/present, {miss} missing.")


if __name__ == "__main__":
    sys.exit(main())
