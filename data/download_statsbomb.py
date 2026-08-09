"""
Download StatsBomb open-data event/lineup/match files for English Premier
League matches.

IMPORTANT COVERAGE CONSTRAINT (verified 2026-08-09 against statsbomb/open-data):
StatsBomb's free open data covers men's EPL for only two seasons:
  - 2015/16 (competition_id=2, season_id=27): full season, 380 matches, all
    20 clubs. This is the only season usable for a general, team-agnostic
    in-game / possession-value model.
  - 2003/04 (competition_id=2, season_id=44): only 38 matches, and every one
    of them involves Arsenal (this is StatsBomb's "Invincibles" release, not
    a full-league release). Not usable for a team-general model without
    introducing severe team bias -- kept for spot-checks / Arsenal-specific
    analysis only, excluded from the possession-value / in-game training set.

Net effect: possession_value/, features/, and models/ (possession-value
model, lineup-aware features, gradient-boosting win-probability model)
have a hard ceiling of ~380 usable matches of event-level training data,
all from a single season. This is
disclosed explicitly in data/README.md and again in the main README's
Limitations section -- it is not a bug to fix later, it is a real
constraint of the free data source.
"""

import json
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parent / "raw" / "statsbomb"
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

EPL_COMPETITION_ID = 2
SEASONS = {
    27: "2015_16",  # full season, all clubs -- primary usable dataset
    44: "2003_04",  # Arsenal-only subset -- kept but excluded from training
}


def fetch_json(session: requests.Session, url: str):
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def save_json(obj, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj))


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "matchstate-research/0.1 (data acquisition script)"})

    for season_id, label in SEASONS.items():
        print(f"\n=== Season {label} (season_id={season_id}) ===")
        matches_url = f"{BASE}/matches/{EPL_COMPETITION_ID}/{season_id}.json"
        matches = fetch_json(session, matches_url)
        save_json(matches, RAW_DIR / "matches" / f"{EPL_COMPETITION_ID}_{season_id}.json")
        print(f"  matches index: {len(matches)} matches")

        events_dir = RAW_DIR / "events"
        lineups_dir = RAW_DIR / "lineups"
        events_dir.mkdir(parents=True, exist_ok=True)
        lineups_dir.mkdir(parents=True, exist_ok=True)

        n_ok, n_skip, n_err = 0, 0, 0
        for i, m in enumerate(matches, 1):
            match_id = m["match_id"]
            ev_dest = events_dir / f"{match_id}.json"
            lu_dest = lineups_dir / f"{match_id}.json"

            if ev_dest.exists() and lu_dest.exists():
                n_skip += 1
            else:
                try:
                    events = fetch_json(session, f"{BASE}/events/{match_id}.json")
                    save_json(events, ev_dest)
                    lineups = fetch_json(session, f"{BASE}/lineups/{match_id}.json")
                    save_json(lineups, lu_dest)
                    n_ok += 1
                except requests.HTTPError as e:
                    print(f"  [err] match {match_id}: {e}")
                    n_err += 1
                time.sleep(0.05)

            if i % 50 == 0 or i == len(matches):
                print(f"  progress: {i}/{len(matches)} (ok={n_ok} skip={n_skip} err={n_err})")

        print(f"  done: {n_ok} downloaded, {n_skip} already present, {n_err} errors")


if __name__ == "__main__":
    sys.exit(main())
