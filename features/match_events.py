"""
Precomputes, for every match, the small summary the backend actually
needs to serve match detail/trajectory/big-moments endpoints: formations,
red cards, substitutions, and goals. Saved as one JSON artifact.

This exists specifically so the deployed backend never needs live access
to StatsBomb's raw event files (~929MB across 418 matches) -- those stay
local-only (re-downloadable via data/download_statsbomb.py), while this
~1MB summary ships with the deployment. Reuses the exact parsing logic
already in features/lineups.py and features/state.py -- this is a
caching/packaging step, not a new source of truth.
"""

import json
from pathlib import Path

from features.lineups import parse_match
from features.state import _goal_events
from possession_value.data import season_match_ids

OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "match_events_2015_16.json"


def build_match_events() -> dict:
    events_by_match = {}
    for match_id in season_match_ids():
        parsed = parse_match(match_id)
        events_by_match[str(match_id)] = {
            "formations": parsed["formations"],
            "red_cards": parsed["red_cards"],
            "substitutions": parsed["substitutions"],
            "match_end_minute": parsed["match_end_minute"],
            "goals": _goal_events(match_id),
        }
    return events_by_match


if __name__ == "__main__":
    data = build_match_events()
    OUT_PATH.write_text(json.dumps(data))
    print(f"{len(data)} matches -> {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB)")
