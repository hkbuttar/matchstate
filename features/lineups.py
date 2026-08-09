"""
Parse StatsBomb lineup/formation, substitution, and card events for a
match, and derive each player's actual minutes played -- needed both for
the substitution-quality feature (Step 5) and for normalizing player
contributions to a per-90 basis (features/player_quality.py).

Both "Red Card" and "Second Yellow" dismissals end a player's match and
are treated identically as a red-card event -- the team is down to 10 men
either way, which is what the in-game features care about.
"""

import json
from pathlib import Path

from possession_value.data import RAW_DIR

RED_CARD_TYPES = {"Red Card", "Second Yellow"}


def parse_match(match_id: int) -> dict:
    events = json.load(open(RAW_DIR / "events" / f"{match_id}.json"))

    starting_players: dict[str, list[dict]] = {}
    formations: dict[str, int] = {}
    for e in events:
        if e["type"]["name"] == "Starting XI":
            team = e["team"]["name"]
            formations[team] = e["tactics"]["formation"]
            starting_players[team] = [
                {"player_id": p["player"]["id"], "player_name": p["player"]["name"], "position": p["position"]["name"]}
                for p in e["tactics"]["lineup"]
            ]

    substitutions = []
    for e in events:
        if e["type"]["name"] == "Substitution":
            substitutions.append(
                {
                    "team": e["team"]["name"],
                    "minute": e["minute"],
                    "second": e["second"],
                    "player_off_id": e["player"]["id"],
                    "player_off_name": e["player"]["name"],
                    "player_on_id": e["substitution"]["replacement"]["id"],
                    "player_on_name": e["substitution"]["replacement"]["name"],
                }
            )

    red_cards = []
    for e in events:
        card = None
        if e["type"]["name"] == "Foul Committed":
            card = e.get("foul_committed", {}).get("card")
        elif e["type"]["name"] == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card")
        if card and card["name"] in RED_CARD_TYPES:
            red_cards.append(
                {
                    "team": e["team"]["name"],
                    "minute": e["minute"],
                    "second": e["second"],
                    "player_id": e["player"]["id"],
                    "player_name": e["player"]["name"],
                    "card_type": card["name"],
                }
            )

    match_end_minute = max((e["minute"] for e in events), default=90)

    return {
        "starting_players": starting_players,
        "formations": formations,
        "substitutions": substitutions,
        "red_cards": red_cards,
        "match_end_minute": match_end_minute,
    }


def player_minutes(parsed: dict) -> dict[int, dict]:
    """{player_id: {team, name, minutes}} for every player who appeared."""
    sub_off_minute = {s["player_off_id"]: s["minute"] for s in parsed["substitutions"]}
    sub_on_minute = {s["player_on_id"]: s["minute"] for s in parsed["substitutions"]}
    red_card_minute: dict[int, int] = {}
    for r in parsed["red_cards"]:
        red_card_minute[r["player_id"]] = min(red_card_minute.get(r["player_id"], r["minute"]), r["minute"])

    end_of_match = parsed["match_end_minute"]
    minutes: dict[int, dict] = {}

    for team, players in parsed["starting_players"].items():
        for p in players:
            pid = p["player_id"]
            end = min(sub_off_minute.get(pid, end_of_match), red_card_minute.get(pid, end_of_match), end_of_match)
            minutes[pid] = {"team": team, "name": p["player_name"], "minutes": max(0, end)}

    for s in parsed["substitutions"]:
        pid = s["player_on_id"]
        start = s["minute"]
        end = min(sub_off_minute.get(pid, end_of_match), red_card_minute.get(pid, end_of_match), end_of_match)
        minutes[pid] = {"team": s["team"], "name": s["player_on_name"], "minutes": max(0, end - start)}

    return minutes
