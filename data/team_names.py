"""
StatsBomb and football-data.co.uk use different naming conventions for
the same clubs (e.g. "Manchester United" vs "Man United"). This maps
StatsBomb's names -> football-data.co.uk's names so match records from
the two sources can be joined. Verified against the full 20-club list for
the 2015/16 season; recheck when extending to other seasons; if a name
appears that isn't in this dict, the code should raise loudly, not
silently drop the match.
"""

STATSBOMB_TO_FOOTBALL_DATA = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Southampton": "Southampton",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
}

FOOTBALL_DATA_TO_STATSBOMB = {v: k for k, v in STATSBOMB_TO_FOOTBALL_DATA.items()}


def to_football_data(statsbomb_name: str) -> str:
    if statsbomb_name not in STATSBOMB_TO_FOOTBALL_DATA:
        raise KeyError(f"No football-data.co.uk mapping for StatsBomb team name {statsbomb_name!r}")
    return STATSBOMB_TO_FOOTBALL_DATA[statsbomb_name]


def to_statsbomb(football_data_name: str) -> str:
    if football_data_name not in FOOTBALL_DATA_TO_STATSBOMB:
        raise KeyError(f"No StatsBomb mapping for football-data.co.uk team name {football_data_name!r}")
    return FOOTBALL_DATA_TO_STATSBOMB[football_data_name]
