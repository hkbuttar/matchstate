"""Serves Step 2's per-season fitted Dixon-Coles parameters (all 33
EPL seasons) -- e.g. for a frontend view of home-advantage or team
strength trends over time."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/seasons", tags=["seasons"])
PARAMS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "dixon_coles"


@router.get("")
def list_seasons():
    seasons = []
    for path in sorted(PARAMS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        seasons.append({
            "season": data["season"],
            "n_matches": data["n_matches"],
            "home_adv": data["home_adv"],
            "rho": data["rho"],
        })
    return seasons


@router.get("/{season}")
def season_detail(season: str):
    path = PARAMS_DIR / f"{season.replace('/', '_')}.json"
    if not path.exists():
        raise HTTPException(404, f"No fitted parameters for season '{season}'")
    return json.loads(path.read_text())
