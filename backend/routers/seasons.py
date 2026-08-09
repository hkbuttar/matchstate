"""Serves baseline/'s per-season fitted Dixon-Coles parameters (all 33
EPL seasons) -- e.g. for a frontend view of home-advantage or team
strength trends over time -- plus bayesian/'s within-season hierarchical
Bayesian team-strength trajectory for 2015/16 (the one season with a
production Bayesian fit, from backend/state.py)."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.state import state

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


@router.get("/2015-16/bayesian-trajectory")
def bayesian_trajectory():
    """bayesian/'s within-season hierarchical Bayesian strength trajectory
    (partially-pooled random walk over 8 periods) for the one season
    with a production Bayesian fit. Long-format rows: team, period,
    attack, defense."""
    return state.bayesian_model.strength_trajectory().to_dict(orient="records")
