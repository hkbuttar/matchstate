"""
Serves the already-computed, held-out-evaluated results from Steps 3,
6, 8, 9, 10, 11 as-is -- these come from the 285/95 train/test split
used for honest evaluation, distinct from the full-season production
models backend/state.py fits for the trajectory endpoints (see that
module's docstring). No computation happens here, just reading the JSON
artifacts each step already produced.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/results", tags=["results"])
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

FILES = {
    "bayesian-seasons": "bayesian_vs_static.json",           # Step 3: all-33-season comparison
    "gbm-comparison": "step6_comparison.json",                # Step 6: static/bayesian/gbm brier+logloss
    "calibration": "calibration/calibration_results.json",    # Step 8: per-class calibration
    "market-comparison": "step9_market_comparison.json",      # Step 9: market benchmark
    "ingame-bootstrap": "step10_ingame_bootstrap.json",        # Step 10: bootstrap CIs, 95-match test
    "season-walkforward": "step10_season_walkforward.json",    # Step 10: cross-season walk-forward
    "final-comparison": "step11_final_comparison.json",        # Step 11: per-class CI table
}


@router.get("/{name}")
def get_result(name: str):
    if name not in FILES:
        raise HTTPException(404, f"Unknown result set '{name}'. Available: {list(FILES.keys())}")
    path = PROCESSED_DIR / FILES[name]
    if not path.exists():
        raise HTTPException(404, f"{path.name} has not been generated yet -- run the corresponding step's script.")
    return json.loads(path.read_text())


@router.get("")
def list_results():
    return {name: (PROCESSED_DIR / fname).exists() for name, fname in FILES.items()}
