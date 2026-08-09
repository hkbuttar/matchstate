"""
Serves the already-computed, held-out-evaluated results from bayesian/,
models/, calibration/, market/, and backtest/ as-is -- these come from
the 285/95 train/test split used for honest evaluation, distinct from
the full-season production models backend/state.py fits for the
trajectory endpoints (see that module's docstring). No computation
happens here, just reading the JSON artifacts each module already
produced.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/results", tags=["results"])
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

FILES = {
    "bayesian-seasons": "bayesian_vs_static.json",           # bayesian/: all-33-season comparison
    "gbm-comparison": "gbm_vs_baselines.json",                # models/: static/bayesian/gbm brier+logloss
    "calibration": "calibration/calibration_results.json",    # calibration/: per-class calibration
    "market-comparison": "market_comparison.json",      # market/: market benchmark
    "ingame-bootstrap": "ingame_bootstrap.json",        # backtest/: bootstrap CIs, 95-match test
    "season-walkforward": "season_walkforward.json",    # backtest/: cross-season walk-forward
    "final-comparison": "final_comparison.json",        # backtest/: per-class CI table
}


@router.get("/{name}")
def get_result(name: str):
    if name not in FILES:
        raise HTTPException(404, f"Unknown result set '{name}'. Available: {list(FILES.keys())}")
    path = PROCESSED_DIR / FILES[name]
    if not path.exists():
        raise HTTPException(404, f"{path.name} has not been generated yet -- run the corresponding script.")
    return json.loads(path.read_text())


@router.get("")
def list_results():
    return {name: (PROCESSED_DIR / fname).exists() for name, fname in FILES.items()}
