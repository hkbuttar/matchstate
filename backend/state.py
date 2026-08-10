"""
Application state: lightweight models are fit once at API startup, the
precomputed Bayesian posterior means are loaded, and all are kept in memory,
plus lazily-loaded static data artifacts from earlier steps.

Distinction worth being explicit about: these are "production" models,
fit on the FULL 380-match 2015/16 season -- different from the 285/25
train/test split used throughout models/, calibration/, market/, and
backtest/ for honest evaluation. The comparison/calibration/market
endpoints serve those modules' already-computed,
rigorously held-out-evaluated JSON artifacts unchanged; the trajectory
endpoints use these full-season models to give the best available live
prediction for any of the 380 matches, not just the 95 held out for
evaluation. Mixing the two would misrepresent the evaluation results, so
they're kept in genuinely separate code paths, not just separate variables.

Loading happens in a background thread, NOT synchronously in FastAPI's
lifespan startup -- verified directly against uvicorn's own source
(Server.startup() awaits `lifespan.startup()` BEFORE it ever calls
`loop.create_server(...)`), so a slow synchronous lifespan means the
process genuinely never binds its port until loading finishes. On Render
this exceeded the platform's port-scan timeout: NUTS sampling that takes
~6s locally (warm pytensor compilation cache, linked BLAS) took long
enough on a cold, BLAS-degraded Render container that the deploy failed
with "no open ports detected" before our own model-fitting print
statements even appeared in the log -- confirmed from the actual failed
deploy's logs, not assumed. `ready`/`error` let `/health` and the
model-dependent routes report an honest "still starting" state instead of
crashing on a `None` attribute for however long loading takes.
"""

import json
import threading
from pathlib import Path

import pandas as pd

from baseline.data import load_results
from baseline.dixon_coles import DixonColes
from bayesian.frozen import FrozenBayesianStrength
from models.gbm import train_gbm
from possession_value.data import RAW_DIR

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MATCHES_FILE = RAW_DIR / "matches" / "2_27.json"
MATCH_EVENTS_FILE = PROCESSED_DIR / "match_events_2015_16.json"
BAYESIAN_FIT_FILE = PROCESSED_DIR / "bayesian_production_fit.json"


class AppState:
    static_model: DixonColes
    bayesian_model: FrozenBayesianStrength
    gbm_model: object
    features_df: pd.DataFrame
    match_meta: dict[int, dict]
    match_events: dict[str, dict]
    ready: bool = False
    error: str | None = None

    def load(self):
        print("Loading production models for the full 2015/16 season ...")
        results = load_results()
        season_results = results[results["Season"] == "2015/16"]

        self.static_model = DixonColes().fit(season_results)
        # The posterior means are deterministic deployment data, so load
        # the fit produced by bayesian/precompute_production_fit.py rather
        # than compiling PyTensor and running NUTS on every cold start.
        self.bayesian_model = FrozenBayesianStrength.from_json(BAYESIAN_FIT_FILE)

        self.features_df = pd.read_parquet(PROCESSED_DIR / "ingame_features_2015_16.parquet")
        # match-level split for the internal validation set, consistent
        # with models/gbm.py's convention elsewhere in the project
        match_order = self.features_df.drop_duplicates("match_id")["match_id"].tolist()
        fit_matches = set(match_order[: int(len(match_order) * 0.85)])
        fit_df = self.features_df[self.features_df["match_id"].isin(fit_matches)]
        val_df = self.features_df[~self.features_df["match_id"].isin(fit_matches)]
        self.gbm_model = train_gbm(fit_df, val_df)

        matches = json.load(open(MATCHES_FILE))
        self.match_meta = {m["match_id"]: m for m in matches}

        # precomputed per-match goals/red-cards/subs/formations -- avoids
        # needing StatsBomb's raw event files (~929MB across 418 matches)
        # at runtime; see features/match_events.py
        self.match_events = json.loads(MATCH_EVENTS_FILE.read_text())

        print(f"Ready: {self.features_df['match_id'].nunique()} matches available.")
        return self

    def load_in_background(self):
        def _run():
            try:
                self.load()
                self.ready = True
            except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here
                # must be visible on /health rather than silently leaving
                # the process half-initialized.
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"FATAL: background model loading failed: {self.error}")

        threading.Thread(target=_run, daemon=True, name="state-loader").start()


state = AppState()


def ensure_ready() -> None:
    """Call at the top of any route that touches state.static_model /
    bayesian_model / gbm_model / features_df / match_meta / match_events --
    those don't exist as attributes until load() completes, so a request
    arriving during the (now backgrounded) loading window would otherwise
    hit an AttributeError instead of a clean, honest 503."""
    from fastapi import HTTPException

    if state.error is not None:
        raise HTTPException(503, f"Backend failed to start: {state.error}")
    if not state.ready:
        raise HTTPException(503, "Backend is still starting up (fitting models) -- try again in a moment.")
