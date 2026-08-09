"""
FastAPI backend. Serves match state, live in-game win
probability trajectories, big-moment detection, and the held-out
evaluation results from bayesian/, models/, calibration/, market/, and backtest/.

Models are fit once at startup (lifespan context) and kept in memory --
NUTS sampling for the Bayesian model takes a few seconds; refitting per
request would make every trajectory request slow for no benefit, since
nothing about the fitted parameters depends on the request.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import matches, results, seasons
from backend.state import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load()
    yield


app = FastAPI(
    title="MatchState API",
    description="Live win probability model for EPL soccer matches (2015/16 season).",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS_ORIGINS: comma-separated list, e.g. "https://matchstate.vercel.app".
# Defaults to "*" for local development; every endpoint here is read-only
# public sports data (no auth, no user data), so a wildcard is a defensible
# default even so -- but production deployments should still set this to
# the actual frontend origin, both as least-privilege practice and because
# a wildcard origin cannot be combined with credentialed requests.
_origins_env = os.environ.get("CORS_ORIGINS", "*")
_allow_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(matches.router)
app.include_router(results.router)
app.include_router(seasons.router)


@app.get("/health")
def health():
    return {"status": "ok", "matches_loaded": int(state.features_df["match_id"].nunique()) if hasattr(state, "features_df") else 0}
