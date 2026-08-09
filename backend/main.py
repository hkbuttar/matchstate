"""
Step 13: FastAPI backend. Serves match state, live in-game win
probability trajectories, big-moment detection, and the held-out
evaluation results from Steps 3/6/8/9/10/11.

Models are fit once at startup (lifespan context) and kept in memory --
NUTS sampling for the Bayesian model takes a few seconds; refitting per
request would make every trajectory request slow for no benefit, since
nothing about the fitted parameters depends on the request.
"""

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo project scope; tighten before any real deployment
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(matches.router)
app.include_router(results.router)
app.include_router(seasons.router)


@app.get("/health")
def health():
    return {"status": "ok", "matches_loaded": int(state.features_df["match_id"].nunique()) if hasattr(state, "features_df") else 0}
