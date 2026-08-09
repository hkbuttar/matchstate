# Backend (FastAPI)

## Two separate code paths, deliberately kept apart

- **`backend/state.py`** fits "production" static Dixon-Coles, hierarchical
  Bayesian, and gradient boosting models on the **full 380-match** 2015/16
  season at startup (kept in memory -- NUTS sampling takes ~6s, done once,
  not per request). These power the live trajectory/big-moments endpoints
  for any of the 380 matches.
- **`backend/routers/results.py`** serves the already-computed JSON
  artifacts from `bayesian/`, `models/`, `calibration/`, `market/`, and
  `backtest/` verbatim -- those came from the
  285/95 train/test split used specifically for honest, held-out
  evaluation. Mixing the two (e.g. serving "live" predictions as if they
  were the evaluated numbers) would misrepresent the evaluation, so
  they're intentionally separate: production models for the demo
  experience, frozen evaluation artifacts for the honest numbers.

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness + matches-loaded count |
| `GET /matches` | All 380 matches: teams, date, final score |
| `GET /matches/{id}` | Match detail + full event timeline (goals, red cards, subs) |
| `GET /matches/{id}/trajectory` | Per-minute win probability from all 3 production models |
| `GET /matches/{id}/big-moments` | Top win-probability swings, with event annotations |
| `GET /seasons` | `baseline/`'s fitted home-advantage/rho summary, all 33 seasons |
| `GET /seasons/{season}` | Full per-team attack/defense for one season |
| `GET /results` | Index of available evaluation result sets |
| `GET /results/{name}` | One of: `bayesian-seasons`, `gbm-comparison`, `calibration`, `market-comparison`, `ingame-bootstrap`, `season-walkforward`, `final-comparison` |

Interactive docs at `/docs` (FastAPI's auto-generated OpenAPI UI).

## Verified end-to-end

Ran the server locally and exercised every endpoint against real match
data. Worth noting: `/matches/3754217/trajectory` (Chelsea 2-0 Arsenal,
the match with two Arsenal red cards used throughout `features/` and
`models/`) **live-reproduces the big-moment detection headline finding
through the actual API** -- at minute 45, right after Gabriel's red card, `gbm_home_win` jumps to
0.82 while `static_home_win` barely moves (0.26 -> 0.26). Confirms the
API is correctly wired to real model behavior, not just returning
plausible-looking numbers.

Also hit a real, unrelated bug during testing: the first test run picked
port 8123, which turned out to already be in use by an unrelated
pre-existing local service -- curl got a response, but not from this
server (checked and caught before assuming success). Re-ran on port 8734
and verified the response actually came from this app.

## Running

```
uvicorn backend.main:app --reload
```
