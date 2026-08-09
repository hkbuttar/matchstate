# Deployment

Backend on Render (needs a persistent process -- the models are fit once at
startup and kept in memory, which a serverless/cold-per-request platform
would defeat), frontend on Vercel (Next.js's native platform). Deploy the
backend first: the frontend needs its URL, not the other way around.

## What's already true in this repo

- Pushed to GitHub (`origin` -> `github.com/hkbuttar/matchstate`), which
  both Render and Vercel deploy from.
- `render.yaml` (repo root) is a ready-to-apply Render Blueprint for the
  backend.
- The frontend already reads its backend URL from `NEXT_PUBLIC_API_URL`
  (`frontend/lib/api.ts`) rather than hardcoding it.
- The backend already reads its allowed CORS origin from `CORS_ORIGINS`
  (`backend/main.py`) rather than hardcoding a wildcard forever.

## 1. Backend -> Render

1. In the Render dashboard: **New +** -> **Blueprint** -> select this repo.
   Render detects `render.yaml` automatically and proposes the
   `matchstate-api` web service from it.
2. Deploy. First boot takes ~15-20s past "container started" before
   `/health` responds -- that's the three models fitting (NUTS sampling
   included), not a hang. See `render.yaml`'s comments for the free-tier
   cold-start and memory caveats.
3. Once live, copy its URL (`https://matchstate-api-XXXX.onrender.com`
   or similar) -- the frontend needs it next.
4. Verify: `curl https://<render-url>/health` should return
   `{"status":"ok","matches_loaded":380}`.

## 2. Frontend -> Vercel

Deploy from the `frontend/` subdirectory, not the repo root -- this repo
isn't a Next.js project at its root, so Vercel's project settings need to
know that explicitly.

**Via the Vercel dashboard:**
1. **Add New** -> **Project** -> import `hkbuttar/matchstate`.
2. Under **Root Directory**, set it to `frontend` (Vercel won't detect
   Next.js correctly otherwise -- it'll see a repo root with no
   `package.json`).
3. Add an environment variable: `NEXT_PUBLIC_API_URL` = the Render URL
   from step 1.
4. Deploy.

**Via the CLI** (already authenticated locally as `hkbuttar`):
```
cd frontend
vercel link          # first time only -- creates/links the Vercel project
vercel env add NEXT_PUBLIC_API_URL production   # paste the Render URL when prompted
vercel --prod
```

## 3. Tighten CORS (optional but recommended)

Once the Vercel URL is known, go back to the Render service's environment
variables and set `CORS_ORIGINS` to that exact URL (comma-separated if
there's more than one, e.g. a preview + production domain) instead of the
default wildcard. Redeploy the backend for it to take effect. Every
endpoint here is read-only public sports data, so the wildcard default
isn't a real security issue -- this is least-privilege hygiene, not a fix
for a vulnerability.

## Two real bugs found during actual deployment attempts, and fixed

Preparing this file and verifying things locally wasn't enough to catch
either of these -- both only showed up once a real deploy was attempted:

1. **`pip._vendor.resolvelib.resolvers.ResolutionTooDeep`** during the
   Render build. `requirements.txt` used loose `>=` version ranges
   throughout; combined with `pymc`'s large dependency tree (pytensor ->
   numba -> llvmlite), pip's resolver couldn't find a satisfying
   combination in reasonable time. Fixed by pinning every top-level
   dependency to an exact, already-verified-working version (checked
   first that none of the pinned packages are platform-specific --
   `appnope`, a transitive macOS-only dependency, confirmed a *full*
   transitive freeze would have been risky on Render's Linux build, so
   only the top-level packages in `requirements.txt` are pinned).
2. **`ValueError: No objects to concatenate`** at backend startup. All of
   `data/raw/` and `data/processed/` were gitignored, so Render's fresh
   clone had zero data to load -- `load_results()` found no CSV files at
   all. Fixed by un-gitignoring everything except StatsBomb's raw
   event/lineup files (~929MB, and not needed at runtime -- see
   `features/match_events.py`, added specifically to precompute the small
   per-match summary the backend needs instead of parsing raw StatsBomb
   events live per request). See `data/README.md` for the full committed/
   gitignored breakdown.

Both fixes were verified locally (fresh venv install + full test suite
for the first; a locally-run backend hitting the same endpoints for the
second) before being reported as fixed.
