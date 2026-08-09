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

## Not yet done

Nothing has actually been deployed by me -- `render.yaml` and the env-var
wiring are prepared and verified locally, but creating the live Render
service and triggering `vercel --prod` are real, visible actions against
your accounts (they create public URLs and, on Render, consume your
account's usage), so they're left for you to trigger, or for me to trigger
on your explicit go-ahead.
