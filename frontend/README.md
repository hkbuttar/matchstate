# Step 14: Frontend

Next.js 16 (App Router, TypeScript, Tailwind), talking to the Step 13 FastAPI
backend. Framework choice ("React + Vite" vs. "Next.js" vs. "Svelte" vs.
plain HTML/JS) was made explicitly by the user rather than assumed, since
the plan called for a deliberate choice.

## Views

| Route | Shows |
|---|---|
| `/` | Overview: headline stats, links to the other views |
| `/matches` | All 380 matches, sortable table |
| `/matches/[id]` | Live per-minute win-probability trajectory (all 3 models) with goal/red-card annotations, plus the full event timeline |
| `/calibration` | Reliability diagrams per outcome class (raw vs. isotonic), reusing Step 8's actual generated PNGs |
| `/market` | Model-vs-market Brier score, pre-match and in-game, with Step 10's bootstrap confidence intervals and pairwise significance |
| `/seasons` | Home advantage across all 33 seasons + within-season Bayesian team-strength evolution for 2015/16 |

The `/seasons/2015-16/bayesian-trajectory` backend endpoint didn't exist
before this step -- added it (`backend/routers/seasons.py`) since the plan
specifically asks for "hierarchical team-strength evolution over a season,"
which needs the Bayesian model's within-season trajectory, not Step 2's
static per-season summary that was already exposed.

## Design: the `dataviz` skill, applied and validated against real bugs

Loaded the project's `dataviz` skill before writing any chart code and used
its reference palette unchanged (already validated, no need to re-run the
validator on an unmodified copy). Built two custom SVG chart components
(`LineChart`, `BarChart`) rather than a charting library, for exact control
over the skill's mark specs (2px lines, hairline gridlines, legend for 2+
series, crosshair + tooltip, emphasis mode for many-series contexts).

Two real problems were caught by actually rendering the charts (Playwright
screenshots), not just by writing code that looked right:

1. **Colliding end-labels.** The match-detail win-probability chart has all
   three models converge near 1.0 by full time -- their direct end-labels
   overlapped into illegible text, exactly the "converging series" failure
   mode the skill's anti-patterns document warns about. Fixed by adding a
   proper legend (always shown for 2+ series) and dropping direct end-labels
   entirely once a legend exists, rather than trying to nudge overlapping
   text apart.
2. **Clipped label.** After that fix, single-series charts (e.g. home
   advantage by season) still had their one direct label clipped by a fixed
   right margin. Fixed by sizing the right margin to the actual label length
   for the single-series case, per the skill's explicit "never clip, measure
   first" rule.

The 20-team strength-evolution chart uses the skill's **emphasis** pattern
(3 highlighted teams in categorical color, the other 17 in muted gray) --
per the skill's series-count ladder, 20 raw categorical lines would be
illegible and unsafe under any CVD check.

## A real Next.js 16 gotcha, worth recording

`create-next-app` ships a `CLAUDE.md`/`AGENTS.md` warning that this Next.js
version has breaking changes from older training data and should be checked
against its bundled docs before writing code -- correctly, as it turned out.
`params` is now a `Promise` (`await props.params`), and passing a function
prop (e.g. a custom tick-formatter) from a Server Component directly into a
`"use client"` component fails at runtime (React can't serialize a function
across that boundary) -- hit this on `/seasons`' two charts and fixed it by
moving the closures into a small client-only wrapper (`SeasonCharts.tsx`)
that receives only plain, serializable data as props.

## Verified end-to-end

Ran both the FastAPI backend and `next dev` together, fetched every route
with `curl` (checking response bodies for real data, not just status
codes), then used Playwright to screenshot all 6 pages in an actual
Chromium browser and visually inspect them -- which is what caught both
chart bugs above; a `curl`/`grep` pass alone would have missed them
entirely. `npm run build` (production build) and `npx tsc --noEmit` both
pass clean.

## Running

```
# terminal 1, from repo root
uvicorn backend.main:app --reload

# terminal 2
cd frontend && npm run dev
```

Set `NEXT_PUBLIC_API_URL` if the backend isn't on the default
`http://127.0.0.1:8000`.
