# Data Sources & Coverage Constraints

This file documents exactly what was downloaded, from where, and — most
importantly — the hard coverage limits of each source. These constraints
shape what every later step can and can't claim; they are not bugs to
"fix" later.

All raw data lives under `data/raw/` and is gitignored (large,
re-downloadable, and in StatsBomb's case licensed for non-redistribution —
see below). Re-run the scripts in this directory to repopulate it.

## 1. football-data.co.uk — results & closing odds

**Script:** `download_football_data.py`
**Coverage:** English Premier League ("E0"), seasons 1993/94 through 2025/26
(33 seasons, one CSV per season).

- Match results (`FTHG`/`FTAG`/`FTR`, half-time score, referee, shots,
  corners, cards) are present from 1993/94 onward, though the *column set*
  is sparser in early seasons (e.g. shots/corners/cards data start mid-90s
  to early-2000s, not 1993/94).
- **Betting odds are not available for the full history.** Verified
  directly against the downloaded files:
  - 1993/94 – 1999/2000: **no odds columns at all.**
  - 2000/01 onward: William Hill (`WHH/WHD/WHA`) odds appear.
  - Mid-2000s onward: Bet365 (`B365H/D/A`) and other bookmakers are added.
  - Recent seasons (2020s): dozens of odds columns incl. Pinnacle
    (`PSH/PSD/PSA`), Bet365, William Hill, and market-average/max columns.
  - **Practical implication for Step 9 (betting-market benchmark):** the
    market-comparison analysis is only meaningful from ~2000/01 onward, and
    the exact bookmaker(s) available differ by season. Pre-2000 seasons can
    be used for Dixon-Coles / Bayesian model fitting (Steps 2-3) but must be
    excluded from any market-benchmark comparison.
- Early-season CSVs also contain trailing blank rows (a formatting quirk
  of the source, not missing matches) — these are dropped during parsing,
  not treated as missing data.

## 2. StatsBomb open data — event-level data (passes, shots, lineups, subs)

**Script:** `download_statsbomb.py`
**License note:** StatsBomb's open data is free for non-commercial /
research use under their published open-data license, but it should not be
redistributed as a bare mirror — hence it's gitignored here, and the
download script is what should be re-run/shared instead of the data itself.

**Coverage — this is the single biggest constraint on this project.**
Verified directly against the `statsbomb/open-data` competitions index
(2026-08-09): despite StatsBomb covering many competitions (La Liga, both
World Cups, several women's leagues, etc.), their free open data for the
**men's English Premier League covers exactly two seasons**:

| Season | competition_id | season_id | Matches | Coverage |
|---|---|---|---|---|
| 2015/16 | 2 | 27 | 380 | Full season, all 20 clubs (the Leicester title season) |
| 2003/04 | 2 | 44 | 38 | **Arsenal matches only** ("Invincibles" release — every match involves Arsenal) |

**Practical implication for Steps 4-6:** the possession-value model,
lineup-aware features, and gradient-boosting win-probability model can only
be trained on event-level data from **one full, team-general season**
(2015/16, 380 matches). The 2003/04 set is retained for spot-checks /
Arsenal-specific sanity checks but is **excluded from general model
training** — including it would bias the model toward Arsenal-specific
patterns (personnel, tactics, referees who officiated Arsenal games) that
don't generalize.

This is a real ceiling, not a placeholder: 380 matches of full in-game
event trajectories is a modest training set for a granular, timestamped
win-probability model. It will be treated explicitly as a stated
limitation throughout (Steps 11 and 16), and cross-validation in Step 10
will need within-season splitting (e.g. by matchweek or by match) rather
than the cross-season walk-forward validation used for the Dixon-Coles /
Bayesian models, since there's only one season available.

Downloaded per match: `events/{match_id}.json` (every on-ball event with
pitch coordinates and timestamps) and `lineups/{match_id}.json` (starting
XI, formations, and every player who appeared).

## 3. Understat — match-level xG

**Script:** `download_understat.py`
**Coverage:** EPL seasons 2014/15 through 2025/26 (12 seasons), all fully
played.

**Method note:** Understat's league pages no longer embed their
`datesData`/`teamsData` JSON directly in page HTML (the commonly-documented
"regex out the `JSON.parse(...)` blob" scraping approach no longer works —
verified 2026-08-09). The page's own JS instead calls
`GET https://understat.com/getLeagueData/{league}/{season}`, which returns
the same data as JSON directly. The script calls that endpoint.

Each match record includes Understat's own pre-match forecast
(win/draw/loss probabilities) alongside match xG — useful as a second,
independent model-comparison point in Step 11, distinct from both
Dixon-Coles and the betting market.

## Summary table

| Source | Seasons | Granularity | Key constraint |
|---|---|---|---|
| football-data.co.uk | 1993/94–2025/26 (33) | Match result + odds | Odds only from ~2000/01; bookmaker set grows over time |
| StatsBomb | 2015/16 (full) + 2003/04 (Arsenal-only) | Event-level (every pass/shot/sub, timestamped) | Only 380 matches usable for general in-game modeling |
| Understat | 2014/15–2025/26 (12) | Match-level xG + forecast | Match-level only, no event/lineup detail |
