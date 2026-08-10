# MatchState — Live Win Probability Model for EPL Soccer

Live win probability model for English Premier League matches: a Dixon-Coles Poisson baseline, a hierarchical Bayesian team-strength extension, a possession-value (xT) momentum signal, and a gradient boosting model, all benchmarked directly against real betting-market odds with bootstrap-validated calibration. Real data, CPU-only, every judgment call disclosed.

**[View the live demo](https://matchstate.vercel.app/)**

## Motivation

Soccer's win-probability problem is harder than basketball's or football's: three outcomes instead of two, low scoring means a single event can swing the probability dramatically, and match state doesn't reduce to discrete plays the way it does in stop-start sports. This project builds a real, calibrated in-game model that addresses those challenges directly — and benchmarks it not just against a statistical baseline but against a real betting market, the sharpest, most honest test available in sports prediction.

The guiding rule throughout: report findings honestly, including the ones that don't flatter the more sophisticated model. Several of the most useful results here are exactly that kind.

## Repo structure

```
matchstate/
├── data/               data acquisition scripts + coverage documentation
├── baseline/           Dixon-Coles Poisson model
├── bayesian/           hierarchical Bayesian team-strength model
├── possession_value/   xT possession-value model
├── features/           in-game state, lineup/formation, momentum features
├── models/             gradient boosting win-probability model, big-moment detection
├── calibration/        probability calibration, reliability diagrams
├── market/             betting-market benchmark (de-vigged Pinnacle odds)
├── backtest/           walk-forward validation, block bootstrap
├── backend/            FastAPI service
├── frontend/           Next.js dashboard
├── notebooks/          pre-executed research notebook
├── tests/              synthetic-data validation, convergence diagnostics
├── RESULTS.md          the full comparison table and honest findings
├── DEPLOYMENT.md       Render (backend) + Vercel (frontend) deployment guide
└── requirements.txt
```

Every directory above has its own README with full technical detail. This file is the entry point and the synthesis; RESULTS.md is where the actual numbers live.

## Data

Three sources, each with real, disclosed limits — see `data/README.md` for the full detail:

- **football-data.co.uk**: match results and closing odds for all 33 EPL seasons (1993/94–2025/26). Odds coverage only starts around 2000/01, and the set of bookmakers included grows over the years — pre-2000 seasons are usable for goals-based model fitting but not for the betting-market benchmark.
- **StatsBomb open data**: event-level data (every pass, shot, tackle substitution, with pitch coordinates and timestamps) — but only for **one general-purpose EPL season, 2015/16** (380 matches, all 20 clubs; a second available set, 2003/04, covers only Arsenal's matches and is excluded from general-purpose model training to avoid team-identity bias). This is the single constraint that shapes the most interesting findings below: every event-level comparison in this project is capped at a 95-match held-out test set, and that sample size turns out to matter a great deal.
- **Understat**: match-level xG, 2014/15 onward, used as an independent  cross-check.

## Methodology

- **Dixon-Coles Poisson baseline** (`baseline/`): per-team attack/defense strength fit by maximum likelihood, with the Dixon-Coles low-score correction for the well-documented tendency of independent Poisson models to underestimate draws. Fit fresh per season. Extended with a simple, disclosed in-game update: goal rates for the remainder of a match are scaled by time remaining and conditioned on the current score — a constant scoring-rate assumption, chosen for transparency, that later models are measured against.
- **Hierarchical Bayesian team-strength model** (`bayesian/`, via `pymc`): extends the static per-season fit into a partially-pooled random walk across within-season periods — a team's strength early in the season leans on the league-wide distribution; as results accumulate, its own form dominates. Validated with real posterior predictive checks and formal convergence diagnostics, not just point estimates.
- **Possession-value momentum model** (`possession_value/`): an Expected Threat (xT) grid fit by value iteration on 655K parsed StatsBomb actions, crediting every completed pass and carry by how much it increased positional danger — not just shots. Aggregated into a rolling 5-/10-minute momentum signal per team.
- **In-game features** (`features/`): score, running xG differential, possession-value momentum, red cards, and substitutions weighted by each player's season-long attacking contribution (a lineup-aware signal, not just a substitution count).
- **Gradient boosting model** (`models/`): XGBoost trained on those in-game snapshots, deliberately excluding team identity and formation (the latter was tried, measured to leak team identity via near-fixed formation choices, and dropped) so it learns transferable state dynamics rather than memorizing which teams won that season.
- **Big-moment detection** (`models/`): flags the largest win-probability swings per match and uses them as a live sanity check — do the models actually react to goals and red cards the way they should?
- **Calibration** (`calibration/`): Platt scaling and isotonic regression, evaluated separately per outcome class rather than averaged together, since draws are the class most likely to need it.
- **Betting-market benchmark** (`market/`): de-vigged Pinnacle closing odds (chosen over a recreational bookmaker for its lower margin and reputation as the sharpest market), compared pre-match and in-game — the in-game comparison uses a market-prior-plus-naive-time-adjustment baseline so the question "does our real-time update add value" is isolated from "whose pre-match prior is better."
- **Walk-forward validation & block bootstrap** (`backtest/`): every headline comparison is wrapped in match-block bootstrap confidence intervals (blocking by match, not row, since per-minute rows within one match are highly correlated), plus a genuinely separate cross-season walk-forward test of Dixon-Coles across all 33 seasons.

## Results

Full comparison table and the four central questions this project set out to answer are in **[RESULTS.md](RESULTS.md)**. In short:

1. **Does gradient boosting beat the statistical baselines?** Not overall — but it wins clearly in the first 30 minutes of a match, when momentum and card state carry signal the Poisson-family models structurally can't see.
2. **Does it beat the betting market, pre-match or in-game?** Neither, at the confidence this data supports — but the gap shrinks substantially in-game, from a real pre-match deficit to a difference that isn't statistically distinguishable from noise.
3. **Is draw prediction meaningfully worse?** Yes, but specifically through discrimination — no model ever predicted a draw as the most likely pre-match outcome across 95 test matches, despite draws happening 30.5% of the time — not uniformly through calibration error.
4. **Does hierarchical Bayesian updating help over a static fit?** Probably, on average, across 33 seasons — but that directional evidence doesn't clear statistical significance on the smaller, rigorously bootstrapped single-season sample.

Every one of those "not proven at high confidence" results traces back to the same root cause: 380 matches of event-level data is not very much, and the contrast between what that sample size can and can't establish — versus what 8,808 matches of goals-only data can establish cleanly — is arguably this project's most important methodological finding, not a caveat to it.

## Limitations

- **StatsBomb's single-season coverage** caps every event-level model comparison (gradient boosting, in-game Bayesian-vs-static, in-game market-vs-models) at a 95-match held-out test set — not enough statistical power to cleanly separate models whose true performance differs by a few percent, as the bootstrap analysis in `backtest/` demonstrates directly.
- **xG and possession-value are themselves modeled estimates, not ground truth.** Understat's xG and this project's own xT grid are both fitted approximations of "how dangerous was this moment," not measured facts — errors in those models propagate into everything built on top of them.
- **Betting odds reflect market consensus, including public information the model doesn't use** (team news, injuries, weather, money flow) — the market's pre-match edge isn't evidence this project's models are poorly built, it's the expected result of comparing against every other public signal at once.
- **The player-quality proxy behind substitution scoring only credits ball progression**, not defending, aerial ability, or finishing — validated as directionally sound (Özil ranks first in a season he set the Premier League assist record) but genuinely partial.
- **Calibration correction itself needs enough data to be trustworthy** — it measurably worsened draw and away-win calibration here, traced to a 43-match calibration split containing only 9 draw outcomes to learn from.

## Future work

- **A live data feed** for genuine real-time predictions, rather than replaying historical event logs — the entire modeling pipeline here is already built around minute-by-minute state snapshots, so this is substantially a data-plumbing extension, not a modeling one.
- **Extension to other leagues**, both to test whether the findings above (draw discrimination, the early/late-match GBM split, the market gap narrowing in-game) generalize beyond the EPL, and to grow the event-level sample size past StatsBomb's single-season EPL constraint.
- **A multi-season Bayesian strength model with promotion/relegation continuity** — the current within-season random walk resets every season; a model that carries partial information across season boundaries (while still discounting it appropriately for squad turnover) would directly test the walk-forward finding that last season's strength, unadjusted, is a measurably worse predictor than a same-season fit.
- **An explicit ensemble** blending gradient boosting's demonstrated early-match advantage with the statistical baselines' demonstrated late-match reliability, rather than treating them as competitors — a natural next step the results directly motivate but that this project deliberately left as a straight comparison rather than a post-hoc-tuned "win."

## Running it

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

# backend (fits models at startup, ~15-20s)
.venv/bin/uvicorn backend.main:app --reload

# frontend, in a second terminal
cd frontend && npm install && npm run dev

# tests
.venv/bin/pytest tests/ -v
```

See `DEPLOYMENT.md` for deploying the backend to Render and the frontend to Vercel.
