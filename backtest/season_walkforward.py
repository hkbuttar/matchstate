"""
Cross-season walk-forward validation for Dixon-Coles: does last season's
fitted attack/defense strength actually predict next season, with zero
lookahead? This is the one model in the project that CAN be tested this
way -- it only needs goal-level results, available for all 33 seasons
(1993/94-2025/26), unlike the Bayesian/GBM/market comparisons which are
necessarily confined to the single StatsBomb season (see `data/README.md`'s
disclosed coverage constraint).

Promoted/relegated teams (~3 clubs swap every season) have no prior-
season parameters at all -- rather than guess, matches involving any
such team are excluded from the walk-forward evaluation and counted
separately. This is a real, disclosed scope reduction: the walk-forward
number is specifically "how well does last season's strength predict
this season's matches, among clubs that stayed in the league" -- not the
harder, and genuinely different, problem of predicting newly promoted
teams.

Also computes the SAME-SEASON (in-sample, the static Dixon-Coles fit's
original protocol) Brier score restricted to the identical matches, so the
walk-forward number isn't read in a vacuum -- it's compared directly
against "what if you got to fit on the season you're predicting," which
is the honest ceiling.
"""

import json
from pathlib import Path

import numpy as np

from baseline.data import load_results
from baseline.dixon_coles import tau
from scipy.stats import poisson

PARAMS_DIR = Path(__file__).parent.parent / "data" / "processed" / "dixon_coles"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _load_params(season: str) -> dict:
    return json.loads((PARAMS_DIR / f"{season.replace('/', '_')}.json").read_text())


def _match_probs_from_params(params: dict, home: str, away: str, max_goals: int = 10) -> dict | None:
    if home not in params["attack"] or away not in params["attack"]:
        return None
    lam = np.exp(params["attack"][home] + params["defense"][away] + params["home_adv"])
    mu = np.exp(params["attack"][away] + params["defense"][home])
    g = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    rho = params["rho"]
    for x in range(2):
        for y in range(2):
            matrix[x, y] *= tau(x, y, lam, mu, rho)
    matrix = np.clip(matrix, 0, None)
    matrix /= matrix.sum()
    return {
        "home_win": float(np.tril(matrix, -1).sum()),
        "draw": float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, 1).sum()),
    }


def build_walkforward_and_insample():
    results = load_results()
    seasons = sorted(s for s in results["Season"].unique() if (PARAMS_DIR / f"{s.replace('/', '_')}.json").exists())

    rows = []
    n_excluded = 0
    for i in range(1, len(seasons)):
        prev_season, curr_season = seasons[i - 1], seasons[i]
        prev_params = _load_params(prev_season)
        curr_params = _load_params(curr_season)  # in-sample reference, same season
        curr_matches = results[results["Season"] == curr_season]

        for m in curr_matches.itertuples():
            wf_probs = _match_probs_from_params(prev_params, m.HomeTeam, m.AwayTeam)
            if wf_probs is None:
                n_excluded += 1
                continue
            insample_probs = _match_probs_from_params(curr_params, m.HomeTeam, m.AwayTeam)
            actual = 0 if m.FTHG > m.FTAG else (1 if m.FTHG == m.FTAG else 2)
            rows.append(
                {
                    "season": curr_season,
                    "match_key": f"{curr_season}:{m.HomeTeam}-{m.AwayTeam}",
                    "wf_home": wf_probs["home_win"], "wf_draw": wf_probs["draw"], "wf_away": wf_probs["away_win"],
                    "is_home": insample_probs["home_win"], "is_draw": insample_probs["draw"], "is_away": insample_probs["away_win"],
                    "actual": actual,
                }
            )

    print(f"Walk-forward evaluation: {len(rows)} matches with both prior- and current-season parameters "
          f"({n_excluded} matches excluded -- involved a promoted/relegated team with no prior-season fit)")
    return rows


def main(n_boot: int = 2000):
    import numpy as np

    from backtest.block_bootstrap import block_bootstrap_brier, summarize_ci, summarize_diff

    rows = build_walkforward_and_insample()
    seasons_covered = sorted(set(r["season"] for r in rows))
    print(f"Covers seasons: {seasons_covered[0]} .. {seasons_covered[-1]} ({len(seasons_covered)} season-transitions)")

    actual = np.array([r["actual"] for r in rows])
    n = len(rows)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual] = 1.0

    wf_probs = np.array([[r["wf_home"], r["wf_draw"], r["wf_away"]] for r in rows])
    is_probs = np.array([[r["is_home"], r["is_draw"], r["is_away"]] for r in rows])
    sq_err = {
        "walk_forward": np.sum((wf_probs - onehot) ** 2, axis=1),
        "in_sample": np.sum((is_probs - onehot) ** 2, axis=1),
    }

    # use match_key (unique per match) as the bootstrap block -- each row
    # here already IS one match (pre-match probabilities only, no
    # per-minute rows), so this block bootstrap is over independent
    # matches directly, no within-match correlation to worry about here.
    match_ids = np.array([r["match_key"] for r in rows])

    print(f"\nBlock bootstrap ({n_boot} draws, {len(np.unique(match_ids))} matches) ...")
    boot = block_bootstrap_brier(match_ids, sq_err, n_boot=n_boot)

    print("\nBrier score, 95% CI:")
    cis = {}
    for name, vals in sq_err.items():
        ci = summarize_ci(boot[name], vals.mean())
        cis[name] = ci
        print(f"  {name:14s}: {ci['point']:.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    diff = summarize_diff(boot["walk_forward"], boot["in_sample"], sq_err["walk_forward"].mean(), sq_err["in_sample"].mean())
    sig = "SIGNIFICANT" if diff["significant_at_95"] else "not significant"
    print(f"\nwalk_forward - in_sample: {diff['point_diff']:+.4f}  [{diff['ci_lo']:+.4f}, {diff['ci_hi']:+.4f}]  ({sig})")
    print("(Positive means predicting purely from last season's strength is worse than fitting on the season itself --")
    print(" the real, quantified cost of having zero lookahead into the season being predicted.)")

    out = {
        "n_matches": n,
        "n_season_transitions": len(seasons_covered),
        "seasons_covered": seasons_covered,
        "per_model_ci": cis,
        "walkforward_vs_insample_diff": diff,
    }
    (PROCESSED_DIR / "season_walkforward.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {PROCESSED_DIR / 'season_walkforward.json'}")


if __name__ == "__main__":
    main()
