"""
Run the static-vs-dynamic comparison (see bayesian/evaluate.py)
across every complete EPL season, and report the aggregate, honest
verdict -- not just a hand-picked example season.

Also captures basic convergence diagnostics (max r_hat, min ESS, total
divergences) per season so a good/bad Brier-score result can be trusted
(a "win" from a poorly-converged fit isn't a real win).
"""

import json
from pathlib import Path

import arviz as az
import numpy as np

from baseline.data import load_results
from bayesian.evaluate import evaluate_season

OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "bayesian_vs_static.json"


def convergence_summary(trace) -> dict:
    # round_to=None keeps full float precision -- az.summary()'s default
    # rounds r_hat/ess to 2 sig figs, which falsely flags a lot of
    # perfectly fine fits sitting right at the 1.01 boundary.
    summary = az.summary(trace, var_names=["attack", "defense", "home_adv",
                                            "sigma_attack_init", "sigma_defense_init",
                                            "sigma_walk_attack", "sigma_walk_defense"],
                          round_to="none")
    n_divergent = int(trace.sample_stats["diverging"].sum())
    return {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergent": n_divergent,
    }


def main():
    results = load_results()
    season_counts = results.groupby("Season").size()
    seasons = sorted(s for s, n in season_counts.items() if n >= 380)  # skip any incomplete season

    all_results = []
    for season in seasons:
        r = evaluate_season(season, n_periods=8, draws=800, tune=800, chains=4)
        conv = convergence_summary(r["dyn_model"].trace)
        r_clean = {k: v for k, v in r.items() if k != "dyn_model"}
        r_clean["convergence"] = conv
        flag = "" if conv["max_rhat"] < 1.01 and conv["n_divergent"] == 0 else "  [CONVERGENCE WARNING]"
        print(f"  convergence: max_rhat={conv['max_rhat']:.4f} min_ess={conv['min_ess_bulk']:.0f} "
              f"divergences={conv['n_divergent']}{flag}")
        all_results.append(r_clean)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(all_results, indent=2))

    briers_static = np.array([r["static_brier"] for r in all_results])
    briers_dyn = np.array([r["dynamic_brier"] for r in all_results])
    wins_dyn = int((briers_dyn < briers_static).sum())
    n = len(all_results)
    mean_delta = float((briers_dyn - briers_static).mean())  # negative = dynamic better on average

    print("\n" + "=" * 70)
    print(f"SUMMARY across {n} seasons")
    print(f"Dynamic Bayesian had lower (better) Brier score in {wins_dyn}/{n} seasons")
    print(f"Mean Brier score, static   : {briers_static.mean():.4f}")
    print(f"Mean Brier score, dynamic  : {briers_dyn.mean():.4f}")
    print(f"Mean delta (dynamic-static): {mean_delta:+.4f}  ({'dynamic better on average' if mean_delta < 0 else 'static better on average'})")
    n_warn = sum(1 for r in all_results if r["convergence"]["max_rhat"] >= 1.01 or r["convergence"]["n_divergent"] > 0)
    print(f"Seasons with convergence warnings: {n_warn}/{n}")


if __name__ == "__main__":
    main()
