"""
Fit a separate, fixed Dixon-Coles model per EPL season and save the
parameters as JSON. Static per-season strengths are the deliberate scope
of this baseline; bayesian/ extends this to strengths that evolve within
a season.
"""

import json
from pathlib import Path

from baseline.data import load_results
from baseline.dixon_coles import DixonColes

OUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "dixon_coles"


def fit_all_seasons(min_matches: int = 300) -> list[dict]:
    results = load_results()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seasons = sorted(results["Season"].unique())
    summaries = []

    for season in seasons:
        season_df = results[results["Season"] == season]
        if len(season_df) < min_matches:
            print(f"[skip] {season}: only {len(season_df)} matches (incomplete season)")
            continue

        model = DixonColes().fit(season_df)
        out = {
            "season": season,
            "n_matches": len(season_df),
            "home_adv": model.home_adv,
            "rho": model.rho,
            "log_likelihood": model.log_likelihood,
            "attack": model.attack,
            "defense": model.defense,
        }
        dest = OUT_DIR / f"{season.replace('/', '_')}.json"
        dest.write_text(json.dumps(out, indent=2))

        top_attack = sorted(model.attack.items(), key=lambda kv: -kv[1])[:3]
        print(
            f"[ok] {season}: {len(season_df)} matches | home_adv={model.home_adv:+.3f} "
            f"rho={model.rho:+.3f} | strongest attack: {[t for t, _ in top_attack]}"
        )
        summaries.append(out)

    return summaries


if __name__ == "__main__":
    fit_all_seasons()
