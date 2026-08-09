"""
Lightweight, real-world sanity checks for the Dixon-Coles baseline
(formal unit tests against synthetic data belong in tests/, Step 12).
This script re-fits the well-known 2015/16 season (Leicester's title win)
and checks the model's outputs make intuitive sense.
"""

from baseline.data import load_results
from baseline.dixon_coles import DixonColes


def main():
    results = load_results()
    season_df = results[results["Season"] == "2015/16"]
    model = DixonColes().fit(season_df)

    print("=== 2015/16 strength ranking (attack - defense) ===")
    overall = {t: model.attack[t] - model.defense[t] for t in model.teams}
    for i, (team, score) in enumerate(sorted(overall.items(), key=lambda kv: -kv[1]), 1):
        print(f"{i:2d}. {team:20s} attack={model.attack[team]:+.3f} defense={model.defense[team]:+.3f}")

    print(f"\nhome_adv={model.home_adv:+.3f} (expect > 0), rho={model.rho:+.3f}")

    print("\n=== Pre-match probabilities ===")
    for home, away in [("Leicester", "Aston Villa"), ("Man City", "Leicester"), ("Sunderland", "Man City")]:
        p = model.match_probabilities(home, away)
        print(f"{home:12s} vs {away:12s} -> home_win={p['home_win']:.3f} draw={p['draw']:.3f} "
              f"away_win={p['away_win']:.3f} (lambda={p['lambda']:.2f}, mu={p['mu']:.2f})")

    print("\n=== In-game (score-conditional) probabilities: Leicester vs Aston Villa ===")
    print("(Villa finished bottom of the table that season, Leicester won the title)")
    scenarios = [
        (0, 0, 0, "kickoff, 0-0"),
        (45, 1, 0, "half-time, Leicester leading 1-0"),
        (80, 1, 0, "80', Leicester still leading 1-0"),
        (80, 0, 1, "80', Villa somehow leading 0-1"),
        (89, 2, 1, "89', Leicester leading 2-1"),
    ]
    for minute, hg, ag, label in scenarios:
        p = model.in_game_probabilities("Leicester", "Aston Villa", minute, hg, ag)
        print(f"  {label:35s} -> home_win={p['home_win']:.3f} draw={p['draw']:.3f} away_win={p['away_win']:.3f}")

    # Sanity assertions
    assert model.home_adv > 0, "home advantage should be positive"
    p_late_lead = model.in_game_probabilities("Leicester", "Aston Villa", 89, 2, 1)
    assert p_late_lead["home_win"] > 0.95, "a 2-goal lead with 1 minute left should be a near-certain win"
    p_kickoff = model.match_probabilities("Leicester", "Aston Villa")
    p_89 = model.in_game_probabilities("Leicester", "Aston Villa", 89, 1, 0)
    assert p_89["home_win"] > p_kickoff["home_win"], (
        "leading late in the match should raise win probability above the pre-match baseline"
    )
    print("\nAll sanity assertions passed.")


if __name__ == "__main__":
    main()
