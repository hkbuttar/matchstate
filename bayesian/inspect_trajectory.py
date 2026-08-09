"""
Qualitative sanity check for the hierarchical model (Step 3): does a
team's estimated within-season strength trajectory actually move the way
we know, from real football history, that it should?

Leicester City's 2015/16 title-winning season is the clearest available
case: they were a relegation-battle team the year before, opened the
season as outsiders, and became unstoppable through the middle of the
season. A model that's correctly picking up within-season form should
show Leicester's attack/defense trajectory improving over the season.
"""

from baseline.data import load_results
from bayesian.model import HierarchicalDixonColes


def main():
    results = load_results()
    season_df = results[results["Season"] == "2015/16"].sort_values("Date").reset_index(drop=True)

    model = HierarchicalDixonColes(n_periods=8).fit(season_df, draws=800, tune=800, chains=4)
    traj = model.strength_trajectory()

    for team in ["Leicester", "Aston Villa", "Man City"]:
        team_traj = traj[traj["team"] == team].sort_values("period")
        print(f"\n{team} strength by period (0=start of season, 7=end):")
        for _, row in team_traj.iterrows():
            overall = row["attack"] - row["defense"]
            bar = "#" * max(0, int((overall + 0.5) * 20))
            print(f"  period {int(row['period'])}: attack={row['attack']:+.3f} defense={row['defense']:+.3f} "
                  f"overall={overall:+.3f} {bar}")


if __name__ == "__main__":
    main()
