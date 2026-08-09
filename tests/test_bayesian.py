"""
Step 12: hierarchical Bayesian model validation -- convergence
diagnostics as hard assertions (not just eyeballed numbers, as in
bayesian/run_all_seasons.py's printed report), plus a genuine posterior
predictive check (PPC): does simulating matches from the fitted
posterior actually produce data that looks like the real observed
season, not just point-estimate parameters that look plausible?
"""

import numpy as np
import pytest

from baseline.data import load_results
from bayesian.model import HierarchicalDixonColes, assign_periods


@pytest.fixture(scope="module")
def fitted_2015_16():
    results = load_results()
    season_df = results[results["Season"] == "2015/16"].sort_values("Date").reset_index(drop=True)
    model = HierarchicalDixonColes(n_periods=8).fit(season_df, draws=800, tune=800, chains=4, random_seed=123)
    return model, season_df


def test_convergence_diagnostics(fitted_2015_16):
    """Formal version of the check bayesian/run_all_seasons.py already
    performs per-season -- here as hard assertions with round_to='none'
    (the bug caught in Step 3: az.summary's default rounds r_hat to 2dp,
    which falsely flags healthy fits sitting near the 1.01 boundary)."""
    import arviz as az

    model, _ = fitted_2015_16
    summary = az.summary(
        model.trace,
        var_names=["attack", "defense", "home_adv", "sigma_attack_init", "sigma_defense_init",
                   "sigma_walk_attack", "sigma_walk_defense"],
        round_to="none",
    )
    n_divergent = int(model.trace.sample_stats["diverging"].sum())
    total_draws = model.trace.posterior.sizes["chain"] * model.trace.posterior.sizes["draw"]

    assert summary["r_hat"].max() < 1.02, f"max r_hat too high: {summary['r_hat'].max():.4f}"
    assert summary["ess_bulk"].min() > 200, f"min ESS too low: {summary['ess_bulk'].min():.0f}"
    assert n_divergent / total_draws < 0.01, f"too many divergences: {n_divergent}/{total_draws}"


def test_posterior_predictive_check(fitted_2015_16):
    """Genuine PPC: simulate full seasons of goals from posterior draws
    (not just the posterior mean), and check the REAL season's summary
    statistics (mean total goals/match, draw rate) fall within the
    simulated distribution -- i.e. the fitted model, taken as a
    generative process, actually produces data resembling reality."""
    model, season_df = fitted_2015_16
    rng = np.random.default_rng(0)

    period_idx = assign_periods(len(season_df), model.n_periods)
    home_idx = np.array([model.team_idx[t] for t in season_df["HomeTeam"]])
    away_idx = np.array([model.team_idx[t] for t in season_df["AwayTeam"]])

    attack = model.trace.posterior["attack"]  # dims: chain, draw, attack_dim_0 (team), attack_dim_1 (period)
    defense = model.trace.posterior["defense"]
    home_adv = model.trace.posterior["home_adv"]

    n_chains, n_draws = attack.sizes["chain"], attack.sizes["draw"]
    n_sim = 200
    sim_mean_goals, sim_draw_rate = [], []
    for _ in range(n_sim):
        c, d = rng.integers(n_chains), rng.integers(n_draws)
        a = attack.isel(chain=c, draw=d).values  # (n_teams, n_periods)
        f = defense.isel(chain=c, draw=d).values
        h = float(home_adv.isel(chain=c, draw=d).values)

        lam = np.exp(a[home_idx, period_idx] + f[away_idx, period_idx] + h)
        mu = np.exp(a[away_idx, period_idx] + f[home_idx, period_idx])
        sim_hg = rng.poisson(lam)
        sim_ag = rng.poisson(mu)

        sim_mean_goals.append((sim_hg + sim_ag).mean())
        sim_draw_rate.append((sim_hg == sim_ag).mean())

    sim_mean_goals = np.array(sim_mean_goals)
    sim_draw_rate = np.array(sim_draw_rate)

    observed_mean_goals = (season_df["FTHG"] + season_df["FTAG"]).mean()
    observed_draw_rate = (season_df["FTHG"] == season_df["FTAG"]).mean()

    lo_g, hi_g = np.quantile(sim_mean_goals, [0.025, 0.975])
    lo_d, hi_d = np.quantile(sim_draw_rate, [0.025, 0.975])

    assert lo_g <= observed_mean_goals <= hi_g, (
        f"observed mean goals/match {observed_mean_goals:.3f} outside simulated 95% range [{lo_g:.3f}, {hi_g:.3f}]"
    )
    assert lo_d <= observed_draw_rate <= hi_d, (
        f"observed draw rate {observed_draw_rate:.3f} outside simulated 95% range [{lo_d:.3f}, {hi_d:.3f}]"
    )
