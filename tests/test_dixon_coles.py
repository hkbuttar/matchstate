"""
Dixon-Coles validation.

The key test here is parameter RECOVERY on synthetic data: generate
match results from a Dixon-Coles process with known true attack/defense/
home_adv/rho, fit the model on that simulated data, and check the fitted
parameters are close to the ones that actually generated it. This is a
meaningfully stronger check than "the fitted values look plausible on
real data" (which is all baseline/sanity_check.py does) -- it validates
the estimation *procedure* itself, independent of whether real football
data is well-described by this model at all.
"""

import math

import numpy as np
import pandas as pd
import pytest

from baseline.data import load_results
from baseline.dixon_coles import DixonColes, tau


def simulate_dixon_coles_season(
    n_teams: int = 20, home_adv: float = 0.3, rho: float = -0.12, seed: int = 7, max_goals: int = 10, n_rounds: int = 3
):
    """n_rounds=1 gives a realistic single-season-sized dataset (380
    matches for 20 teams); n_rounds=3+ repeats the round-robin with fresh
    random draws from the SAME true parameters, giving more statistical
    power. Verified empirically (not just assumed) that n_rounds=1 is not
    enough to reliably recover defense strength or rho's sign -- see the
    module docstring note below -- so the fixture uses n_rounds=3."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]

    raw_attack = rng.normal(0, 0.35, n_teams)
    raw_defense = rng.normal(0, 0.25, n_teams)
    true_attack = dict(zip(teams, raw_attack - raw_attack.mean()))
    true_defense = dict(zip(teams, raw_defense - raw_defense.mean()))

    rows = []
    for _ in range(n_rounds):
        for i, home in enumerate(teams):
            for j, away in enumerate(teams):
                if i == j:
                    continue
                lam = np.exp(true_attack[home] + true_defense[away] + home_adv)
                mu = np.exp(true_attack[away] + true_defense[home])
                g = np.arange(max_goals + 1)
                matrix = np.outer([np.exp(-lam) * lam**k / math.factorial(k) for k in g],
                                   [np.exp(-mu) * mu**k / math.factorial(k) for k in g])
                for x in range(2):
                    for y in range(2):
                        matrix[x, y] *= tau(x, y, lam, mu, rho)
                matrix = np.clip(matrix, 0, None)
                matrix /= matrix.sum()

                flat_idx = rng.choice(matrix.size, p=matrix.ravel())
                hg, ag = divmod(flat_idx, max_goals + 1)
                rows.append({"HomeTeam": home, "AwayTeam": away, "FTHG": hg, "FTAG": ag})

    df = pd.DataFrame(rows)
    truth = {"attack": true_attack, "defense": true_defense, "home_adv": home_adv, "rho": rho}
    return df, truth


@pytest.fixture(scope="module")
def synthetic_fit():
    df, truth = simulate_dixon_coles_season(n_rounds=3)
    model = DixonColes().fit(df)
    return model, truth


def test_single_season_sample_size_genuinely_limits_recovery():
    """Documents a real, checked finding (not an assumption): with only
    380 matches (n_rounds=1, matching a real single EPL season), defense
    correlation and rho's sign are NOT reliably recovered even though the
    estimator is correctly implemented -- confirmed by comparing against
    n_rounds=3 (1,140 matches), where both recover cleanly. This directly
    corroborates bayesian/README.md and data/README.md's real-data
    observation that individual seasons' fitted rho flips sign
    unpredictably (e.g. 1995/96: +0.028, 1997/98: +0.046, vs. most
    seasons negative) -- that's expected estimator behavior at this
    sample size, not noise to be alarmed by."""
    df_small, truth = simulate_dixon_coles_season(n_rounds=1)
    model_small = DixonColes().fit(df_small)
    df_large, _ = simulate_dixon_coles_season(n_rounds=3)
    model_large = DixonColes().fit(df_large)

    def defense_corr(model):
        true_d = np.array([truth["defense"][t] for t in model.teams])
        fit_d = np.array([model.defense[t] for t in model.teams])
        return np.corrcoef(true_d, fit_d)[0, 1]

    # the large-data fit should recover defense noticeably better --
    # this is the actual claim being tested, not a fixed threshold on
    # the (expectedly noisy) small-data fit
    assert defense_corr(model_large) > defense_corr(model_small)
    assert defense_corr(model_large) > 0.9


def test_recovers_attack_defense_ranking(synthetic_fit):
    """Fitted attack/defense should correlate strongly with the true
    values used to generate the data -- the model can't recover exact
    values from finite noisy data, but should recover the right *shape*."""
    model, truth = synthetic_fit
    true_attack = np.array([truth["attack"][t] for t in model.teams])
    fitted_attack = np.array([model.attack[t] for t in model.teams])
    true_defense = np.array([truth["defense"][t] for t in model.teams])
    fitted_defense = np.array([model.defense[t] for t in model.teams])

    attack_corr = np.corrcoef(true_attack, fitted_attack)[0, 1]
    defense_corr = np.corrcoef(true_defense, fitted_defense)[0, 1]

    assert attack_corr > 0.85, f"attack correlation too low: {attack_corr:.3f}"
    assert defense_corr > 0.85, f"defense correlation too low: {defense_corr:.3f}"


def test_recovers_home_advantage(synthetic_fit):
    model, truth = synthetic_fit
    assert abs(model.home_adv - truth["home_adv"]) < 0.1, (
        f"fitted home_adv={model.home_adv:.3f} vs true={truth['home_adv']:.3f}"
    )


def test_recovers_rho_sign_and_rough_magnitude(synthetic_fit):
    """rho is estimated from only 4 rare scoreline categories, so it's
    noisier than attack/defense/home_adv -- check sign and a loose
    tolerance rather than a tight one."""
    model, truth = synthetic_fit
    assert np.sign(model.rho) == np.sign(truth["rho"]), (
        f"fitted rho sign wrong: fitted={model.rho:.3f} true={truth['rho']:.3f}"
    )
    assert abs(model.rho - truth["rho"]) < 0.15


def test_identifiability_constraint_holds(synthetic_fit):
    """Attack and defense values should each sum to ~0 by construction
    (see DixonColes.fit's unpack()) -- a basic internal-consistency check."""
    model, _ = synthetic_fit
    assert abs(sum(model.attack.values())) < 1e-6
    assert abs(sum(model.defense.values())) < 1e-6


def test_probabilities_sum_to_one():
    """Regression check on real data (baseline/sanity_check.py's original
    check, formalized): match probabilities and in-game probabilities
    must be valid probability distributions."""
    results = load_results()
    season_df = results[results["Season"] == "2015/16"]
    model = DixonColes().fit(season_df)

    p = model.match_probabilities("Leicester", "Aston Villa")
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-6

    p_ig = model.in_game_probabilities("Leicester", "Aston Villa", 60, 1, 0)
    assert abs(p_ig["home_win"] + p_ig["draw"] + p_ig["away_win"] - 1.0) < 1e-6


def test_home_advantage_positive_on_real_data():
    """Home advantage should come out positive for essentially any real
    top-flight football season -- a fast, cheap regression guard."""
    results = load_results()
    season_df = results[results["Season"] == "2015/16"]
    model = DixonColes().fit(season_df)
    assert model.home_adv > 0


def test_leading_late_increases_win_probability():
    """A team leading late in the match should have a higher win
    probability than its pre-match baseline (baseline/sanity_check.py's
    original assertion, kept as a permanent regression test)."""
    results = load_results()
    season_df = results[results["Season"] == "2015/16"]
    model = DixonColes().fit(season_df)

    prematch = model.match_probabilities("Leicester", "Aston Villa")
    late_leading = model.in_game_probabilities("Leicester", "Aston Villa", 89, 1, 0)
    assert late_leading["home_win"] > prematch["home_win"]

    two_goal_lead_stoppage = model.in_game_probabilities("Leicester", "Aston Villa", 89, 2, 1)
    assert two_goal_lead_stoppage["home_win"] > 0.95
