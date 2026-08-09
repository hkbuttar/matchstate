"""
Builds the "naive pre-match-odds-plus-simple-score-adjustment" baseline
the plan specifically asks for: isolate whether our models' in-game
updates genuinely add value beyond just taking the market's pre-match
view and adjusting it for the current score/time, using the same
mechanism our own baselines use.

Method: numerically invert the market's de-vigged (home_win, draw,
away_win) probabilities to find the (lambda, mu) goal-rate pair that a
plain bivariate Poisson model (no Dixon-Coles low-score correction --
matching what we're solving for, not assuming market inefficiency in the
correlation structure) would need to produce those exact probabilities.
Then apply identical in-game time-scaling to baseline.dixon_coles: scale
remaining-match goal rates by time left, condition on the current score.
This isolates ONE variable -- whose PRE-MATCH prior is better, ours or
the market's -- while holding the in-game update mechanism fixed and
identical across both.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


def _score_matrix(lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))


def _match_probs(lam: float, mu: float, max_goals: int = 10) -> np.ndarray:
    m = _score_matrix(lam, mu, max_goals)
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def implied_lambda_mu(p_home: float, p_draw: float, p_away: float, max_goals: int = 10) -> tuple[float, float]:
    """Numerically invert market probabilities to a plain-Poisson (lambda, mu)."""
    target = np.array([p_home, p_draw, p_away])

    def loss(log_params):
        lam, mu = np.exp(log_params)
        return np.sum((_match_probs(lam, mu, max_goals) - target) ** 2)

    res = minimize(loss, np.log([1.3, 1.1]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10})
    lam, mu = np.exp(res.x)
    return float(lam), float(mu)


def market_in_game_probabilities(
    lam: float, mu: float, minute: float, home_goals: int, away_goals: int, match_length: float = 90.0, max_goals: int = 10
) -> dict:
    """Same time-scaling mechanism as baseline.dixon_coles.DixonColes.in_game_probabilities,
    applied to the market-implied pre-match (lambda, mu) instead of a fitted one."""
    remaining_frac = max(0.0, (match_length - minute) / match_length)
    lam_rem, mu_rem = lam * remaining_frac, mu * remaining_frac

    g = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(g, lam_rem), poisson.pmf(g, mu_rem))
    matrix /= matrix.sum()

    final_home = home_goals + g[:, None]
    final_away = away_goals + g[None, :]
    return {
        "home_win": float(matrix[final_home > final_away].sum()),
        "draw": float(matrix[final_home == final_away].sum()),
        "away_win": float(matrix[final_home < final_away].sum()),
    }
