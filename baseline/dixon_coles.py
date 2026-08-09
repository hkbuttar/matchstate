"""
Dixon-Coles Poisson model for football match outcomes.

Reference: Dixon, M.J. and Coles, S.G. (1997), "Modelling Association
Football Scores and Inefficiencies in Football Betting Markets", Journal
of the Royal Statistical Society: Series C, 46(2), 265-280.

Each team gets an attack strength and a defense strength; the home team
gets a fixed home-advantage boost. Goals are modeled as Poisson variables:

    home_goals ~ Poisson(lambda),  lambda = exp(attack[home] + defense[away] + home_adv)
    away_goals ~ Poisson(mu),      mu     = exp(attack[away] + defense[home])

A plain product of two independent Poissons systematically underestimates
how often low-scoring results occur (0-0, 1-0, 0-1, 1-1), because goals
within a match aren't fully independent -- e.g. a team leading 1-0 late
tends to play more conservatively, suppressing further scoring for both
sides. Dixon-Coles correct for this with a multiplicative adjustment tau,
applied only to those four scorelines and controlled by a single extra
parameter rho, fit jointly with the attack/defense strengths.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColes:
    """Fits fixed (non-time-varying) attack/defense strengths on a set of
    match results -- typically one season at a time. See bayesian/ for the
    hierarchical, within-season time-varying extension (Step 3)."""

    def __init__(self):
        self.teams: list[str] | None = None
        self.attack: dict[str, float] | None = None
        self.defense: dict[str, float] | None = None
        self.home_adv: float | None = None
        self.rho: float | None = None
        self.max_goals = 10

    def fit(self, results: pd.DataFrame, max_goals: int = 10) -> "DixonColes":
        """results needs columns: HomeTeam, AwayTeam, FTHG, FTAG."""
        self.max_goals = max_goals
        teams = sorted(set(results["HomeTeam"]) | set(results["AwayTeam"]))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        home_idx = results["HomeTeam"].map(idx).to_numpy()
        away_idx = results["AwayTeam"].map(idx).to_numpy()
        hg = results["FTHG"].to_numpy()
        ag = results["FTAG"].to_numpy()
        low_score_mask = (hg <= 1) & (ag <= 1)

        def unpack(params):
            # n-1 free attack params + n-1 free defense params, each set
            # constrained to sum to zero (last team's value = -sum of the
            # rest) so the model is identifiable -- otherwise you could add
            # a constant to every attack score and subtract it from every
            # defense score with no change in likelihood.
            attack = np.empty(n)
            attack[:-1] = params[: n - 1]
            attack[-1] = -attack[:-1].sum()
            defense = np.empty(n)
            defense[:-1] = params[n - 1 : 2 * n - 2]
            defense[-1] = -defense[:-1].sum()
            home_adv = params[2 * n - 2]
            rho = params[2 * n - 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            mu = np.exp(attack[away_idx] + defense[home_idx])

            ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)

            tau_vals = np.ones(len(hg))
            for i in np.nonzero(low_score_mask)[0]:
                tau_vals[i] = tau(hg[i], ag[i], lam[i], mu[i], rho)
            tau_vals = np.clip(tau_vals, 1e-10, None)  # guard mid-optimization
            ll = ll + np.log(tau_vals)
            return -ll.sum()

        x0 = np.zeros(2 * n)
        res = minimize(neg_log_likelihood, x0, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit did not converge: {res.message}")

        attack, defense, home_adv, rho = unpack(res.x)
        self.teams = teams
        self.attack = dict(zip(teams, attack.tolist()))
        self.defense = dict(zip(teams, defense.tolist()))
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.log_likelihood = -res.fun
        return self

    def _rates(self, home_team: str, away_team: str) -> tuple[float, float]:
        lam = np.exp(self.attack[home_team] + self.defense[away_team] + self.home_adv)
        mu = np.exp(self.attack[away_team] + self.defense[home_team])
        return lam, mu

    def score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Returns a (max_goals+1) x (max_goals+1) matrix of
        P(home_goals=i, away_goals=j) for the full 90 minutes."""
        lam, mu = self._rates(home_team, away_team)
        g = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= tau(x, y, lam, mu, self.rho)
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix

    def match_probabilities(self, home_team: str, away_team: str) -> dict:
        """Pre-match win/draw/loss probabilities (Step 2's core deliverable)."""
        matrix = self.score_matrix(home_team, away_team)
        lam, mu = self._rates(home_team, away_team)
        return {
            "home_win": float(np.tril(matrix, -1).sum()),
            "draw": float(np.trace(matrix)),
            "away_win": float(np.triu(matrix, 1).sum()),
            "lambda": float(lam),
            "mu": float(mu),
        }

    def in_game_probabilities(
        self,
        home_team: str,
        away_team: str,
        minute: float,
        home_goals: int,
        away_goals: int,
        match_length: float = 90.0,
    ) -> dict:
        """
        Score-conditional win/draw/loss probability given the match has
        reached `minute` at score (home_goals, away_goals).

        Modeling choice (disclosed): goals for the *remainder* of the match
        are assumed Poisson at the same rate as the full-match model,
        scaled down proportionally to time remaining (a constant scoring
        rate across the 90 minutes). Real scoring rate rises somewhat in
        the closing minutes of matches; this constant-rate assumption is
        the transparent, analytically simple choice for this baseline, and
        exactly the kind of simplification the state-aware models in
        Steps 4-6 are meant to improve on. The same tau low-score
        correction is re-applied to the remaining-goals distribution.
        """
        remaining_frac = max(0.0, (match_length - minute) / match_length)
        lam, mu = self._rates(home_team, away_team)
        lam_rem, mu_rem = lam * remaining_frac, mu * remaining_frac

        g = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(g, lam_rem), poisson.pmf(g, mu_rem))
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= tau(x, y, lam_rem, mu_rem, self.rho)
        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()

        # Sum over remaining-goal outcomes, bucketed by final result.
        final_home = home_goals + g[:, None]
        final_away = away_goals + g[None, :]
        home_win = matrix[final_home > final_away].sum()
        draw = matrix[final_home == final_away].sum()
        away_win = matrix[final_home < final_away].sum()

        return {
            "home_win": float(home_win),
            "draw": float(draw),
            "away_win": float(away_win),
            "lambda_remaining": float(lam_rem),
            "mu_remaining": float(mu_rem),
        }
