"""
A "frozen" hierarchical Bayesian strength model: same prediction
interface as HierarchicalDixonColes (match_probabilities,
in_game_probabilities, strength_trajectory), but backed by plain numpy
arrays loaded from a precomputed JSON artifact instead of a live PyMC
trace -- no PyTensor/NUTS involved at inference time at all.

Why this exists: backend/state.py used to call
HierarchicalDixonColes(...).fit(...) directly at server startup. That's
fast locally (~6s, warm pytensor compile cache, linked BLAS) but a real
Render deploy showed NUTS sampling taking dramatically longer there --
confirmed from the deploy's own logs, which included "PyTensor could not
link to a BLAS installation. Operations that might benefit from BLAS
will be severely degraded." There's no good reason to pay that cost on
every server boot in production: the fitted parameters don't depend on
anything about the request or the deploy environment, so they're fit
ONCE (bayesian/precompute_production_fit.py, run locally where BLAS
works normally) and shipped as a small JSON artifact instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson


class FrozenBayesianStrength:
    def __init__(self, teams: list[str], attack: np.ndarray, defense: np.ndarray, home_adv: float):
        """attack/defense: shape (n_teams, n_periods)."""
        self.teams = teams
        self.attack = attack
        self.defense = defense
        self.home_adv = home_adv
        self.n_periods = attack.shape[1]

    def latest_strength(self) -> dict:
        return {
            "attack": dict(zip(self.teams, self.attack[:, -1].tolist())),
            "defense": dict(zip(self.teams, self.defense[:, -1].tolist())),
            "home_adv": self.home_adv,
        }

    def strength_trajectory(self):
        import pandas as pd

        rows = []
        for i, team in enumerate(self.teams):
            for p in range(self.n_periods):
                rows.append({"team": team, "period": p, "attack": float(self.attack[i, p]), "defense": float(self.defense[i, p])})
        return pd.DataFrame(rows)

    def match_probabilities(self, home_team: str, away_team: str, max_goals: int = 10) -> dict:
        s = self.latest_strength()
        lam = np.exp(s["attack"][home_team] + s["defense"][away_team] + s["home_adv"])
        mu = np.exp(s["attack"][away_team] + s["defense"][home_team])
        g = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        matrix /= matrix.sum()
        return {
            "home_win": float(np.tril(matrix, -1).sum()),
            "draw": float(np.trace(matrix)),
            "away_win": float(np.triu(matrix, 1).sum()),
            "lambda": float(lam),
            "mu": float(mu),
        }

    def in_game_probabilities(
        self, home_team: str, away_team: str, minute: float, home_goals: int, away_goals: int,
        match_length: float = 90.0, max_goals: int = 10,
    ) -> dict:
        remaining_frac = max(0.0, (match_length - minute) / match_length)
        s = self.latest_strength()
        lam = np.exp(s["attack"][home_team] + s["defense"][away_team] + s["home_adv"])
        mu = np.exp(s["attack"][away_team] + s["defense"][home_team])
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

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({
            "teams": self.teams,
            "attack": self.attack.tolist(),
            "defense": self.defense.tolist(),
            "home_adv": self.home_adv,
        }))

    @classmethod
    def from_json(cls, path: Path) -> "FrozenBayesianStrength":
        data = json.loads(Path(path).read_text())
        return cls(
            teams=data["teams"],
            attack=np.array(data["attack"]),
            defense=np.array(data["defense"]),
            home_adv=data["home_adv"],
        )

    @classmethod
    def from_fitted(cls, model) -> "FrozenBayesianStrength":
        """Extract the posterior-mean arrays from an already-fitted
        HierarchicalDixonColes (live PyMC trace) into this lightweight,
        trace-free form."""
        attack_mean = model.trace.posterior["attack"].mean(dim=("chain", "draw")).values
        defense_mean = model.trace.posterior["defense"].mean(dim=("chain", "draw")).values
        home_adv_mean = float(model.trace.posterior["home_adv"].mean())
        return cls(teams=model.teams, attack=attack_mean, defense=defense_mean, home_adv=home_adv_mean)
