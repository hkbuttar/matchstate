"""
Hierarchical Bayesian extension of the Dixon-Coles baseline (Step 2):
team attack/defense strengths are (a) partially pooled across teams via
shared hyperparameters, and (b) allowed to evolve within a season via a
random walk over chronological "periods", rather than fixed for the whole
season.

Partial pooling means a team's strength in period 0 (few/no matches
observed yet) is informed by the league-wide distribution of strengths
(sigma_attack_init, sigma_defense_init are shared hyperparameters); as
periods accumulate, each team's own results increasingly dominate its
own estimate via the random-walk innovations. This directly addresses the
small-sample-at-season-start problem that a single fixed per-season
Dixon-Coles fit can't.

Disclosed simplification: unlike baseline.dixon_coles.DixonColes, this
model omits the Dixon-Coles low-score correction (tau/rho), to keep the
NUTS sampling problem lower-dimensional and faster to fit. Its likely
effect is small relative to the time-varying-strength effect under test
here, but this is a real, disclosed simplification, not an oversight --
see bayesian/README.md.
"""

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import poisson


def assign_periods(n_matches: int, n_periods: int) -> np.ndarray:
    """Chronological equal-count bins: match i (0-indexed, pre-sorted by
    date) gets floor(i * n_periods / n_matches), clipped to the last bin."""
    idx = np.arange(n_matches)
    periods = (idx * n_periods) // n_matches
    return np.minimum(periods, n_periods - 1)


class HierarchicalDixonColes:
    def __init__(self, n_periods: int = 8):
        self.n_periods = n_periods
        self.teams = None
        self.team_idx = None
        self.trace = None
        self.model = None

    def fit(
        self,
        results: pd.DataFrame,
        draws: int = 800,
        tune: int = 800,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> "HierarchicalDixonColes":
        results = results.sort_values("Date").reset_index(drop=True)
        teams = sorted(set(results["HomeTeam"]) | set(results["AwayTeam"]))
        n_teams = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        home_idx = results["HomeTeam"].map(idx).to_numpy()
        away_idx = results["AwayTeam"].map(idx).to_numpy()
        hg = results["FTHG"].to_numpy()
        ag = results["FTAG"].to_numpy()
        period_idx = assign_periods(len(results), self.n_periods)

        with pm.Model() as model:
            sigma_attack_init = pm.HalfNormal("sigma_attack_init", 1.0)
            sigma_defense_init = pm.HalfNormal("sigma_defense_init", 1.0)
            sigma_walk_attack = pm.HalfNormal("sigma_walk_attack", 0.15)
            sigma_walk_defense = pm.HalfNormal("sigma_walk_defense", 0.15)

            attack_init_raw = pm.Normal("attack_init_raw", 0, 1, shape=n_teams)
            defense_init_raw = pm.Normal("defense_init_raw", 0, 1, shape=n_teams)
            attack_innov_raw = pm.Normal("attack_innov_raw", 0, 1, shape=(n_teams, self.n_periods - 1))
            defense_innov_raw = pm.Normal("defense_innov_raw", 0, 1, shape=(n_teams, self.n_periods - 1))

            attack_init = attack_init_raw * sigma_attack_init
            defense_init = defense_init_raw * sigma_defense_init
            attack_innov = attack_innov_raw * sigma_walk_attack
            defense_innov = defense_innov_raw * sigma_walk_defense

            attack_raw = pm.math.concatenate(
                [attack_init[:, None], attack_init[:, None] + pm.math.cumsum(attack_innov, axis=1)], axis=1
            )
            defense_raw = pm.math.concatenate(
                [defense_init[:, None], defense_init[:, None] + pm.math.cumsum(defense_innov, axis=1)], axis=1
            )
            # Zero-sum each period (identifiability: otherwise a constant
            # could shift every attack score up and every defense score
            # down with no change in likelihood -- same issue as Step 2).
            attack = pm.Deterministic("attack", attack_raw - attack_raw.mean(axis=0, keepdims=True))
            defense = pm.Deterministic("defense", defense_raw - defense_raw.mean(axis=0, keepdims=True))

            home_adv = pm.Normal("home_adv", 0.3, 0.2)

            lam = pm.math.exp(attack[home_idx, period_idx] + defense[away_idx, period_idx] + home_adv)
            mu = pm.math.exp(attack[away_idx, period_idx] + defense[home_idx, period_idx])

            pm.Poisson("home_goals_obs", mu=lam, observed=hg)
            pm.Poisson("away_goals_obs", mu=mu, observed=ag)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=True,
            )

        self.teams = teams
        self.team_idx = idx
        self.model = model
        self.trace = trace
        return self

    def strength_trajectory(self) -> pd.DataFrame:
        """Posterior-mean attack/defense per team per period, tidy long form."""
        attack_mean = self.trace.posterior["attack"].mean(dim=("chain", "draw")).values
        defense_mean = self.trace.posterior["defense"].mean(dim=("chain", "draw")).values
        rows = []
        for i, team in enumerate(self.teams):
            for p in range(self.n_periods):
                rows.append(
                    {"team": team, "period": p, "attack": float(attack_mean[i, p]), "defense": float(defense_mean[i, p])}
                )
        return pd.DataFrame(rows)

    def latest_strength(self) -> dict:
        """Posterior-mean attack/defense at the LAST fitted period -- the
        model's 'current form' estimate, used to forecast future matches
        via random-walk persistence (a random walk's best h-step-ahead
        forecast is simply its last observed state)."""
        attack_mean = self.trace.posterior["attack"].mean(dim=("chain", "draw")).values[:, -1]
        defense_mean = self.trace.posterior["defense"].mean(dim=("chain", "draw")).values[:, -1]
        home_adv_mean = float(self.trace.posterior["home_adv"].mean())
        return {
            "attack": dict(zip(self.teams, attack_mean.tolist())),
            "defense": dict(zip(self.teams, defense_mean.tolist())),
            "home_adv": home_adv_mean,
        }

    def match_probabilities(self, home_team: str, away_team: str, max_goals: int = 10) -> dict:
        """Predict a match using the latest (most recent period) strengths."""
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
