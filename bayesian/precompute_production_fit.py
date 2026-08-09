"""
Fits the hierarchical Bayesian model on the full 2015/16 season ONCE
(locally, where BLAS is linked and pytensor's C-compilation cache is
warm) and saves the posterior-mean arrays as a small JSON artifact via
FrozenBayesianStrength. backend/state.py loads this instead of calling
HierarchicalDixonColes(...).fit(...) at server startup -- see
bayesian/frozen.py's module docstring for why: NUTS sampling that takes
~6s here took long enough on Render's BLAS-less container to leave the
service stuck reporting "starting" well past any reasonable deploy
window. The fitted parameters don't depend on the deploy environment,
so there's no reason to pay that cost on every boot.

Re-run this whenever the underlying model or data changes; commit the
output (data/processed/bayesian_production_fit.json) like the other
data/processed/ artifacts.
"""

from pathlib import Path

from baseline.data import load_results
from bayesian.frozen import FrozenBayesianStrength
from bayesian.model import HierarchicalDixonColes

OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "bayesian_production_fit.json"


def main():
    results = load_results()
    season_results = results[results["Season"] == "2015/16"]

    print("Fitting HierarchicalDixonColes on full 2015/16 season (this runs once, locally) ...")
    model = HierarchicalDixonColes(n_periods=8).fit(season_results, draws=800, tune=800, chains=4, random_seed=42)

    frozen = FrozenBayesianStrength.from_fitted(model)
    frozen.to_json(OUT_PATH)
    print(f"Saved {len(frozen.teams)} teams x {frozen.n_periods} periods -> {OUT_PATH}")


if __name__ == "__main__":
    main()
