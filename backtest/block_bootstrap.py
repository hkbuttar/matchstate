"""
Block bootstrap for confidence intervals on Brier score, blocking by
match rather than by row.

Every comparison in bayesian/, models/, calibration/, and market/
reported a single point estimate on a held-out set. But per-minute rows
within one match are highly correlated (they share the same outcome and
much of the same trajectory) -- calibration/ already ran into this directly (a 43-match
calibration set has only 9 independent draw outcomes despite ~900 "draw"
rows). Treating each row as an independent bootstrap unit would
understate uncertainty. Resampling whole matches (with replacement) and
including all of a resampled match's rows preserves that within-match
correlation structure.

For paired model comparisons (is model A actually better than model B,
not just numerically lower on this one test set?), the same resampled
match indices are used for every model in a given bootstrap draw, so the
per-draw difference A-B is a valid paired comparison -- differencing
cancels shared match-to-match difficulty rather than comparing unrelated
resamples.
"""

import numpy as np


def block_bootstrap_brier(
    match_ids: np.ndarray,
    squared_error_by_model: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    match_ids: shape (n_rows,) -- which match each row belongs to.
    squared_error_by_model: {model_name: array of shape (n_rows,)}, the
      per-row multi-class squared error (sum over classes of (pred-actual)^2)
      -- mean of this is the Brier score.
    Returns {model_name: array of shape (n_boot,)} -- bootstrap distribution
      of the mean (Brier score) under match-block resampling.
    """
    rng = np.random.default_rng(seed)
    unique_matches = np.unique(match_ids)
    n_matches = len(unique_matches)
    match_to_rows = {m: np.where(match_ids == m)[0] for m in unique_matches}

    boot = {name: np.empty(n_boot) for name in squared_error_by_model}
    for b in range(n_boot):
        sampled = rng.choice(unique_matches, size=n_matches, replace=True)
        row_idx = np.concatenate([match_to_rows[m] for m in sampled])
        for name, vals in squared_error_by_model.items():
            boot[name][b] = vals[row_idx].mean()
    return boot


def summarize_ci(boot_dist: np.ndarray, point_estimate: float, ci: float = 0.95) -> dict:
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_dist, [alpha, 1 - alpha])
    return {"point": float(point_estimate), "ci_lo": float(lo), "ci_hi": float(hi)}


def summarize_diff(boot_a: np.ndarray, boot_b: np.ndarray, point_a: float, point_b: float, ci: float = 0.95) -> dict:
    """Paired difference (a - b); negative means a has lower (better) Brier."""
    diff_dist = boot_a - boot_b
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diff_dist, [alpha, 1 - alpha])
    significant = not (lo <= 0 <= hi)
    return {
        "point_diff": float(point_a - point_b),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "significant_at_95": bool(significant),
    }
