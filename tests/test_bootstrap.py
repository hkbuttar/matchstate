"""
Block bootstrap procedure validation.

Three checks: (1) does resampling by match actually produce wider,
more honest confidence intervals than naively resampling by row, on data
constructed to have strong within-match correlation -- this is the whole
reason backtest/ uses match blocks instead of row-level bootstrap; (2) does
a real, planted difference between two synthetic models get correctly
flagged as significant; (3) approximate coverage -- across many
repeated synthetic experiments with a known true Brier score, does the
95% CI actually contain the truth close to 95% of the time?
"""

import numpy as np

from backtest.block_bootstrap import block_bootstrap_brier, summarize_ci, summarize_diff


def _make_correlated_matches(n_matches: int, rows_per_match: int, true_brier: float, seed: int):
    """Each match gets ONE noisy per-match squared-error value, repeated
    across all its rows (strong within-match correlation, zero
    within-match noise) -- rows are NOT independent, which is exactly
    the structure calibration/'s and market/'s per-minute data has (all
    rows in a match share the match's outcome)."""
    rng = np.random.default_rng(seed)
    match_ids = np.repeat(np.arange(n_matches), rows_per_match)
    per_match_value = np.clip(rng.normal(true_brier, 0.15, n_matches), 0, 1)
    values = np.repeat(per_match_value, rows_per_match)
    return match_ids, values


def test_block_bootstrap_is_wider_than_naive_row_bootstrap():
    """The core justification for match-blocking (calibration/'s and
    backtest/'s design choice): on data with strong within-match correlation, a naive
    row-level bootstrap should understate uncertainty relative to a
    proper match-block bootstrap."""
    match_ids, values = _make_correlated_matches(n_matches=50, rows_per_match=90, true_brier=0.5, seed=42)

    block_boot = block_bootstrap_brier(match_ids, {"model": values}, n_boot=1000, seed=1)
    block_ci = summarize_ci(block_boot["model"], values.mean())
    block_width = block_ci["ci_hi"] - block_ci["ci_lo"]

    # naive row-level bootstrap: resample individual ROW indices, ignoring match structure
    rng = np.random.default_rng(2)
    n_rows = len(values)
    naive_means = np.array([values[rng.integers(0, n_rows, n_rows)].mean() for _ in range(1000)])
    naive_ci = summarize_ci(naive_means, values.mean())
    naive_width = naive_ci["ci_hi"] - naive_ci["ci_lo"]

    assert block_width > naive_width * 2, (
        f"block bootstrap CI (width={block_width:.4f}) should be substantially wider than "
        f"naive row bootstrap (width={naive_width:.4f}) on strongly match-correlated data"
    )


def test_detects_planted_difference():
    """Model B is constructed to be worse by a real, fixed margin --
    verify the paired bootstrap correctly flags this as significant,
    with the correct sign."""
    n_matches, rows_per_match = 80, 90
    match_ids = np.repeat(np.arange(n_matches), rows_per_match)
    rng = np.random.default_rng(5)
    per_match_a = np.clip(rng.normal(0.45, 0.08, n_matches), 0, 1)
    per_match_b = np.clip(per_match_a + 0.08, 0, 1)  # B is reliably worse by 0.08
    a_vals = np.repeat(per_match_a, rows_per_match)
    b_vals = np.repeat(per_match_b, rows_per_match)

    boot = block_bootstrap_brier(match_ids, {"a": a_vals, "b": b_vals}, n_boot=2000, seed=7)
    diff = summarize_diff(boot["a"], boot["b"], a_vals.mean(), b_vals.mean())

    assert diff["significant_at_95"]
    assert diff["point_diff"] < 0  # a is better (lower Brier) than b
    assert diff["ci_hi"] < 0  # entire CI below zero


def test_no_false_positive_on_identical_models():
    """Two identical models should NOT show a significant difference --
    a false-positive-rate sanity check."""
    match_ids, values = _make_correlated_matches(n_matches=60, rows_per_match=90, true_brier=0.5, seed=11)
    boot = block_bootstrap_brier(match_ids, {"a": values, "b": values.copy()}, n_boot=1000, seed=12)
    diff = summarize_diff(boot["a"], boot["b"], values.mean(), values.mean())
    assert not diff["significant_at_95"]
    assert diff["point_diff"] == 0.0


def test_approximate_ci_coverage():
    """Across many independent synthetic experiments with a known true
    mean, the 95% CI should contain the truth roughly 95% of the time
    (loose Monte Carlo tolerance given a modest number of repetitions)."""
    true_brier = 0.55
    n_experiments = 150
    contains_truth = 0
    for i in range(n_experiments):
        match_ids, values = _make_correlated_matches(n_matches=40, rows_per_match=20, true_brier=true_brier, seed=1000 + i)
        boot = block_bootstrap_brier(match_ids, {"model": values}, n_boot=300, seed=2000 + i)
        ci = summarize_ci(boot["model"], values.mean())
        if ci["ci_lo"] <= true_brier <= ci["ci_hi"]:
            contains_truth += 1

    coverage = contains_truth / n_experiments
    assert 0.85 <= coverage <= 1.0, f"empirical coverage {coverage:.2f} far from nominal 0.95"
