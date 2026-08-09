"""
Reliability diagrams and per-class calibration metrics.

For each outcome class (one-vs-rest), bins predicted probability into
equal-width bins and compares mean predicted probability against observed
frequency of that class in each bin -- the classic reliability-diagram
construction. Expected Calibration Error (ECE) is the bin-count-weighted
mean absolute gap between predicted and observed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_LABELS = ["Home Win", "Draw", "Away Win"]


def reliability_curve(probs: np.ndarray, actual_indicator: np.ndarray, n_bins: int = 10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bin_edges) - 1, 0, n_bins - 1)

    bin_mean_pred, bin_obs_freq, bin_counts = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_mean_pred.append(probs[mask].mean())
        bin_obs_freq.append(actual_indicator[mask].mean())
        bin_counts.append(mask.sum())

    return np.array(bin_mean_pred), np.array(bin_obs_freq), np.array(bin_counts)


def expected_calibration_error(probs: np.ndarray, actual_indicator: np.ndarray, n_bins: int = 10) -> float:
    mean_pred, obs_freq, counts = reliability_curve(probs, actual_indicator, n_bins)
    if len(counts) == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(mean_pred - obs_freq)) / counts.sum())


def class_brier(probs: np.ndarray, actual_indicator: np.ndarray) -> float:
    return float(np.mean((probs - actual_indicator) ** 2))


def plot_reliability_grid(
    model_name: str,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    actual_class: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c in range(3):
        ax = axes[c]
        indicator = (actual_class == c).astype(float)

        for probs, style, label in [
            (raw_probs[:, c], {"color": "tab:red", "marker": "o", "linestyle": "--"}, "raw"),
            (calibrated_probs[:, c], {"color": "tab:blue", "marker": "s", "linestyle": "-"}, "calibrated"),
        ]:
            mean_pred, obs_freq, counts = reliability_curve(probs, indicator, n_bins)
            ece = expected_calibration_error(probs, indicator, n_bins)
            ax.plot(mean_pred, obs_freq, label=f"{label} (ECE={ece:.3f})", **style)

        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(CLASS_LABELS[c])
        ax.legend(fontsize=8)

    fig.suptitle(f"Reliability diagram: {model_name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
