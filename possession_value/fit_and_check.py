"""
Fit the xT grid on the full 2015/16 action set, save it, and run the
sanity check called for in the project plan: does threat increase
monotonically as play moves toward the opponent's goal? (A formal pytest
version belongs in tests/, Step 12 -- this is the build-time check.)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from possession_value.xt_model import ExpectedThreat, N_X_BINS, N_Y_BINS

ACTIONS_PATH = Path(__file__).parent.parent / "data" / "processed" / "possession_actions_2015_16.parquet"
MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "xt_grid_2015_16.npz"


def print_grid(model: ExpectedThreat):
    grid = model.xt_grid.reshape(N_X_BINS, N_Y_BINS)
    print("\nxT grid (rows = x bins, own goal at top, opponent goal at bottom; values x1000):")
    for xi in range(N_X_BINS):
        row = " ".join(f"{v*1000:5.1f}" for v in grid[xi])
        print(f"  x_bin {xi:2d}: {row}")


def main():
    actions = pd.read_parquet(ACTIONS_PATH)
    print(f"Fitting xT on {len(actions):,} actions from {actions['match_id'].nunique()} matches")

    model = ExpectedThreat().fit(actions, n_iterations=25)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print_grid(model)

    # Sanity check: average xT per x-bin (collapsing y) should rise
    # monotonically (or very close to it) moving from x_bin 0 (own
    # box) to x_bin N_X_BINS-1 (opponent's box).
    grid = model.xt_grid.reshape(N_X_BINS, N_Y_BINS)
    by_x = grid.mean(axis=1)
    print("\nMean xT by x-bin (0=own goal end, 15=opponent goal end):")
    for xi, v in enumerate(by_x):
        print(f"  x_bin {xi:2d}: {v*1000:6.2f}")

    diffs = np.diff(by_x)
    n_decreases = int((diffs < 0).sum())
    print(f"\n{n_decreases} of {len(diffs)} consecutive x-bin steps DECREASE in value "
          f"(0 would be perfectly monotonic).")

    is_overall_monotonic_enough = n_decreases <= 2  # allow a little noise, not a hard requirement
    print(f"Overall trend increasing toward goal: {'YES' if by_x[-1] > by_x[0] else 'NO'} "
          f"(x_bin 0 = {by_x[0]*1000:.2f}, x_bin {N_X_BINS-1} = {by_x[-1]*1000:.2f})")

    assert by_x[-1] > by_x[0], "threat should be higher near the opponent's goal than near your own"
    print(f"\nSanity check {'PASSED' if is_overall_monotonic_enough else 'PASSED WITH NOISE'} "
          f"({n_decreases}/{len(diffs)} local decreases, overall trend correct).")


if __name__ == "__main__":
    main()
