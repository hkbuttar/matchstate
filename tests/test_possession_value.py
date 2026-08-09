"""
Step 12: possession-value (xT) model validation.

Two checks: (1) a synthetic test with a hand-constructed action set where
the correct zone ordering is known by construction, validating the value
iteration algorithm itself; (2) the real-data monotonicity check from
Step 4 (possession_value/fit_and_check.py), formalized as a permanent
regression test on the actual saved model artifact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from possession_value.xt_model import ExpectedThreat, N_X_BINS, N_Y_BINS, zone_of

MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "xt_grid_2015_16.npz"


def test_synthetic_value_iteration_recovers_known_ordering():
    """Construct a toy pitch where all shots happen from the zone
    nearest goal (fixed value by construction) and successful moves go
    one zone closer to goal, WITH a chance of turnover (failed pass,
    sequence ends at 0) that shrinks as you approach goal. The value
    iteration should assign strictly higher xT to zones closer to goal --
    a property that follows directly from the algorithm's definition, so
    if this fails, the algorithm (not real football data) is wrong.

    (First version of this test used deterministic, always-succeeding
    transitions with no turnover risk -- every zone came out with
    identical value, 0.3 flat across the board. That's actually *correct*
    behavior for this model, not a bug: with a guaranteed eventual shot
    and no discounting, position doesn't matter. It's specifically the
    possibility of a turnover -- and that possibility being larger far
    from goal -- that creates the monotonic gradient real football data
    shows. Fixed by adding turnover risk explicitly, rather than assuming
    the naive construction would work.)
    """
    rng = np.random.default_rng(0)
    rows = []

    y_mid = 40.0
    n_attempts = 400
    for k in range(N_X_BINS):
        x = (k + 0.5) * (120 / N_X_BINS)
        p_success = 0.5 + 0.03 * k  # 0.50 at own end, 0.95 near goal
        successes = rng.random(n_attempts) < p_success
        for success in successes:
            if k == N_X_BINS - 1:
                rows.append({"event_type": "Shot", "start_x": x, "start_y": y_mid,
                             "end_x": x, "end_y": y_mid, "success": True, "shot_xg": 0.3})
            elif success:
                x_end = (k + 1.5) * (120 / N_X_BINS)
                rows.append({"event_type": "Pass", "start_x": x, "start_y": y_mid,
                             "end_x": x_end, "end_y": y_mid, "success": True, "shot_xg": np.nan})
            else:
                rows.append({"event_type": "Pass", "start_x": x, "start_y": y_mid,
                             "end_x": x, "end_y": y_mid - 5, "success": False, "shot_xg": np.nan})

    actions = pd.DataFrame(rows)
    model = ExpectedThreat().fit(actions, n_iterations=50)

    grid = model.xt_grid.reshape(N_X_BINS, N_Y_BINS)
    y_bin = zone_of(np.array([y_mid]), np.array([y_mid]))[0] % N_Y_BINS
    values_along_conveyor = grid[:, y_bin]

    assert np.all(np.diff(values_along_conveyor) > 0), (
        f"xT should strictly increase along the conveyor toward goal, got {values_along_conveyor}"
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="fitted xT grid not built yet (run possession_value/fit_and_check.py)")
def test_real_grid_monotonic_toward_goal():
    """Regression test for Step 4's real-data finding: mean xT per x-bin
    should be (near-)monotonically increasing from own goal to opponent's
    goal. Originally found 0/15 decreases on the full fit -- allow a
    small amount of noise rather than requiring exact monotonicity, since
    this is real data, not the synthetic case above."""
    model = ExpectedThreat.load(MODEL_PATH)
    grid = model.xt_grid.reshape(N_X_BINS, N_Y_BINS)
    by_x = grid.mean(axis=1)

    assert by_x[-1] > by_x[0], "threat should be higher near the opponent's goal than near your own"
    n_decreases = int((np.diff(by_x) < 0).sum())
    assert n_decreases <= 2, f"too many non-monotonic steps toward goal: {n_decreases}/15"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="fitted xT grid not built yet (run possession_value/fit_and_check.py)")
def test_action_value_is_zero_for_no_movement():
    """A pass that starts and ends in the same zone should have ~zero
    credited value -- a basic sanity property of the xT[end]-xT[start]
    definition."""
    model = ExpectedThreat.load(MODEL_PATH)
    same_spot = pd.DataFrame({"start_x": [60.0], "start_y": [40.0], "end_x": [60.0], "end_y": [40.0]})
    value = model.action_value(same_spot)
    assert abs(value[0]) < 1e-9
