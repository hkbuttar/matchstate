"""
Expected Threat (xT) model: a simplified possession-value model in the
style of Karun Singh's xT (https://karun.in/blog/expected-threat.html).

The pitch is divided into a grid of zones. Each zone gets a "threat"
value, defined recursively:

    xT[z] = shot_prob[z] * avg_shot_value[z]
          + move_prob[z] * sum_over_dest( transition_prob[z, dest] * xT[dest] )

In words: the threat of having the ball in zone z equals the chance you
shoot from there (times how good a shot from there tends to be) plus the
chance you successfully move the ball onward (times the threat of
wherever it typically goes). This is a Bellman/value-iteration equation
-- solved here by simple fixed-point iteration from xT=0 until it stops
changing.

The value credited to a specific completed pass or carry is then
xT[end_zone] - xT[start_zone]: how much closer to a goal that specific
action moved the team's positional threat.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_X_BINS = 16
N_Y_BINS = 12


def zone_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.clip((x / 120 * N_X_BINS).astype(int), 0, N_X_BINS - 1)
    yi = np.clip((y / 80 * N_Y_BINS).astype(int), 0, N_Y_BINS - 1)
    return xi * N_Y_BINS + yi  # flatten (x,y) -> single zone index


def zone_to_xy(zone: int) -> tuple[int, int]:
    return zone // N_Y_BINS, zone % N_Y_BINS


class ExpectedThreat:
    def __init__(self):
        self.xt_grid = None  # shape (N_X_BINS*N_Y_BINS,)
        self.shot_prob = None
        self.move_prob = None
        self.avg_shot_value = None
        self.transition = None

    def fit(self, actions: pd.DataFrame, n_iterations: int = 25) -> "ExpectedThreat":
        n_zones = N_X_BINS * N_Y_BINS

        actions = actions.copy()
        actions["start_zone"] = zone_of(actions["start_x"].to_numpy(), actions["start_y"].to_numpy())

        total_count = np.bincount(actions["start_zone"], minlength=n_zones).astype(float)

        shots = actions[actions["event_type"] == "Shot"]
        shot_count = np.bincount(shots["start_zone"], minlength=n_zones).astype(float)
        shot_value_sum = np.bincount(shots["start_zone"], weights=shots["shot_xg"], minlength=n_zones)
        avg_shot_value = np.divide(shot_value_sum, shot_count, out=np.zeros(n_zones), where=shot_count > 0)

        moves = actions[(actions["event_type"].isin(["Pass", "Carry"])) & (actions["success"])]
        move_count = np.bincount(moves["start_zone"], minlength=n_zones).astype(float)
        end_zone = zone_of(moves["end_x"].to_numpy(), moves["end_y"].to_numpy())

        transition = np.zeros((n_zones, n_zones))
        np.add.at(transition, (moves["start_zone"].to_numpy(), end_zone), 1)
        row_sums = transition.sum(axis=1, keepdims=True)
        transition = np.divide(transition, row_sums, out=np.zeros_like(transition), where=row_sums > 0)

        shot_prob = np.divide(shot_count, total_count, out=np.zeros(n_zones), where=total_count > 0)
        move_prob = np.divide(move_count, total_count, out=np.zeros(n_zones), where=total_count > 0)

        xt = np.zeros(n_zones)
        for _ in range(n_iterations):
            xt = shot_prob * avg_shot_value + move_prob * (transition @ xt)

        self.xt_grid = xt
        self.shot_prob = shot_prob
        self.move_prob = move_prob
        self.avg_shot_value = avg_shot_value
        self.transition = transition
        self.total_count = total_count
        return self

    def value_at(self, x, y) -> np.ndarray:
        return self.xt_grid[zone_of(np.asarray(x), np.asarray(y))]

    def action_value(self, actions: pd.DataFrame) -> np.ndarray:
        """xT[end] - xT[start] for each row (only meaningful for successful pass/carry)."""
        start_v = self.value_at(actions["start_x"].to_numpy(), actions["start_y"].to_numpy())
        end_v = self.value_at(actions["end_x"].to_numpy(), actions["end_y"].to_numpy())
        return end_v - start_v

    def grid_as_dataframe(self) -> pd.DataFrame:
        rows = []
        for z in range(N_X_BINS * N_Y_BINS):
            xi, yi = zone_to_xy(z)
            rows.append(
                {
                    "zone": z,
                    "x_bin": xi,
                    "y_bin": yi,
                    "xt": self.xt_grid[z],
                    "shot_prob": self.shot_prob[z],
                    "move_prob": self.move_prob[z],
                    "avg_shot_value": self.avg_shot_value[z],
                    "n_actions": self.total_count[z],
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: Path):
        np.savez(
            path,
            xt_grid=self.xt_grid,
            shot_prob=self.shot_prob,
            move_prob=self.move_prob,
            avg_shot_value=self.avg_shot_value,
            transition=self.transition,
            total_count=self.total_count,
        )

    @classmethod
    def load(cls, path: Path) -> "ExpectedThreat":
        data = np.load(path)
        model = cls()
        model.xt_grid = data["xt_grid"]
        model.shot_prob = data["shot_prob"]
        model.move_prob = data["move_prob"]
        model.avg_shot_value = data["avg_shot_value"]
        model.transition = data["transition"]
        model.total_count = data["total_count"]
        return model
