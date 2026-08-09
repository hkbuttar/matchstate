"""
Platt scaling and isotonic regression for 3-class probabilities.

Both are fit one-vs-rest: for each class c independently, fit a
calibrator mapping raw P(c) -> corrected P(c) using the class-c indicator
as the target, then renormalize the 3 calibrated probabilities to sum to
1. This is the standard, transparent way to extend binary calibration
methods to the multi-class case -- simpler than a joint multinomial
recalibration, and easy to inspect per class, which is exactly what this
step needs (per-class reporting, not an averaged fix).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class OneVsRestCalibrator:
    method: str

    def __init__(self):
        self.calibrators = []

    def fit(self, raw_probs: np.ndarray, actual_class: np.ndarray) -> "OneVsRestCalibrator":
        n_classes = raw_probs.shape[1]
        self.calibrators = []
        for c in range(n_classes):
            indicator = (actual_class == c).astype(float)
            self.calibrators.append(self._fit_one(raw_probs[:, c], indicator))
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        calibrated = np.column_stack([self._transform_one(cal, raw_probs[:, c]) for c, cal in enumerate(self.calibrators)])
        calibrated = np.clip(calibrated, 1e-6, None)
        return calibrated / calibrated.sum(axis=1, keepdims=True)

    def _fit_one(self, raw_col, indicator):
        raise NotImplementedError

    def _transform_one(self, cal, raw_col):
        raise NotImplementedError


class PlattCalibrator(OneVsRestCalibrator):
    method = "platt"

    def _fit_one(self, raw_col, indicator):
        lr = LogisticRegression()
        lr.fit(raw_col.reshape(-1, 1), indicator)
        return lr

    def _transform_one(self, cal, raw_col):
        return cal.predict_proba(raw_col.reshape(-1, 1))[:, 1]


class IsotonicCalibrator(OneVsRestCalibrator):
    method = "isotonic"

    def _fit_one(self, raw_col, indicator):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_col, indicator)
        return iso

    def _transform_one(self, cal, raw_col):
        return cal.predict(raw_col)
