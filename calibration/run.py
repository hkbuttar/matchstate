"""
Step 8: fit Platt scaling and isotonic regression per model (on the 43-
match calibration split), evaluate before/after on the 95-match test
split, per outcome class -- reported separately, since draws are the
class most likely to be miscalibrated and averaging would hide that.
"""

import json
from pathlib import Path

import numpy as np

from calibration.calibrate import IsotonicCalibrator, PlattCalibrator
from calibration.data import CLASS_ORDER, build_splits_and_predictions
from calibration.reliability import class_brier, expected_calibration_error, plot_reliability_grid

OUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "calibration"
PLOTS_DIR = Path(__file__).parent.parent / "calibration" / "plots"


def evaluate_model(name: str, cal_probs: np.ndarray, cal_actual: np.ndarray, test_probs: np.ndarray, test_actual: np.ndarray) -> dict:
    result = {"model": name, "classes": {}}

    platt = PlattCalibrator().fit(cal_probs, cal_actual)
    isotonic = IsotonicCalibrator().fit(cal_probs, cal_actual)
    platt_test = platt.transform(test_probs)
    isotonic_test = isotonic.transform(test_probs)

    for c, class_name in enumerate(CLASS_ORDER):
        indicator = (test_actual == c).astype(float)
        raw_brier = class_brier(test_probs[:, c], indicator)
        raw_ece = expected_calibration_error(test_probs[:, c], indicator)
        platt_brier = class_brier(platt_test[:, c], indicator)
        platt_ece = expected_calibration_error(platt_test[:, c], indicator)
        iso_brier = class_brier(isotonic_test[:, c], indicator)
        iso_ece = expected_calibration_error(isotonic_test[:, c], indicator)

        result["classes"][class_name] = {
            "raw": {"brier": raw_brier, "ece": raw_ece},
            "platt": {"brier": platt_brier, "ece": platt_ece},
            "isotonic": {"brier": iso_brier, "ece": iso_ece},
        }
        print(f"  {class_name:10s} raw: Brier={raw_brier:.4f} ECE={raw_ece:.4f}  |  "
              f"Platt: Brier={platt_brier:.4f} ECE={platt_ece:.4f}  |  "
              f"Isotonic: Brier={iso_brier:.4f} ECE={iso_ece:.4f}")

    best_method = "isotonic" if np.mean([result["classes"][c]["isotonic"]["ece"] for c in CLASS_ORDER]) < \
                                 np.mean([result["classes"][c]["platt"]["ece"] for c in CLASS_ORDER]) else "platt"
    best_probs = isotonic_test if best_method == "isotonic" else platt_test
    plot_reliability_grid(name, test_probs, best_probs, test_actual, PLOTS_DIR / f"{name}_reliability.png")
    result["best_method_by_mean_ece"] = best_method
    return result


def main():
    predictions = build_splits_and_predictions()
    all_results = []
    for name in ["static", "bayesian", "gbm"]:
        print(f"\n=== {name} ===")
        r = evaluate_model(
            name,
            predictions["cal"][name], predictions["cal"]["actual"],
            predictions["test"][name], predictions["test"]["actual"],
        )
        all_results.append(r)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {OUT_DIR / 'calibration_results.json'}")
    print(f"Reliability diagrams saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
