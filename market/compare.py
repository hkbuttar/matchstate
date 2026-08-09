"""
Step 9's core deliverable, in two genuinely distinct parts:

1. PRE-MATCH: our three models vs. the real (de-vigged Pinnacle closing
   odds) market, on the 95 held-out test matches. Expectation, stated in
   the plan itself: the market wins here, and that's normal, not a
   failure -- betting markets are hard to beat pre-match.

2. IN-GAME: the actually interesting, isolated question -- does our
   models' real-time (score/time-conditioned) update add genuine value
   beyond just taking the market's pre-match view and adjusting it
   naively for the current score, using the exact same adjustment
   mechanism our own baseline uses? This isolates one variable (whose
   prior is better) while holding the update mechanism fixed.
"""

import json
from pathlib import Path

import numpy as np

from calibration.data import CLASS_ORDER, build_splits_and_predictions
from data.team_names import to_football_data
from market.implied_strength import implied_lambda_mu, market_in_game_probabilities
from market.odds import load_market_probabilities

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def brier_logloss(pred: np.ndarray, actual_class: np.ndarray) -> tuple[float, float]:
    n = len(actual_class)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), actual_class] = 1.0
    pred = np.clip(pred, 1e-10, 1.0)
    brier = np.mean(np.sum((pred - onehot) ** 2, axis=1))
    logloss = np.mean(-np.log(pred[np.arange(n), actual_class]))
    return float(brier), float(logloss)


def main():
    predictions, splits = build_splits_and_predictions()
    test_actual = predictions["test"]["actual"]
    test_df = splits["test"]  # exact row order predictions['test'][...] was computed from
    assert len(test_df) == len(test_actual), "row alignment mismatch between calibration split and features"

    # --- market odds, joined onto test matches by team name ---
    odds = load_market_probabilities()
    odds_lookup = {(r.HomeTeam, r.AwayTeam): r for r in odds.itertuples()}

    test_matches = test_df[["match_id", "home_team", "away_team"]].drop_duplicates()
    market_lam_mu = {}
    market_prematch = {}
    missing = []
    for r in test_matches.itertuples():
        key = (to_football_data(r.home_team), to_football_data(r.away_team))
        if key not in odds_lookup:
            missing.append(key)
            continue
        row = odds_lookup[key]
        p = (row.market_home_win, row.market_draw, row.market_away_win)
        market_prematch[r.match_id] = p
        market_lam_mu[r.match_id] = implied_lambda_mu(*p)
    print(f"Matched market odds for {len(market_prematch)}/{len(test_matches)} test matches "
          f"({len(missing)} missing: {missing})")

    # ============ PART 1: PRE-MATCH COMPARISON ============
    print("\n=== PRE-MATCH comparison (95 test matches, minute=0) ===")
    minute0_mask = (test_df["minute"] == 0).to_numpy()
    minute0_idx = np.where(minute0_mask)[0]
    minute0_match_ids = test_df.iloc[minute0_idx]["match_id"].to_numpy()

    has_market = np.array([mid in market_prematch for mid in minute0_match_ids])
    idx_with_market = minute0_idx[has_market]
    actual0 = test_actual[idx_with_market]

    market_probs0 = np.array([market_prematch[mid] for mid in minute0_match_ids[has_market]])
    mb, ml = brier_logloss(market_probs0, actual0)
    print(f"  {'Market (Pinnacle, de-vigged)':28s}: Brier={mb:.4f}  LogLoss={ml:.4f}")

    prematch_summary = {"market": {"brier": mb, "logloss": ml}}
    for name in ["static", "bayesian", "gbm"]:
        probs0 = predictions["test"][name][idx_with_market]
        b, l = brier_logloss(probs0, actual0)
        print(f"  {name:28s}: Brier={b:.4f}  LogLoss={l:.4f}")
        prematch_summary[name] = {"brier": b, "logloss": l}

    # ============ PART 2: IN-GAME COMPARISON ============
    print("\n=== IN-GAME comparison (all match-minutes, market+naive-adjustment baseline) ===")
    has_market_row = test_df["match_id"].isin(market_lam_mu.keys()).to_numpy()
    sub_df = test_df[has_market_row].reset_index(drop=True)
    sub_actual = test_actual[has_market_row]

    market_ingame_probs = []
    for r in sub_df.itertuples():
        lam, mu = market_lam_mu[r.match_id]
        p = market_in_game_probabilities(lam, mu, r.minute, r.home_goals, r.away_goals)
        market_ingame_probs.append([p[c] for c in CLASS_ORDER])
    market_ingame_probs = np.array(market_ingame_probs)

    mb_ig, ml_ig = brier_logloss(market_ingame_probs, sub_actual)
    print(f"  {'Market + naive score/time adj':28s}: Brier={mb_ig:.4f}  LogLoss={ml_ig:.4f}")
    ingame_summary = {"market": {"brier": mb_ig, "logloss": ml_ig}}
    for name in ["static", "bayesian", "gbm"]:
        probs = predictions["test"][name][has_market_row]
        b, l = brier_logloss(probs, sub_actual)
        print(f"  {name:28s}: Brier={b:.4f}  LogLoss={l:.4f}")
        ingame_summary[name] = {"brier": b, "logloss": l}

    print("\nBrier score by match phase (in-game):")
    minutes = sub_df["minute"].to_numpy()
    for lo, hi, label in [(0, 30, "0-30'"), (30, 60, "30-60'"), (60, 200, "60'+")]:
        mask = (minutes >= lo) & (minutes < hi)
        line = f"  {label:8s}"
        line += f"  Market={brier_logloss(market_ingame_probs[mask], sub_actual[mask])[0]:.4f}"
        for name in ["static", "bayesian", "gbm"]:
            probs = predictions["test"][name][has_market_row][mask]
            line += f"  {name}={brier_logloss(probs, sub_actual[mask])[0]:.4f}"
        print(line)

    out = {"prematch": prematch_summary, "ingame_overall": ingame_summary}
    (PROCESSED_DIR / "step9_market_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved summary to {PROCESSED_DIR / 'step9_market_comparison.json'}")


if __name__ == "__main__":
    main()
