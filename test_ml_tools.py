"""
Quick checks for ml_tools, on made-up games only (no real games, no names).

Run it with:  python test_ml_tools.py
"""

import numpy as np
import pandas as pd

from ml_tools import (
    FEATURE_COLUMNS, calibration_table, make_features,
    permutation_importance_table, split_by_date, three_class_log_loss,
    tune_and_compare,
)


def make_games(n, seed=3):
    """
    n made-up games, one per day in date order. The score depends strongly
    on rating_diff through a known logistic curve; the other columns are noise.
    """
    rng = np.random.default_rng(seed)
    rating_diff = rng.integers(-300, 301, size=n)
    p = 1 / (1 + np.exp(-(1.2 * rating_diff / 100)))
    u = rng.random(n)
    # Win if u < p - 0.03, draw in a thin band around it, loss otherwise.
    score = np.where(u < p - 0.03, 1.0, np.where(u < p + 0.03, 0.5, 0.0))
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "colour": rng.choice(["white", "black"], size=n),
        "my_rating": rng.integers(1200, 1800, size=n),
        "rating_diff": rating_diff,
        "base_seconds": rng.choice([60, 180, 600], size=n),
        "increment_seconds": rng.choice([0, 2], size=n),
        "opening": rng.choice(["Made-Up-Opening", "Pretend-Defense-Line"], size=n),
        "score": score,
    })


# Test 1: changing the result of game i must not change row i's features,
# but it may change row i + 1 (which can see game i's result).
games = make_games(50)
before = make_features(games)
i = 10
changed = games.copy()
changed.loc[i, "score"] = 1.0 - changed.loc[i, "score"] if changed.loc[i, "score"] != 0.5 else 0.0
after = make_features(changed)
assert before.loc[: i, FEATURE_COLUMNS].equals(after.loc[: i, FEATURE_COLUMNS])
assert not before.loc[i + 1, FEATURE_COLUMNS].equals(after.loc[i + 1, FEATURE_COLUMNS])
assert before.loc[0, "form_20"] == 0.5 and before.loc[0, "last_score"] == 0.5
print("Test 1 passed: a game's own result never leaks into its features")

# Test 2: loss_streak on a hand-made sequence of results.
# Results:          L  L  W  L  L  L  D  L
# Losses before:    0  1  2  0  1  2  3  0
games = make_games(8)
games["score"] = [0, 0, 1, 0, 0, 0, 0.5, 0]
features = make_features(games)
assert list(features["loss_streak"]) == [0, 1, 2, 0, 1, 2, 3, 0]
print("Test 2 passed: loss_streak counts the losses in a row before each game")

# Test 3: tune_and_compare on 600 made-up games.
games = make_games(600)
results, info = tune_and_compare(games)
assert set(results["model"]) == {
    "Logistic regression", "Random forest", "Gradient boosting", "Constant (train mean)",
}
assert len(results) == 4
brier = dict(zip(results["model"], results["test_brier"]))
assert brier[info["best_model_name"]] < brier["Constant (train mean)"]
assert info["train_end"] <= info["test_start"]
assert info["train_games"] + info["test_games"] == 600
print(f"Test 3 passed: best model ({info['best_model_name']}) test Brier "
      f"{brier[info['best_model_name']]:.4f} beats constant "
      f"{brier['Constant (train mean)']:.4f}")

# Test 4: calibration_table splits the games into equal-count bins.
rng = np.random.default_rng(5)
predicted = rng.random(103)
actual = rng.choice([0.0, 0.5, 1.0], size=103)
table = calibration_table(predicted, actual, bins=5)
assert len(table) == 5
assert table["games"].sum() == 103
assert table["games"].max() - table["games"].min() <= 1
print("Test 4 passed: calibration_table gives 5 bins holding all 103 games")

# Test 5: on the made-up data, rating_diff should matter most.
_, test = split_by_date(make_features(games))
importance = permutation_importance_table(
    info["best_pipeline"], test[FEATURE_COLUMNS], test["score"]
)
assert importance.loc[0, "feature"] == "rating_diff"
print(f"Test 5 passed: rating_diff is the most important feature "
      f"(importance {importance.loc[0, 'importance']:.4f})")

# Test 6: log loss matches a hand calculation. Columns are loss, draw, win.
# Log loss is the average of -log(probability given to what really happened).
labels = np.array(["loss", "draw", "win"])
proba = np.array([
    [0.7, 0.1, 0.2],    # really a loss: probability 0.7
    [0.3, 0.5, 0.2],    # really a draw: probability 0.5
    [0.1, 0.1, 0.8],    # really a win:  probability 0.8
])
by_hand = -np.mean(np.log([0.7, 0.5, 0.8]))
assert abs(three_class_log_loss(labels, proba) - by_hand) < 1e-12
print(f"Test 6 passed: log loss {by_hand:.4f} matches the hand calculation")
