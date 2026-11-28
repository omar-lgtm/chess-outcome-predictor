"""
Quick checks for general_experiments, on made-up data only.

Run it with:  python test_general_experiments.py
"""

import math

import numpy as np
import pandas as pd

from general_experiments import (
    CLASSES, F0, F1, F2, F3, FEATURE_SETS, build_features, check_no_leakage,
    drop_duplicate_ids, encode_labels, model_proba, multiclass_brier,
    multiclass_log_loss, parse_increment, reorder_proba,
)


def made_up_games():
    """Three invented games with every raw column the CSV has."""
    return pd.DataFrame({
        "id": ["g1", "g2", "g3"],
        "rated": [True, False, True],
        "created_at": [1.0, 2.0, 3.0],
        "last_move_at": [1.5, 2.5, 3.5],
        "turns": [40, 61, 25],
        "victory_status": ["mate", "resign", "draw"],
        "winner": ["white", "black", "draw"],
        "increment_code": ["10+0", "15+10", "oops"],
        "white_id": ["player_a", "player_b", "player_c"],
        "white_rating": [1500, 1600, 1700],
        "black_id": ["player_d", "player_e", "player_f"],
        "black_rating": [1400, 1650, 1700],
        "moves": ["e4 e5", "d4 d5", "c4 c5"],
        "opening_eco": ["C20", "D00", "A10"],
        "opening_name": ["Kings Pawn", "Queens Pawn", "English"],
        "opening_ply": [2, 2, 1],
    })


# Test 1: increment codes.
assert parse_increment("10+0") == (10.0, 0.0)
assert parse_increment("15+10") == (15.0, 10.0)
for bad in ("oops", "10", "", None, "a+b", "1+2+3"):
    base, inc = parse_increment(bad)
    assert math.isnan(base) and math.isnan(inc)
print("Test 1 passed: 10+0 and 15+10 parse; malformed codes give missing values")

# Test 2: the leakage guard.
games = made_up_games()
X = build_features(games, F3)              # allowed columns only: no error
assert "rating_diff" in X.columns and len(X.columns) == len(F3)
assert X.loc[2, "base_minutes"] != X.loc[2, "base_minutes"]   # "oops" became NaN
for forbidden in ("turns", "victory_status", "moves", "white_id", "id",
                  "opening_name", "created_at", "game_duration"):
    leaky = X.assign(**{forbidden: 1})
    try:
        check_no_leakage(leaky)
        raise AssertionError(f"leakage guard missed {forbidden}")
    except ValueError:
        pass
try:
    build_features(games, F2 + ["turns"])
    raise AssertionError("build_features let a forbidden column through")
except (ValueError, KeyError):
    pass
print("Test 2 passed: the leakage guard rejects every forbidden column")

# Test 3: feature groups are nested.
groups = [F0, F1, F2, F3]
for smaller, larger in zip(groups, groups[1:]):
    assert set(smaller) < set(larger)
assert list(FEATURE_SETS.values()) == groups
print("Test 3 passed: F0, F1, F2 and F3 are nested, each a superset of the one before")

# Test 4: duplicates keep the first row of each id.
dupes = pd.DataFrame({"id": ["a", "b", "a", "c", "b"], "marker": [1, 2, 3, 4, 5]})
kept, dropped = drop_duplicate_ids(dupes)
assert dropped == 2
assert list(kept["id"]) == ["a", "b", "c"]
assert list(kept["marker"]) == [1, 2, 4]
print("Test 4 passed: duplicate ids are dropped and the first row is kept")

# Test 5: column order for log loss, with a hand calculation.
# A model whose classes_ are in a different order: white, draw, black.
model_classes = ["white", "draw", "black"]
raw = np.array([
    [0.6, 0.1, 0.3],    # really white: probability 0.6
    [0.2, 0.1, 0.7],    # really black: probability 0.7
    [0.3, 0.4, 0.3],    # really a draw: probability 0.4
])
y = np.array(["white", "black", "draw"])
ordered = reorder_proba(raw, model_classes)
assert list(ordered[0]) == [0.3, 0.1, 0.6]           # now black, draw, white
by_hand = -(math.log(0.6) + math.log(0.7) + math.log(0.4)) / 3
assert abs(multiclass_log_loss(y, ordered) - by_hand) < 1e-12
# Without reordering the answer would be wrong: that is the old bug.
assert abs(multiclass_log_loss(y, raw) - by_hand) > 0.1
# Brier: sum over classes of (p - y)^2, averaged over games.
brier_by_hand = ((0.3**2 + 0.1**2 + 0.4**2) + (0.3**2 + 0.1**2 + 0.2**2)
                 + (0.3**2 + 0.6**2 + 0.3**2)) / 3
assert abs(multiclass_brier(y, ordered) - brier_by_hand) < 1e-12
assert CLASSES == ["black", "draw", "white"]
print(f"Test 5 passed: log loss {by_hand:.4f} and Brier {brier_by_hand:.4f} match the "
      "hand calculation after reordering")


# Test 6: number labels (what our models use) and name labels (the saved
# model) both end up in black, draw, white order.
class FakeModel:
    def __init__(self, classes, row):
        self.classes_ = np.array(classes)
        self.row = row

    def predict_proba(self, X):
        return np.array([self.row] * len(X))


assert list(encode_labels(["white", "black", "draw"])) == [2, 0, 1]
by_number = FakeModel([2, 1, 0], [0.6, 0.1, 0.3])          # white, draw, black
by_name = FakeModel(["white", "draw", "black"], [0.6, 0.1, 0.3])
assert list(model_proba(by_number, [0])[0]) == [0.3, 0.1, 0.6]
assert list(model_proba(by_name, [0])[0]) == [0.3, 0.1, 0.6]
print("Test 6 passed: number and name class labels both map to black, draw, white")
