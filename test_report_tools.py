"""
Quick checks for report_tools.

Part 1 runs tests on small made-up tables (no real games, no names).
Part 2 runs the whole pipeline on the sample file and prints the reports.
Nothing is saved to a file.

Run it with:  python test_report_tools.py
"""

import numpy as np
import pandas as pd

from pgn_tools import parse_pgn
from report_tools import (
    add_expected, add_groupings, clean_games, group_report,
    rating_by_month, wilson_interval,
)

SAMPLE_FILE = "data/chipsa69-white.pgn"


# ---------------------------------------------------------------------------
# Part 1: tests on made-up data
# ---------------------------------------------------------------------------

# Test 1: Wilson interval for 50 points out of 100 games.
low, high = wilson_interval(50, 100)
assert abs(low - 0.404) < 0.005 and abs(high - 0.596) < 0.005
print(f"Test 1 passed: wilson_interval(50, 100) = ({low:.3f}, {high:.3f})")

# Test 2: no games at all gives the widest possible interval.
assert wilson_interval(0, 0) == (0.0, 1.0)
print("Test 2 passed: wilson_interval(0, 0) = (0.0, 1.0)")

# Test 3: the fit recovers a known curve.
# We pick true values for a, b and c, generate random games from them,
# and check that add_expected finds b again.
rng = np.random.default_rng(42)
true_a, true_b, true_c = 0.1, 0.8, 0.3
n = 400
rating_diff = rng.integers(-200, 201, size=n)
colour = rng.choice(["white", "black"], size=n)
is_white = (colour == "white").astype(float)
p_win = 1 / (1 + np.exp(-(true_a + true_b * rating_diff / 100 + true_c * is_white)))
score = (rng.random(n) < p_win).astype(float)   # 1 = win, 0 = loss
games = pd.DataFrame({"rating_diff": rating_diff, "colour": colour, "score": score})
_, params = add_expected(games)
assert params["b"] > 0
assert abs(params["b"] - true_b) < 0.3
assert params["c"] is not None  # both colours have 30+ games, so c is used
print(f"Test 3 passed: fitted b = {params['b']:.3f} (true value {true_b})")

# Test 4: opening A scores 80%, opening B 40%, with the same rating gaps.
# Overall the model expects about 60%, so A is above and B below that.
gaps = np.linspace(-100, 100, 200).round()
scores_a = np.array([1.0] * 160 + [0.0] * 40)    # 80% won
scores_b = np.array([1.0] * 80 + [0.0] * 120)    # 40% won
rng.shuffle(scores_a)
rng.shuffle(scores_b)
games = pd.DataFrame({
    "opening": ["A"] * 200 + ["B"] * 200,
    "rating_diff": np.concatenate([gaps, gaps]),
    "colour": "white",
    "score": np.concatenate([scores_a, scores_b]),
})
games, params = add_expected(games)
assert params["c"] is None  # only one colour, so c is left out
report = group_report(games, "opening").set_index("group")
assert report.loc["A", "verdict"] == "above expectation"
assert report.loc["B", "verdict"] == "below expectation"
print("Test 4 passed: opening A is above expectation, B is below")

# Test 5: a group with only 10 games gets "too few games".
games = pd.DataFrame({
    "opening": ["C"] * 10,
    "score": [1.0] * 10,
    "expected": [0.5] * 10,
})
report = group_report(games, "opening")
assert report.loc[0, "games"] == 10
assert report.loc[0, "verdict"] == "too few games"
print("Test 5 passed: a 10-game group is marked 'too few games'")

# Test 6: clean_games drops flagged rows and the counts add up.
games = pd.DataFrame({
    "stale_rating":    [False, True,  False, True,  False],
    "repeat_opponent": [False, False, True,  True,  False],
})
clean, info = clean_games(games)
assert len(clean) == 2
assert not clean["stale_rating"].any() and not clean["repeat_opponent"].any()
assert info == {"total": 5, "kept": 2, "set_aside": 3}
assert info["kept"] + info["set_aside"] == info["total"]
print("Test 6 passed: clean_games keeps 2 of 5 rows")

# Test 7: add_groupings on hand-written rows.
games = pd.DataFrame({
    "opening": [
        "Scandinavian-Defense-Mieses-Kotrc-Variation",
        "Kings-Pawn",
        "Polish",
        "French-Defense",
    ],
    "date": pd.to_datetime(["2023-05-01", "2024-01-01", "2024-12-31", "2025-06-15"]),
    "base_seconds": [60, 120, 180, 600],
})
games = add_groupings(games)
assert list(games["opening_family"]) == [
    "Scandinavian-Defense", "Kings-Pawn", "Polish", "French-Defense",
]
assert list(games["year"]) == [2023, 2024, 2024, 2025]
assert list(games["time_group"]) == [
    "1 min or less", "2 min", "3-5 min", "more than 5 min",
]
print("Test 7 passed: add_groupings gives the expected family, year and time group")
print()


# ---------------------------------------------------------------------------
# Part 2: the whole pipeline on the sample file
# ---------------------------------------------------------------------------

with open(SAMPLE_FILE, encoding="utf-8") as f:
    df, _ = parse_pgn(f.read())   # the info dict holds the username, so it is not printed

df, clean_info = clean_games(df)
df = add_groupings(df)
df, params = add_expected(df)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 3)

print("Clean games:", clean_info)
print("Fitted parameters:", params)
print()
print("By opening family (top 12 by games):")
print(group_report(df, "opening_family", min_games=40).head(12))
print()
print("By time group:")
print(group_report(df, "time_group"))
print()
print("By year:")
print(group_report(df, "year"))
print()
monthly = rating_by_month(df)
print("Rating by month (first 3 and last 3):")
print(pd.concat([monthly.head(3), monthly.tail(3)]))
