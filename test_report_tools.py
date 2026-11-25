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
    add_expected, add_groupings, build_report, clean_games, group_report,
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

# Test 8: opening_family handles both naming styles.
games = pd.DataFrame({
    "opening": [
        "Imaginary Defense: Long Variation, Sub Line",   # Lichess style
        "Imaginary Defense",                             # Lichess, no colon
        "Pretend-Gambit-Accepted-Main-Line",             # Chess.com style
    ],
    "date": pd.to_datetime(["2024-01-01"] * 3),
    "base_seconds": [180] * 3,
})
games = add_groupings(games)
assert list(games["opening_family"]) == [
    "Imaginary Defense", "Imaginary Defense", "Pretend-Gambit",
]
print("Test 8 passed: opening_family works for colon and hyphen names")


# Tests 9 to 11: build_report on a made-up PGN file.
def make_pgn(n_games, seed=7):
    """
    Build PGN text with n_games invented games for the player "Alice".
    30 invented opponents (Opp01 to Opp30) take turns, so each appears
    fewer than 20 times in 300 games. Ratings, dates, colours, results and
    time controls all vary.
    """
    rng = np.random.default_rng(seed)
    openings = ["Imaginary Defense: Main Line", "Pretend Opening: Side Line",
                "Made-Up-Gambit-Declined"]
    time_controls = ["60", "180+2", "600"]
    parts = []
    for i in range(n_games):
        opponent = f"Opp{i % 30 + 1:02d}"
        alice_elo = 1500 + i + int(rng.integers(-20, 21))
        opp_elo = alice_elo + int(rng.integers(-200, 201))
        if i % 2 == 0:
            white, black, white_elo, black_elo = "Alice", opponent, alice_elo, opp_elo
        else:
            white, black, white_elo, black_elo = opponent, "Alice", opp_elo, alice_elo
        result = rng.choice(["1-0", "0-1", "1/2-1/2"], p=[0.47, 0.47, 0.06])
        year = 2022 + i // 100                 # 2022, 2023 and 2024
        month = i % 12 + 1
        day = i % 28 + 1
        parts.append(
            f'[Event "Made-up game"]\n'
            f'[White "{white}"]\n[Black "{black}"]\n'
            f'[Result "{result}"]\n'
            f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n'
            f'[UTCDate "{year}.{month:02d}.{day:02d}"]\n'
            f'[TimeControl "{time_controls[i % 3]}"]\n'
            f'[ECO "A00"]\n[Opening "{openings[i % 3]}"]\n'
            f'\n1. e4 e5 {result}\n\n'
        )
    return "".join(parts)


report = build_report(make_pgn(300))
assert set(report) == {
    "parse_info", "clean_info", "params", "overall", "opening_table",
    "time_table", "year_table", "rating_months",
}
# The player's name is returned as written in the file.
assert report["parse_info"]["player"] == "Alice"
assert report["clean_info"] == {"total": 300, "kept": 300, "set_aside": 0}
assert report["params"]["c"] is not None          # both colours, 150 each
assert list(report["year_table"]["group"]) == [2022, 2023, 2024]
# No opponent name may appear anywhere in the result.
everything = " ".join(
    value.to_string() if isinstance(value, pd.DataFrame) else str(value)
    for value in report.values()
)
assert "Opp" not in everything
print("Test 9 passed: build_report returns all parts and no opponent names")

# Test 10: a 5-game file is too small for a report.
try:
    build_report(make_pgn(5))
    raise AssertionError("expected a ValueError for a 5-game file")
except ValueError:
    pass
print("Test 10 passed: a 5-game file raises ValueError")

# Test 11: an empty string has no games.
try:
    build_report("")
    raise AssertionError("expected a ValueError for an empty file")
except ValueError:
    pass
print("Test 11 passed: an empty file raises ValueError")
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
