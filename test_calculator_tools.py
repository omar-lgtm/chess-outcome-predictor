"""
Quick checks for calculator_tools and the "Your personal odds" section,
on made-up games only (no real games, no names).

Run it with:  python test_calculator_tools.py
"""

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis_page import GAMES_KEY, games_fingerprint
from calculator_tools import (
    bootstrap_param_sets, default_rating, general_odds, option_summary,
    personal_prediction, similar_record, support_range,
)
from report_tools import add_expected, wilson_interval


def make_games(n, seed, colours=("white", "black")):
    """
    n made-up games, one per day in date order. The score follows a known
    logistic curve of the rating gap. Ratings drift upwards over time.
    """
    rng = np.random.default_rng(seed)
    rating_diff = rng.integers(-200, 201, size=n)
    p = 1 / (1 + np.exp(-(0.1 + 1.0 * rating_diff / 100)))
    u = rng.random(n)
    score = np.where(u < p - 0.03, 1.0, np.where(u < p + 0.03, 0.5, 0.0))
    my_rating = 1200 + np.arange(n) + rng.integers(-20, 21, size=n)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "colour": rng.choice(list(colours), size=n),
        "my_rating": my_rating,
        "opp_rating": my_rating - rating_diff,
        "rating_diff": rating_diff,
        "score": score,
        "time_group": rng.choice(["1 min or less", "3-5 min"], size=n),
    })


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------

# Test 1: default_rating.
games = pd.DataFrame({
    "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
    "time_group": ["2 min", "3-5 min", "2 min", "3-5 min"],
    "my_rating": [1000, 1100, 1050, 1200],
})
assert default_rating(games, "2 min") == 1050          # latest 2 min game
assert default_rating(games, "3-5 min") == 1200
assert default_rating(games, "1 min or less") == 1200  # no games: latest overall
assert default_rating(games) == 1200                   # no group: latest overall
assert default_rating(games.iloc[0:0]) is None          # empty table
print("Test 1 passed: default_rating picks the latest rating, with fallbacks")


# Test 2: general_odds with a fake model that records what it receives.
class FakeModel:
    classes_ = np.array(["black", "draw", "white"])

    def __init__(self):
        self.received = []

    def predict_proba(self, X):
        self.received.append(float(X["rating_diff"].iloc[0]))
        return np.array([[0.3, 0.1, 0.6]])


fake = FakeModel()
as_white = general_odds(fake, 1600, 1500, "white")
as_black = general_odds(fake, 1600, 1500, "black")
assert fake.received == [100.0, -100.0]     # black: white minus black is -100
for odds in (as_white, as_black):
    assert abs(odds["win"] + odds["draw"] + odds["loss"] - 1) < 1e-12
assert as_white["win"] == 0.6 and as_white["loss"] == 0.3
assert as_black["win"] == 0.3 and as_black["loss"] == 0.6
assert abs(as_white["expected_score"] - 0.65) < 1e-12
print("Test 2 passed: general_odds flips the rating difference for Black")

# Test 3: similar_record on a hand-made table.
games = pd.DataFrame({
    "colour":      ["white"] * 6 + ["black", "white"],
    "time_group":  ["2 min"] * 6 + ["2 min", "3-5 min"],
    "rating_diff": [0, 10, -20, 40, 60, 200, 0, 0],
    "score":       [1, 0.5, 0, 1, 1, 1, 1, 1],
})
# Window 50 around gap 0: gaps 0, 10, -20 and 40 match (4 games, 2.5 points).
rec = similar_record(games, "white", "2 min", 0, window=50, min_games=3)
assert rec["n"] == 4 and rec["score"] == 2.5 / 4
assert (rec["low"], rec["high"]) == wilson_interval(2.5, 4)
assert rec["low"] <= rec["score"] <= rec["high"]
assert rec["window_used"] == 50 and not rec["widened"] and rec["enough"]
# Needing 5 games: widen to 75, which adds the gap-60 game.
rec = similar_record(games, "white", "2 min", 0, window=50, min_games=5)
assert rec["n"] == 5 and rec["window_used"] == 75 and rec["widened"] and rec["enough"]
# Needing 10 games: no window is enough; the widest (150) is reported.
rec = similar_record(games, "white", "2 min", 0, window=50, min_games=10)
assert rec["n"] == 5 and rec["window_used"] == 150 and not rec["enough"]
# Any time group also counts the 3-5 min game.
rec = similar_record(games, "white", None, 0, window=50, min_games=3)
assert rec["n"] == 5
print("Test 3 passed: similar_record counts, widens and flags too few games")

# Test 4: bootstrap is repeatable, and the point matches add_expected.
games = make_games(400, seed=1)
first = bootstrap_param_sets(games, n_boot=100, seed=7)
second = bootstrap_param_sets(games, n_boot=100, seed=7)
assert first == second
with_expected, _ = add_expected(games)
row = with_expected.iloc[(with_expected["rating_diff"] - 20).abs().argmin()]
prediction = personal_prediction(games, first, int(row["rating_diff"]), row["colour"])
assert prediction["status"] == "ok"
assert abs(prediction["point"] - row["expected"]) < 1e-9
print("Test 4 passed: same seed, same fits; point equals add_expected exactly")

# Test 5: every status of personal_prediction.
few = make_games(150, seed=2)
result = personal_prediction(few, None, 0, "white")
assert result["status"] == "too_few_games" and "50 more" in result["message"]

low, high = support_range(games)
result = personal_prediction(games, first, int(high) + 50, "white")
assert result["status"] == "outside_range" and result["point"] is None

only_white = make_games(300, seed=3, colours=("white",))
white_info = bootstrap_param_sets(only_white, n_boot=50)
result = personal_prediction(only_white, white_info, 0, "black")
assert result["status"] == "no_games_for_colour"

result = personal_prediction(games, first, 0, "white")
assert result["status"] == "ok"
assert result["low"] <= result["point"] <= result["high"]
print("Test 5 passed: too few games, outside range, missing colour and ok")

# Test 6: option_summary.
summary = option_summary(games)
assert summary["n_games"] == 400
assert sum(count for _, count in summary["time_groups"]) == 400
assert summary["time_groups"][0][1] >= summary["time_groups"][-1][1]
assert {colour for colour, _ in summary["colours"]} == {"white", "black"}
print("Test 6 passed: option_summary lists colours and time groups with counts")


# ---------------------------------------------------------------------------
# Screen tests with Streamlit's AppTest
# ---------------------------------------------------------------------------

def app_with_games(games):
    """The app with made-up games placed in session state before the run."""
    at = AppTest.from_file("app.py", default_timeout=120)
    if games is not None:
        at.session_state[GAMES_KEY] = {
            "games": games,
            "info": {"games_used": len(games), "clean_games": len(games)},
            "fingerprint": games_fingerprint(games),
        }
    return at.run()


def shown_text(at):
    return " ".join(str(el.value) for kind in ("info", "warning", "caption")
                    for el in getattr(at, kind))


# Test 7: no games yet.
at = app_with_games(None)
assert not at.exception
assert "analyse your games" in shown_text(at).lower()
print("Test 7 passed: with no games, the note appears and nothing raises")

# Test 8: 300 games, all three columns.
at = app_with_games(make_games(300, seed=4))
assert not at.exception
subheaders = [s.value for s in at.subheader]
for title in ("General model", "Your personal curve", "Your record in similar games"):
    assert title in subheaders
labels = [m.label for m in at.metric]
assert "Expected score" in labels
# Win, draw and loss are shown as a stacked bar with a written legend.
bars = [m.value for m in at.markdown if 'role="img"' in str(m.value)]
assert any('aria-label="Win ' in bar for bar in bars)
print("Test 8 passed: with 300 games, all three columns appear")

# Test 9: 100 games, too few for the personal curve.
at = app_with_games(make_games(100, seed=5))
assert not at.exception
assert "needs at least 200 games" in shown_text(at)
print("Test 9 passed: with 100 games, the personal column says more games are needed")

# Test 10: a gap outside the range.
at = app_with_games(make_games(300, seed=4))
opponent = at.number_input(key="odds_opp_Any_white")
opponent.set_value(max(100, opponent.value - 600)).run()
assert not at.exception
assert "outside the range" in shown_text(at)
print("Test 10 passed: a gap outside the range shows the range message")
