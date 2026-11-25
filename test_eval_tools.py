"""
Quick checks for eval_tools (and the optional weights in report_tools),
on made-up games only (no real games, no names).

Run it with:  python test_eval_tools.py
"""

import numpy as np
import pandas as pd

import eval_tools
from eval_tools import (
    CV_NAME, bootstrap_prediction_interval, compare_with_elo_and_recency,
    elo_expected, learning_curve, paired_bootstrap_diff, recency_name,
    recency_weights,
)
from report_tools import fit_expected_params


def make_games(n, seed, slope=1.0, new_slope=None):
    """
    n made-up games, one per day in date order, both colours. The score
    follows a known logistic curve of rating_diff with the given slope.
    If new_slope is given, the newer half of the games uses it instead.
    """
    rng = np.random.default_rng(seed)
    rating_diff = rng.integers(-300, 301, size=n)
    slopes = np.full(n, slope)
    if new_slope is not None:
        slopes[n // 2:] = new_slope
    p = 1 / (1 + np.exp(-(0.1 + slopes * rating_diff / 100)))
    u = rng.random(n)
    # Win if u < p - 0.03, draw in a thin band around it, loss otherwise.
    score = np.where(u < p - 0.03, 1.0, np.where(u < p + 0.03, 0.5, 0.0))
    return pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "colour": rng.choice(["white", "black"], size=n),
        "rating_diff": rating_diff,
        "score": score,
    })


# Test 1: the Elo formula.
assert elo_expected(0) == 0.5
assert abs(elo_expected(400) - 0.909) < 0.001
assert abs(elo_expected(100) + elo_expected(-100) - 1) < 1e-12
assert np.allclose(elo_expected(np.array([0, 400])), [0.5, 10 / 11])
print(f"Test 1 passed: elo_expected(0) = 0.5, elo_expected(400) = {elo_expected(400):.4f}")

# Test 2: recency weights halve every half-life.
# Three dates: 200 days, 100 days and 0 days before 2025-01-01.
dates = pd.Timestamp("2025-01-01") - pd.to_timedelta([200, 100, 0], unit="D")
weights = recency_weights(dates, half_life_days=100)
assert weights[2] == 1.0
assert abs(weights[1] - 0.5) < 1e-12
assert abs(weights[0] - 0.25) < 1e-12
assert ((weights > 0) & (weights <= 1)).all()
print("Test 2 passed: recency weights are 1.0, 0.5 and 0.25 at 0, 1 and 2 half-lives")

# Test 3: weights in the fitting code.
games = make_games(600, seed=21)
unweighted = fit_expected_params(games)
equal_weights = fit_expected_params(games, sample_weight=np.full(600, 0.7))
for key in ("a", "b", "c"):
    assert abs(unweighted[key] - equal_weights[key]) < 1e-6
# Numbers from the same fit made with the code BEFORE weights were added.
before = {"a": -0.06178650417475816, "b": 1.213008141134838, "c": -0.056726046152977175}
for key in ("a", "b", "c"):
    assert abs(unweighted[key] - before[key]) < 1e-12
print("Test 3 passed: equal weights match the unweighted fit, which matches the old code")

# Test 4: compare_with_elo_and_recency on 600 made-up games.
half_lives = (180, 365, 730)
results, info = compare_with_elo_and_recency(games, half_lives=half_lives)
expected_names = (
    {"Constant", "Elo (scale 400)", "Personal curve", CV_NAME}
    | {recency_name(h) for h in half_lives}
)
assert set(results["predictor"]) == expected_names
assert len(results) == len(expected_names) == 7
assert info["train_end"] <= info["test_start"]
assert info["train_games"] + info["test_games"] == 600
assert info["chosen_half_life"] in half_lives
assert set(info["test_predictions"].columns) == expected_names
assert len(info["test_predictions"]) == len(info["test_actual"]) == info["test_games"]
print(f"Test 4 passed: 7 predictors, CV chose a half-life of {info['chosen_half_life']} days")

# Test 5: the relationship changes halfway through (weak, then strong).
# Giving recent games more weight should predict the newer games better.
drifting = make_games(1000, seed=8, slope=0.3, new_slope=2.0)
results, info = compare_with_elo_and_recency(drifting, half_lives=(60, 180, 365))
brier = dict(zip(results["predictor"], results["brier"]))
assert brier[CV_NAME] < brier["Personal curve"]
print(f"Test 5 passed: recency weighted Brier {brier[CV_NAME]:.4f} beats "
      f"unweighted {brier['Personal curve']:.4f} (half-life {info['chosen_half_life']})")

# Test 6: paired bootstrap.
rng = np.random.default_rng(4)
actual = rng.choice([0.0, 1.0], size=300)
pred_b = np.full(300, 0.5)
same = paired_bootstrap_diff(pred_b, pred_b, actual)
assert same == {"mean_diff": 0.0, "low": 0.0, "high": 0.0}
# Made-up, clearly better: leans towards the real result, with random noise
# so its advantage differs from game to game (sometimes it is worse).
pred_a = np.clip(np.where(actual == 1.0, 0.7, 0.3) + rng.normal(0, 0.25, size=300), 0, 1)
better = paired_bootstrap_diff(pred_a, pred_b, actual)
assert better["mean_diff"] < 0 and better["high"] < 0
assert better["low"] < better["high"]              # a real range, not one value
print(f"Test 6 passed: a clearly better predictor has a range entirely below 0 "
      f"({better['low']:.4f} to {better['high']:.4f})")

# Test 7: bootstrap prediction interval.
source = make_games(2000, seed=30)
small = bootstrap_prediction_interval(source.iloc[:100], 100, "white", n_boot=300)
large = bootstrap_prediction_interval(source, 100, "white", n_boot=300)
for interval in (small, large):
    assert interval["low"] <= interval["point"] <= interval["high"]
assert small["n_games"] == 100 and large["n_games"] == 2000
assert (large["high"] - large["low"]) < (small["high"] - small["low"])
again = bootstrap_prediction_interval(source.iloc[:100], 100, "white", n_boot=300)
assert again == small
weighted = bootstrap_prediction_interval(source, 100, "white", n_boot=100, half_life_days=365)
assert weighted["low"] <= weighted["point"] <= weighted["high"]
print(f"Test 7 passed: range width {small['high'] - small['low']:.3f} with 100 games, "
      f"{large['high'] - large['low']:.3f} with 2000, same seed gives the same result")

# Test 8: learning curve on 1500 made-up games (1050 train, 450 test).
games = make_games(1500, seed=12)
curve, info = learning_curve(games)
by_n = curve.set_index("n")
assert list(curve["n"]) == [50, 100, 200, 400, 800, 1050]   # 1600 dropped, 1050 added
assert by_n.loc[1050, "personal_mean_brier"] < by_n.loc[50, "personal_mean_brier"]
band = by_n["personal_p90"] - by_n["personal_p10"]
assert band.loc[1050] < band.loc[50]
assert band.loc[800] < band.loc[50]          # also for the largest repeated size
assert by_n.loc[50, "repeats_ok"] + by_n.loc[50, "repeats_failed"] == 30
assert by_n.loc[1050, "repeats_ok"] == 1
again, _ = learning_curve(games)
assert again.equals(curve)
print(f"Test 8 passed: mean Brier {by_n.loc[50, 'personal_mean_brier']:.4f} at n=50, "
      f"{by_n.loc[1050, 'personal_mean_brier']:.4f} at n=1050; band "
      f"{band.loc[50]:.4f} -> {band.loc[1050]:.4f}; same seed, same curve")

# Test 9: sizes larger than the train set are dropped, full size comes last.
curve, info = learning_curve(make_games(600, seed=13), sizes=(50, 420, 5000), n_repeats=5)
assert info["train_games"] == 420
assert list(curve["n"]) == [50, 420]      # 420 is the full train size, listed once
print("Test 9 passed: too-large sizes are dropped and the full train size is last")

# Test 10: no fitting on test rows. Swap in a "spy" that records every row
# the fitting code receives, run the learning curve, then put the original back.
recorded_index, recorded_dates = [], []
original_fit = eval_tools.fit_expected_params


def spy_fit(games, sample_weight=None):
    recorded_index.extend(games.index)
    recorded_dates.extend(games["date"])
    return original_fit(games, sample_weight)


shuffled = make_games(600, seed=14).sample(frac=1, random_state=0)   # input order mixed up
eval_tools.fit_expected_params = spy_fit
try:
    curve, info = learning_curve(shuffled, sizes=(50, 100), n_repeats=5)
finally:
    eval_tools.fit_expected_params = original_fit
assert len(recorded_index) > 0
assert set(recorded_index) <= set(range(info["train_games"]))
assert max(recorded_dates) <= pd.Timestamp(info["train_end"])
assert max(recorded_dates) < pd.Timestamp(info["test_start"])
print(f"Test 10 passed: {len(recorded_index)} rows went into fits, all of them train rows")

# Test 11: a fake general model that always says 0.5, on 50/50 wins and losses.
coin = make_games(600, seed=15)
coin["score"] = np.random.default_rng(15).choice([0.0, 1.0], size=600)
curve, info = learning_curve(coin, sizes=(50,), n_repeats=5,
                             general_expected=lambda d: np.full(len(d), 0.5))
assert abs(info["general_brier"] - 0.25) < 1e-12
curve, info = learning_curve(coin, sizes=(50,), n_repeats=5)
assert info["general_brier"] is None and info["first_n_beating_general"] is None
print("Test 11 passed: general model always saying 0.5 scores 0.25; None when not given")

# Test 12: fits that fail are counted, not crashed on.
# 95% of games are wins, so many random 5-game subsets are all wins.
mostly_wins = make_games(600, seed=16)
mostly_wins["score"] = np.where(np.random.default_rng(16).random(600) < 0.95, 1.0, 0.0)
curve, info = learning_curve(mostly_wins, sizes=(5,), n_repeats=30)
row = curve.set_index("n").loc[5]
assert row["repeats_failed"] > 0
assert row["repeats_ok"] + row["repeats_failed"] == 30
# Every train game a win: every fit fails, including the full train size.
all_wins = make_games(600, seed=17)
all_wins["score"] = 1.0
curve, info = learning_curve(all_wins, sizes=(5,), n_repeats=10)
assert list(curve["repeats_ok"]) == [0, 0]
assert list(curve["repeats_failed"]) == [10, 1]
assert curve["personal_mean_brier"].isna().all()
assert info["first_n_beating_elo"] is None
print(f"Test 12 passed: {int(row['repeats_failed'])} of 30 tiny fits failed and were "
      f"counted; an all-wins file does not crash")
