"""
Quick checks for live_model_tools, on made-up games and tiny made-up models.

Run it with:  python test_live_model_tools.py
"""

import json
import tempfile
from pathlib import Path

import chess
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from inplay_tools import feature_groups, split_rows
from live_model_tools import (
    META_FILE, POSITION_ONLY_FILE, WITH_RATINGS_FILE, build_live_rows, load_bundle,
    ordered_features, predict_curve, terminal_result,
)

GROUPS = feature_groups()
WITH_RATINGS = GROUPS["P3"] + ["ply"]
POSITION_ONLY = [c for c in GROUPS["P3"] if c not in GROUPS["P0"]] + ["ply"]
PRE_GAME = {"white_rating": 1500, "black_rating": 1450, "base_minutes": 10,
            "increment_seconds": 0, "rated": 1}

# A made-up 24-ply game (Ruy Lopez style moves) and a variation of it.
GAME = ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O "
        "h3 Nb8 d4 Nbd7 Nbd2 Bb7 Bc2 Re8")
OTHER = ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 d4 exd4 e5 Ne4 Qe2 Nc5 "
         "Bxc6 dxc6 Nxd4 O-O Nc3 Ne6 Nxe6 Bxe6")

# Test 1: rows at 5, 10, ... and never past the end of the game.
rows = build_live_rows(0, GAME, PRE_GAME, step=5, max_ply=120)
assert [r["ply"] for r in rows] == [5, 10, 15, 20]          # 24 plies: no row at 25
assert all(r["game_index"] == 0 for r in rows)
short = build_live_rows(1, "e4 e5 Nf3 Nc6 Bb5 a6 Ba4", PRE_GAME)   # 7 plies
assert [r["ply"] for r in short] == [5]
assert build_live_rows(2, "e4 e5 Ke3 Nc6 Bb5 a6", PRE_GAME) == []   # illegal Ke3
assert [r["ply"] for r in build_live_rows(3, GAME, PRE_GAME, max_ply=12)] == [5, 10]
print("Test 1 passed: rows every 5 plies, none beyond the game; bad moves give no rows")

# Test 2: leakage. GAME and OTHER share their first 10 plies only.
row_a = pd.DataFrame(build_live_rows(0, GAME, PRE_GAME))
row_b = pd.DataFrame(build_live_rows(0, OTHER, PRE_GAME))
at_10_a = row_a[row_a["ply"] == 10][WITH_RATINGS].iloc[0]
at_10_b = row_b[row_b["ply"] == 10][WITH_RATINGS].iloc[0]
assert at_10_a.equals(at_10_b)
assert not row_a[row_a["ply"] == 15][WITH_RATINGS].iloc[0].equals(
    row_b[row_b["ply"] == 15][WITH_RATINGS].iloc[0])
print("Test 2 passed: the row at ply 10 is identical when only later moves differ")

# Test 3: split by game, and GroupKFold never splits a game.
all_rows = pd.DataFrame([r for i in range(12) for r in build_live_rows(i, GAME, PRE_GAME)])
development, test = split_rows(all_rows, development_games=range(0, 9), test_games=range(9, 12))
assert set(development["game_index"]).isdisjoint(set(test["game_index"]))
for fit_idx, check_idx in GroupKFold(n_splits=3).split(
        development, groups=development["game_index"]):
    fit_games = set(development["game_index"].iloc[fit_idx])
    check_games = set(development["game_index"].iloc[check_idx])
    assert fit_games.isdisjoint(check_games)
print("Test 3 passed: no game is in both train and test, or in both parts of a fold")


# A tiny made-up bundle: two small models trained on random numbers with
# the real feature names, plus the metadata.
def tiny_model(columns, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(150, len(columns))), columns=columns)
    y = rng.integers(0, 3, size=150)            # 0 black, 1 draw, 2 white
    return HistGradientBoostingClassifier(max_iter=5, random_state=0).fit(X, y)


meta = {
    "classes": {"0": "black", "1": "draw", "2": "white"},
    "features_with_ratings": WITH_RATINGS,
    "features_position_only": POSITION_ONLY,
    "ply_step": 5, "max_ply": 120, "sklearn_version": sklearn.__version__,
}
bundle = {"with_ratings": tiny_model(WITH_RATINGS, 1),
          "position_only": tiny_model(POSITION_ONLY, 2), "meta": meta}

# Test 4: save, reload, identical predictions; load_bundle checks the version.
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    joblib.dump(bundle["with_ratings"], folder / WITH_RATINGS_FILE)
    joblib.dump(bundle["position_only"], folder / POSITION_ONLY_FILE)
    (folder / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    loaded = load_bundle(folder)
    X = pd.DataFrame(np.random.default_rng(3).normal(size=(20, len(WITH_RATINGS))),
                     columns=WITH_RATINGS)
    assert np.array_equal(loaded["with_ratings"].predict_proba(X),
                          bundle["with_ratings"].predict_proba(X))
    (folder / META_FILE).write_text(json.dumps({**meta, "sklearn_version": "0.0.1"}),
                                    encoding="utf-8")
    try:
        load_bundle(folder)
        raise AssertionError("load_bundle accepted a different scikit-learn version")
    except RuntimeError:
        pass
print("Test 4 passed: a saved model reloads with identical predictions; version mismatch is refused")

# Test 5: predict_curve.
curve = predict_curve(GAME, 1500, 1450, 10, 0, 1, bundle=bundle)
assert list(curve["ply"]) == list(range(0, 25))              # 0 to the last ply
assert np.allclose(curve[["p_black", "p_draw", "p_white"]].sum(axis=1), 1)
assert np.allclose(curve["expected_score_white"], curve["p_white"] + 0.5 * curve["p_draw"])
# Move numbers and a result marker are accepted.
numbered = predict_curve("1. e4 e5 2. Nf3 Nc6 3.Bb5 a6 1-0", 1500, 1450, bundle=bundle)
assert list(numbered["ply"]) == list(range(0, 7))
for bad in ("", "   ", "e4 e5 Ke3", "1. e4 zz9"):
    try:
        predict_curve(bad, 1500, 1450, bundle=bundle)
        raise AssertionError(f"predict_curve accepted bad input {bad!r}")
    except ValueError:
        pass
# A missing rating switches to the position-only model.
with_r = predict_curve(GAME, 1500, 1450, bundle=bundle)
without = predict_curve(GAME, 1500, None, bundle=bundle)
assert not np.allclose(with_r[["p_black", "p_draw", "p_white"]], without[["p_black", "p_draw", "p_white"]])
position_only_direct = predict_curve(GAME, None, None, bundle=bundle)
assert np.allclose(without[["p_black", "p_draw", "p_white"]],
                   position_only_direct[["p_black", "p_draw", "p_white"]])
print("Test 5 passed: one row per ply, probabilities sum to 1, bad input is refused, "
      "a missing rating uses the position-only model")

# Test 6: the feature-order guard.
shuffled = list(reversed(WITH_RATINGS))
try:
    ordered_features(pd.DataFrame(columns=WITH_RATINGS), shuffled, bundle["with_ratings"])
    raise AssertionError("the guard accepted a different column order")
except ValueError:
    pass
try:
    ordered_features(pd.DataFrame(columns=WITH_RATINGS[:-1]), WITH_RATINGS)
    raise AssertionError("the guard accepted a missing column")
except ValueError:
    pass
print("Test 6 passed: the feature-order guard refuses a wrong order or a missing column")

# Test 7: beyond_training is True only for plies after the stored max ply,
# and missing time-control values reach the model as NaN.
assert not curve["beyond_training"].any()                    # 24 plies, max_ply 120
short_meta = {**bundle, "meta": {**meta, "max_ply": 10}}
flagged = predict_curve(GAME, 1500, 1450, 10, 0, 1, bundle=short_meta)
assert flagged["beyond_training"].dtype == bool
assert list(flagged["ply"][flagged["beyond_training"]]) == list(range(11, 25))
assert np.allclose(flagged[["p_black", "p_draw", "p_white"]], curve[["p_black", "p_draw", "p_white"]])
missing_time = predict_curve(GAME, 1500, 1450, bundle=bundle)   # no time control or rated
assert np.allclose(missing_time[["p_black", "p_draw", "p_white"]].sum(axis=1), 1)
print("Test 7 passed: beyond_training flags only plies after max_ply; missing values are allowed")

# Test 8: finished positions get the real result; before_training marks plies 0 to 4.
OPERA = ("1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5 6.Bc4 Nf6 7.Qb3 Qe7 "
         "8.Nc3 c6 9.Bg5 b5 10.Nxb5 cxb5 11.Bxb5+ Nbd7 12.O-O-O Rd8 13.Rxd7 Rxd7 "
         "14.Rd1 Qe6 15.Bxd7+ Nxd7 16.Qb8+ Nxb8 17.Rd8#")
# Sam Loyd's ten-move stalemate (legal; Black to move has no moves and is not in check).
LOYD = ("1.e3 a5 2.Qh5 Ra6 3.Qxa5 h5 4.Qxc7 Rah6 5.h4 f6 6.Qxd7+ Kf7 7.Qxb7 Qd3 "
        "8.Qxb8 Qh7 9.Qxc8 Kg6 10.Qe6")
opera = predict_curve(OPERA, bundle=bundle)
last = opera.iloc[-1]
assert last["terminal_result"] == "white"
assert (last["p_white"], last["p_draw"], last["p_black"]) == (1.0, 0.0, 0.0)
assert last["expected_score_white"] == 1.0
assert opera["terminal_result"].iloc[:-1].isna().all()     # only the final ply is finished
loyd = predict_curve(LOYD, bundle=bundle)
assert loyd["terminal_result"].iloc[-1] == "draw" and loyd["p_draw"].iloc[-1] == 1.0
assert loyd["expected_score_white"].iloc[-1] == 0.5
assert terminal_result(chess.Board("8/8/4k3/8/8/3K4/8/8 w - - 0 1")) == "draw"   # two kings
assert terminal_result(chess.Board()) is None
assert list(opera["before_training"].iloc[:6]) == [True] * 5 + [False]
assert not opera["before_training"].iloc[5:].any()
print("Test 8 passed: checkmate, stalemate and bare kings give the real result; "
      "before_training is True for plies 0 to 4 only")

# Test 9: with the saved models, non-terminal rows are unchanged from the
# values printed before this change (Opera Game, no ratings, 3 decimals).
real = predict_curve(OPERA, bundle=load_bundle("."))
expected = {10: (0.525, 0.048, 0.427), 20: (0.575, 0.038, 0.388), 30: (0.610, 0.034, 0.356)}
for ply, values in expected.items():
    row = real.iloc[ply]
    assert tuple(round(row[c], 3) for c in ("p_white", "p_draw", "p_black")) == values, ply
print("Test 9 passed: the saved models give the same numbers as before at plies 10, 20 and 30")
