"""
Logic for the "Your personal odds" calculator (no Streamlit code here).

Everything works on the clean games table made by the "Analyse your games"
tab (pgn_tools.parse_pgn, then report_tools.clean_games and add_groupings).
In that table, rating_diff is my rating minus my opponent's rating and
colour is "white" or "black". In this file, "gap" means the same thing as
rating_diff: my rating minus the opponent's rating.
"""

import numpy as np
import pandas as pd

from eval_tools import fit_or_none
from report_tools import MIN_GAMES_PER_COLOUR, predict_expected, wilson_interval

MIN_GAMES = 200          # fewest clean games before the personal curve is shown
MIN_SIMILAR = 20         # fewest similar games before a similar-games record is shown
GAP_WINDOW = 50          # similar games: rating gap within this many points
WIDER_WINDOWS = (75, 100, 150)   # windows to try, in order, if too few games match
SUPPORT_LOW_PCT = 5      # the personal curve is only shown between these
SUPPORT_HIGH_PCT = 95    # percentiles of the player's own rating gaps
MIN_PER_COLOUR = 30      # colour term only if each colour has this many games

# The fitting code in report_tools applies the same colour rule with its own
# constant; this check makes sure the two never drift apart.
assert MIN_PER_COLOUR == MIN_GAMES_PER_COLOUR


def default_rating(df, time_group=None):
    """
    The player's most recent rating, to pre-fill the rating box.

    Uses the latest game in the chosen time group. If no time group is
    chosen, or it has no games, uses the latest game overall.
    Returns None for an empty table.
    """
    if len(df) == 0:
        return None
    games = df.sort_values("date", kind="stable")
    if time_group is not None:
        in_group = games[games["time_group"] == time_group]
        if len(in_group) > 0:
            return int(in_group["my_rating"].iloc[-1])
    return int(games["my_rating"].iloc[-1])


def option_summary(df):
    """
    What choices make sense for this player: which colours and time groups
    appear in the file, with how many games each, most games first.
    """
    colours = df["colour"].value_counts()
    time_groups = df["time_group"].value_counts()
    return {
        "colours": [(colour, int(count)) for colour, count in colours.items()],
        "time_groups": [(group, int(count)) for group, count in time_groups.items()],
        "n_games": len(df),
    }


def support_range(df):
    """
    The range of rating gaps where the player has most of their games:
    the 5th to 95th percentile. Outside it, the personal curve would be
    guessing beyond the data, so it is not shown there.
    """
    low = float(np.percentile(df["rating_diff"], SUPPORT_LOW_PCT))
    high = float(np.percentile(df["rating_diff"], SUPPORT_HIGH_PCT))
    return low, high


def general_odds(model, my_rating, opp_rating, colour):
    """
    Win, draw and loss chances from the general Lichess model, from the
    player's side, plus the expected score (a draw counts as half a point).

    The saved model takes white rating minus black rating. So for a white
    player that is my rating minus the opponent's, and for a black player
    it is the opponent's rating minus mine.
    """
    if colour == "white":
        white_minus_black = my_rating - opp_rating
    else:
        white_minus_black = opp_rating - my_rating
    proba = model.predict_proba(pd.DataFrame({"rating_diff": [white_minus_black]}))[0]
    classes = list(model.classes_)
    p_white = float(proba[classes.index("white")])
    p_black = float(proba[classes.index("black")])
    p_draw = float(proba[classes.index("draw")])

    win, loss = (p_white, p_black) if colour == "white" else (p_black, p_white)
    return {
        "win": win,
        "draw": p_draw,
        "loss": loss,
        "expected_score": win + 0.5 * p_draw,
    }


def similar_record(df, colour, time_group, gap, window=GAP_WINDOW, min_games=MIN_SIMILAR):
    """
    The player's actual results in games like this one: same colour, same
    time group (None means any) and a rating gap within +/- window points.

    If fewer than min_games match, the window is widened step by step
    (WIDER_WINDOWS) until enough games match. If even the widest window has
    too few, its result is returned with enough = False.
    """
    games = df[df["colour"] == colour]
    if time_group is not None:
        games = games[games["time_group"] == time_group]

    def record(w):
        matching = games[(games["rating_diff"] - gap).abs() <= w]
        n = len(matching)
        points = float(matching["score"].sum())
        low, high = wilson_interval(points, n)
        return {
            "n": n,
            "score": points / n if n else None,
            "low": low,
            "high": high,
            "window_used": w,
        }

    result = record(window)
    if result["n"] < min_games:
        for wider in WIDER_WINDOWS:
            if wider <= window:
                continue          # only ever widen, never narrow
            result = record(wider)
            if result["n"] >= min_games:
                break

    result["widened"] = result["window_used"] != window
    result["enough"] = result["n"] >= min_games
    return result


def bootstrap_param_sets(df, n_boot=500, seed=42):
    """
    Fit the personal curve once on all games (point_params), and again on
    n_boot resamples of the games drawn with replacement (sets). The spread
    of the resampled curves gives the range shown in the calculator.

    The fitting code applies the colour rule (colour term only if each
    colour has at least MIN_PER_COLOUR games). A resample whose fit fails is
    skipped; if more than 10 percent fail, the range would not be reliable,
    so an error is raised. Meant to run once per uploaded file.
    """
    df = df.reset_index(drop=True)
    point_params = fit_or_none(df)
    if point_params is None:
        raise ValueError("The personal curve could not be fitted to these games.")

    rng = np.random.default_rng(seed)
    sets = []
    failures = 0
    for _ in range(n_boot):
        rows = rng.integers(0, len(df), size=len(df))
        params = fit_or_none(df.iloc[rows])
        if params is None:
            failures += 1
        else:
            sets.append(params)

    if failures > 0.1 * n_boot:
        raise ValueError(
            f"{failures} of {n_boot} resampled fits failed, which is more than "
            "10 percent, so the range would not be reliable."
        )
    return {"point_params": point_params, "sets": sets, "n_games": len(df)}


def personal_prediction(df, param_info, gap, colour):
    """
    The personal curve's expected score for one game, with a 95% range.

    status says whether a number can be shown:
    - "ok": point, low and high are filled in.
    - "too_few_games": fewer than MIN_GAMES clean games.
    - "no_games_for_colour": the file has no games with this colour.
    - "outside_range": the gap is outside the range where the player has
      most of their games (support_range).
    param_info is the output of bootstrap_param_sets; it may be None when
    there are too few games.
    """
    n = len(df)
    support_low, support_high = support_range(df) if n else (None, None)
    result = {
        "status": "ok",
        "point": None,
        "low": None,
        "high": None,
        "n_games": n,
        "support_low": support_low,
        "support_high": support_high,
        "message": "",
    }

    if n < MIN_GAMES:
        result["status"] = "too_few_games"
        result["message"] = (
            f"The personal curve needs at least {MIN_GAMES} games. You have "
            f"{n}, so {MIN_GAMES - n} more are needed."
        )
        return result

    if (df["colour"] == colour).sum() == 0:
        result["status"] = "no_games_for_colour"
        result["message"] = (
            f"Your file has no games as {colour.capitalize()}, so the personal "
            "curve cannot predict games with that colour."
        )
        return result

    if not support_low <= gap <= support_high:
        result["status"] = "outside_range"
        result["message"] = (
            f"A rating gap of {gap:+.0f} is outside the range where you have most "
            f"of your games ({support_low:+.0f} to {support_high:+.0f}), so the "
            "personal curve would only be guessing. Try a gap inside that range."
        )
        return result

    # The same formula as add_expected, through predict_expected.
    game = pd.DataFrame({"rating_diff": [gap], "colour": [colour]})
    result["point"] = float(predict_expected(game, param_info["point_params"])[0])
    with np.errstate(over="ignore"):
        estimates = [float(predict_expected(game, params)[0]) for params in param_info["sets"]]
    result["low"] = float(np.percentile(estimates, 2.5))
    result["high"] = float(np.percentile(estimates, 97.5))
    return result
