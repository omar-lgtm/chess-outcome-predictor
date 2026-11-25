"""
Tools for turning a player's games into a simple performance report.

All functions take the DataFrame returned by pgn_tools.parse_pgn().
A typical pipeline is:

    df, _ = parse_pgn(text)
    df, clean_info = clean_games(df)
    df = add_groupings(df)
    df, params = add_expected(df)
    report = group_report(df, "opening_family")

The idea: first predict each game's expected score from the rating gap
(and colour), then check whether some groups of games (an opening, a time
control, a year) score clearly above or below what the ratings predicted.

build_report() runs all of these steps in one call.
"""

import math

import numpy as np
import pandas as pd

from pgn_tools import parse_pgn

# Fewest games per colour needed before the colour term is added to the model.
MIN_GAMES_PER_COLOUR = 30

# Fewest clean games needed before build_report will make a report.
MIN_CLEAN_GAMES = 100


def clean_games(df):
    """
    Keep only games that are not flagged stale_rating or repeat_opponent.

    Returns (clean_df, info) where info counts the total, kept and
    set-aside games.
    """
    keep = ~df["stale_rating"] & ~df["repeat_opponent"]
    clean_df = df[keep].reset_index(drop=True)
    info = {
        "total": len(df),
        "kept": len(clean_df),
        "set_aside": len(df) - len(clean_df),
    }
    return clean_df, info


def fit_logistic(X, y, max_iter=50, ridge=1e-6):
    """
    Fit a logistic regression with Newton-Raphson, using numpy only.

    X: 2D array, one row per game, one column per term (first column all 1s).
    y: targets between 0 and 1 (here 1, 0.5 or 0, so draws count as half).

    Each step moves the coefficients using the gradient (how wrong we are)
    and the Hessian (how curved the error is). The tiny ridge added to the
    Hessian's diagonal stops it from being impossible to invert.
    """
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(X @ beta)))         # current predictions
        gradient = X.T @ (y - p)
        weights = p * (1 - p)
        hessian = X.T @ (X * weights[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:            # stop once it has settled
            break
    return beta


def add_expected(df):
    """
    Add an "expected" column: the predicted score for each game.

    Model: expected = 1 / (1 + exp(-(a + b * rating_diff/100 + c * is_white)))
    The colour term c is only used if both colours have at least 30 games.

    Returns (df_with_expected, {"a": ..., "b": ..., "c": ... or None}).
    """
    df = df.copy()
    is_white = (df["colour"] == "white").astype(float).to_numpy()
    use_colour = (
        is_white.sum() >= MIN_GAMES_PER_COLOUR
        and (1 - is_white).sum() >= MIN_GAMES_PER_COLOUR
    )

    # Build the X matrix: a column of 1s (for a), the rating gap in units
    # of 100 points (for b), and is_white (for c) if it is used.
    columns = [np.ones(len(df)), df["rating_diff"].to_numpy() / 100]
    if use_colour:
        columns.append(is_white)
    X = np.column_stack(columns)
    y = df["score"].to_numpy(dtype=float)

    beta = fit_logistic(X, y)
    df["expected"] = 1 / (1 + np.exp(-(X @ beta)))

    params = {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "c": float(beta[2]) if use_colour else None,
    }
    return df, params


def wilson_interval(points, n, z=1.96):
    """
    95% Wilson confidence interval for a score rate (points / n).

    This gives a range that the "true" score rate is likely to be in.
    It behaves better than the simple p +/- 2 standard errors when n is
    small or p is close to 0 or 1.
    """
    if n == 0:
        return (0.0, 1.0)
    p = points / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (centre - half, centre + half)


def group_report(df, by, min_games=40):
    """
    Compare actual and expected score for each group in column `by`.

    df must already have the "expected" column (from add_expected).
    The verdict says whether the expected score falls outside the 95%
    interval around the actual score, i.e. whether the difference is
    bigger than chance alone would usually give.
    """
    rows = []
    for group, games in df.groupby(by):
        n = len(games)
        points = games["score"].sum()
        score = points / n
        expected = games["expected"].mean()
        ci_low, ci_high = wilson_interval(points, n)

        if n < min_games:
            verdict = "too few games"
        elif expected < ci_low:
            verdict = "above expectation"
        elif expected > ci_high:
            verdict = "below expectation"
        else:
            verdict = "in line"

        rows.append({
            "group": group,
            "games": n,
            "score": score,
            "expected": expected,
            "difference": score - expected,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "verdict": verdict,
        })

    report = pd.DataFrame(rows, columns=[
        "group", "games", "score", "expected", "difference",
        "ci_low", "ci_high", "verdict",
    ])
    return report.sort_values("games", ascending=False, kind="stable").reset_index(drop=True)


def opening_family(name):
    """
    Short family name for an opening.

    Lichess style, with a colon: the text before the first colon.
        "Sicilian Defense: Najdorf Variation" -> "Sicilian Defense"
    Chess.com style, with hyphens: the first two hyphen-separated words.
        "Scandinavian-Defense-Mieses-Kotrc-Variation" -> "Scandinavian-Defense"
    """
    if not isinstance(name, str):
        return name  # keep missing values as they are
    if ":" in name:
        return name.split(":")[0].strip()
    return "-".join(name.split("-")[:2])


def time_group(base_seconds):
    """Put a game into a time control bucket by its starting clock time."""
    if base_seconds <= 60:
        return "1 min or less"
    if base_seconds <= 120:
        return "2 min"
    if base_seconds <= 300:
        return "3-5 min"
    return "more than 5 min"


def add_groupings(df):
    """Add the opening_family, year and time_group columns."""
    df = df.copy()
    df["opening_family"] = df["opening"].apply(opening_family)
    df["year"] = df["date"].dt.year
    df["time_group"] = df["base_seconds"].apply(time_group)
    return df


def rating_by_month(df):
    """Number of games and average rating for each calendar month."""
    # to_period("M") turns a date into its month; to_timestamp() then gives
    # the first day of that month.
    month = df["date"].dt.to_period("M").dt.to_timestamp()
    result = (
        df.groupby(month)
        .agg(games=("my_rating", "size"), my_rating=("my_rating", "mean"))
        .reset_index()
        .rename(columns={"date": "month"})
    )
    return result.sort_values("month").reset_index(drop=True)


def build_report(text, player=None, min_games=40):
    """
    Run the whole pipeline on PGN text and return everything the report needs.

    Raises ValueError with a short message if there is not enough to report on.
    The result holds only the player's own name (in parse_info), never
    opponent names: parse_pgn already keeps those out of its output.
    """
    df, parse_info = parse_pgn(text, player)

    if parse_info["games_in_file"] == 0:
        raise ValueError("No games were found. Please check that this is a PGN file.")
    if parse_info["games_used"] == 0:
        raise ValueError(
            "None of the games could be used. Check the username, and that "
            "the games have ratings and a normal result."
        )

    df, clean_info = clean_games(df)
    if clean_info["kept"] < MIN_CLEAN_GAMES:
        raise ValueError(
            f"Only {clean_info['kept']} usable games were left after cleaning. "
            f"At least {MIN_CLEAN_GAMES} are needed for a meaningful report."
        )

    df, params = add_expected(df)
    df = add_groupings(df)

    overall = {
        "games": len(df),
        "score": float(df["score"].mean()),
        "expected": float(df["expected"].mean()),
    }
    year_table = (
        group_report(df, "year", min_games)
        .sort_values("group")
        .reset_index(drop=True)
    )

    return {
        "parse_info": parse_info,
        "clean_info": clean_info,
        "params": params,
        "overall": overall,
        "opening_table": group_report(df, "opening_family", min_games),
        "time_table": group_report(df, "time_group", min_games),
        "year_table": year_table,
        "rating_months": rating_by_month(df),
    }
