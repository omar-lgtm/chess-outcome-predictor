"""
Tools for checking how much to trust the personal expected-score curve.

They compare the personal curve with the standard Elo formula, test whether
giving recent games more weight helps, measure whether one predictor really
beats another (paired bootstrap), and put a range around a single
prediction (bootstrap prediction interval).

The fitting and prediction code is reused from report_tools.py.
"""

import math

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from report_tools import brier_score, fit_expected_params, predict_expected


def elo_expected(gap, scale=400.0):
    """
    Expected score from the standard Elo formula, with no fitting:
        1 / (1 + 10 ** (-gap / scale))
    gap is my rating minus my opponent's. Works on a number or an array.
    """
    result = 1 / (1 + 10 ** (-np.asarray(gap, dtype=float) / scale))
    return float(result) if result.ndim == 0 else result


def recency_weights(dates, half_life_days, reference_date=None):
    """
    Weight for each game: 0.5 ** (age_days / half_life_days).

    age_days is how many days before reference_date the game was played
    (reference_date defaults to the latest date). A game one half-life old
    counts half as much as the newest, two half-lives old a quarter, and so
    on. Dates after reference_date are treated as age 0, so every weight is
    between 0 and 1.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if reference_date is None:
        reference_date = dates.max()
    age_days = (pd.Timestamp(reference_date) - dates).dt.total_seconds() / 86400
    age_days = age_days.clip(lower=0).to_numpy()
    return 0.5 ** (age_days / half_life_days)


def recency_name(half_life):
    return f"Personal curve, recency weighted (half-life = {half_life} days)"


CV_NAME = "Personal curve, recency weighted (CV-chosen)"


def compare_with_elo_and_recency(df, train_fraction=0.7, half_lives=(180, 365, 730), n_splits=5):
    """
    Compare the personal curve with the Elo formula and with recency-weighted
    versions of itself, on later games that were not used for any choice.

    The split is the same as compare_predictors in report_tools.py: games
    sorted by date, the first int(train_fraction * len) are train and the
    rest are test. Recency weights are measured from the newest train game.

    Returns (results_df sorted by Brier score, info). info["test_predictions"]
    holds every predictor's prediction for each test game and
    info["test_actual"] the real scores, for use with paired_bootstrap_diff.
    """
    data = df.copy().sort_values("date", kind="stable").reset_index(drop=True)
    n_train = int(train_fraction * len(data))
    train = data.iloc[:n_train].reset_index(drop=True)
    test = data.iloc[n_train:].reset_index(drop=True)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Not enough games to split into train and test sets.")

    predictions = {}

    # Constant: the average train score for every test game.
    predictions["Constant"] = np.full(len(test), train["score"].mean())

    # Elo: the standard formula, nothing fitted.
    predictions["Elo (scale 400)"] = elo_expected(test["rating_diff"].to_numpy())

    # Personal curve, every train game counting the same.
    personal_params = fit_expected_params(train)
    predictions["Personal curve"] = predict_expected(test, personal_params)

    # Personal curve with recent train games counting more, one per half-life.
    for half_life in half_lives:
        weights = recency_weights(train["date"], half_life)
        params = fit_expected_params(train, sample_weight=weights)
        predictions[recency_name(half_life)] = predict_expected(test, params)

    # Choose a half-life by cross-validation on the TRAIN rows only.
    # TimeSeriesSplit always fits on earlier rows and checks on the rows
    # straight after them, just like the real train/test split.
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(train))
    cv_brier = {}
    for half_life in half_lives:
        fold_briers = []
        for fit_rows, check_rows in folds:
            fit_games = train.iloc[fit_rows]
            check_games = train.iloc[check_rows]
            # Step 1: weights measured from the newest game in this fold's fit rows.
            weights = recency_weights(fit_games["date"], half_life)
            # Step 2: fit on the fold's fit rows with those weights.
            params = fit_expected_params(fit_games, sample_weight=weights)
            # Step 3: score the fold's check rows.
            predicted = predict_expected(check_games, params)
            fold_briers.append(brier_score(predicted, check_games["score"]))
        # Step 4: average the folds for this half-life.
        cv_brier[half_life] = float(np.mean(fold_briers))
    # Step 5: the half-life with the lowest average Brier score wins.
    chosen_half_life = min(half_lives, key=lambda h: cv_brier[h])
    predictions[CV_NAME] = predictions[recency_name(chosen_half_life)]

    # Score every predictor on the test games.
    actual = test["score"].to_numpy(dtype=float)
    results = pd.DataFrame([
        {
            "predictor": name,
            "brier": brier_score(predicted, actual),
            "mean_predicted": float(np.mean(predicted)),
            "mean_actual": float(np.mean(actual)),
        }
        for name, predicted in predictions.items()
    ])
    results = results.sort_values("brier", kind="stable").reset_index(drop=True)

    info = {
        "train_games": len(train),
        "test_games": len(test),
        "train_start": str(train["date"].iloc[0].date()),
        "train_end": str(train["date"].iloc[-1].date()),
        "test_start": str(test["date"].iloc[0].date()),
        "test_end": str(test["date"].iloc[-1].date()),
        "chosen_half_life": chosen_half_life,
        "cv_brier_by_half_life": cv_brier,
        "personal_params": personal_params,
        # The Elo formula, written like the personal curve, has this slope
        # per 100 rating points: ln(10) / 400 * 100.
        "elo_slope_per_100": 100 * math.log(10) / 400,
        "test_predictions": pd.DataFrame(predictions),
        "test_actual": pd.Series(actual, name="score"),
    }
    return results, info


def paired_bootstrap_diff(pred_a, pred_b, actual, n_boot=2000, seed=42):
    """
    How sure can we be that predictor A beats predictor B?

    mean_diff is Brier(A) minus Brier(B) on all games; negative means A is
    better. To see how much that could move by luck, the games are resampled
    with replacement n_boot times, using the SAME games for both predictors
    each time, and low/high are the 2.5th and 97.5th percentiles of the
    difference. If the whole range is below 0, A is reliably better.
    """
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Each game's squared-error difference between A and B.
    diff_per_game = (pred_a - actual) ** 2 - (pred_b - actual) ** 2

    rng = np.random.default_rng(seed)
    n = len(actual)
    # One row of resampled game numbers per bootstrap round.
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = diff_per_game[indices].mean(axis=1)

    return {
        "mean_diff": float(diff_per_game.mean()),
        "low": float(np.percentile(boot_diffs, 2.5)),
        "high": float(np.percentile(boot_diffs, 97.5)),
    }


def fit_or_none(games):
    """
    Fit the personal curve, or return None if the fit is not usable.

    A fit counts as failed if it raises an error, gives a non-finite number,
    or if every game in the subset has the same score (all wins, for
    example). In that last case the fitting code does not raise an error,
    but the curve it returns is meaningless, so it is treated as a failure.
    """
    if games["score"].nunique() < 2:
        return None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            params = fit_expected_params(games)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    numbers = [params["a"], params["b"]] + ([params["c"]] if params["c"] is not None else [])
    if not np.all(np.isfinite(numbers)):
        return None
    return params


def learning_curve(df, sizes=(50, 100, 200, 400, 800, 1600), train_fraction=0.7,
                   n_repeats=30, seed=42, general_expected=None):
    """
    How many games does the personal curve need before it predicts well?

    The games are split by date as before (earliest train_fraction are train,
    the rest are a fixed test block). For each size n, n train games are
    picked at random (without replacement), the personal curve is fitted on
    just those, and its Brier score is measured on the test block. This is
    repeated n_repeats times per size to show how much luck is involved.
    The full train set is always the last size, fitted once.

    Returns (curve_df, info). curve_df has one row per size; info compares
    the curve with the Elo formula and, if given, the general model.
    """
    data = df.copy().sort_values("date", kind="stable").reset_index(drop=True)
    n_train = int(train_fraction * len(data))
    train = data.iloc[:n_train]
    test = data.iloc[n_train:]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Not enough games to split into train and test sets.")
    actual = test["score"].to_numpy(dtype=float)

    # Keep sizes smaller than the train set, then add the full train size.
    sizes = sorted({n for n in sizes if n < n_train}) + [n_train]

    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        # The full train size has only one possible subset, so fit it once.
        repeats = 1 if n == n_train else n_repeats
        personal_briers, constant_briers, slopes = [], [], []
        failed = 0
        for _ in range(repeats):
            # Step 1: pick n train rows (all of them for the full size).
            if n == n_train:
                subset = train
            else:
                subset = train.iloc[rng.choice(n_train, size=n, replace=False)]

            # Step 2: the constant predictor, from this subset's average score.
            constant = np.full(len(test), subset["score"].mean())
            constant_briers.append(brier_score(constant, actual))

            # Step 3: fit the personal curve on the subset and score it.
            params = fit_or_none(subset)
            if params is None:
                failed += 1
                continue
            # A tiny subset can be perfectly separable (every win at a bigger
            # rating gap than every loss), giving a huge slope. The prediction
            # then rounds to exactly 0 or 1, which is correct but makes numpy
            # warn about overflow, so that warning is silenced here only.
            with np.errstate(over="ignore"):
                predicted = predict_expected(test, params)
            personal_briers.append(brier_score(predicted, actual))
            slopes.append(params["b"])

        ok = len(personal_briers)
        rows.append({
            "n": n,
            "repeats_ok": ok,
            "repeats_failed": failed,
            "personal_mean_brier": float(np.mean(personal_briers)) if ok else np.nan,
            "personal_p10": float(np.percentile(personal_briers, 10)) if ok else np.nan,
            "personal_p90": float(np.percentile(personal_briers, 90)) if ok else np.nan,
            "constant_mean_brier": float(np.mean(constant_briers)),
            "personal_mean_slope_b": float(np.mean(slopes)) if ok else np.nan,
        })
    curve = pd.DataFrame(rows)

    # Baselines that do not depend on n.
    elo_brier = brier_score(elo_expected(test["rating_diff"].to_numpy()), actual)
    general_brier = None
    if general_expected is not None:
        general_brier = brier_score(np.asarray(general_expected(test), dtype=float), actual)

    def first_n(beats):
        """Smallest n where beats is True, or None."""
        winners = curve.loc[beats, "n"]
        return int(winners.iloc[0]) if len(winners) else None

    mean = curve["personal_mean_brier"]     # NaN never counts as beating anything
    info = {
        "train_games": len(train),
        "test_games": len(test),
        "train_start": str(train["date"].iloc[0].date()),
        "train_end": str(train["date"].iloc[-1].date()),
        "test_start": str(test["date"].iloc[0].date()),
        "test_end": str(test["date"].iloc[-1].date()),
        "elo_brier": elo_brier,
        "general_brier": general_brier,
        "first_n_beating_elo": first_n(mean < elo_brier),
        "first_n_beating_general": (
            first_n(mean < general_brier) if general_brier is not None else None
        ),
        "first_n_beating_constant": first_n(mean < curve["constant_mean_brier"]),
    }
    return curve, info


def bootstrap_prediction_interval(train_df, gap, colour, n_boot=500, seed=42, half_life_days=None):
    """
    Expected score for one future game, with a 95% range.

    gap is my rating minus my opponent's, colour is "white" or "black".
    point uses a fit on all of train_df. The range comes from refitting on
    n_boot resamples of train_df's games (with replacement) and taking the
    2.5th and 97.5th percentiles of their predictions. If half_life_days is
    given, every fit gives recent games more weight.
    """
    train_df = train_df.reset_index(drop=True)
    game = pd.DataFrame({"rating_diff": [gap], "colour": [colour]})
    weights = None
    if half_life_days is not None:
        weights = recency_weights(train_df["date"], half_life_days)

    point = float(predict_expected(game, fit_expected_params(train_df, weights))[0])

    rng = np.random.default_rng(seed)
    n = len(train_df)
    estimates = []
    failures = 0
    for _ in range(n_boot):
        rows = rng.integers(0, n, size=n)
        sample = train_df.iloc[rows]
        sample_weights = None if weights is None else weights[rows]
        try:
            params = fit_expected_params(sample, sample_weights)
            estimate = float(predict_expected(game, params)[0])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            failures += 1
            continue
        if not np.isfinite(estimate):
            failures += 1
            continue
        estimates.append(estimate)

    if failures > 0.1 * n_boot:
        raise ValueError(
            f"{failures} of {n_boot} bootstrap fits failed, which is more than "
            "10 percent, so the range would not be reliable."
        )

    return {
        "point": point,
        "low": float(np.percentile(estimates, 2.5)),
        "high": float(np.percentile(estimates, 97.5)),
        "n_games": n,
    }
