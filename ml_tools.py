"""
Machine learning tools for predicting a player's game results.

These build on report_tools. Starting from the cleaned games of one player,
they add features (including recent form), tune a few scikit-learn models
with time-ordered cross-validation on the earlier games, and then score each
model once on the later games that were never used for any choice.

A typical run:

    df, _ = parse_pgn(text)
    df, _ = clean_games(df)
    results, info = tune_and_compare(df)
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from report_tools import add_groupings, brier_score

NUMERIC_COLUMNS = [
    "rating_diff", "is_white", "my_rating", "base_seconds", "increment_seconds",
    "day_of_week", "form_20", "last_score", "loss_streak",
]
CATEGORY_COLUMNS = ["time_group", "opening_family"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORY_COLUMNS

# The three outcomes the models predict, and how a score maps to each.
LABELS = ["loss", "draw", "win"]
SCORE_TO_LABEL = {0.0: "loss", 0.5: "draw", 1.0: "win"}

# A small, fixed set of settings to try for each model.
SETTINGS_GRID = {
    "Logistic regression": [{"C": 0.1}, {"C": 1.0}],
    "Random forest": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (4, 8) for leaf in (10, 30)
    ],
    "Gradient boosting": [
        {"max_depth": depth, "learning_rate": rate, "max_iter": 200}
        for depth in (2, 3) for rate in (0.03, 0.1)
    ],
}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def make_features(df):
    """
    Build the model inputs, one row per game, sorted by date.

    Returns the FEATURE_COLUMNS plus "date" and "score", which are kept so
    the games can be split by date and so there is a target to predict.
    The form features (form_20, last_score, loss_streak) only look at
    EARLIER games, never at the game itself, so they don't leak its result.
    """
    data = df.copy()
    if "time_group" not in data.columns or "opening_family" not in data.columns:
        data = add_groupings(data)
    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    features = pd.DataFrame({
        "rating_diff": data["rating_diff"],
        "is_white": (data["colour"] == "white").astype(int),
        "my_rating": data["my_rating"],
        "base_seconds": data["base_seconds"],
        "increment_seconds": data["increment_seconds"],
        "time_group": data["time_group"],
        # A missing opening becomes its own category rather than an error.
        "opening_family": data["opening_family"].fillna("Unknown"),
        "day_of_week": data["date"].dt.dayofweek,    # 0 = Monday
    })

    # shift(1) moves every score down one row, so each row only sees the
    # score of the game BEFORE it. The first game has nothing before it.
    previous = data["score"].shift(1)
    features["form_20"] = previous.rolling(20, min_periods=1).mean().fillna(0.5)
    features["last_score"] = previous.fillna(0.5)

    # Count the losses in a row just before each game.
    streaks = []
    streak = 0
    for score in data["score"]:
        streaks.append(streak)          # recorded before this game's result
        streak = streak + 1 if score == 0 else 0
    features["loss_streak"] = streaks

    features["date"] = data["date"]
    features["score"] = data["score"]
    return features


def split_by_date(features, train_fraction=0.7):
    """Earliest train_fraction of the rows are train, the rest are test."""
    n_train = int(train_fraction * len(features))
    return features.iloc[:n_train], features.iloc[n_train:]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models(random_state=42):
    """
    One scikit-learn Pipeline per model. Each pipeline first prepares the
    columns (scales the numbers, one-hot encodes the categories) and then
    fits the model, so the same steps are applied to train and test data.
    """
    def preparation():
        return ColumnTransformer([
            ("numbers", StandardScaler(), NUMERIC_COLUMNS),
            # Categories with fewer than 30 games are merged into one
            # "infrequent" column; categories never seen in training are ignored.
            ("categories",
             OneHotEncoder(handle_unknown="ignore", min_frequency=30,
                           sparse_output=False),
             CATEGORY_COLUMNS),
        ])

    return {
        "Logistic regression": Pipeline([
            ("prep", preparation()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Random forest": Pipeline([
            ("prep", preparation()),
            ("model", RandomForestClassifier(
                class_weight=None, n_jobs=1, random_state=random_state)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", preparation()),
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]),
    }


def score_labels(scores):
    """Turn scores (0, 0.5, 1) into labels ("loss", "draw", "win")."""
    return pd.Series(scores).astype(float).map(SCORE_TO_LABEL).to_numpy()


def class_probabilities(pipeline, X):
    """
    Probabilities for loss, draw and win, always in that order.
    If the model never saw a class in training (for example no draws),
    that class gets probability 0.
    """
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    columns = [
        proba[:, classes.index(label)] if label in classes else np.zeros(len(X))
        for label in LABELS
    ]
    return np.column_stack(columns)


def three_class_log_loss(y_labels, proba):
    """
    Log loss for labels ("loss"/"draw"/"win") and probabilities whose
    columns are in LABELS order (loss, draw, win).

    sklearn's log_loss expects the columns in alphabetical label order
    (draw, loss, win), so the columns are reordered before calling it.
    """
    order = sorted(LABELS)
    reordered = proba[:, [LABELS.index(label) for label in order]]
    return float(log_loss(y_labels, reordered, labels=order))


def predicted_score(pipeline, X):
    """Expected score for each game: P(win) + 0.5 * P(draw)."""
    proba = class_probabilities(pipeline, X)
    return proba[:, LABELS.index("win")] + 0.5 * proba[:, LABELS.index("draw")]


# ---------------------------------------------------------------------------
# Tuning and comparison
# ---------------------------------------------------------------------------

def settings_text(settings):
    """{"max_depth": 4, "min_samples_leaf": 10} -> "max_depth=4, min_samples_leaf=10"."""
    return ", ".join(f"{key}={value}" for key, value in settings.items())


def tune_and_compare(df, train_fraction=0.7, n_splits=5):
    """
    Tune each model on the earlier games and test it once on the later games.

    Steps:
    1. Make features and split by date into train (earlier) and test (later).
    2. For each model and each setting in its small grid, run time-ordered
       cross-validation on the TRAIN rows only and record the mean Brier score.
    3. Keep each model's best setting, refit it on all train rows, and
       score it once on the test rows.
    The best model is chosen by its cross-validation score, not by the
    test score, so the test rows are never used for any choice.

    Returns (results_df sorted by test_brier, info dict). info also holds
    the refitted best pipeline under "best_pipeline".
    """
    # Step 1: features, then split by date.
    features = make_features(df)
    train, test = split_by_date(features, train_fraction)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Not enough games to split into train and test sets.")

    X_train, y_train = train[FEATURE_COLUMNS], score_labels(train["score"])
    X_test, y_test = test[FEATURE_COLUMNS], score_labels(test["score"])
    train_scores = train["score"].to_numpy(dtype=float)
    test_scores = test["score"].to_numpy(dtype=float)

    # TimeSeriesSplit always trains on earlier rows and validates on the
    # rows straight after them, so no fold peeks into its own future.
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(X_train))

    rows = []
    fitted = {}
    for name, base_pipeline in build_models().items():
        # Step 2: try every setting in the grid with cross-validation.
        best_settings, best_cv = None, np.inf
        for settings in SETTINGS_GRID[name]:
            fold_briers = []
            for fit_rows, check_rows in folds:
                pipeline = clone(base_pipeline).set_params(
                    **{f"model__{key}": value for key, value in settings.items()}
                )
                pipeline.fit(X_train.iloc[fit_rows], y_train[fit_rows])
                predicted = predicted_score(pipeline, X_train.iloc[check_rows])
                fold_briers.append(brier_score(predicted, train_scores[check_rows]))
            cv_brier = float(np.mean(fold_briers))
            if cv_brier < best_cv:
                best_settings, best_cv = settings, cv_brier

        # Step 3: refit with the best settings on all train rows, test once.
        pipeline = clone(base_pipeline).set_params(
            **{f"model__{key}": value for key, value in best_settings.items()}
        )
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        predicted = predicted_score(pipeline, X_test)
        rows.append({
            "model": name,
            "cv_brier": best_cv,
            "test_brier": brier_score(predicted, test_scores),
            "test_log_loss": three_class_log_loss(
                y_test, class_probabilities(pipeline, X_test)
            ),
            "mean_predicted": float(np.mean(predicted)),
            "mean_actual": float(np.mean(test_scores)),
            "best_settings": settings_text(best_settings),
        })

    # The constant baseline: always predict the train average. Its CV score
    # uses the same folds, predicting each fold's own training average, and
    # its log loss uses how often each outcome happened in the train rows.
    constant_cv = np.mean([
        brier_score(np.full(len(check_rows), train_scores[fit_rows].mean()),
                    train_scores[check_rows])
        for fit_rows, check_rows in folds
    ])
    class_shares = np.array([np.mean(y_train == label) for label in LABELS])
    rows.append({
        "model": "Constant (train mean)",
        "cv_brier": float(constant_cv),
        "test_brier": brier_score(np.full(len(test), train_scores.mean()), test_scores),
        "test_log_loss": three_class_log_loss(
            y_test, np.tile(class_shares, (len(test), 1))
        ),
        "mean_predicted": float(train_scores.mean()),
        "mean_actual": float(np.mean(test_scores)),
        "best_settings": "-",
    })

    results = pd.DataFrame(rows).sort_values("test_brier", kind="stable")
    results = results.reset_index(drop=True)

    # The best model is picked by cross-validation, never by the test score.
    model_rows = results[results["model"] != "Constant (train mean)"]
    best_name = model_rows.loc[model_rows["cv_brier"].idxmin(), "model"]

    info = {
        "train_games": len(train),
        "test_games": len(test),
        "train_start": str(train["date"].iloc[0].date()),
        "train_end": str(train["date"].iloc[-1].date()),
        "test_start": str(test["date"].iloc[0].date()),
        "test_end": str(test["date"].iloc[-1].date()),
        "best_model_name": best_name,
        "best_pipeline": fitted[best_name],
    }
    return results, info


# ---------------------------------------------------------------------------
# Understanding a fitted model
# ---------------------------------------------------------------------------

def permutation_importance_table(pipeline, X_test, y_test_score, n_repeats=10, random_state=42):
    """
    How much each feature matters to a fitted pipeline.

    Each original column of X_test (before one-hot encoding) is shuffled in
    turn, which breaks its link with the result. "importance" is how much the
    test Brier score gets worse on average when that column is shuffled.
    Bigger means the model relies on it more; near 0 means it barely matters.
    """
    rng = np.random.default_rng(random_state)
    actual = np.asarray(y_test_score, dtype=float)
    baseline = brier_score(predicted_score(pipeline, X_test), actual)

    rows = []
    for column in X_test.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = X_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = brier_score(predicted_score(pipeline, shuffled), actual)
            increases.append(score - baseline)
        rows.append({"feature": column, "importance": float(np.mean(increases))})

    table = pd.DataFrame(rows).sort_values("importance", ascending=False, kind="stable")
    return table.reset_index(drop=True)


def calibration_table(predicted, actual, bins=5):
    """
    Check whether predicted scores match reality.

    Games are sorted by predicted score and cut into `bins` groups of (nearly)
    equal size. If the model is well calibrated, mean_predicted and
    mean_actual are close in every row.
    """
    predicted = pd.Series(np.asarray(predicted, dtype=float))
    actual = pd.Series(np.asarray(actual, dtype=float))
    # rank(method="first") breaks ties, so every bin gets the same count
    # even when many predictions are identical.
    bin_number = pd.qcut(predicted.rank(method="first"), bins, labels=False) + 1
    table = (
        pd.DataFrame({"bin": bin_number, "predicted": predicted, "actual": actual})
        .groupby("bin")
        .agg(games=("actual", "size"),
             mean_predicted=("predicted", "mean"),
             mean_actual=("actual", "mean"))
        .reset_index()
    )
    return table
