"""
Model-comparison experiment for the general "Predict a game" model.

Uses the Kaggle Lichess games file (data/games.csv, 20,058 rows) to compare
several scikit-learn models and feature sets against the saved model
(chess_model.joblib), on exactly the same test games.

Rules followed:
- Duplicate game ids are dropped (first row kept) BEFORE splitting.
- No feature may be known only after the game (see FORBIDDEN_COLUMNS).
- Settings are chosen only by 5-fold stratified cross-validation on the
  development set (train + validation, 85%). The 15% test set is used only
  to report scores, never to choose anything.
- No model file is saved. chess_model.joblib is only loaded and scored.

Run it with:  python general_experiments.py
Its main results table is saved to docs/general_model_results.csv.
"""

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/games.csv")
MODEL_PATH = Path("chess_model.joblib")
RESULTS_PATH = Path("docs/general_model_results.csv")
RANDOM_STATE = 42

# The three outcomes, in alphabetical order. Every probability table in this
# file is put into exactly this column order before it is scored, because
# sklearn's log_loss reads columns in sorted label order (a mismatch here
# caused a bug in this project before).
CLASSES = ["black", "draw", "white"]
assert CLASSES == sorted(CLASSES)

# Columns known only after the game has finished (or that identify games and
# players). None of these may ever be used as a feature.
FORBIDDEN_COLUMNS = {
    "victory_status", "turns", "moves", "last_move_at", "created_at",
    "white_id", "black_id", "id", "opening_name",
}

# Nested feature groups: each one adds to the one before.
# F3 adds the opening. The opening is only known once the opening phase of
# the game has been played, so F3 is EARLY-GAME information, not pre-game.
F0 = ["rating_diff"]
F1 = F0 + ["white_rating", "black_rating", "mean_rating"]
F2 = F1 + ["base_minutes", "increment_seconds", "rated"]
F3 = F2 + ["opening_eco", "opening_ply"]
FEATURE_SETS = {
    "F0 rating gap only": F0,
    "F1 both ratings": F1,
    "F2 time and rated": F2,
    "F3 opening (early-game)": F3,
}
CATEGORY_COLUMNS = ["opening_eco"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def parse_increment(code):
    """
    Split an increment code like "10+0" or "15+10" into
    (base_minutes, increment_seconds). Malformed values give (nan, nan)
    instead of crashing; the pipelines fill those in with the median.
    """
    try:
        base, increment = str(code).strip().split("+")
        return float(int(base)), float(int(increment))
    except (ValueError, AttributeError):
        return np.nan, np.nan


def drop_duplicate_ids(df):
    """Keep the first row of each game id. Returns (df, number of rows dropped)."""
    deduplicated = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return deduplicated, len(df) - len(deduplicated)


def check_no_leakage(X):
    """Raise an error if any forbidden column (or a duration column) is in X."""
    bad = [c for c in X.columns if c in FORBIDDEN_COLUMNS or "duration" in c.lower()]
    if bad:
        raise ValueError(f"Forbidden columns in the feature table: {bad}")


def build_features(df, columns=F3):
    """
    Build the feature table from the raw games, keeping only `columns`.
    Every derived column comes only from pre-game (or, for F3, early-game)
    information. The leakage guard is run on the result.
    """
    features = pd.DataFrame(index=df.index)
    features["rating_diff"] = df["white_rating"] - df["black_rating"]
    features["white_rating"] = df["white_rating"]
    features["black_rating"] = df["black_rating"]
    features["mean_rating"] = (df["white_rating"] + df["black_rating"]) / 2
    parsed = df["increment_code"].map(parse_increment)
    features["base_minutes"] = [base for base, _ in parsed]
    features["increment_seconds"] = [inc for _, inc in parsed]
    features["rated"] = df["rated"].astype(int)
    features["opening_eco"] = df["opening_eco"].astype(str)
    features["opening_ply"] = df["opening_ply"]
    X = features[list(columns)]
    check_no_leakage(X)
    return X


def split_like_original(df):
    """
    70/15/15 split, stratified on winner, random_state=42: first 30% is set
    aside, then that 30% is split in half into validation and test.
    Returns (development = train + validation, test).
    """
    train, rest = train_test_split(
        df, test_size=0.30, stratify=df["winner"], random_state=RANDOM_STATE)
    validation, test = train_test_split(
        rest, test_size=0.50, stratify=rest["winner"], random_state=RANDOM_STATE)
    development = pd.concat([train, validation])
    return development, test


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def reorder_proba(proba, from_classes):
    """Put probability columns into CLASSES order (black, draw, white)."""
    from_classes = list(from_classes)
    return np.asarray(proba)[:, [from_classes.index(c) for c in CLASSES]]


def encode_labels(labels):
    """
    Text labels to numbers: black 0, draw 1, white 2 (the CLASSES order).
    The models are trained on numbers because MLPClassifier with
    early_stopping in scikit-learn 1.6.1 crashes on text labels.
    """
    return np.array([CLASSES.index(label) for label in labels])


def model_proba(model, X):
    """
    Probabilities from a fitted model, always in CLASSES order.
    Works whether the model's classes_ are numbers (our models) or names
    (the saved model).
    """
    names = [CLASSES[c] if isinstance(c, (int, np.integer)) else c for c in model.classes_]
    return reorder_proba(model.predict_proba(X), names)


def per_game_log_loss(y, proba):
    """-log(probability given to what really happened), for each game."""
    y = np.asarray(y)
    rows = np.arange(len(y))
    cols = np.array([CLASSES.index(label) for label in y])
    return -np.log(np.clip(proba[rows, cols], 1e-15, 1))


def multiclass_log_loss(y, proba):
    """Average log loss; proba columns must be in CLASSES order."""
    return float(per_game_log_loss(y, proba).mean())


def multiclass_brier(y, proba):
    """Mean over games of the sum over classes of (p - y)^2."""
    one_hot = (np.asarray(y)[:, None] == np.array(CLASSES)[None, :]).astype(float)
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def scores(y, proba):
    """Accuracy, log loss, Brier and draw recall from CLASSES-ordered probabilities."""
    y = np.asarray(y)
    predicted = np.array(CLASSES)[proba.argmax(axis=1)]
    is_draw = y == "draw"
    return {
        "test_accuracy": float((predicted == y).mean()),
        "test_log_loss": multiclass_log_loss(y, proba),
        "test_brier": multiclass_brier(y, proba),
        "draw_recall": float((predicted[is_draw] == "draw").mean()) if is_draw.any() else np.nan,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def preprocessor(columns, scale):
    """
    Column preparation for a pipeline: numbers get missing values filled with
    the median (and are scaled for models that need it); opening_eco is
    one-hot encoded with codes seen fewer than 30 times grouped together.
    """
    numeric = [c for c in columns if c not in CATEGORY_COLUMNS]
    categories = [c for c in columns if c in CATEGORY_COLUMNS]
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    parts = [("numbers", Pipeline(numeric_steps), numeric)]
    if categories:
        parts.append(("categories", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False,
        ), categories))
    return ColumnTransformer(parts)


def make_pipeline(name, columns, settings):
    """One model with its preprocessing, for the given settings."""
    if name == "Logistic regression":
        model, scale = LogisticRegression(max_iter=2000, **settings), True
    elif name == "Random forest":
        model = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, **settings)
        scale = False
    elif name == "Gradient boosting":
        model = HistGradientBoostingClassifier(
            max_iter=300, early_stopping=True, random_state=RANDOM_STATE, **settings)
        scale = False
    elif name == "Neural net (MLP)":
        model = MLPClassifier(
            early_stopping=True, max_iter=500, random_state=RANDOM_STATE, **settings)
        scale = True
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("prep", preprocessor(columns, scale)), ("model", model)])


# A small, fixed grid of settings per model.
GRIDS = {
    "Logistic regression": [{"C": c} for c in (0.1, 1, 10)],
    "Random forest": [
        {"max_depth": d, "min_samples_leaf": leaf} for d in (6, 12) for leaf in (5, 20)
    ],
    "Gradient boosting": [
        {"learning_rate": rate, "max_depth": d} for rate in (0.05, 0.1) for d in (3, 5)
    ],
    "Neural net (MLP)": [
        {"hidden_layer_sizes": h, "alpha": a} for h in ((32,), (64, 32)) for a in (0.0001, 0.01)
    ],
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cv_log_loss(pipeline, X, y):
    """Mean 5-fold cross-validated log loss on the development set."""
    # sklearn's "neg_log_loss" scorer matches columns to classes_ itself.
    return float(-cross_val_score(pipeline, X, y, cv=CV, scoring="neg_log_loss").mean())


def settings_text(settings):
    return ", ".join(f"{k}={v}" for k, v in settings.items()) if settings else "-"


# ---------------------------------------------------------------------------
# Extra analysis
# ---------------------------------------------------------------------------

def calibration_table(p_white, actual_white, bins=5):
    """P(white wins) in equal-count bins: mean predicted vs how often white won."""
    p = pd.Series(p_white)
    bin_number = pd.qcut(p.rank(method="first"), bins, labels=False) + 1
    return (
        pd.DataFrame({"bin": bin_number, "predicted": p, "actual": np.asarray(actual_white, float)})
        .groupby("bin")
        .agg(games=("actual", "size"), mean_predicted=("predicted", "mean"),
             mean_actual=("actual", "mean"))
        .reset_index()
    )


def permutation_importance_log_loss(model, X, y, n_repeats=5, seed=RANDOM_STATE):
    """
    Shuffle each ORIGINAL feature column in turn (before one-hot encoding)
    and record how much the test log loss gets worse. Bigger means more
    important.
    """
    rng = np.random.default_rng(seed)
    baseline = multiclass_log_loss(y, model_proba(model, X))
    rows = []
    for column in X.columns:
        increases = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            increases.append(multiclass_log_loss(y, model_proba(model, shuffled)) - baseline)
        rows.append({"feature": column, "log_loss_increase": float(np.mean(increases))})
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False).reset_index(drop=True)


def paired_bootstrap(loss_a, loss_b, n_boot=1000, seed=RANDOM_STATE):
    """
    Difference in mean per-game log loss (A minus B; negative means A is
    better), with a 95% range from resampling the same test games for both.
    """
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return {"mean_diff": float(diff.mean()),
            "low": float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5))}


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def main():
    # scikit-learn 1.6.1 with the installed SciPy prints a harmless
    # "Unknown solver options: iprint" warning on every logistic fit.
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    # Step 1: load and check the data, before any modelling.
    raw = pd.read_csv(DATA_PATH)
    print("=== Data checks ===")
    print("shape:", raw.shape)
    print("winner counts:", raw["winner"].value_counts().to_dict())
    print("duplicated game ids (rows after the first of each id):", int(raw["id"].duplicated().sum()))
    print("fully duplicated rows:", int(raw.duplicated().sum()))
    print("missing values per column:", raw.isna().sum().to_dict())
    games, dropped = drop_duplicate_ids(raw)
    print(f"Dropped {dropped} rows with a repeated game id (kept the first). "
          f"{len(games)} games remain.")

    # Step 2: split exactly like the original recipe.
    development, test = split_like_original(games)
    print(f"\nDevelopment set (train + validation): {len(development)} games; "
          f"test set: {len(test)} games.")
    y_dev = development["winner"].to_numpy()
    y_test = test["winner"].to_numpy()
    y_dev_codes = encode_labels(y_dev)      # what the models are trained on
    print("Note: F3 uses the opening, which is only known after the opening phase "
          "of the game, so F3 is early-game information, not pre-game information.")

    rows = []          # the main results table
    timings = []

    def add_row(model_name, feature_set, cv_loss, proba, settings="-"):
        rows.append({"model": model_name, "feature_set": feature_set,
                     "cv_log_loss": cv_loss, **scores(y_test, proba),
                     "settings": settings})

    # Step 3: baselines.
    # (a) Always predict white; probabilities are the development set's class shares.
    shares = np.array([np.mean(y_dev == c) for c in CLASSES])
    add_row("Always white (class shares)", "none", np.nan, np.tile(shares, (len(test), 1)))
    # (b) The saved model, on exactly these test games.
    saved = joblib.load(MODEL_PATH)
    saved_X_test = pd.DataFrame({"rating_diff": test["white_rating"] - test["black_rating"]})
    saved_proba = model_proba(saved, saved_X_test)
    add_row("Saved model (chess_model.joblib)", "F0 rating gap only", np.nan, saved_proba)
    print(f"Saved model accuracy on this test set: {scores(y_test, saved_proba)['test_accuracy']:.4f} "
          "(reported originally: 0.6195)")

    # Step 4: tune each model on F3 by cross-validation on the development set.
    X_dev = build_features(development, F3)
    X_test = build_features(test, F3)
    best = {}          # model name -> (settings, cv log loss, fitted pipeline)
    for name, grid in GRIDS.items():
        start = time.perf_counter()
        cv_by_setting = []
        for settings in grid:
            # 5-fold CV log loss for this setting, development rows only.
            cv_by_setting.append((cv_log_loss(make_pipeline(name, F3, settings), X_dev, y_dev_codes), settings))
        cv_loss, settings = min(cv_by_setting, key=lambda pair: pair[0])
        # Refit the chosen setting on the whole development set, then score once on test.
        fitted = make_pipeline(name, F3, settings).fit(X_dev, y_dev_codes)
        best[name] = (settings, cv_loss, fitted)
        add_row(name, "F3 opening (early-game)", cv_loss, model_proba(fitted, X_test), settings_text(settings))
        seconds = time.perf_counter() - start
        timings.append((name, seconds))
        print(f"{name}: best {settings_text(settings)}, CV log loss {cv_loss:.4f} ({seconds:.0f} s)")

    # Soft voting of the best logistic regression, random forest and gradient boosting.
    start = time.perf_counter()
    voting = VotingClassifier([
        ("lr", make_pipeline("Logistic regression", F3, best["Logistic regression"][0])),
        ("rf", make_pipeline("Random forest", F3, best["Random forest"][0])),
        ("hgb", make_pipeline("Gradient boosting", F3, best["Gradient boosting"][0])),
    ], voting="soft")
    voting_cv = cv_log_loss(voting, X_dev, y_dev_codes)
    voting.fit(X_dev, y_dev_codes)
    add_row("Soft voting (LR + RF + GB)", "F3 opening (early-game)", voting_cv, model_proba(voting, X_test))
    timings.append(("Soft voting", time.perf_counter() - start))
    print(f"Soft voting: CV log loss {voting_cv:.4f} ({timings[-1][1]:.0f} s)")

    # Calibrate the best single model (lowest CV log loss among the four).
    # Isotonic needs plenty of data per fold; with over 10,000 development
    # games it is used, otherwise the smoother sigmoid method.
    start = time.perf_counter()
    best_single = min(GRIDS, key=lambda n: best[n][1])
    method = "isotonic" if len(development) >= 10_000 else "sigmoid"
    calibrated = CalibratedClassifierCV(
        make_pipeline(best_single, F3, best[best_single][0]), method=method, cv=5)
    calibrated_cv = cv_log_loss(calibrated, X_dev, y_dev_codes)
    calibrated.fit(X_dev, y_dev_codes)
    calibrated_name = f"{best_single} + {method} calibration"
    add_row(calibrated_name, "F3 opening (early-game)", calibrated_cv,
            model_proba(calibrated, X_test), settings_text(best[best_single][0]))
    timings.append(("Calibrated " + best_single, time.perf_counter() - start))
    print(f"{calibrated_name}: CV log loss {calibrated_cv:.4f} "
          f"(uncalibrated {best[best_single][1]:.4f}) ({timings[-1][1]:.0f} s)")

    # Step 5: ablation. Logistic regression and gradient boosting, with the
    # settings chosen above, on each feature group.
    print("\n=== Ablation (settings from the F3 tuning) ===")
    fitted_f0_lr = None
    for name in ("Logistic regression", "Gradient boosting"):
        for set_name, columns in FEATURE_SETS.items():
            if set_name.startswith("F3"):
                continue     # already in the table from the tuning step
            start = time.perf_counter()
            Xd, Xt = build_features(development, columns), build_features(test, columns)
            pipe = make_pipeline(name, columns, best[name][0])
            cv_loss = cv_log_loss(pipe, Xd, y_dev_codes)
            pipe.fit(Xd, y_dev_codes)
            add_row(name, set_name, cv_loss, model_proba(pipe, Xt), settings_text(best[name][0]))
            if name == "Logistic regression" and set_name.startswith("F0"):
                fitted_f0_lr = (pipe, Xt)
            print(f"{name} on {set_name}: CV log loss {cv_loss:.4f} "
                  f"({time.perf_counter() - start:.0f} s)")

    results = pd.DataFrame(rows)
    print("\n=== Results on the test set (sorted by test log loss) ===")
    print(results.sort_values("test_log_loss").round(
        {"cv_log_loss": 4, "test_accuracy": 3, "test_log_loss": 4, "test_brier": 4, "draw_recall": 3}
    ).to_string(index=False))

    print("\n=== Time taken ===")
    for name, seconds in timings:
        print(f"{name}: {seconds:.1f} s")

    # Save the main table (no game ids or player ids are in it).
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    saved_columns = ["model", "feature_set", "cv_log_loss", "test_accuracy",
                     "test_log_loss", "test_brier", "draw_recall"]
    results[saved_columns].round(4).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH}")

    # Step 6: extra analysis for the best model by DEVELOPMENT CV log loss
    # (among the F3 rows), never by test score.
    candidates = {name: (cv, model) for name, (_, cv, model) in best.items()}
    candidates["Soft voting (LR + RF + GB)"] = (voting_cv, voting)
    candidates[calibrated_name] = (calibrated_cv, calibrated)
    best_name = min(candidates, key=lambda n: candidates[n][0])
    best_model = candidates[best_name][1]
    best_proba = model_proba(best_model, X_test)
    print(f"\n=== Extra analysis: {best_name} (lowest CV log loss, {candidates[best_name][0]:.4f}) ===")

    predicted = np.array(CLASSES)[best_proba.argmax(axis=1)]
    matrix = pd.DataFrame(confusion_matrix(y_test, predicted, labels=CLASSES),
                          index=[f"actual {c}" for c in CLASSES],
                          columns=[f"predicted {c}" for c in CLASSES])
    print("\nConfusion matrix (test set):")
    print(matrix.to_string())

    print("\nCalibration of P(white wins), 5 equal-count bins (test set):")
    print(calibration_table(best_proba[:, CLASSES.index("white")], y_test == "white")
          .round(3).to_string(index=False))

    print("\nPermutation importance (test set, increase in log loss, 5 repeats):")
    print(permutation_importance_log_loss(best_model, X_test, y_test).round(4).to_string(index=False))

    best_losses = per_game_log_loss(y_test, best_proba)
    print("\nPaired bootstrap of per-game log loss (1000 resamples, A minus B, negative = A better):")
    lr_f0_model, lr_f0_X = fitted_f0_lr
    for label, other in [
        ("saved model", per_game_log_loss(y_test, saved_proba)),
        ("logistic regression on F0", per_game_log_loss(y_test, model_proba(lr_f0_model, lr_f0_X))),
    ]:
        r = paired_bootstrap(best_losses, other)
        print(f"{best_name} vs {label}: mean_diff {r['mean_diff']:.4f}, "
              f"95% range {r['low']:.4f} to {r['high']:.4f}")


if __name__ == "__main__":
    main()
