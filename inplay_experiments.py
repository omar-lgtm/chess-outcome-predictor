"""
In-game win-probability experiment: predict a game's final result from the
position after the first N plies (half-moves).

Steps:
1. Load data/games.csv, drop repeated game ids, and split at GAME level
   (70/15/15 stratified on winner, random_state=42) with the shared helpers
   from general_experiments.py. All checkpoint rows of a game stay in its split.
2. Replay each game once and record the position after N = 10, 20, 30, 40
   and 60 plies (plus N = 0, the start position: pre-game information only).
   The feature table is cached in data/inplay_features.csv (git-ignored).
3. For each N, tune logistic regression, gradient boosting and a small
   neural net by 5-fold cross-validation on the development rows only, then
   score once on the test rows, with baselines and a feature-group ablation.

The set of games differs between N: a game that ends before N plies has no
row at N. So scores at different N are not directly comparable, and the
game length is never used as a feature.

Run it with:  python inplay_experiments.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from general_experiments import (
    CLASSES, DATA_PATH, GRIDS, calibration_table, cv_log_loss, drop_duplicate_ids,
    encode_labels, make_pipeline, model_proba, paired_bootstrap, parse_increment,
    per_game_log_loss, permutation_importance_log_loss, scores, settings_text,
    split_like_original,
)
from inplay_tools import build_checkpoint_rows, check_no_leakage, feature_groups, split_rows

CACHE_PATH = Path("data/inplay_features.csv")
RESULTS_PATH = Path("docs/inplay_results.csv")
CURVE_PATH = Path("docs/inplay_curve.png")
CHECKPOINTS = (10, 20, 30, 40, 60)
MODELS = ("Logistic regression", "Gradient boosting", "Neural net (MLP)")
GROUPS = feature_groups()
META_COLUMNS = ["game_index", "ply", "winner"]


def load_or_build_features(games):
    """
    Replay every game once, or reuse the cached table if it exists and has
    exactly the expected columns. The table holds a row-number game index
    (never the real id), the checkpoint N, the label and the features.
    """
    expected = META_COLUMNS + GROUPS["P3"]
    if CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH)
        if list(cached.columns) == expected:
            print(f"Reusing the cached feature table {CACHE_PATH} ({len(cached)} rows).")
            return cached, None, 0.0
        print("The cached feature table has different columns, so it is rebuilt.")
    print("Replaying the games (this takes a few minutes)...")
    start = time.perf_counter()
    rows, skipped = build_checkpoint_rows(games, CHECKPOINTS, progress_every=2000)
    seconds = time.perf_counter() - start
    rows = rows[expected]
    CACHE_PATH.parent.mkdir(exist_ok=True)
    rows.to_csv(CACHE_PATH, index=False)
    print(f"Replay took {seconds:.0f} s. Saved {CACHE_PATH} ({len(rows)} rows).")
    return rows, skipped, seconds


def material_rule_accuracy(rows):
    """Predict white if White is ahead in material, black if behind, white if level."""
    predicted = np.where(rows["material_balance"] < 0, "black", "white")
    return float((predicted == rows["winner"].to_numpy()).mean())


def main():
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)   # known, harmless
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    run_start = time.perf_counter()

    # --- 1. Data and the game-level split ---------------------------------------
    games, dropped = drop_duplicate_ids(pd.read_csv(DATA_PATH))
    print(f"Loaded {len(games) + dropped} rows; dropped {dropped} repeated game ids; "
          f"{len(games)} games remain.")
    parsed = games["increment_code"].map(parse_increment)
    games["base_minutes"] = [base for base, _ in parsed]
    games["increment_seconds"] = [inc for _, inc in parsed]
    games["rated"] = games["rated"].astype(int)
    # Only these columns are passed on: the moves (read only up to ply 60),
    # the pre-game columns and the label. No id, turns or end-of-game column.
    games = games[["moves", "white_rating", "black_rating", "base_minutes",
                   "increment_seconds", "rated", "winner"]]
    development_games, test_games = split_like_original(games)

    # --- 2. Positions after N plies --------------------------------------------------
    rows, skipped, replay_seconds = load_or_build_features(games)
    if skipped is not None:
        print(f"Games skipped because their moves could not be parsed: {skipped}")
    check_no_leakage(GROUPS["P3"])
    dev_rows, test_rows = split_rows(rows, development_games.index, test_games.index)

    print("\nNOTE: the set of games differs between N. A game that ends before N plies "
          "has no row at N, so scores at different N are not directly comparable. "
          "No information about game length is used as a feature.")
    print("\n=== Games per checkpoint ===")
    total_games = rows.loc[rows["ply"] == 0, "game_index"].nunique()
    for n in (0,) + CHECKPOINTS:
        n_dev = int((dev_rows["ply"] == n).sum())
        n_test = int((test_rows["ply"] == n).sum())
        print(f"N = {n:2d}: {n_dev + n_test:5d} games reach it "
              f"({(n_dev + n_test) / total_games:.1%} of all), train {n_dev}, test {n_test}")

    results = []
    fitted = {}         # (N, model, group) -> (pipeline, X_test, y_test)

    def add(n, model, group, cv_loss, test_scores, n_train, n_test):
        results.append({"N": n, "model": model, "feature_group": group, "cv_log_loss": cv_loss,
                        **test_scores, "n_train": n_train, "n_test": n_test})

    def fit_and_score(n, name, group, settings, dev, test):
        """CV on development rows, refit on all of them, score once on test."""
        columns = GROUPS[group]
        X_dev, X_test = dev[columns], test[columns]
        y_dev = encode_labels(dev["winner"])
        pipe = make_pipeline(name, columns, settings)
        cv_loss = cv_log_loss(pipe, X_dev, y_dev)
        pipe.fit(X_dev, y_dev)
        y_test = test["winner"].to_numpy()
        add(n, name, group, cv_loss, scores(y_test, model_proba(pipe, X_test)), len(dev), len(test))
        fitted[(n, name, group)] = (pipe, X_test, y_test)
        return cv_loss

    best_by_n = {}
    for n in (0,) + CHECKPOINTS:
        start = time.perf_counter()
        dev = dev_rows[dev_rows["ply"] == n].reset_index(drop=True)
        test = test_rows[test_rows["ply"] == n].reset_index(drop=True)
        y_test = test["winner"].to_numpy()
        print(f"\n--- N = {n} ({len(dev)} train, {len(test)} test) ---", flush=True)

        # Baselines: always white, with the training rows' class shares as probabilities.
        shares = np.array([np.mean(dev["winner"].to_numpy() == c) for c in CLASSES])
        add(n, "Always white (class shares)", "none", np.nan,
            scores(y_test, np.tile(shares, (len(test), 1))), len(dev), len(test))
        # The material rule gives a guess but no probabilities, so only accuracy.
        add(n, "Material rule", "material_balance", np.nan,
            {"test_accuracy": material_rule_accuracy(test), "test_log_loss": np.nan,
             "test_brier": np.nan, "draw_recall": np.nan}, len(dev), len(test))

        # Tune each model on P3 (all features) by 5-fold CV on development rows.
        # At N = 0 the position is always the start position, so only the
        # ratings models (P0) are fitted there, as the pre-game reference.
        chosen = {}
        for name in MODELS:
            grid_cv = []
            for settings in GRIDS[name]:
                group = "P0" if n == 0 else "P3"
                X_dev = dev[GROUPS[group]]
                grid_cv.append((cv_log_loss(make_pipeline(name, GROUPS[group], settings),
                                            X_dev, encode_labels(dev["winner"])), settings))
            chosen[name] = min(grid_cv, key=lambda pair: pair[0])[1]
            group = "P0" if n == 0 else "P3"
            cv_loss = fit_and_score(n, name, group, chosen[name], dev, test)
            print(f"{name}: {settings_text(chosen[name])}, CV log loss {cv_loss:.4f}", flush=True)

        # Ablation: logistic regression and gradient boosting on P0, P1, P2
        # (P3 is already done), with the settings chosen above. Logistic
        # regression on P0 is also the ratings-only baseline.
        if n > 0:
            for name in ("Logistic regression", "Gradient boosting"):
                for group in ("P0", "P1", "P2"):
                    fit_and_score(n, name, group, chosen[name], dev, test)

        # Best model at this N by CV log loss among the tuned models (never by test).
        group = "P0" if n == 0 else "P3"
        tuned = [r for r in results if r["N"] == n and r["model"] in MODELS and r["feature_group"] == group]
        best = min(tuned, key=lambda r: r["cv_log_loss"])
        best_by_n[n] = (best["model"], group)
        print(f"Best at N = {n} by CV: {best['model']} ({time.perf_counter() - start:.0f} s for this N)")

    table = pd.DataFrame(results)
    print("\n=== Results on the test rows ===")
    rounding = {"cv_log_loss": 4, "test_accuracy": 3, "test_log_loss": 4, "test_brier": 4, "draw_recall": 3}
    print(table.round(rounding).to_string(index=False))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    table.round(4).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH}")

    # --- Extra analysis at N = 30, and bootstraps at N = 10, 30 and 60 --------------
    name, group = best_by_n[30]
    pipe, X_test, y_test = fitted[(30, name, group)]
    proba = model_proba(pipe, X_test)
    print(f"\n=== Extra analysis at N = 30: {name} on {group} (best by CV) ===")
    predicted = np.array(CLASSES)[proba.argmax(axis=1)]
    matrix = pd.crosstab(pd.Series(y_test, name="actual"), pd.Series(predicted, name="predicted"))
    print("Confusion matrix:")
    print(matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_string())
    print("\nCalibration of P(white wins), 5 equal-count bins:")
    print(calibration_table(proba[:, CLASSES.index("white")], y_test == "white").round(3).to_string(index=False))
    print("\nPermutation importance (increase in log loss, 5 repeats, top 10):")
    start = time.perf_counter()
    print(permutation_importance_log_loss(pipe, X_test, y_test).head(10).round(4).to_string(index=False))
    print(f"(took {time.perf_counter() - start:.0f} s)")

    print("\nPaired bootstrap, best model minus ratings-only logistic regression "
          "(per-game log loss, 1000 resamples, negative = best model better):")
    for n in (10, 30, 60):
        best_name, best_group = best_by_n[n]
        b_pipe, b_X, b_y = fitted[(n, best_name, best_group)]
        r_pipe, r_X, r_y = fitted[(n, "Logistic regression", "P0")]
        r = paired_bootstrap(per_game_log_loss(b_y, model_proba(b_pipe, b_X)),
                             per_game_log_loss(r_y, model_proba(r_pipe, r_X)))
        print(f"N = {n}: {best_name} minus ratings-only: {r['mean_diff']:.4f} "
              f"(95% range {r['low']:.4f} to {r['high']:.4f})")

    # --- Chart ------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed, so no chart was made.")
    else:
        ns = (0,) + CHECKPOINTS

        def accuracy(model, group):
            return [float(table[(table["N"] == n) & (table["model"] == model(n))
                                & (table["feature_group"] == group(n))]["test_accuracy"].iloc[0])
                    for n in ns]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ns, accuracy(lambda n: best_by_n[n][0], lambda n: best_by_n[n][1]),
                marker="o", color="#2F6F5E", label="Best model by CV")
        ax.plot(ns, accuracy(lambda n: "Logistic regression", lambda n: "P0"),
                marker="o", color="#8A8F98", label="Ratings only (logistic regression)")
        ax.plot(ns, accuracy(lambda n: "Material rule", lambda n: "material_balance"),
                marker="s", color="#B5533C", label="Material rule")
        ax.plot(ns, accuracy(lambda n: "Always white (class shares)", lambda n: "none"),
                linestyle="--", color="#1F2933", label="Always white")
        ax.set_xlabel("Plies played (N)")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Predicting the result from the position after N plies\n"
                     "(the games differ between N: shorter games drop out)")
        ax.set_xticks(ns)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(CURVE_PATH, dpi=120)
        print(f"\nSaved {CURVE_PATH}")

    print(f"\nTotal run time: {time.perf_counter() - run_start:.0f} s "
          f"(of which game replay: {replay_seconds:.0f} s)")


if __name__ == "__main__":
    main()
