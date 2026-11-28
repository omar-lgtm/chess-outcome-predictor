"""
"Live game" win-probability model: ONE model for every ply of a game.

Steps:
1. Load data/games.csv, drop repeated ids, split at GAME level (70/15/15
   stratified on winner, random_state=42) with the shared helpers. Every row
   of a game stays in that game's split.
2. Replay each game once and record the position every 5 plies up to ply 120.
   The table is cached in data/live_features.csv (git-ignored).
3. Tune gradient boosting by 3-fold GroupKFold on game_index over the
   development rows; fit a WITH RATINGS and a POSITION ONLY model, plus a
   pooled logistic regression and a ratings-only baseline; score once on the
   test rows; save the two gradient boosting models and their metadata.
4. Report by ply bucket, compare with the per-checkpoint models, check
   positions between the grid plies, calibration, behaviour over the game,
   paired bootstrap by game and permutation importance.

Run it with:  python live_model_experiments.py
"""

import datetime
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from general_experiments import (
    CLASSES, DATA_PATH, calibration_table, drop_duplicate_ids, encode_labels,
    make_pipeline, model_proba, multiclass_log_loss, paired_bootstrap, parse_increment,
    per_game_log_loss, permutation_importance_log_loss, scores, settings_text,
    split_like_original,
)
from inplay_tools import check_no_leakage, feature_groups, split_rows
from live_model_tools import (
    META_FILE, POSITION_ONLY_FILE, WITH_RATINGS_FILE, build_live_rows, rows_at_plies,
)

CACHE_PATH = Path("data/live_features.csv")
RESULTS_PATH = Path("docs/live_model_results.csv")
CURVE_PATH = Path("docs/live_model_curve.png")
INPLAY_RESULTS = Path("docs/inplay_results.csv")
STEP, MAX_PLY = 5, 120
OFF_GRID_PLIES = list(range(7, 118, 5))        # 7, 12, ..., 117
MAX_MODEL_MB = 20
RANDOM_STATE = 42

GROUPS = feature_groups()
WITH_RATINGS = GROUPS["P3"] + ["ply"]
POSITION_ONLY = [c for c in GROUPS["P3"] if c not in GROUPS["P0"]] + ["ply"]
RATINGS_ONLY = GROUPS["P0"] + ["ply"]
GRID = [{"learning_rate": rate, "max_depth": depth} for rate in (0.05, 0.1) for depth in (4, 6)]
BUCKETS = [("5 to 20", 20), ("25 to 40", 40), ("45 to 60", 60), ("65 to 90", 90), ("95 to 120", 120)]
PRE_GAME_COLUMNS = ["white_rating", "black_rating", "base_minutes", "increment_seconds", "rated"]


def gradient_boosting(settings):
    """
    The gradient boosting model. sklearn's built-in early stopping holds out
    a random 10 percent of ROWS, which would put positions from the same game
    on both sides, so it is switched off and a fixed max_iter is used.
    """
    return HistGradientBoostingClassifier(
        early_stopping=False, max_iter=300, random_state=RANDOM_STATE, **settings)


def bucket_of(ply):
    """Name of the ply bucket a ply falls in (by its upper limit)."""
    for name, upper in BUCKETS:
        if ply <= upper:
            return name
    return BUCKETS[-1][0]


def material_rule(rows):
    """White if White is ahead in material, black if behind, white if level."""
    return np.where(rows["material_balance"] < 0, "black", "white")


def expected_calibration_error(table):
    """Mean absolute gap between predicted and actual, weighted by games per bin."""
    return float((table["games"] * (table["mean_predicted"] - table["mean_actual"]).abs()).sum()
                 / table["games"].sum())


def load_games():
    """Deduplicated games with the parsed time control; no ids or end-of-game columns kept."""
    games, dropped = drop_duplicate_ids(pd.read_csv(DATA_PATH))
    parsed = games["increment_code"].map(parse_increment)
    games["base_minutes"] = [base for base, _ in parsed]
    games["increment_seconds"] = [inc for _, inc in parsed]
    games["rated"] = games["rated"].astype(int)
    return games[["moves", "winner"] + PRE_GAME_COLUMNS], dropped


def replay(games, make_rows, progress_every=2000):
    """Replay every game with make_rows(index, game) and collect the rows."""
    rows, skipped = [], 0
    for done, (index, game) in enumerate(games.iterrows(), start=1):
        if done % progress_every == 0:
            print(f"  replayed {done} of {len(games)} games", flush=True)
        game_rows = make_rows(index, game)
        if not game_rows:
            skipped += 1
        for row in game_rows:
            row["winner"] = game["winner"]
        rows.extend(game_rows)
    return pd.DataFrame(rows), skipped


def main():
    from scipy.optimize import OptimizeWarning
    warnings.filterwarnings("ignore", category=OptimizeWarning)   # known, harmless
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    run_start = time.perf_counter()
    rounding = {"accuracy": 3, "log_loss": 4, "brier": 4}

    # --- a) Data, game-level split, replay ---------------------------------------
    games, dropped = load_games()
    print(f"Loaded games; dropped {dropped} repeated ids; {len(games)} games remain.")
    development_games, test_games = split_like_original(games)

    expected_columns = ["game_index", "ply", "winner"] + GROUPS["P3"]
    replay_seconds = 0.0
    if CACHE_PATH.exists() and list(pd.read_csv(CACHE_PATH, nrows=0).columns) == expected_columns:
        rows = pd.read_csv(CACHE_PATH)
        print(f"Reusing the cached feature table {CACHE_PATH}.")
    else:
        print(f"Replaying the games, a row every {STEP} plies up to ply {MAX_PLY}...", flush=True)
        start = time.perf_counter()
        rows, skipped = replay(games, lambda i, g: build_live_rows(
            i, g["moves"], g[PRE_GAME_COLUMNS].to_dict(), STEP, MAX_PLY))
        replay_seconds = time.perf_counter() - start
        rows = rows[expected_columns]
        rows.to_csv(CACHE_PATH, index=False)
        print(f"Replay took {replay_seconds:.0f} s; games with no rows "
              f"(unparseable, or shorter than {STEP} plies): {skipped}.")
    dev, test = split_rows(rows, development_games.index, test_games.index)
    dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
    print(f"Rows: {len(rows)} from {rows['game_index'].nunique()} games. "
          f"Development: {len(dev)} rows, {dev['game_index'].nunique()} games. "
          f"Test: {len(test)} rows, {test['game_index'].nunique()} games.")

    # --- b) Feature sets ---------------------------------------------------------------
    for columns in (WITH_RATINGS, POSITION_ONLY, RATINGS_ONLY):
        check_no_leakage(columns)
    print(f"WITH RATINGS: {len(WITH_RATINGS)} features; POSITION ONLY: {len(POSITION_ONLY)}; "
          f"ratings-only baseline: {len(RATINGS_ONLY)}. All pass the leakage guard.")
    y_dev = encode_labels(dev["winner"])
    y_test = test["winner"].to_numpy()

    # --- c) Tune gradient boosting with GroupKFold on game_index -------------------------
    folds = list(GroupKFold(n_splits=3).split(dev, y_dev, groups=dev["game_index"]))
    print("\n=== Tuning gradient boosting (3-fold GroupKFold by game, WITH RATINGS) ===", flush=True)
    tune_start = time.perf_counter()
    tuning = []
    for settings in GRID:
        fold_losses = []
        for fit_idx, check_idx in folds:
            start = time.perf_counter()
            model = gradient_boosting(settings).fit(dev.loc[fit_idx, WITH_RATINGS], y_dev[fit_idx])
            proba = model_proba(model, dev.loc[check_idx, WITH_RATINGS])
            fold_losses.append(multiclass_log_loss(dev.loc[check_idx, "winner"], proba))
            print(f"  {settings_text(settings)}: fold log loss {fold_losses[-1]:.4f} "
                  f"({time.perf_counter() - start:.0f} s)", flush=True)
        tuning.append((float(np.mean(fold_losses)), settings))
        print(f"{settings_text(settings)}: mean CV log loss {tuning[-1][0]:.4f}", flush=True)
    cv_loss, best_settings = min(tuning, key=lambda pair: pair[0])
    print(f"Chosen: {settings_text(best_settings)} (CV log loss {cv_loss:.4f}); "
          f"tuning took {time.perf_counter() - tune_start:.0f} s")

    # --- d) Refit on all development rows, score once on test, save --------------------------
    print("\n=== Final models (fit on all development rows) ===", flush=True)
    models = {}
    for name, columns, estimator in [
        ("GB with ratings", WITH_RATINGS, gradient_boosting(best_settings)),
        ("GB position only", POSITION_ONLY, gradient_boosting(best_settings)),
        ("Pooled logistic regression", WITH_RATINGS,
         make_pipeline("Logistic regression", WITH_RATINGS, {"C": 0.1})),
        ("Ratings-only logistic regression", RATINGS_ONLY,
         make_pipeline("Logistic regression", RATINGS_ONLY, {"C": 0.1})),
    ]:
        start = time.perf_counter()
        models[name] = (estimator.fit(dev[columns], y_dev), columns)
        print(f"{name}: fitted in {time.perf_counter() - start:.0f} s", flush=True)

    joblib.dump(models["GB with ratings"][0], WITH_RATINGS_FILE)
    joblib.dump(models["GB position only"][0], POSITION_ONLY_FILE)
    meta = {
        "classes": {str(i): label for i, label in enumerate(CLASSES)},
        "features_with_ratings": WITH_RATINGS,
        "features_position_only": POSITION_ONLY,
        "ply_step": STEP,
        "max_ply": MAX_PLY,
        "sklearn_version": sklearn.__version__,
        "date": datetime.date.today().isoformat(),
        "settings": {"max_iter": 300, "early_stopping": False, **best_settings},
    }
    Path(META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for path in (WITH_RATINGS_FILE, POSITION_ONLY_FILE, META_FILE):
        print(f"Saved {path}: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    for path in (WITH_RATINGS_FILE, POSITION_ONLY_FILE):
        if os.path.getsize(path) > MAX_MODEL_MB * 1024 * 1024:
            raise SystemExit(f"STOP: {path} is larger than {MAX_MODEL_MB} MB.")

    # --- e) Test scores by ply bucket ------------------------------------------------------
    test["bucket"] = test["ply"].map(bucket_of)
    proba = {name: model_proba(model, test[columns]) for name, (model, columns) in models.items()}
    results = []
    for name, p in proba.items():
        for bucket in [b for b, _ in BUCKETS] + ["overall"]:
            mask = np.ones(len(test), bool) if bucket == "overall" else (test["bucket"] == bucket).to_numpy()
            s = scores(y_test[mask], p[mask])
            results.append({"ply_bucket": bucket, "model": name, "accuracy": s["test_accuracy"],
                            "log_loss": s["test_log_loss"], "brier": s["test_brier"],
                            "n_rows": int(mask.sum())})
    rule = material_rule(test)
    for bucket in [b for b, _ in BUCKETS] + ["overall"]:
        mask = np.ones(len(test), bool) if bucket == "overall" else (test["bucket"] == bucket).to_numpy()
        results.append({"ply_bucket": bucket, "model": "Material rule",
                        "accuracy": float((rule[mask] == y_test[mask]).mean()),
                        "log_loss": np.nan, "brier": np.nan, "n_rows": int(mask.sum())})
    table = pd.DataFrame(results)
    print("\n=== Test results by ply bucket ===")
    print(table.round(rounding).to_string(index=False))
    table.round(4).to_csv(RESULTS_PATH, index=False)
    print(f"Saved {RESULTS_PATH}")

    # --- f) Comparison with the per-checkpoint models -------------------------------------
    print("\n=== Pooled models vs per-checkpoint models (same test rows at each N) ===")
    inplay = pd.read_csv(INPLAY_RESULTS)
    for n in (10, 20, 30, 40, 60):
        mask = (test["ply"] == n).to_numpy()
        line = f"N = {n:2d} ({mask.sum()} test rows): "
        for per_name, pooled_name in [("Gradient boosting", "GB with ratings"),
                                      ("Logistic regression", "Pooled logistic regression")]:
            ref = inplay[(inplay["N"] == n) & (inplay["model"] == per_name)
                         & (inplay["feature_group"] == "P3")].iloc[0]
            s = scores(y_test[mask], proba[pooled_name][mask])
            line += (f"| {per_name}: per-N {ref['test_log_loss']:.4f} / {ref['test_accuracy']:.3f}, "
                     f"pooled {s['test_log_loss']:.4f} / {s['test_accuracy']:.3f} ")
            if int(ref["n_test"]) != int(mask.sum()):
                line += f"[row count differs: per-N file has {int(ref['n_test'])}] "
        print(line)
    print("(each pair is log loss / accuracy)")

    # --- g) Positions between the grid plies (test games only) ------------------------------
    print("\n=== Interpolation check: plies 7, 12, ..., 117 (test games only) ===", flush=True)
    start = time.perf_counter()
    off_grid, _ = replay(games.loc[test_games.index], lambda i, g: rows_at_plies(
        i, g["moves"], g[PRE_GAME_COLUMNS].to_dict(), OFF_GRID_PLIES), progress_every=1000)
    off_proba = model_proba(models["GB with ratings"][0], off_grid[WITH_RATINGS])
    off_grid["bucket"] = off_grid["ply"].map(bucket_of)
    print(f"Built {len(off_grid)} off-grid rows in {time.perf_counter() - start:.0f} s")
    for bucket, _ in BUCKETS:
        on_mask = (test["bucket"] == bucket).to_numpy()
        off_mask = (off_grid["bucket"] == bucket).to_numpy()
        on_loss = multiclass_log_loss(y_test[on_mask], proba["GB with ratings"][on_mask])
        off_loss = multiclass_log_loss(off_grid["winner"].to_numpy()[off_mask], off_proba[off_mask])
        print(f"{bucket:>10}: on-grid log loss {on_loss:.4f} ({on_mask.sum()} rows), "
              f"off-grid {off_loss:.4f} ({off_mask.sum()} rows)")

    # --- h) Calibration ------------------------------------------------------------------------
    print("\n=== Calibration of P(white wins), WITH RATINGS model ===")
    p_white = proba["GB with ratings"][:, CLASSES.index("white")]
    is_white = y_test == "white"
    eces = {}
    for bucket in ["overall"] + [b for b, _ in BUCKETS]:
        mask = np.ones(len(test), bool) if bucket == "overall" else (test["bucket"] == bucket).to_numpy()
        cal = calibration_table(p_white[mask], is_white[mask])
        eces[bucket] = expected_calibration_error(cal)
        print(f"\n{bucket}: expected calibration error {eces[bucket]:.4f}")
        print(cal.round(3).to_string(index=False))
    if max(eces.values()) > 0.03:
        print("\nSome bucket has a calibration error above 0.03, so an isotonic-calibrated "
              "version is fitted for comparison (not saved).", flush=True)
        start = time.perf_counter()
        calibrated = CalibratedClassifierCV(gradient_boosting(best_settings), method="isotonic", cv=folds)
        calibrated.fit(dev[WITH_RATINGS], y_dev)
        cal_proba = model_proba(calibrated, test[WITH_RATINGS])
        print(f"(fitted in {time.perf_counter() - start:.0f} s)")
        for bucket in ["overall"] + [b for b, _ in BUCKETS]:
            mask = np.ones(len(test), bool) if bucket == "overall" else (test["bucket"] == bucket).to_numpy()
            after = expected_calibration_error(calibration_table(
                cal_proba[mask, CLASSES.index("white")], is_white[mask]))
            print(f"{bucket:>10}: ECE {eces[bucket]:.4f} -> {after:.4f}; log loss "
                  f"{multiclass_log_loss(y_test[mask], proba['GB with ratings'][mask]):.4f} -> "
                  f"{multiclass_log_loss(y_test[mask], cal_proba[mask]):.4f}")
    else:
        print("\nNo bucket has a calibration error above 0.03, so no calibrated version is fitted.")

    # --- i) Behaviour over the game ------------------------------------------------------------
    print("\n=== Behaviour over the game (test games, WITH RATINGS) ===")
    behaviour = test[["game_index", "ply"]].copy()
    rows_idx = np.arange(len(test))
    behaviour["p_actual"] = proba["GB with ratings"][rows_idx, [CLASSES.index(w) for w in y_test]]
    behaviour["p_white"] = p_white
    behaviour = behaviour.sort_values(["game_index", "ply"])
    previous = behaviour.groupby("game_index")[["ply", "p_white"]].shift(1)
    consecutive = previous["ply"] == behaviour["ply"] - STEP
    behaviour["change"] = np.where(consecutive, (behaviour["p_white"] - previous["p_white"]).abs(), np.nan)
    by_ply = behaviour.groupby("ply").agg(rows=("p_actual", "size"),
                                          mean_p_actual=("p_actual", "mean"),
                                          mean_abs_change=("change", "mean")).reset_index()
    print(by_ply.round(4).to_string(index=False))

    # --- j) Paired bootstrap by game -------------------------------------------------------------
    print("\n=== Paired bootstrap by game (1000 resamples of test games, seed 42) ===")
    losses = pd.DataFrame({"game_index": test["game_index"]})
    for name in ("GB with ratings", "GB position only", "Ratings-only logistic regression"):
        losses[name] = per_game_log_loss(y_test, proba[name])
    per_game = losses.groupby("game_index").mean()     # each game's mean log loss over its rows
    for a, b in [("GB with ratings", "Ratings-only logistic regression"),
                 ("GB with ratings", "GB position only")]:
        r = paired_bootstrap(per_game[a].to_numpy(), per_game[b].to_numpy())
        print(f"{a} minus {b}: {r['mean_diff']:.4f} (95% range {r['low']:.4f} to {r['high']:.4f}; "
              "negative = first is better)")

    # --- k) Permutation importance ---------------------------------------------------------------
    print("\n=== Permutation importance, WITH RATINGS (test rows, 3 repeats, top 10) ===", flush=True)
    start = time.perf_counter()
    importance = permutation_importance_log_loss(
        models["GB with ratings"][0], test[WITH_RATINGS], y_test, n_repeats=3)
    print(importance.head(10).round(4).to_string(index=False))
    print(f"(took {time.perf_counter() - start:.0f} s)")

    # --- l) Chart --------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed, so no chart was made.")
    else:
        plies = sorted(test["ply"].unique())
        by = test["ply"].to_numpy()
        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, colour in [("GB with ratings", "#2F6F5E"), ("GB position only", "#6FA38F"),
                             ("Ratings-only logistic regression", "#8A8F98")]:
            predicted = np.array(CLASSES)[proba[name].argmax(axis=1)]
            left.plot(plies, [(predicted[by == p] == y_test[by == p]).mean() for p in plies],
                      marker=".", color=colour, label=name)
            right.plot(plies, [proba[name][by == p][np.arange((by == p).sum()),
                               [CLASSES.index(w) for w in y_test[by == p]]].mean() for p in plies],
                       marker=".", color=colour, label=name)
        left.plot(plies, [(rule[by == p] == y_test[by == p]).mean() for p in plies],
                  marker=".", color="#B5533C", label="Material rule")
        left.set(xlabel="Ply", ylabel="Test accuracy", title="Accuracy by ply (test games)")
        right.set(xlabel="Ply", ylabel="Mean probability of the actual result",
                  title="Probability given to what really happened")
        for ax in (left, right):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Live model: the games differ between plies (shorter games drop out)", fontsize=10)
        fig.tight_layout()
        fig.savefig(CURVE_PATH, dpi=120)
        print(f"\nSaved {CURVE_PATH}")

    print(f"\nTotal run time: {time.perf_counter() - run_start:.0f} s "
          f"(game replay {replay_seconds:.0f} s, tuning included)")


if __name__ == "__main__":
    main()
