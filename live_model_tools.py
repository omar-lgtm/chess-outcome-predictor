"""
Tools for the "Live game" win-probability model: one model that predicts
the final result from the position at ANY ply of a game, and a helper that
returns the whole probability curve for one game.

Features come from inplay_tools.position_features (only the board after
that many plies plus the pre-game columns) and the ply number. Nothing from
later moves, the game length or how the game ended is ever used.

No Streamlit code here.
"""

import json
import re
from pathlib import Path

import chess
import joblib
import pandas as pd
import sklearn

from inplay_tools import parse_moves, position_features

WITH_RATINGS_FILE = "live_model_with_ratings.joblib"
POSITION_ONLY_FILE = "live_model_position_only.joblib"
META_FILE = "live_model_meta.json"

PRE_GAME_KEYS = ("white_rating", "black_rating", "base_minutes", "increment_seconds", "rated")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def clean_move_tokens(moves_text):
    """
    Split SAN text into move tokens, dropping move numbers ("1.", "12...",
    or the "1." in "1.e4") and result markers ("1-0", "0-1", "1/2-1/2", "*").
    """
    tokens = []
    for token in str(moves_text).split():
        if token in RESULT_TOKENS or re.fullmatch(r"\d+\.+", token):
            continue
        tokens.append(re.sub(r"^\d+\.+", "", token))
    return tokens


def rows_at_plies(game_index, moves_text, pre_game_columns, plies):
    """
    Replay one game and record the features after exactly each ply in
    `plies` that the game reaches. Only moves up to the largest wanted ply
    are read. Returns [] if those moves cannot be parsed.
    """
    plies = sorted(set(plies))
    if not plies:
        return []
    moves = parse_moves(" ".join(clean_move_tokens(moves_text)[: plies[-1]]))
    if moves is None:
        return []
    wanted = set(plies)
    pre_game = {key: pre_game_columns.get(key) for key in PRE_GAME_KEYS}
    board = chess.Board()
    rows = []
    for ply, move in enumerate(moves, start=1):
        board.push(move)
        if ply in wanted:
            rows.append({"game_index": game_index, "ply": ply,
                         **position_features(board, **pre_game)})
    return rows


def build_live_rows(game_index, moves_text, pre_game_columns, step=5, max_ply=120):
    """
    Feature rows for one game at plies step, 2*step, ... up to max_ply (and
    never beyond the end of the game). Each row has the position features,
    "ply" and "game_index". Returns [] if the moves cannot be parsed.
    """
    return rows_at_plies(game_index, moves_text, pre_game_columns,
                         range(step, max_ply + 1, step))


def ordered_features(table, feature_names, model=None):
    """
    Feature-order guard: return the columns in exactly the stored order.
    Raises ValueError if a column is missing, or if the model was trained
    with a different column order than the one stored in the metadata.
    """
    missing = [name for name in feature_names if name not in table.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    trained_on = getattr(model, "feature_names_in_", None)
    if trained_on is not None and list(trained_on) != list(feature_names):
        raise ValueError("The model was trained with a different feature order "
                         "than the one stored in live_model_meta.json.")
    return table[list(feature_names)]


def load_bundle(path_prefix="."):
    """
    Load both saved models and the metadata. Raises an error if the
    scikit-learn version recorded when the models were saved differs from
    the installed one, because pickled models may then behave differently.
    """
    folder = Path(path_prefix)
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    if meta["sklearn_version"] != sklearn.__version__:
        raise RuntimeError(
            f"The live models were saved with scikit-learn {meta['sklearn_version']}, "
            f"but scikit-learn {sklearn.__version__} is installed. Install the same "
            "version or retrain the models.")
    return {
        "with_ratings": joblib.load(folder / WITH_RATINGS_FILE),
        "position_only": joblib.load(folder / POSITION_ONLY_FILE),
        "meta": meta,
    }


def terminal_result(board):
    """
    The real result if the game is over on this board, else None:
    "white" or "black" for checkmate (the side that delivered it), "draw"
    for stalemate or insufficient material.
    """
    if board.is_checkmate():
        return "black" if board.turn == chess.WHITE else "white"   # side to move is mated
    if board.is_stalemate() or board.is_insufficient_material():
        return "draw"
    return None


def predict_curve(moves_text, white_rating=None, black_rating=None, base_minutes=None,
                  increment_seconds=None, rated=None, bundle=None, max_ply=None):
    """
    Win probabilities after every ply of one game, from 0 (the start
    position) to the last ply (or max_ply). Returns a DataFrame with ply,
    p_black, p_draw, p_white, expected_score_white (p_white + 0.5 * p_draw),
    terminal_result, before_training and beyond_training.

    Uses the with-ratings model when both ratings are given, otherwise the
    position-only model. Raises ValueError for empty or illegal moves.

    Finished positions: where the board is checkmate, stalemate or
    insufficient material, the model's numbers are replaced by the real
    result (probability 1 for that result, 0 for the others), and
    terminal_result says "white", "black" or "draw". It is None on every
    other row, and those rows keep the model's numbers unchanged.

    Training range: the models were trained on plies from the meta
    "min_ply" (5 if it is not stored) to the meta "max_ply". before_training
    is True for earlier plies and beyond_training for later ones. Both are
    still predicted, but the model has not seen positions like them.
    """
    if bundle is None:
        bundle = load_bundle()
    tokens = clean_move_tokens(moves_text)
    if not tokens:
        raise ValueError("No moves were given.")
    if max_ply is not None:
        tokens = tokens[:max_ply]

    meta = bundle["meta"]
    use_ratings = white_rating is not None and black_rating is not None
    model = bundle["with_ratings"] if use_ratings else bundle["position_only"]
    feature_names = meta["features_with_ratings" if use_ratings else "features_position_only"]
    pre_game = dict(white_rating=white_rating, black_rating=black_rating,
                    base_minutes=base_minutes, increment_seconds=increment_seconds, rated=rated)

    # Replay move by move, recording the position before any move (ply 0)
    # and after each move.
    board = chess.Board()
    rows = [{"ply": 0, **position_features(board, **pre_game)}]
    results = [terminal_result(board)]
    for ply, san in enumerate(tokens, start=1):
        try:
            board.push(board.parse_san(san))
        except ValueError:
            raise ValueError(f"Move {ply} ('{san}') is illegal or unreadable.") from None
        rows.append({"ply": ply, **position_features(board, **pre_game)})
        results.append(terminal_result(board))

    table = pd.DataFrame(rows).astype(float)
    X = ordered_features(table, feature_names, model)
    proba = model.predict_proba(X)
    # The models are trained on number labels; the metadata maps them to names.
    names = [meta["classes"][str(int(c))] for c in model.classes_]
    curve = pd.DataFrame({"ply": table["ply"].astype(int)})
    for label in ("black", "draw", "white"):
        curve[f"p_{label}"] = proba[:, names.index(label)]

    # Finished positions get the real result instead of the model's guess.
    curve["terminal_result"] = pd.Series(results, dtype=object)
    for index, result in enumerate(results):
        if result is not None:
            for label in ("black", "draw", "white"):
                curve.loc[index, f"p_{label}"] = 1.0 if label == result else 0.0

    curve["expected_score_white"] = curve["p_white"] + 0.5 * curve["p_draw"]
    # True for plies earlier or later than any the models were trained on.
    curve["before_training"] = curve["ply"] < meta.get("min_ply", 5)
    curve["beyond_training"] = curve["ply"] > meta["max_ply"]
    return curve
