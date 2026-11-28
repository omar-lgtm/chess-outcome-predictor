"""
Quick checks for inplay_tools, on made-up positions and move lists only.

Run it with:  python test_inplay_tools.py
"""

import chess
import pandas as pd

from inplay_tools import (
    build_checkpoint_rows, check_no_leakage, feature_groups, parse_moves,
    position_features, split_rows,
)

GROUPS = feature_groups()
P3 = GROUPS["P3"]


def features(fen=None):
    board = chess.Board(fen) if fen else chess.Board()
    return position_features(board, 1500, 1400, 10, 0, 1)


def made_up_games(move_lists):
    """A games table with invented ratings and the given move lists."""
    n = len(move_lists)
    return pd.DataFrame({
        "moves": move_lists,
        "white_rating": [1500] * n, "black_rating": [1450] * n,
        "base_minutes": [10.0] * n, "increment_seconds": [0.0] * n, "rated": [1] * n,
        "winner": ["white"] * n,
    })


# Test 1: the start position.
f = features()
assert f["pawns_white"] == 8 and f["pawns_black"] == 8
assert f["material_balance"] == 0 and f["material_total"] == 2 * (8 + 6 + 6 + 10 + 9)
assert f["white_to_move"] == 1
assert f["mobility_white"] == 20 and f["mobility_black"] == 20
assert f["hanging_count_white"] == 0 and f["hanging_count_black"] == 0
assert f["castled_approx_white"] == 0 and f["castled_approx_black"] == 0
for side in ("white", "black"):
    assert f[f"can_castle_kingside_{side}"] == 1 and f[f"can_castle_queenside_{side}"] == 1
    assert f[f"king_on_start_{side}"] == 1 and f[f"minors_off_start_{side}"] == 0
assert f["rating_diff"] == 100 and f["is_check"] == 0
print("Test 1 passed: start position has 8 pawns each, balance 0, 20 moves, nothing hanging")

# Test 2: the main leakage test. Two made-up games share their first 10
# plies and then differ completely. Features after 10 plies must be equal.
shared = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7"
game_a = shared + " Re1 b5 Bb3 d6 c3 O-O"
game_b = shared + " d4 exd4 e5 Ne4 Qe2 Nc5 Bxc6 dxc6"
rows, skipped = build_checkpoint_rows(made_up_games([game_a, game_b]), checkpoints=(10,))
assert skipped == 0
at_10 = rows[rows["ply"] == 10]
assert len(at_10) == 2
assert at_10.iloc[0][P3].equals(at_10.iloc[1][P3])
# And the rows really do differ later, so the test is not trivial.
rows_later, _ = build_checkpoint_rows(made_up_games([game_a, game_b]), checkpoints=(14,))
later = rows_later[rows_later["ply"] == 14]
assert not later.iloc[0][P3].equals(later.iloc[1][P3])
print("Test 2 passed: features after 10 plies ignore every move after ply 10")

# Test 3: hanging pieces. White bishop b5 is attacked by the a6 pawn and not
# defended (hanging, worth 3). White knight e5 is attacked by the d6 pawn but
# defended by the d4 pawn (not hanging). Black's a6 pawn is attacked by the
# bishop and not defended (hanging, worth 1).
f = features("7k/8/p2p4/1B2N3/3P4/8/8/4K3 w - - 0 1")
assert (f["hanging_count_white"], f["hanging_value_white"]) == (1, 3)
assert (f["hanging_count_black"], f["hanging_value_black"]) == (1, 1)
print("Test 3 passed: one undefended attacked piece per side is counted, the defended one is not")

# Test 4: pawn structure.
f = features("4k3/p7/8/4P3/8/8/8/4K3 w - - 0 1")          # e5 and a7, nothing in the way
assert f["passed_pawns_white"] == 1 and f["passed_pawns_black"] == 1
f = features("4k3/p7/5p2/4P3/8/8/8/4K3 w - - 0 1")        # black f6 stops e5 being passed
assert f["passed_pawns_white"] == 0
f = features("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1")         # c2 and c3: doubled and isolated
assert f["doubled_pawns_white"] == 1 and f["isolated_pawns_white"] == 2
f = features("4k3/8/8/8/8/2P5/2PP4/4K3 w - - 0 1")        # add d2: no longer isolated
assert f["doubled_pawns_white"] == 1 and f["isolated_pawns_white"] == 0
print("Test 4 passed: passed, doubled and isolated pawns are counted correctly")

# Test 5: castling. 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.O-O Nf6
board = chess.Board()
for move in parse_moves("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6"):
    board.push(move)
f = position_features(board)
assert f["castled_approx_white"] == 1 and f["castled_approx_black"] == 0
assert f["can_castle_kingside_white"] == 0 and f["can_castle_queenside_white"] == 0
assert f["can_castle_kingside_black"] == 1
assert f["minors_off_start_white"] == 2 and f["minors_off_start_black"] == 3
print("Test 5 passed: White has castled and lost castling rights; Black has not")

# Test 6: illegal or unreadable moves.
assert parse_moves("e4 e5 Ke3") is None          # king cannot reach e3
assert parse_moves("e4 e5 Zz9") is None          # not a move at all
assert len(parse_moves("e4 e5 Nf3")) == 3
print("Test 6 passed: parse_moves returns None for an illegal move")

# Test 7: nested groups and the leakage guard.
order = [GROUPS["P0"], GROUPS["P1"], GROUPS["P2"], GROUPS["P3"]]
for smaller, larger in zip(order, order[1:]):
    assert set(smaller) < set(larger)
check_no_leakage(P3)                              # the real features pass
for forbidden in ("turns", "victory_status", "id", "white_id", "last_move_at",
                  "created_at", "moves", "game_duration", "total_plies"):
    try:
        check_no_leakage(P3 + [forbidden])
        raise AssertionError(f"guard missed {forbidden}")
    except ValueError:
        pass
print("Test 7 passed: P0 to P3 are nested and the guard rejects forbidden names")

# Test 8: the split is by game. Eight made-up games, all reaching ply 10.
games = made_up_games([shared + " Re1 b5"] * 8)
rows, _ = build_checkpoint_rows(games, checkpoints=(10,))
development, test = split_rows(rows, development_games=[0, 1, 2, 3, 4, 5], test_games=[6, 7])
assert set(development["game_index"]).isdisjoint(set(test["game_index"]))
assert len(development) + len(test) == len(rows)
assert set(test["game_index"]) == {6, 7}
try:
    split_rows(rows, development_games=[0, 1, 2], test_games=[2, 3])
    raise AssertionError("split_rows accepted a game in both sets")
except ValueError:
    pass
print("Test 8 passed: every game's rows stay in one split, and overlap is refused")
