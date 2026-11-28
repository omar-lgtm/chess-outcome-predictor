"""
Tools for in-game win probability: describe a chess position after the
first N plies (half-moves) with simple numbers, so a model can predict the
final result from the position so far.

Every feature is computed ONLY from the board after N plies and the
pre-game columns (ratings, time control, rated). Nothing from later moves,
the game length, how the game ended, or any id is ever used.

Uses python-chess (import chess). No Streamlit code here.
"""

import chess
import pandas as pd

# Piece values in pawn units, used for material and hanging pieces.
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PIECE_NAMES = {chess.PAWN: "pawns", chess.KNIGHT: "knights", chess.BISHOP: "bishops",
               chess.ROOK: "rooks", chess.QUEEN: "queens"}
SIDES = {"white": chess.WHITE, "black": chess.BLACK}
CENTRE = [chess.D4, chess.E4, chess.D5, chess.E5]

# Starting squares, used to see which pieces have left home.
START_KING = {chess.WHITE: chess.E1, chess.BLACK: chess.E8}
START_QUEEN = {chess.WHITE: chess.D1, chess.BLACK: chess.D8}
START_MINORS = {
    chess.WHITE: [(chess.B1, chess.KNIGHT), (chess.G1, chess.KNIGHT),
                  (chess.C1, chess.BISHOP), (chess.F1, chess.BISHOP)],
    chess.BLACK: [(chess.B8, chess.KNIGHT), (chess.G8, chess.KNIGHT),
                  (chess.C8, chess.BISHOP), (chess.F8, chess.BISHOP)],
}
CASTLED_SQUARES = {chess.WHITE: (chess.G1, chess.C1), chess.BLACK: (chess.G8, chess.C8)}

# Names that must never be features: they are only known after the game,
# identify games or players, or describe moves after the position.
FORBIDDEN_FEATURES = {
    "victory_status", "turns", "id", "white_id", "black_id", "last_move_at",
    "created_at", "moves", "winner", "opening_name",
}
FORBIDDEN_WORDS = ("duration", "game_length", "total_plies", "future")


def parse_moves(moves_text):
    """
    Turn the space-separated SAN moves from the CSV ("e4 e5 Nf3 ...") into a
    list of chess.Move. Returns None if any move is illegal or unreadable.
    """
    board = chess.Board()
    moves = []
    for san in str(moves_text).split():
        try:
            move = board.parse_san(san)
        except ValueError:          # illegal, ambiguous or unreadable move
            return None
        board.push(move)
        moves.append(move)
    return moves


# ---------------------------------------------------------------------------
# Pawn structure helpers
# ---------------------------------------------------------------------------

def pawn_files(board, colour):
    """List of file numbers (0 = a ... 7 = h) of this side's pawns, one per pawn."""
    return [chess.square_file(sq) for sq in board.pieces(chess.PAWN, colour)]


def doubled_pawns(board, colour):
    """
    Doubled pawns: for every file, each pawn beyond the first on that file
    counts once (two pawns on the c-file give 1, three give 2).
    """
    files = pawn_files(board, colour)
    return sum(max(0, files.count(f) - 1) for f in set(files))


def isolated_pawns(board, colour):
    """Isolated pawns: pawns with no pawn of the same side on either neighbouring file."""
    files = pawn_files(board, colour)
    return sum(1 for f in files if (f - 1) not in files and (f + 1) not in files)


def passed_pawns(board, colour):
    """
    Passed pawns: pawns with no enemy pawn in front of them on the same or a
    neighbouring file (in front means towards the enemy's side of the board).
    """
    enemy = [(chess.square_file(sq), chess.square_rank(sq))
             for sq in board.pieces(chess.PAWN, not colour)]
    count = 0
    for sq in board.pieces(chess.PAWN, colour):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = (lambda er: er > r) if colour == chess.WHITE else (lambda er: er < r)
        if not any(abs(ef - f) <= 1 and ahead(er) for ef, er in enemy):
            count += 1
    return count


def mobility(board, colour):
    """
    How many moves this side could make if it were its turn (pseudo-legal:
    moves that may leave the king in check are still counted). A copy of the
    board is used, with the en passant square cleared, so the real board is
    never changed.
    """
    copy = board.copy(stack=False)
    copy.turn = colour
    copy.ep_square = None
    return copy.pseudo_legal_moves.count()


def hanging(board, colour):
    """
    This side's "hanging" pieces: non-king pieces attacked by the enemy and
    not defended by their own side. Returns (count, total value in pawns).
    """
    count, value = 0, 0
    for piece_type, piece_value in PIECE_VALUES.items():
        for sq in board.pieces(piece_type, colour):
            if board.is_attacked_by(not colour, sq) and not board.is_attacked_by(colour, sq):
                count += 1
                value += piece_value
    return count, value


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def position_features(board, white_rating=None, black_rating=None, base_minutes=None,
                      increment_seconds=None, rated=None):
    """
    All features for one position, as a dict. Uses only the board given
    (the position after N plies) and the pre-game columns passed in.
    """
    f = {}

    # --- P0 ratings and game settings (known before the game) ---------------
    f["rating_diff"] = (white_rating - black_rating
                        if white_rating is not None and black_rating is not None else None)
    f["white_rating"] = white_rating
    f["black_rating"] = black_rating
    f["base_minutes"] = base_minutes            # starting clock in minutes
    f["increment_seconds"] = increment_seconds  # seconds added per move
    f["rated"] = rated                          # 1 rated game, 0 casual

    # --- P1 material ------------------------------------------------------------
    totals = {}
    for side, colour in SIDES.items():
        total = 0
        for piece_type, name in PIECE_NAMES.items():
            n = len(board.pieces(piece_type, colour))
            f[f"{name}_{side}"] = n                 # how many of this piece are left
            total += n * PIECE_VALUES[piece_type]
        totals[side] = total
    f["material_balance"] = totals["white"] - totals["black"]   # + means White is ahead
    f["material_total"] = totals["white"] + totals["black"]     # all material left, no kings

    # --- P2 position --------------------------------------------------------------
    f["white_to_move"] = int(board.turn == chess.WHITE)        # side to move
    for side, colour in SIDES.items():
        # Castling rights still available, per wing.
        f[f"can_castle_kingside_{side}"] = int(board.has_kingside_castling_rights(colour))
        f[f"can_castle_queenside_{side}"] = int(board.has_queenside_castling_rights(colour))
        king = board.king(colour)
        f[f"king_file_{side}"] = chess.square_file(king)   # 0 = a-file ... 7 = h-file
        f[f"king_rank_{side}"] = chess.square_rank(king)   # 0 = rank 1 ... 7 = rank 8
        f[f"king_on_start_{side}"] = int(king == START_KING[colour])
        # Rough "has castled": king sits on the square castling puts it on.
        f[f"castled_approx_{side}"] = int(king in CASTLED_SQUARES[colour])
        # Minor pieces no longer on their starting squares (moved or captured).
        f[f"minors_off_start_{side}"] = sum(
            1 for sq, piece_type in START_MINORS[colour]
            if board.piece_at(sq) != chess.Piece(piece_type, colour))
        f[f"queen_off_start_{side}"] = int(
            board.piece_at(START_QUEEN[colour]) != chess.Piece(chess.QUEEN, colour))
        # Own pawns on the four centre squares d4, e4, d5, e5.
        f[f"centre_pawns_{side}"] = sum(
            1 for sq in CENTRE if board.piece_at(sq) == chess.Piece(chess.PAWN, colour))
        # How many times this side attacks the four centre squares, added up.
        f[f"centre_attacks_{side}"] = sum(len(board.attackers(colour, sq)) for sq in CENTRE)
        f[f"mobility_{side}"] = mobility(board, colour)
        f[f"doubled_pawns_{side}"] = doubled_pawns(board, colour)
        f[f"isolated_pawns_{side}"] = isolated_pawns(board, colour)
        f[f"passed_pawns_{side}"] = passed_pawns(board, colour)

    # --- P3 tactics -----------------------------------------------------------------
    f["is_check"] = int(board.is_check())      # the side to move is in check
    for side, colour in SIDES.items():
        count, value = hanging(board, colour)
        f[f"hanging_count_{side}"] = count
        f[f"hanging_value_{side}"] = value
    return f


def feature_groups():
    """P0..P3 feature names, nested: each group contains the one before."""
    names = list(position_features(chess.Board(), 1500, 1500, 10, 0, 1))
    p0 = names[: names.index("pawns_white")]
    p1 = names[: names.index("white_to_move")]
    p2 = names[: names.index("is_check")]
    p3 = names
    return {"P0": p0, "P1": p1, "P2": p2, "P3": p3}


def check_no_leakage(columns):
    """Raise an error if a forbidden name appears among the feature columns."""
    bad = [c for c in columns
           if c in FORBIDDEN_FEATURES or any(word in c.lower() for word in FORBIDDEN_WORDS)]
    if bad:
        raise ValueError(f"Forbidden columns in the feature table: {bad}")


# ---------------------------------------------------------------------------
# Checkpoint rows and the game-level split
# ---------------------------------------------------------------------------

def build_checkpoint_rows(games, checkpoints=(10, 20, 30, 40, 60), progress_every=None):
    """
    Replay each game once and record the position features after exactly N
    plies, for every N the game reaches. Also adds an N = 0 row (the start
    position, i.e. pre-game information only) for every game.

    games must have: moves, white_rating, black_rating, base_minutes,
    increment_seconds, rated, winner. The game index (its row label in
    games) is stored instead of any real id.

    Only the first max(checkpoints) moves are read, so nothing after the
    last checkpoint is ever looked at. Games whose moves in that range cannot
    be parsed are skipped. Returns (rows DataFrame, number of games skipped).
    """
    last = max(checkpoints)
    wanted = set(checkpoints)
    rows = []
    skipped = 0
    for done, (index, game) in enumerate(games.iterrows(), start=1):
        if progress_every and done % progress_every == 0:
            print(f"  replayed {done} of {len(games)} games", flush=True)
        moves = parse_moves(" ".join(str(game["moves"]).split()[:last]))
        if moves is None:
            skipped += 1
            continue
        pre_game = dict(white_rating=game["white_rating"], black_rating=game["black_rating"],
                        base_minutes=game["base_minutes"],
                        increment_seconds=game["increment_seconds"], rated=game["rated"])
        board = chess.Board()
        rows.append({"game_index": index, "ply": 0, "winner": game["winner"],
                     **position_features(board, **pre_game)})
        for ply, move in enumerate(moves, start=1):
            board.push(move)
            if ply in wanted:
                rows.append({"game_index": index, "ply": ply, "winner": game["winner"],
                             **position_features(board, **pre_game)})
    return pd.DataFrame(rows), skipped


def split_rows(rows, development_games, test_games):
    """
    Put every checkpoint row in the same split as its game. Raises an error
    if a game index is listed in both splits. Returns (development rows, test rows).
    """
    development_games, test_games = set(development_games), set(test_games)
    overlap = development_games & test_games
    if overlap:
        raise ValueError(f"{len(overlap)} games are in both the development and test sets.")
    return (rows[rows["game_index"].isin(development_games)],
            rows[rows["game_index"].isin(test_games)])
