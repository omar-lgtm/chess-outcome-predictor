"""
Tools for reading chess games from PGN text (Lichess or Chess.com exports).

The main function is parse_pgn(). It reads only the header lines of each
game (players, ratings, result, date, time control, opening) and turns them
into a pandas DataFrame with one row per game, seen from one player's side.
The moves themselves are not parsed.

Example:
    with open("data/my_games.pgn", encoding="utf-8") as f:
        df, info = parse_pgn(f.read())
"""

import re
from collections import Counter

import pandas as pd

# Matches one header line, for example:  [WhiteElo "1500"]
# Group 1 is the tag name (WhiteElo), group 2 is the value (1500).
HEADER_RE = re.compile(r'^\[(\w+) "(.*?)"\]', re.M)

# Splits the text just before every line that starts with [Event .
# The (?=...) part is a "lookahead": it splits without removing the text.
GAME_SPLIT_RE = re.compile(r"^(?=\[Event )", re.M)

# Result header value -> (result, score) from WHITE's point of view.
WHITE_RESULTS = {
    "1-0": ("win", 1.0),
    "1/2-1/2": ("draw", 0.5),
    "0-1": ("loss", 0.0),
}

# Used to flip a result when the player has the black pieces.
FLIPPED = {"win": "loss", "draw": "draw", "loss": "win"}

COLUMNS = [
    "date", "colour", "my_rating", "opp_rating", "rating_diff", "result",
    "score", "time_control", "base_seconds", "increment_seconds", "eco",
    "opening",
]

# A run of at least this many games in a row with exactly the same two
# ratings is flagged as stale_rating (ratings may not have been updating,
# for example in unrated games).
STALE_RUN_LENGTH = 10

# An opponent met in at least this many games is flagged as repeat_opponent.
REPEAT_OPPONENT_GAMES = 20


def split_games(text):
    """Split PGN text into a list of games, each as a dict of headers."""
    games = []
    for chunk in GAME_SPLIT_RE.split(text):
        headers = dict(HEADER_RE.findall(chunk))
        if headers:  # skip empty pieces, such as text before the first game
            games.append(headers)
    return games


def most_common_player(games):
    """Return the name that appears most often as White or Black."""
    counts = Counter()
    spelling = {}  # lower-case name -> name as written in the file
    for headers in games:
        for side in ("White", "Black"):
            name = headers.get(side)
            if name:
                counts[name.lower()] += 1
                spelling.setdefault(name.lower(), name)
    if not counts:
        return None
    top_name, _ = counts.most_common(1)[0]
    return spelling[top_name]


def parse_time_control(value):
    """
    Turn a time control string into (base_seconds, increment_seconds).

    "60+1"    -> (60, 1)       60 seconds plus 1 second per move
    "600"     -> (600, 0)      10 minutes, no increment
    "1/86400" -> (86400, 0)    correspondence: 1 move per day

    Returns None if the value is "-" (no clock) or can't be read.
    """
    match = re.fullmatch(r"(\d+)\+(\d+)", value)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(\d+)", value)
    if match:
        return int(match.group(1)), 0
    match = re.fullmatch(r"\d+/(\d+)", value)
    if match:
        return int(match.group(1)), 0
    return None


def opening_name(headers):
    """Pick the best available opening name for one game."""
    # Lichess games have an "Opening" header with the full name.
    if headers.get("Opening"):
        return headers["Opening"]

    # Chess.com games have a URL like
    # https://www.chess.com/openings/French-Defense-Exchange-Variation-3...exd5
    # We take the last part of the URL and cut it before the first
    # "-" followed by a digit, which is where the move numbers start.
    url = headers.get("ECOUrl")
    if url:
        last_part = url.rstrip("/").split("/")[-1]
        return re.split(r"-(?=\d)", last_part, maxsplit=1)[0]

    # Otherwise fall back to the ECO code (for example "C00"), if any.
    return headers.get("ECO")


def parse_pgn(text: str, player: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Read PGN text and return (games DataFrame, info dict).

    text:   the full text of a PGN file (not a file path).
    player: whose point of view to use. If None, the name that appears
            most often in the file is used. Matching ignores upper/lower case.

    Games are skipped (and counted in info["skipped"]) when the player is
    not in the game, the result is not 1-0, 0-1 or 1/2-1/2, a rating is
    missing or not a number, there is no usable time control, or the date
    can't be read.

    Two True/False columns flag games that may need care:
    - stale_rating: part of a run of 10+ games in a row (date order) with
      exactly the same my_rating and opp_rating.
    - repeat_opponent: the opponent appears in 20+ of the player's games
      in the file.
    """
    games = split_games(text)
    if player is None:
        player = most_common_player(games)

    rows = []
    skipped = Counter()

    # Opponent names are only kept inside this function, to count how often
    # each one appears. They are never put in the DataFrame or the info dict.
    opponent_counts = Counter()  # lower-case name -> games against them
    row_opponents = []           # opponent of each row in rows, same order

    for headers in games:
        white = headers.get("White", "")
        black = headers.get("Black", "")

        # 1. Which side is the player on?
        if player is not None and white.lower() == player.lower():
            colour = "white"
        elif player is not None and black.lower() == player.lower():
            colour = "black"
        else:
            skipped["player_not_in_game"] += 1
            continue

        # Count the opponent for every game the player is in, including
        # games skipped below, since the rule is "appears in the file".
        opponent = (black if colour == "white" else white).lower()
        opponent_counts[opponent] += 1

        # 2. Result must be a finished game.
        if headers.get("Result") not in WHITE_RESULTS:
            skipped["bad_result"] += 1
            continue

        # 3. Both ratings must be whole numbers ("?" or empty is skipped).
        white_elo = headers.get("WhiteElo", "")
        black_elo = headers.get("BlackElo", "")
        if not (white_elo.isdigit() and black_elo.isdigit()):
            skipped["missing_rating"] += 1
            continue

        # 4. Time control must be readable ("-" means no clock).
        time_control = headers.get("TimeControl", "")
        clock = parse_time_control(time_control)
        if clock is None:
            skipped["bad_time_control"] += 1
            continue

        # 5. Date: prefer UTCDate, fall back to Date. PGN dates look
        #    like 2024.03.15; unknown parts are written as "??".
        date_text = headers.get("UTCDate") or headers.get("Date", "")
        date = pd.to_datetime(date_text, format="%Y.%m.%d", errors="coerce")
        if pd.isna(date):
            skipped["bad_date"] += 1
            continue

        # Everything is valid, so build the row from the player's side.
        result, score = WHITE_RESULTS[headers["Result"]]
        if colour == "white":
            my_rating, opp_rating = int(white_elo), int(black_elo)
        else:
            my_rating, opp_rating = int(black_elo), int(white_elo)
            result = FLIPPED[result]
            score = 1.0 - score

        rows.append({
            "date": date,
            "colour": colour,
            "my_rating": my_rating,
            "opp_rating": opp_rating,
            "rating_diff": my_rating - opp_rating,
            "result": result,
            "score": score,
            "time_control": time_control,
            "base_seconds": clock[0],
            "increment_seconds": clock[1],
            "eco": headers.get("ECO"),
            "opening": opening_name(headers),
        })
        row_opponents.append(opponent)

    # columns=COLUMNS keeps the column order fixed, even if no rows were made.
    df = pd.DataFrame(rows, columns=COLUMNS)

    # repeat_opponent: look up how many games each row's opponent played.
    # This is done before sorting, while rows and row_opponents still line up.
    df["repeat_opponent"] = [
        opponent_counts[name] >= REPEAT_OPPONENT_GAMES for name in row_opponents
    ]

    # A "stable" sort keeps games from the same day in their file order.
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # stale_rating: find runs of games in a row with the same two ratings.
    # changed is True where either rating differs from the previous game.
    # cumsum() then gives every run its own number (0, 1, 2, ...), and
    # transform("size") gives each game the length of the run it is in.
    changed = (df["my_rating"] != df["my_rating"].shift()) | (
        df["opp_rating"] != df["opp_rating"].shift()
    )
    run_id = changed.cumsum()
    run_length = df.groupby(run_id)["my_rating"].transform("size")
    df["stale_rating"] = (run_length >= STALE_RUN_LENGTH).astype(bool)
    df["repeat_opponent"] = df["repeat_opponent"].astype(bool)

    info = {
        "player": player,
        "games_in_file": len(games),
        "games_used": len(df),
        "skipped": dict(skipped),
        "stale_rating_games": int(df["stale_rating"].sum()),
        "repeat_opponent_games": int(df["repeat_opponent"].sum()),
    }
    return df, info
