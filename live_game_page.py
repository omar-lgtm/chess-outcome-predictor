"""
The "Live game" tab of the Streamlit app.

The user pastes one game (a bare move list or a full PGN). The app shows
the estimated chance of a white win, a draw and a black win after every
ply, from live_model_tools.predict_curve, with the board at any chosen ply.

Privacy: only the moves and four headers (WhiteElo, BlackElo, TimeControl
and Event, for rated or casual) are read. Player names, sites, dates and
every other header are ignored and never shown or cached.
"""

import hashlib
import io
import re
from pathlib import Path

import altair as alt
import chess
import chess.pgn
import chess.svg
import pandas as pd
import streamlit as st

from analysis_page import RESULT_COLOURS
from live_model_tools import load_bundle, predict_curve

# The model files sit next to this file, wherever the app is started from.
APP_DIR = Path(__file__).resolve().parent

MAX_BYTES = 100 * 1024   # 100 KB of pasted text
MAX_PLIES = 300
MIN_RATING, MAX_RATING = 100, 3500

TEXT_KEY = "live_game_text"

# A famous public-domain game (Morphy, Paris 1858), used as the example.
OPERA_GAME = (
    "1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5 6.Bc4 Nf6 7.Qb3 Qe7 "
    "8.Nc3 c6 9.Bg5 b5 10.Nxb5 cxb5 11.Bxb5+ Nbd7 12.O-O-O Rd8 13.Rxd7 Rxd7 "
    "14.Rd1 Qe6 15.Bxd7+ Nxd7 16.Qb8+ Nxb8 17.Rd8#"
)

# Chart labels, in stacking order from the bottom, with the app's colours.
OUTCOMES = [
    ("p_white", "White wins", RESULT_COLOURS["win"]),
    ("p_draw", "Draw", RESULT_COLOURS["draw"]),
    ("p_black", "Black wins", RESULT_COLOURS["loss"]),
]


class QuietGameBuilder(chess.pgn.GameBuilder):
    """
    The normal PGN reader, except that it does not write errors to the log.
    python-chess's own message includes the game's headers (player names),
    so errors are only recorded on the game, where this page checks them.
    """

    def handle_error(self, error):
        self.game.errors.append(error)


def header_rating(value):
    """A header rating as an int, or None if missing, "?" or out of range."""
    value = (value or "").strip()
    if not value.isdigit():
        return None
    rating = int(value)
    return rating if MIN_RATING <= rating <= MAX_RATING else None


def parse_time_control(value):
    """
    PGN TimeControl in seconds ("600+5", or "600" for no increment) to
    (base_minutes, increment_seconds). Anything else gives (None, None).
    """
    value = (value or "").strip()
    match = re.fullmatch(r"(\d+)(?:\+(\d+))?", value)
    if not match:
        return None, None
    return int(match.group(1)) / 60, int(match.group(2) or 0)


def parse_rated(event):
    """1 for a Lichess "Rated ..." event, 0 for "Casual ...", else None."""
    event = (event or "").strip().lower()
    if event.startswith("rated"):
        return 1
    if event.startswith("casual"):
        return 0
    return None


def parse_game_text(text):
    """
    Read the first game in pasted text. Returns a dict with the main-line
    moves (chess.Move and SAN), the wanted header values and whether more
    games followed. Raises ValueError with a friendly message otherwise.
    """
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError("This text is larger than 100 KB. Please paste a single game.")
    stream = io.StringIO(text)
    game = chess.pgn.read_game(stream, Visitor=QuietGameBuilder)
    if game is None:
        raise ValueError("No game was found in the text. Paste a move list or a PGN.")
    if game.board() != chess.Board():
        raise ValueError("Only games from the normal starting position are supported.")

    moves = list(game.mainline_moves())
    if game.errors:
        raise ValueError(
            f"Move {len(moves) + 1} (half-move number) is illegal or could not be "
            "read, so the game cannot be analysed. Please check the moves."
        )
    if not moves:
        raise ValueError("No moves were found in the text. Paste a move list or a PGN.")
    if len(moves) > MAX_PLIES:
        raise ValueError(
            f"This game has {len(moves)} half-moves; the limit is {MAX_PLIES}. "
            "Please paste a shorter game."
        )

    board = chess.Board()
    san_moves = []
    for move in moves:
        san_moves.append(board.san(move))
        board.push(move)

    headers = game.headers
    base_minutes, increment_seconds = parse_time_control(headers.get("TimeControl"))
    return {
        "moves": moves,
        "san": san_moves,
        "white_rating": header_rating(headers.get("WhiteElo")),
        "black_rating": header_rating(headers.get("BlackElo")),
        "base_minutes": base_minutes,
        "increment_seconds": increment_seconds,
        "rated": parse_rated(headers.get("Event")),
        "more_games": chess.pgn.read_game(stream, Visitor=QuietGameBuilder) is not None,
    }


def move_label(ply, san):
    """'14. Qb3' for a white move, '14... Qe7' for a black move."""
    number = (ply + 1) // 2
    return f"{number}. {san}" if ply % 2 == 1 else f"{number}... {san}"


def finished_game_line(board):
    """The line shown for a finished position, for example "Checkmate: White wins"."""
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate: {winner} wins"
    if board.is_stalemate():
        return "Draw: stalemate"
    return "Draw: insufficient material"


def biggest_swings(curve, san_moves, count=3):
    """
    The plies with the largest change in White's expected score from the
    previous ply, largest first, as (ply, label, side, before, after).
    Plies before the model's range are skipped, and so is the first ply in
    the range, because its "before" value would be outside the range.
    """
    change = curve["expected_score_white"].diff().abs()
    before = curve["before_training"]
    usable = ~before & ~before.shift(1, fill_value=True)
    swings = []
    for ply in change[usable].nlargest(count).index:
        side = "White" if ply % 2 == 1 else "Black"
        swings.append((int(ply), move_label(ply, san_moves[ply - 1]), side,
                       curve["expected_score_white"].iloc[ply - 1],
                       curve["expected_score_white"].iloc[ply]))
    return swings


# The models are loaded the first time a game is analysed, then kept.
@st.cache_resource(show_spinner="Loading the live-game models...")
def get_bundle():
    return load_bundle(APP_DIR)


# Keyed only on the SAN moves and the numbers, never on the pasted text.
@st.cache_data(show_spinner="Working out the win probabilities...")
def cached_curve(moves_text, white_rating, black_rating, base_minutes,
                 increment_seconds, rated):
    return predict_curve(moves_text, white_rating, black_rating, base_minutes,
                         increment_seconds, rated, bundle=get_bundle())


def load_example():
    """Button callback: put the example game in the text area."""
    st.session_state[TEXT_KEY] = OPERA_GAME


def curve_chart(curve, selected_ply):
    """Stacked area chart of the three outcomes by ply, with a ply marker."""
    long = curve.melt(id_vars="ply", value_vars=[c for c, _, _ in OUTCOMES],
                      var_name="column", value_name="probability")
    names = {column: label for column, label, _ in OUTCOMES}
    long["outcome"] = long["column"].map(names)
    long["order"] = long["column"].map({c: i for i, (c, _, _) in enumerate(OUTCOMES)})
    last_ply = int(curve["ply"].max())
    x_scale = alt.Scale(domain=[0, max(last_ply, 1)], nice=False)

    area = alt.Chart(long).mark_area().encode(
        x=alt.X("ply:Q", title="Ply (half-moves played)", scale=x_scale),
        y=alt.Y("probability:Q", stack="zero", title="Probability",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color(
            "outcome:N", title="Result",
            scale=alt.Scale(domain=[label for _, label, _ in OUTCOMES],
                            range=[colour for _, _, colour in OUTCOMES]),
        ),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("ply:Q", title="Ply"), alt.Tooltip("outcome:N", title="Result"),
                 alt.Tooltip("probability:Q", title="Probability", format=".0%")],
    )
    layers = [area]

    # Light shading over the plies outside the model's training range: the
    # first few plies (label at the bottom) and plies after 120 (label at
    # the top), so the two labels never sit on top of each other.
    regions = []
    before = curve.loc[curve["before_training"], "ply"]
    if len(before):
        in_range = curve.loc[~curve["before_training"], "ply"]
        end = int(in_range.min()) if len(in_range) else last_ply
        regions.append((0, end, "bottom", 280))
    beyond = curve.loc[curve["beyond_training"], "ply"]
    if len(beyond):
        regions.append((int(beyond.min()) - 1, last_ply, "top", 0))
    for start, end, baseline, y in regions:
        shade = pd.DataFrame({"start": [start], "end": [end]})
        layers.append(alt.Chart(shade).mark_rect(color="#FFFFFF", opacity=0.45).encode(
            x="start:Q", x2="end:Q"))
        layers.append(alt.Chart(shade).mark_text(
            align="left", baseline=baseline, dx=4, dy=4 if y == 0 else -4,
            fontSize=11, color="#1F2933",
        ).encode(x="start:Q", y=alt.value(y),
                 text=alt.value("outside the data the model learned from")))

    layers.append(alt.Chart(pd.DataFrame({"ply": [selected_ply]})).mark_rule(
        color="#1F2933", strokeWidth=2).encode(x="ply:Q"))

    in_range = curve[~curve["before_training"]]
    last = curve.iloc[-1]
    description = (
        "Stacked area chart of the estimated chances of a white win, a draw and "
        f"a black win after each of the {last_ply} half-moves. "
    )
    if len(in_range):
        first = in_range.iloc[0]
        description += (
            f"At ply {int(first['ply'])}, where the model starts: White "
            f"{first['p_white']:.0%}, draw {first['p_draw']:.0%}, Black {first['p_black']:.0%}. "
        )
    description += (
        f"At the end: White {last['p_white']:.0%}, draw {last['p_draw']:.0%}, "
        f"Black {last['p_black']:.0%}. A vertical line marks ply {selected_ply}."
    )
    return alt.layer(*layers).properties(height=280), description


def render():
    """Draw the whole "Live game" tab."""
    st.header("Live game")
    st.write(
        "Paste a game to see how the estimated chances of a white win, a draw "
        "and a black win change move by move."
    )

    st.button("Load example game (the Opera Game, 1858)", on_click=load_example,
              key="live_load_example")
    st.caption(
        "This game ends with a famous queen sacrifice and a mating attack. The "
        "model only sees material and simple position features, so it can "
        "underrate White in the middle of the attack. That is a real limit of "
        "this model: it does not search for tactics the way a chess engine does."
    )
    text = st.text_area(
        "Game (a move list or a full PGN)", key=TEXT_KEY, height=150,
        placeholder="1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 ...",
    )
    if not text.strip():
        st.caption("Paste a game above, or load the example game, to begin.")
        return

    try:
        game = parse_game_text(text)
    except ValueError as problem:
        st.error(str(problem))
        return
    if game["more_games"]:
        st.info("The text holds more than one game. Only the first game is used.")

    # --- Ratings and time control ----------------------------------------
    # The widget keys include a short fingerprint of the text, so the boxes
    # are pre-filled again when a different game is pasted. The fingerprint
    # is a hash, not the text itself.
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    have_both = game["white_rating"] is not None and game["black_rating"] is not None
    use_ratings = st.checkbox("Use ratings", value=have_both, key=f"live_use_ratings_{tag}")
    left, right = st.columns(2)
    white_rating = left.number_input(
        "White rating", min_value=MIN_RATING, max_value=MAX_RATING, step=10,
        value=game["white_rating"], key=f"live_white_{tag}", disabled=not use_ratings,
    )
    black_rating = right.number_input(
        "Black rating", min_value=MIN_RATING, max_value=MAX_RATING, step=10,
        value=game["black_rating"], key=f"live_black_{tag}", disabled=not use_ratings,
    )
    if use_ratings and white_rating is not None and black_rating is not None:
        ratings = (int(white_rating), int(black_rating))
        st.info("Using the ratings model: the position plus the two ratings "
                f"(White {ratings[0]}, Black {ratings[1]}).")
    else:
        ratings = (None, None)
        reason = "a rating is missing" if use_ratings else "ratings are switched off"
        st.info(f"Using the position-only model, because {reason}. It looks at the board only.")

    if game["base_minutes"] is not None:
        st.caption(f"Time control from the PGN: {game['base_minutes']:g} min "
                   f"+ {game['increment_seconds']} s per move.")
    else:
        st.caption("No time control was found, so the model treats it as unknown.")

    # --- The curve ------------------------------------------------------------
    try:
        curve = cached_curve(" ".join(game["san"]), *ratings, game["base_minutes"],
                             game["increment_seconds"], game["rated"])
    except (RuntimeError, OSError) as problem:
        st.error(f"The live-game models could not be loaded. {problem}")
        return
    except ValueError as problem:
        st.error(str(problem))
        return

    last_ply = len(game["san"])
    selected = st.slider("Ply (half-moves played)", 0, last_ply, value=last_ply,
                         key=f"live_ply_{tag}")

    chart, description = curve_chart(curve, selected)
    # alt gives screen readers the same summary as the caption.
    st.altair_chart(chart, width="stretch", alt=description)
    st.caption(description)
    if curve["beyond_training"].any():
        first_beyond = int(curve.loc[curve["beyond_training"], "ply"].min())
        st.warning(
            f"From ply {first_beyond} on, the game is longer than any position the "
            "model learned from (shaded in the chart), so treat those estimates "
            "with extra caution."
        )

    # --- The chosen ply ------------------------------------------------------
    row = curve.iloc[selected]
    if selected == 0:
        st.subheader("Starting position")
    else:
        st.subheader(f"After {move_label(selected, game['san'][selected - 1])}")
    board = chess.Board()
    for move in game["moves"][:selected]:
        board.push(move)
    last_move = game["moves"][selected - 1] if selected else None

    # A finished game shows its real result; the first few plies show no
    # estimate; every other ply shows the model's three numbers.
    if pd.notna(row["terminal_result"]):
        st.success(finished_game_line(board))
    elif row["before_training"]:
        start = int(curve.loc[~curve["before_training"], "ply"].min()
                    if (~curve["before_training"]).any() else len(curve))
        st.info(f"No estimate yet: the model starts at move {(start + 1) // 2} (ply {start})")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("White wins", f"{row['p_white']:.0%}")
        col2.metric("Draw", f"{row['p_draw']:.0%}")
        col3.metric("Black wins", f"{row['p_black']:.0%}")
    # st.image accepts an SVG string in this Streamlit version.
    st.image(chess.svg.board(board, lastmove=last_move, size=380), width=380,
             caption=f"Board after ply {selected}",
             alt=f"Chess board after ply {selected}")

    # --- Biggest swings ------------------------------------------------------
    st.subheader("Biggest swings in the estimate")
    lines = [
        f"- {label} ({side}): White's expected score moved from {before:.0%} to {after:.0%}"
        for _, label, side, before, after in biggest_swings(curve, game["san"])
    ]
    if lines:
        st.markdown("\n".join(lines))
    else:
        st.write("The game is too short to show any swings.")
    st.caption("A swing is not necessarily a mistake: an exchange in progress can "
               "look lopsided until it finishes.")

    with st.expander("How to read this"):
        st.markdown(
            "- These are estimates from simple position features (material, "
            "pieces under attack, mobility, pawn structure and king safety). "
            "This is not a chess engine.\n"
            "- The model learned from about 19,000 Lichess games.\n"
            "- When it says 70%, about 70% of similar positions in held-out test "
            "games were won by White (this was checked with a calibration test).\n"
            "- The estimate is less reliable after move 60 (ply 120), because "
            "there is much less data there.\n"
            "- With ratings, the model also uses the rating gap between the players.\n"
            "- Nothing you paste is stored or shown to anyone else, and player "
            "names in a pasted PGN are ignored."
        )
