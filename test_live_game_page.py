"""
Checks for the "Live game" tab, on made-up input and the public-domain
Opera Game only (no real game files, no real names).

Run it with:  python test_live_game_page.py
"""

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from analysis_page import GAMES_KEY, games_fingerprint
from live_game_page import (
    APP_DIR, MAX_PLIES, OPERA_GAME, TEXT_KEY, biggest_swings, parse_game_text,
    parse_time_control,
)
from live_model_tools import load_bundle, predict_curve

# A legal knight shuffle: 4 plies per round that bring the board back.
SHUFFLE = "Nf3 Nf6 Ng1 Ng8 "
# Sam Loyd's ten-move stalemate game (public domain).
LOYD = ("1.e3 a5 2.Qh5 Ra6 3.Qxa5 h5 4.Qxc7 Rah6 5.h4 f6 6.Qxd7+ Kf7 7.Qxb7 Qd3 "
        "8.Qxb8 Qh7 9.Qxc8 Kg6 10.Qe6")
bundle = load_bundle(APP_DIR)


def made_up_pgn(white_elo, black_elo, moves="1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 *"):
    """A PGN with made-up names and headers around the given moves."""
    return (
        '[Event "Rated Blitz game"]\n[Site "Somewhere"]\n[Date "2020.01.01"]\n'
        '[White "Zebedee Testplayer"]\n[Black "Quentin Madeupname"]\n'
        f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n'
        '[TimeControl "300+3"]\n\n' + moves
    )


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------

# Test 1: parsing the pasted text.
game = parse_game_text(OPERA_GAME)
assert len(game["san"]) == 33 and game["san"][-1] == "Rd8#"
assert game["white_rating"] is None and not game["more_games"]
pgn = parse_game_text(made_up_pgn("1800", "?") + "\n\n1.d4 d5 *")
assert pgn["white_rating"] == 1800 and pgn["black_rating"] is None
assert pgn["rated"] == 1 and pgn["more_games"]
assert parse_time_control("600+5") == (10, 5)
assert parse_time_control("600") == (10, 0)
assert parse_time_control("-") == (None, None)
for bad in ("hello world", "e4 e5 Ke3", "e4 " * 40000, SHUFFLE * 80):
    try:
        parse_game_text(bad)
        raise AssertionError("parse_game_text accepted bad input")
    except ValueError:
        pass
print("Test 1 passed: moves, wanted headers and limits are read correctly")


# ---------------------------------------------------------------------------
# Screen tests with Streamlit's AppTest
# ---------------------------------------------------------------------------

def new_app(games=None):
    at = AppTest.from_file("app.py", default_timeout=120)
    if games is not None:
        at.session_state[GAMES_KEY] = {
            "games": games,
            "info": {"games_used": len(games), "clean_games": len(games)},
            "fingerprint": games_fingerprint(games),
        }
    return at.run()


def paste(text):
    at = new_app()
    at.text_area(key=TEXT_KEY).input(text).run()
    return at


def shown_text(at):
    """Every message, caption, heading and markdown text on the page."""
    kinds = ("info", "warning", "error", "caption", "subheader", "header", "markdown")
    return " ".join(str(el.value) for kind in kinds for el in getattr(at, kind))


def charts(at):
    """The charts in the "Live game" tab (the third tab)."""
    return at.tabs[2].get("vega_lite_chart")


# Test 2: no input gives a short prompt and no error.
at = new_app()
assert not at.exception and not at.error
assert "Paste a game above" in shown_text(at)
assert not charts(at)
print("Test 2 passed: with no input, a short prompt and no error")

# Test 3: the example button fills the text area; chart, slider and board appear.
at = new_app()
at.button(key="live_load_example").click().run()
assert not at.exception and not at.error
assert at.text_area(key=TEXT_KEY).value == OPERA_GAME
assert len(charts(at)) == 1
slider = [s for s in at.slider if s.label.startswith("Ply")][0]
assert slider.value == 33 and slider.max == 33
assert len(at.tabs[2].get("image")) == 1                 # the board
assert "After 17. Rd8#" in shown_text(at)
assert "Biggest swings in the estimate" in shown_text(at)
slider.set_value(14).run()
assert not at.exception and "After 7... Qe7" in shown_text(at)
print("Test 3 passed: the example fills the box and the chart, slider and board appear")

# Test 4: illegal moves give a friendly error and no exception.
at = paste("1. e4 e5 2. Ke3 Nc6")
assert not at.exception
assert len(at.error) == 1 and "illegal" in at.error[0].value
print("Test 4 passed: illegal moves give a friendly message")

# Test 5: input over the limits is rejected with a message.
at = paste("e4 " * 40000)
assert not at.exception and "100 KB" in at.error[0].value
at = paste(SHUFFLE * 80)
assert not at.exception and f"limit is {MAX_PLIES}" in at.error[0].value
print("Test 5 passed: text over 100 KB or games over 300 plies are refused")

# Test 6: which model is used, and names never appear.
at = new_app()
at.button(key="live_load_example").click().run()
assert "Using the position-only model" in shown_text(at)
at = paste(made_up_pgn("1800", "1800"))
assert not at.exception
text = shown_text(at)
assert "Using the ratings model" in text and "White 1800, Black 1800" in text
assert "5 min" in text
for private in ("Zebedee", "Quentin", "Somewhere", "2020"):
    assert private not in text
checkbox = [c for c in at.checkbox if c.label == "Use ratings"][0]
checkbox.uncheck().run()
assert "Using the position-only model, because ratings are switched off" in shown_text(at)
at = paste(made_up_pgn("1800", "?"))
assert not [c for c in at.checkbox if c.label == "Use ratings"][0].value
assert "Using the position-only model" in shown_text(at)
print("Test 6 passed: the page says which model is used; names, sites and dates are not shown")

# Test 7: a game longer than 60 moves is flagged beyond the training range.
at = paste(SHUFFLE * 32)                    # 128 plies
assert not at.exception and not at.error
assert "From ply 121 on" in shown_text(at)
assert "outside the data the model learned from" in charts(at)[0].proto.spec
print("Test 7 passed: plies after 120 are flagged and shaded")

# Test 8: tabs 1 and 2 still work with a games table in session state.
rng = np.random.default_rng(4)
n = 300
gap = rng.integers(-200, 201, size=n)
my_rating = 1200 + np.arange(n)
games = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "colour": rng.choice(["white", "black"], size=n),
    "my_rating": my_rating, "opp_rating": my_rating - gap, "rating_diff": gap,
    "score": rng.choice([0.0, 0.5, 1.0], size=n),
    "time_group": rng.choice(["1 min or less", "3-5 min"], size=n),
})
at = new_app(games)
assert not at.exception
assert [t.label for t in at.tabs] == ["Predict a game", "Analyse your games", "Live game"]
assert {"White wins", "Black wins", "Draw"} <= {m.label for m in at.tabs[0].metric}
assert "General model" in [s.value for s in at.tabs[0].subheader]
assert "Analyse your games" in [h.value for h in at.tabs[1].header]
assert "Live game" in [h.value for h in at.tabs[2].header]
print("Test 8 passed: the first two tabs still work next to the new tab")

# Test 9: finished games, the first plies and the example caption.
at = new_app()
at.button(key="live_load_example").click().run()
assert not at.exception
assert "famous queen sacrifice" in shown_text(at)
assert [s.value for s in at.tabs[2].success] == ["Checkmate: White wins"]     # last ply is shown first
assert not at.tabs[2].metric                                          # no model numbers
slider = [s for s in at.slider if s.label.startswith("Ply")][0]
slider.set_value(2).run()
assert not at.exception
assert "No estimate yet: the model starts at move 3 (ply 5)" in shown_text(at)
assert not at.tabs[2].metric
slider.set_value(5).run()
assert not at.exception and len(at.tabs[2].metric) == 3
assert "outside the data the model learned from" in charts(at)[0].proto.spec
at = paste(LOYD)
assert not at.exception and [s.value for s in at.tabs[2].success] == ["Draw: stalemate"]
print("Test 9 passed: checkmate and stalemate lines, no estimate before ply 5, caption shown")

# Test 10: the swings skip plies 0 to 5, even when those changes are the biggest.
fake = pd.DataFrame({"ply": range(9)})
fake["expected_score_white"] = [0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 0.1, 0.3, 0.2]
fake["before_training"] = fake["ply"] < 5
swings = biggest_swings(fake, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"])
assert [ply for ply, *_ in swings] == [7, 6, 8]
assert swings[0][1:3] == ("4. Ba4", "White")
for at_ply in (OPERA_GAME, LOYD):
    curve = predict_curve(" ".join(parse_game_text(at_ply)["san"]), bundle=bundle)
    plies = [ply for ply, *_ in biggest_swings(curve, parse_game_text(at_ply)["san"])]
    assert len(plies) == 3 and min(plies) > 5
print("Test 10 passed: the biggest swings never use plies before the model's range")
