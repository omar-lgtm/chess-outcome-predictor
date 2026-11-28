"""
Chess Outcome Predictor - a small Streamlit app.

Enter the ratings of the white and black players, and the app shows the
probability of a white win, a black win, or a draw. The predictions come
from a logistic regression model trained on about 19,000 Lichess games.

Run it with:  streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

import analysis_page
import live_game_page
from analysis_page import RESULT_COLOURS
from calculator_tools import (
    MIN_GAMES, MIN_SIMILAR, bootstrap_param_sets, default_rating, general_odds,
    option_summary, personal_prediction, similar_record,
)

# Must be the first Streamlit command on the page.
st.set_page_config(page_title="Chess outcome predictor", layout="centered")

MODEL_PATH = "chess_model.joblib"
CHART_PATH = "images/accuracy_by_gap.png"

# Where the personal curve's fitted parameters are kept for this session,
# with the fingerprint of the games they were fitted on.
PARAMS_KEY = "my_games_params"


# st.cache_resource makes Streamlit load the model only once.
# Without it, the model would be reloaded every time the user changes an input.
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


def outcome_bar(parts):
    """
    Draw one horizontal bar split into coloured parts, with the percentages
    written underneath (the words are there too, so colour is not needed).

    parts is a list of (label, probability, colour key) from this file only,
    for example [("Win", 0.55, "win"), ...]. The HTML is built only from
    these fixed labels and numbers formatted here, never from user text.
    """
    segments = "".join(
        f'<div style="width:{100 * p:.2f}%;background:{RESULT_COLOURS[key]};"></div>'
        for _, p, key in parts
    )
    legend = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="display:inline-block;width:0.7em;height:0.7em;'
        f'background:{RESULT_COLOURS[key]};border-radius:2px;margin-right:0.3em;"></span>'
        f"{label} {p:.1%}"
        for label, p, key in parts
    )
    summary = ", ".join(f"{label} {p:.1%}" for label, p, _ in parts)
    st.markdown(
        f'<div role="img" aria-label="{summary}" style="display:flex;height:1.1rem;'
        f'border-radius:4px;overflow:hidden;margin:0.25rem 0 0.4rem 0;">{segments}</div>'
        f'<div style="font-size:0.9rem;">{legend}</div>',
        unsafe_allow_html=True,
    )


def personal_params(stored):
    """
    The bootstrap fits for the stored games, computed once per file.

    They are kept in session state next to the games' fingerprint (number
    of games plus first and last date), so a rerun only compares that small
    fingerprint instead of hashing the whole table. Returns (param_info,
    error message or None). param_info is None when there are too few games.
    """
    games = stored["games"]
    saved = st.session_state.get(PARAMS_KEY)
    if saved is None or saved["fingerprint"] != stored["fingerprint"]:
        param_info, error = None, None
        if len(games) >= MIN_GAMES:
            with st.spinner("Fitting your personal curve..."):
                try:
                    param_info = bootstrap_param_sets(games)
                except ValueError as problem:
                    error = str(problem)
        saved = {"fingerprint": stored["fingerprint"], "param_info": param_info, "error": error}
        st.session_state[PARAMS_KEY] = saved
    return saved["param_info"], saved["error"]


def render_personal_odds():
    """The "Your personal odds" section at the bottom of the first tab."""
    st.header("Your personal odds")
    stored = st.session_state.get(analysis_page.GAMES_KEY)
    if stored is None:
        st.info(
            "To see odds based on your own games, upload and analyse your games "
            "in the \"Analyse your games\" tab first."
        )
        return

    games = stored["games"]
    summary = option_summary(games)

    # --- Choices -----------------------------------------------------------
    colour_label = st.radio("Your colour", ["White", "Black"], horizontal=True,
                            key="odds_colour")
    colour = colour_label.lower()

    counts = dict(summary["time_groups"])
    time_label = st.selectbox(
        "Time control",
        ["Any"] + list(counts),
        format_func=lambda g: (
            f"Any ({summary['n_games']} games)" if g == "Any" else f"{g} ({counts[g]} games)"
        ),
        key="odds_time",
    )
    time_group = None if time_label == "Any" else time_label

    # The keys include the time group and colour, so the boxes are
    # pre-filled again whenever those choices change.
    start = min(max(default_rating(games, time_group), 100), 3000)
    left, right = st.columns(2)
    my_rating = left.number_input(
        "Your rating", min_value=100, max_value=3000, value=start, step=10,
        key=f"odds_my_{time_label}_{colour}",
    )
    opp_rating = right.number_input(
        "Opponent rating", min_value=100, max_value=3000, value=start, step=10,
        key=f"odds_opp_{time_label}_{colour}",
    )
    gap = my_rating - opp_rating
    st.caption(f"Rating gap (your rating minus your opponent's): {gap:+d}")

    # --- The three answers -------------------------------------------------
    param_info, fit_error = personal_params(stored)
    general = general_odds(model, my_rating, opp_rating, colour)
    personal = personal_prediction(games, param_info, gap, colour)
    similar = similar_record(games, colour, time_group, gap)

    # Each answer sits in its own bordered card. Each card leads with one
    # large number, with the detail in smaller text underneath.
    col1, col2, col3 = st.columns(3)
    with col1, st.container(border=True):
        st.subheader("General model")
        st.metric("Expected score", f"{general['expected_score']:.1%}")
        outcome_bar([
            ("Win", general["win"], "win"),
            ("Draw", general["draw"], "draw"),
            ("Loss", general["loss"], "loss"),
        ])
        st.caption("Trained on about 19,000 Lichess games, not on your own games.")

    with col2, st.container(border=True):
        st.subheader("Your personal curve")
        if fit_error:
            st.warning(fit_error)
        elif personal["status"] == "ok":
            st.metric("Expected score", f"{personal['point']:.1%}")
            st.caption(
                f"Likely range {personal['low']:.1%} to {personal['high']:.1%}, "
                f"from {personal['n_games']} of your games."
            )
        else:
            st.info(personal["message"])

    with col3, st.container(border=True):
        st.subheader("Your record in similar games")
        where = f"as {colour_label}" + (f", {time_group}" if time_group else "")
        if similar["enough"]:
            st.metric("Score", f"{similar['score']:.1%}")
            st.caption(
                f"Range {similar['low']:.1%} to {similar['high']:.1%}, from "
                f"{similar['n']} games {where} with a rating gap within "
                f"{similar['window_used']} points of this one."
            )
        else:
            st.info(
                f"Not enough similar games: {similar['n']} games {where} with a "
                f"rating gap within {similar['window_used']} points of this one, "
                f"and at least {MIN_SIMILAR} are needed."
            )

    with st.expander("How to read this"):
        st.markdown(
            "- **Expected score** counts a win as 1 point and a draw as half a "
            "point, so 60% means about 0.6 points per game on average.\n"
            "- **Your personal curve** uses only the rating gap, not the time "
            "control, because testing showed that time control and other "
            "features added nothing measurable.\n"
            "- **The range** comes from refitting the curve on many resampled "
            "versions of your own games.\n"
            f"- The personal number is hidden when you have fewer than {MIN_GAMES} "
            "games, or when the rating gap is outside the range where you have "
            "most of your games.\n"
            "- **Your record in similar games** is simply your average score in "
            "past games like this one.\n"
            "- Chess.com and Lichess ratings are on different scales, so the "
            "general model may not fit Chess.com ratings well."
        )

# ---------------------------------------------------------------------------
# Page header (shown once, above the tabs)
# ---------------------------------------------------------------------------
st.title("Chess Outcome Predictor")
st.write(
    "Predicts the result of a Lichess game (white win, black win or draw) "
    "from the two players' ratings."
)

tab1, tab2, tab3 = st.tabs(["Predict a game", "Analyse your games", "Live game"])

with tab1:
    # ---------------------------------------------------------------------------
    # Inputs: one number box per player, side by side
    # ---------------------------------------------------------------------------
    left, right = st.columns(2)
    with left:
        white_rating = st.number_input(
            "White rating", min_value=100, max_value=3000, value=1500, step=10
        )
    with right:
        black_rating = st.number_input(
            "Black rating", min_value=100, max_value=3000, value=1500, step=10
        )

    # The model was trained on a single feature: white rating minus black rating.
    rating_diff = white_rating - black_rating
    st.caption(f"Rating difference (white minus black): {rating_diff:+d}")

    # ---------------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------------
    # The model expects a DataFrame with a column named "rating_diff".
    features = pd.DataFrame({"rating_diff": [rating_diff]})

    # predict_proba returns one row of probabilities, in the order of
    # model.classes_, which is ['black', 'draw', 'white'].
    # Zipping the two together gives a dictionary like {'black': 0.45, ...},
    # so we do not have to rely on remembering the order.
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))

    white_win = probabilities["white"]
    black_win = probabilities["black"]
    draw = probabilities["draw"]

    # Show the three probabilities as large numbers.
    col1, col2, col3 = st.columns(3)
    col1.metric("White wins", f"{white_win:.1%}")
    col2.metric("Black wins", f"{black_win:.1%}")
    col3.metric("Draw", f"{draw:.1%}")

    # The same three numbers as one stacked bar, from White's side:
    # a white win uses the win colour and a black win the loss colour.
    outcome_bar([
        ("White wins", white_win, "win"),
        ("Draw", draw, "draw"),
        ("Black wins", black_win, "loss"),
    ])

    # One-line summary. We only compare white and black, because the model
    # almost never gives a draw the highest probability.
    if white_win > black_win:
        st.success(f"White is more likely to win ({white_win:.1%}).")
    elif black_win > white_win:
        st.success(f"Black is more likely to win ({black_win:.1%}).")
    else:
        st.info("White and black are equally likely to win.")

    # ---------------------------------------------------------------------------
    # About this model
    # ---------------------------------------------------------------------------
    st.header("About this model")
    st.write(
        "- **Model:** logistic regression on one feature, the rating difference.\n"
        "- **Test accuracy:** 61.95% (always guessing white scores about 50%).\n"
        "- **Draws:** only about 5% of games end in a draw, and they cannot be "
        "predicted from pre-game information. Treat the draw probability as a "
        "rough base rate."
    )
    st.image(
        CHART_PATH,
        caption="Validation accuracy by rating gap. The bigger the gap between "
        "the players, the easier the result is to predict.",
    )

    # ---------------------------------------------------------------------------
    # Your personal odds (uses the games from the "Analyse your games" tab)
    # ---------------------------------------------------------------------------
    render_personal_odds()

with tab2:
    analysis_page.render()

with tab3:
    live_game_page.render()
