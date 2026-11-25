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

MODEL_PATH = "chess_model.joblib"
CHART_PATH = "images/accuracy_by_gap.png"


# st.cache_resource makes Streamlit load the model only once.
# Without it, the model would be reloaded every time the user changes an input.
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------------------------------------------------------------
# Page header (shown once, above both tabs)
# ---------------------------------------------------------------------------
st.title("Chess Outcome Predictor")
st.write(
    "Predicts the result of a Lichess game (white win, black win or draw) "
    "from the two players' ratings."
)

tab1, tab2 = st.tabs(["Predict a game", "Analyse your games"])

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

    # A simple bar chart of the same three numbers (shown as percentages).
    chart_data = pd.DataFrame(
        {"Probability (%)": [white_win * 100, black_win * 100, draw * 100]},
        index=["White wins", "Black wins", "Draw"],
    )
    st.bar_chart(chart_data)

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

with tab2:
    analysis_page.render()
