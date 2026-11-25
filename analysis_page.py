"""
The "Analyse your games" tab of the Streamlit app.

The user uploads a PGN file of their own games. The app reads it with
report_tools.build_report() and shows how their results compare with what
their rating gaps alone would predict, by opening, time control and year.
"""

import pandas as pd
import streamlit as st

from report_tools import build_report

MAX_FILE_MB = 30

# How each verdict from group_report is shown to the user. "Possibly" is
# there because some flags will be luck when there are many rows.
VERDICT_LABELS = {
    "above expectation": "Possibly above expectation",
    "below expectation": "Possibly below expectation",
    "in line": "In line",
    "too few games": "Too few games to say",
}


# st.cache_data remembers the result for the same file text and username,
# so the report is not rebuilt every time the page reruns.
# If build_report raises ValueError, nothing is cached and the error is
# passed on to the caller.
@st.cache_data(show_spinner="Building your report...")
def cached_report(text, player):
    return build_report(text, player)


def display_table(table):
    """Turn a group_report table into a tidy table for display."""
    return pd.DataFrame({
        # str() stops years being shown with a thousands separator (2,024).
        "Group": table["group"].astype(str),
        "Games": table["games"],
        "Score": table["score"].map("{:.2f}".format),
        "Expected": table["expected"].map("{:.2f}".format),
        "Difference": table["difference"].map("{:+.2f}".format),
        "Range": [
            f"{low:.2f} to {high:.2f}"
            for low, high in zip(table["ci_low"], table["ci_high"])
        ],
        "Verdict": table["verdict"].map(VERDICT_LABELS),
    })


def render():
    """Draw the whole "Analyse your games" tab."""
    st.header("Analyse your games")
    st.write(
        "Upload a PGN file of your own games from Lichess or Chess.com to see "
        "where you score better or worse than your ratings predict."
    )
    st.info(
        "Privacy: your file is used only to build this report and is not "
        "saved by this app. It is, however, processed on the app's server."
    )

    uploaded = st.file_uploader("PGN file", type=["pgn"])
    username = st.text_input(
        "Your username (optional)",
        help="Leave blank to use the name that appears most often in the file.",
    )

    if uploaded is None:
        return

    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.warning(
            f"This file is larger than {MAX_FILE_MB} MB. Please upload a "
            "smaller file, for example the games from one year."
        )
        return

    # errors="replace" swaps any unreadable bytes for a placeholder
    # character instead of crashing.
    text = uploaded.getvalue().decode("utf-8", errors="replace")
    player = username.strip() or None   # blank means auto-detect

    try:
        report = cached_report(text, player)
    except ValueError as error:
        st.warning(str(error))
        return

    show_report(report)


def show_report(report):
    """Show the metrics, chart, tables and explanation for one report."""
    parse_info = report["parse_info"]
    clean_info = report["clean_info"]
    overall = report["overall"]

    # --- Summary numbers ---------------------------------------------------
    st.subheader(f"Report for {parse_info['player']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games in file", parse_info["games_in_file"])
    col2.metric("Games used", clean_info["kept"])
    col3.metric("Set aside", clean_info["set_aside"])
    col4.metric("Overall score", f"{overall['score']:.2f}")
    unreadable = parse_info["games_in_file"] - parse_info["games_used"]
    if unreadable > 0:
        st.caption(
            f"{unreadable} games in the file were skipped before cleaning "
            "(other players' games, missing ratings, unusual results or dates)."
        )

    # --- Rating over time --------------------------------------------------
    st.subheader("Your rating by month")
    st.line_chart(report["rating_months"], x="month", y="my_rating")

    # --- Tables ------------------------------------------------------------
    st.subheader("By opening")
    st.dataframe(display_table(report["opening_table"]), hide_index=True)

    st.subheader("By time control")
    st.dataframe(display_table(report["time_table"]), hide_index=True)

    st.subheader("By year")
    st.dataframe(display_table(report["year_table"]), hide_index=True)

    # --- Explanation -------------------------------------------------------
    st.subheader("How to read this")
    st.markdown(
        "- **Score** is your average points per game: 1 for a win, 0.5 for a "
        "draw, 0 for a loss.\n"
        "- **Expected** is the score your rating gap alone predicts for those "
        "games.\n"
        "- **Range** is where your true score is likely to be, given how many "
        "games there are. Fewer games give a wider range.\n"
        "- A group is only flagged as above or below expectation when the "
        "difference is larger than chance would usually explain. With many "
        "rows, some flags will still be luck, so treat them as hints.\n"
        "- The openings are the openings played in each game, which your "
        "opponent may have chosen.\n"
        "- **Set aside** games are games against an opponent you played very "
        "often, or runs of games where the ratings did not change (placeholder "
        "ratings). They are left out because they do not reflect normal "
        "rated play."
    )
