"""
The "Analyse your games" tab of the Streamlit app.

The user uploads a PGN file of their own games. The app reads it with
report_tools.build_report() and shows how their results compare with what
their rating gaps alone would predict, by opening, time control and year.
"""

import pandas as pd
import streamlit as st

from pgn_tools import parse_pgn
from report_tools import add_groupings, build_report, clean_games

MAX_FILE_MB = 30   # keep in step with maxUploadSize in .streamlit/config.toml

# One set of result colours for the whole app. The words are always shown
# too, so nothing depends on colour alone.
PRIMARY_COLOUR = "#2F6F5E"
RESULT_COLOURS = {"win": "#2F6F5E", "draw": "#8A8F98", "loss": "#B5533C"}
# Light tints of the same colours, used behind dark text in the tables.
RESULT_TINTS = {"win": "#D5E2DF", "draw": "#E8E9EA", "loss": "#F0DDD8"}

# Which tint goes with each verdict as shown to the user.
VERDICT_TINTS = {
    "Possibly above expectation": RESULT_TINTS["win"],
    "In line": RESULT_TINTS["draw"],
    "Possibly below expectation": RESULT_TINTS["loss"],
}

# Where the clean games are kept for this browser session, so the
# "Predict a game" tab can use them. The value is a dict with "games"
# (the clean games table, no names), "info" and "fingerprint".
GAMES_KEY = "my_games"
# The upload whose games the user cleared, so they are not stored again
# until a new file is uploaded.
CLEARED_KEY = "my_games_cleared_upload"

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


# The clean games table, made with the same steps as the report.
# parse_pgn never puts opponent names in the table, and the username is
# left out here, so the stored table holds no names at all.
@st.cache_data(show_spinner=False)
def cached_clean_games(text, player):
    df, parse_info = parse_pgn(text, player)
    df, clean_info = clean_games(df)
    df = add_groupings(df)
    info = {"games_used": parse_info["games_used"], "clean_games": clean_info["kept"]}
    return df, info


def games_fingerprint(df):
    """A short label for a games table: count plus first and last date."""
    return (len(df), str(df["date"].min().date()), str(df["date"].max().date()))


def store_games(text, player, upload_id):
    """
    Keep the clean games in session state for the "Predict a game" tab,
    unless the user cleared this upload. Reruns the page once when the
    stored games change, because the first tab is drawn before this one.
    """
    if st.session_state.get(CLEARED_KEY) == upload_id:
        return
    games, info = cached_clean_games(text, player)
    fingerprint = games_fingerprint(games)
    current = st.session_state.get(GAMES_KEY)
    if current is None or current["fingerprint"] != fingerprint:
        st.session_state[GAMES_KEY] = {
            "games": games, "info": info, "fingerprint": fingerprint,
        }
        st.rerun()


def session_games_controls(upload_id):
    """A small note and a button to remove the games from this session."""
    st.subheader("Your games in this session")
    if GAMES_KEY not in st.session_state:
        st.caption("Your games have been cleared from this session.")
        return
    st.caption(
        "Your cleaned games are kept in memory for this session so the "
        "\"Predict a game\" tab can show your personal odds."
    )
    if st.button("Clear my games from this session"):
        st.session_state.pop(GAMES_KEY, None)
        st.session_state[CLEARED_KEY] = upload_id
        st.rerun()


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


def verdict_style(verdict):
    """CSS for one Verdict cell: a soft tint, dark text. No tint for "too few"."""
    tint = VERDICT_TINTS.get(verdict)
    return f"background-color: {tint}; color: #1F2933" if tint else ""


def styled_table(table):
    """The display table with the Verdict column tinted by result colour."""
    return display_table(table).style.map(verdict_style, subset=["Verdict"])


def render():
    """Draw the whole "Analyse your games" tab."""
    st.header("Analyse your games")
    st.write(
        "Upload a PGN file of your own games from Lichess or Chess.com to see "
        "where you score better or worse than your ratings predict."
    )
    st.info(
        "Privacy: your file is used only to build this report and your "
        "personal odds, and is not saved by this app. Your cleaned games, "
        "without any names, are kept in memory only for this browser session, "
        "and you can clear them below the report. The file is, however, "
        "processed on the app's server."
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

    # Each new upload gets its own id; used to remember a "clear" click.
    upload_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    store_games(text, player, upload_id)

    show_report(report)
    session_games_controls(upload_id)


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
    st.line_chart(report["rating_months"], x="month", y="my_rating", color=PRIMARY_COLOUR)

    # --- Tables ------------------------------------------------------------
    st.subheader("By opening")
    st.dataframe(styled_table(report["opening_table"]), hide_index=True)

    st.subheader("By time control")
    st.dataframe(styled_table(report["time_table"]), hide_index=True)

    st.subheader("By year")
    st.dataframe(styled_table(report["year_table"]), hide_index=True)

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
