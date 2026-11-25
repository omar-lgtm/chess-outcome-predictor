"""
Quick checks for pgn_tools.parse_pgn.

Part 1 runs three small tests on made-up games written below.
Part 2 reads the real sample file and prints a summary of it.

Run it with:  python test_pgn_tools.py
"""

import pandas as pd

from pgn_tools import parse_pgn

SAMPLE_FILE = "data/chipsa69-white.pgn"


# ---------------------------------------------------------------------------
# Part 1: tests on tiny made-up games (invented names and ratings)
# ---------------------------------------------------------------------------

def make_game(white, black, white_elo, black_elo, result):
    """Build the text of one fake PGN game with just the headers we need."""
    return (
        f'[Event "Test game"]\n'
        f'[White "{white}"]\n'
        f'[Black "{black}"]\n'
        f'[Result "{result}"]\n'
        f'[WhiteElo "{white_elo}"]\n'
        f'[BlackElo "{black_elo}"]\n'
        f'[UTCDate "2024.01.15"]\n'
        f'[TimeControl "180+2"]\n'
        f'[ECO "C00"]\n'
        f'\n1. e4 e6 {result}\n\n'
    )


# Test 1: the player has White and wins.
# The name is given in lower case to check that matching ignores case.
df, info = parse_pgn(make_game("Alice", "Bob", 1600, 1500, "1-0"), player="alice")
row = df.iloc[0]
assert info["games_used"] == 1
assert row["colour"] == "white"
assert row["result"] == "win" and row["score"] == 1.0
assert row["my_rating"] == 1600 and row["rating_diff"] == 100
assert row["base_seconds"] == 180 and row["increment_seconds"] == 2
print("Test 1 passed: player is White and wins")

# Test 2: the player has Black and Black wins (0-1), so from the
# player's point of view the result must be flipped to a win.
df, info = parse_pgn(make_game("Bob", "Alice", 1500, 1600, "0-1"), player="Alice")
row = df.iloc[0]
assert row["colour"] == "black"
assert row["result"] == "win" and row["score"] == 1.0
assert row["my_rating"] == 1600 and row["opp_rating"] == 1500
assert row["rating_diff"] == 100
print("Test 2 passed: player is Black and wins, result is flipped")

# Test 3: a game with an unknown rating ("?") must be skipped and counted.
df, info = parse_pgn(make_game("Alice", "Bob", "?", 1500, "1-0"), player="Alice")
assert len(df) == 0
assert info["games_used"] == 0
assert info["skipped"] == {"missing_rating": 1}
print("Test 3 passed: game with a '?' rating is skipped")

# Test 4: 10 games in a row with identical ratings must all be flagged
# stale_rating. One earlier game with different ratings must not be.
pgn = make_game("Alice", "Bob", 1450, 1480, "1-0")
for _ in range(10):
    pgn += make_game("Alice", "Bob", 1500, 1500, "1-0")
df, info = parse_pgn(pgn, player="Alice")
assert len(df) == 11
assert not df["stale_rating"].iloc[0]
assert df["stale_rating"].iloc[1:].all()
assert info["stale_rating_games"] == 10
print("Test 4 passed: 10 games with identical ratings are flagged stale_rating")

# Test 5: the same opponent 20 times (written in different upper/lower
# case) must be flagged repeat_opponent; a one-off opponent must not be.
# Ratings change every game so stale_rating doesn't get involved.
pgn = ""
for i in range(20):
    opponent = "Bob" if i % 2 == 0 else "BOB"
    pgn += make_game("Alice", opponent, 1500 + i, 1400 + i, "1-0")
pgn += make_game("Alice", "Carol", 1600, 1550, "1-0")
df, info = parse_pgn(pgn, player="Alice")
assert len(df) == 21
assert df["repeat_opponent"].iloc[:20].all()
assert not df["repeat_opponent"].iloc[20]
assert info["repeat_opponent_games"] == 20
assert "Bob" not in str(info) and "Carol" not in str(info)
print("Test 5 passed: opponent met 20 times is flagged repeat_opponent")
print()


# ---------------------------------------------------------------------------
# Part 2: summary of the real sample file
# ---------------------------------------------------------------------------

with open(SAMPLE_FILE, encoding="utf-8") as f:
    df, info = parse_pgn(f.read())

# Show every column when printing, instead of hiding some with "...".
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("Info:", info)
print("Shape (rows, columns):", df.shape)
print()
print("Missing values per column:")
print(df.isna().sum())
print()
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print()
print("Results:")
print(df["result"].value_counts())
print()
print("First rows:")
print(df.head())
print()

# How many games each flag catches, and how many are left with no flag.
stale = df["stale_rating"]
repeat = df["repeat_opponent"]
print("Flagged stale_rating:   ", stale.sum())
print("Flagged repeat_opponent:", repeat.sum())
print("Flagged by both:        ", (stale & repeat).sum())
print("Flagged by neither:     ", (~stale & ~repeat).sum())
print()
print("Results for games flagged by neither:")
print(df.loc[~stale & ~repeat, "result"].value_counts())
