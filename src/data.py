"""Loading and cleaning the data.

The paper pulled its data straight from the Spotify Web API, but the audio
features endpoint was shut off for new apps in Nov 2024, so we can't redo that.
Instead we use two public Kaggle datasets that were collected from the same API:

* spotify_songs.csv   - ~32k rows of (track, playlist) with the audio features
* data_w_genres.csv   - ~28k artists with Spotify's genre tags

Like the paper, genre tags belong to the *artist*, and a track gets the genres
of its artist. We do NOT use the playlist_genre / playlist_subgenre columns in
spotify_songs.csv, because those describe the playlist itself, which would leak
the answer into the features.
"""
import ast

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_artist_genres():
    """Return {lowercased artist name: sorted list of genre tags}."""
    g = pd.read_csv(config.ARTIST_GENRES_CSV, usecols=["artists", "genres"])
    g["genres"] = g["genres"].map(ast.literal_eval)
    lookup = {}
    for artist, genres in zip(g["artists"].str.lower().str.strip(), g["genres"]):
        # a handful of artists appear twice with different capitalisation -> merge
        lookup.setdefault(artist, set()).update(genres)
    return {a: sorted(gs) for a, gs in lookup.items() if gs}


def genres_for(artists, lookup):
    """Union of genre tags over a list of artist names, joined with '|'."""
    tags = set()
    for a in artists:
        tags.update(lookup.get(str(a).lower().strip(), []))
    return "|".join(sorted(tags))


def load_playlists(playlists, verbose=True):
    """Build a clean (track, playlist) table for the given {playlist_id: name} dict.

    Returns (df, log) where log records how many rows each cleaning step removed,
    so we can quote the numbers in the write-up.
    """
    raw = pd.read_csv(config.SONGS_CSV)
    df = raw[raw["playlist_id"].isin(playlists)].copy()
    df["playlist"] = df["playlist_id"].map(playlists)
    log = {"rows_selected": len(df)}

    # 1. rows with any missing audio feature are useless to us
    before = len(df)
    df = df.dropna(subset=config.AUDIO_FEATURES + ["track_name", "track_artist"])
    log["dropped_missing"] = before - len(df)

    # 2. the same track listed twice in one playlist
    before = len(df)
    df = df.drop_duplicates(subset=["playlist_id", "track_id"])
    # ...or the same song under two different track IDs (single vs album release)
    df["_song"] = df["track_name"].str.lower().str.strip() + " | " + df["track_artist"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["playlist_id", "_song"])
    log["dropped_duplicates_within_playlist"] = before - len(df)

    # 3. songs that sit in more than one of our playlists. The paper points out this
    #    makes the problem multi-label; for a single-label classifier the same x with
    #    two different y's is just noise, so we drop them (and report how many).
    before = len(df)
    n_lists = df.groupby("_song")["playlist_id"].transform("nunique")
    log["songs_in_multiple_playlists"] = int(df.loc[n_lists > 1, "_song"].nunique())
    df = df[n_lists == 1]
    log["dropped_multi_playlist_rows"] = before - len(df)

    # 4. attach artist genre tags (one-hot encoding happens later, inside the
    #    model pipeline, so that it is only ever fitted on training data)
    lookup = load_artist_genres()
    df["genres"] = df["track_artist"].map(lambda a: genres_for([a], lookup))
    df["has_genre"] = df["genres"] != ""

    df = df[["track_id", "track_name", "track_artist", "playlist", *config.AUDIO_FEATURES, "genres", "has_genre"]]
    df = df.sort_values(["playlist", "track_name"]).reset_index(drop=True)
    log["final_tracks"] = len(df)
    log["tracks_with_genre_tags"] = round(float(df["has_genre"].mean()), 3)

    if verbose:
        print("  cleaning:", ", ".join(f"{k}={v}" for k, v in log.items()))
    return df, log


def load_paper_toy():
    """The paper's own 13 playlists, rebuilt by scripts/build_paper_toy.py."""
    if not config.PAPER_TOY_CSV.exists():
        raise FileNotFoundError(
            f"{config.PAPER_TOY_CSV} is missing - run `python scripts/build_paper_toy.py` first"
        )
    df = pd.read_csv(config.PAPER_TOY_CSV, keep_default_na=False)
    df["has_genre"] = df["genres"] != ""
    return df


def add_split(df, seed=config.SEED, test_size=config.TEST_SIZE):
    """Stratified 80/20 split, stored as a column so Person B can reuse the exact same split."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df["playlist"], random_state=seed
    )
    df = df.copy()
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def save_processed(df, name):
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def load_processed(name="main"):
    """Load a processed dataset (main / easy / hard / paper_toy) with its train/test split.

    Example (this is what the neural net / demo code should use):

        from src.data import load_processed
        from src.features import make_features
        df = load_processed("main")
        train, test = df[df.split == "train"], df[df.split == "test"]
    """
    path = config.PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python run_person_a.py` first")
    df = pd.read_csv(path, keep_default_na=False)
    df["has_genre"] = df["genres"] != ""
    return df
