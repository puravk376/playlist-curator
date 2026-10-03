"""Shared settings for the whole project.

Everything that both of us (Person A and Person B) need to agree on lives here:
which playlists we use, which features, the random seed and the file paths.
If you change something here, re-run `python run_person_a.py` so the processed
data and the train/test split get rebuilt.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SONGS_CSV = RAW_DIR / "spotify_songs.csv"          # Kaggle "30000 Spotify Songs"
ARTIST_GENRES_CSV = RAW_DIR / "data_w_genres.csv"  # Kaggle "Spotify Dataset 1921-2020", artist -> genres
PAPER_TOY_JSON = RAW_DIR / "paper_toy_playlists.json"  # the paper's own 13 playlists (authors' repo)
PAPER_TOY_CSV = RAW_DIR / "paper_toy_tracks.csv"       # built by scripts/build_paper_toy.py

SEED = 42
TEST_SIZE = 0.20  # the paper used an 80/20 split
CV_FOLDS = 5      # the paper used 5-fold cross validation

# The same 10 audio features the paper used (section 3 of the report).
# Spotify also gives instrumentalness and duration, but the paper left them out,
# so we do too to keep the comparison fair.
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "liveness", "valence", "tempo",
]

# Main experiment: 13 playlists, same count as the paper's "toy set".
# We picked them to cover a range of moods/genres, and deliberately kept a few
# pairs that should be hard to tell apart (Jazz Vibes vs Lush Lofi, the two rock
# lists, Southern Hip Hop vs Trap Nation), like the paper's Swagger / Kitchen Swagger.
# "Jazz Vibes" is literally the same Spotify playlist the paper used.
# Keys are Spotify playlist IDs (names are not unique in the dataset).
MAIN_PLAYLISTS = {
    "37i9dQZF1DX0SM0LYsmbMT": "Jazz Vibes",
    "37i9dQZF1DXc8kgYqQLMfH": "Lush Lofi",
    "37i9dQZF1DX8SfyqmSFDwe": "Reggaeton Classics",
    "37i9dQZF1DX6ThddIjWuGT": "Latin Pop Classics",
    "37i9dQZF1DX68H8ZujdnN7": "'80s Hard Rock",
    "37i9dQZF1DWXRqgorJj26U": "Rock Classics",
    "4EYSGTuqe9cVfSVpX4gtGv": "Permanent Wave",
    "3ykXidKLz1eYPvuGoFlD1e": "New Jack Swing",
    "78RRvktrPMSqAoCI21mNOe": "Neo Soul Music",
    "4lcyWQDOzPfcbZrcBI3FOW": "Southern Hip Hop",
    "5aPwKjwNHr6dnCejLcPTVx": "Trap Nation",
    "7kyvBmlc1uSqsTL0EuNLrx": "Electropop Hits",
    "5Bx5niVgi3qGQQw06C0RKq": "Big Room House",
}

# The paper's Figure 3 compares a "separable" user (Jacob) with an "overlapping"
# one (Myles). We recreate that with two 5-playlist sets.
EASY_PLAYLISTS = {
    "37i9dQZF1DX0SM0LYsmbMT": "Jazz Vibes",
    "37i9dQZF1DX8SfyqmSFDwe": "Reggaeton Classics",
    "37i9dQZF1DX68H8ZujdnN7": "'80s Hard Rock",
    "5Bx5niVgi3qGQQw06C0RKq": "Big Room House",
    "78RRvktrPMSqAoCI21mNOe": "Neo Soul Music",
}
HARD_PLAYLISTS = {
    "4lcyWQDOzPfcbZrcBI3FOW": "Southern Hip Hop",
    "37i9dQZF1DWUFmyho2wkQU": "Hip-Hop Drive",
    "37i9dQZF1DWT5MrZnPU1zD": "Hip Hop Controller",
    "2NMW1nwQYSVlXd26uLenX6": "90's Southern Hip Hop",
    "0ZRwrJ2EDGyKR6YgQPWXeO": "Gangster Rap Workout",
}
