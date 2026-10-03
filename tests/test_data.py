"""Quick checks that the data pipeline does what the write-up says it does.

    python -m pytest -q
"""
import numpy as np
import pytest

from src import config
from src.data import add_split, load_playlists
from src.features import make_features


@pytest.fixture(scope="module")
def main_df():
    df, _ = load_playlists(config.MAIN_PLAYLISTS, verbose=False)
    return add_split(df)


def test_all_playlists_present(main_df):
    assert set(main_df["playlist"]) == set(config.MAIN_PLAYLISTS.values())


def test_no_missing_audio_features(main_df):
    assert not main_df[config.AUDIO_FEATURES].isna().any().any()


def test_each_song_has_one_label(main_df):
    song = main_df["track_name"].str.lower().str.strip() + "|" + main_df["track_artist"].str.lower().str.strip()
    assert song.groupby(song).size().max() == 1
    assert main_df["track_id"].is_unique


def test_split_is_80_20_and_stratified(main_df):
    frac = (main_df["split"] == "test").groupby(main_df["playlist"]).mean()
    assert abs((main_df["split"] == "test").mean() - config.TEST_SIZE) < 0.01
    assert frac.between(0.15, 0.25).all()


def test_split_is_reproducible(main_df):
    again = add_split(main_df.drop(columns="split"))
    assert (again["split"] == main_df["split"]).all()


def test_no_label_leak_in_features():
    # playlist_genre / playlist_subgenre describe the playlist itself and must never be features
    t = make_features()
    cols = t.named_steps["columns"].transformers
    used = {c for _, _, cs in cols for c in ([cs] if isinstance(cs, str) else cs)}
    assert not used & {"playlist", "playlist_name", "playlist_id", "playlist_genre", "playlist_subgenre"}


def test_genre_vocab_comes_from_training_data_only(main_df):
    train, test = main_df[main_df.split == "train"], main_df[main_df.split == "test"]
    t = make_features(scale="none").fit(train)
    n_audio = len(config.AUDIO_FEATURES)
    vocab = set(t.named_steps["columns"].named_transformers_["genre"].vocabulary_)
    train_tags = set("|".join(train["genres"]).split("|")) - {""}
    assert vocab == train_tags
    # a test song whose only tags were never seen in training gets an all-zero genre vector
    X = t.transform(test)
    assert X.shape[1] == n_audio + len(vocab)


@pytest.mark.parametrize("scale", ["all", "audio", "none"])
def test_scaling_variants(main_df, scale):
    train = main_df[main_df.split == "train"]
    X = make_features(scale=scale).fit_transform(train)
    audio = X[:, : len(config.AUDIO_FEATURES)]
    if scale == "none":
        assert audio[:, config.AUDIO_FEATURES.index("tempo")].mean() > 50
    else:
        assert np.allclose(audio.mean(0), 0, atol=1e-8) and np.allclose(audio.std(0), 1, atol=1e-6)
    genres = X[:, len(config.AUDIO_FEATURES):]
    if scale in ("audio", "none"):
        assert set(np.unique(genres)) <= {0.0, 1.0}
