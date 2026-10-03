"""Turning the cleaned table into a feature matrix.

x = [10 audio features | one-hot genre tags]   (same layout as the paper's Fig. 1)

Everything is wrapped in scikit-learn transformers so that the genre vocabulary
and the scaler are fitted on the training fold only. If we fitted them on the
whole dataset first, information from the test songs would leak into training.
"""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def split_genres(s):
    # module-level (not a lambda) so fitted pipelines can be pickled for the demo
    return s.split("|") if s else []


SCALING_OPTIONS = {
    "all": "standardise every column (paper)",
    "audio": "standardise audio only, genres stay 0/1",
    "none": "raw / unscaled",
}


def make_features(scale="all", use_genres=True, min_genre_count=1):
    """Return an (unfitted) transformer: DataFrame -> dense feature matrix.

    scale="all"   standardise every column (subtract mean, divide by std), which is
                  exactly the preprocessing step the paper tested. True means the same.
    scale="audio" standardise only the 10 audio features and leave the one-hot genre
                  columns as 0/1. Standardising a rare 0/1 column turns its 1s into huge
                  values (a tag 2 songs out of 1000 have becomes ~22), which hurts
                  distance-based models like the RBF / poly SVM.
    scale="none"  raw features. False means the same.
    use_genres=False audio features only (useful as an ablation).
    """
    scale = {True: "all", False: "none"}.get(scale, scale)
    if scale not in SCALING_OPTIONS:
        raise ValueError(f"scale must be one of {list(SCALING_OPTIONS)}, got {scale!r}")

    audio = StandardScaler() if scale == "audio" else "passthrough"
    parts = [("audio", audio, config.AUDIO_FEATURES)]
    if use_genres:
        genre_vec = CountVectorizer(
            tokenizer=split_genres, token_pattern=None, lowercase=False,
            binary=True, min_df=min_genre_count,
        )
        parts.append(("genre", genre_vec, "genres"))  # a string, not a list: CountVectorizer wants 1-D input
    columns = ColumnTransformer(parts, sparse_threshold=0.0)  # always dense, StandardScaler needs it

    steps = [("columns", columns)]
    if scale == "all":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def feature_names(fitted_features):
    """Column names of a fitted make_features() transformer."""
    return list(fitted_features.named_steps["columns"].get_feature_names_out())


def make_model(estimator, scale="all", use_genres=True):
    """Features + classifier in one pipeline, so cross-validation never leaks."""
    return Pipeline([("features", make_features(scale, use_genres)), ("clf", estimator)])
