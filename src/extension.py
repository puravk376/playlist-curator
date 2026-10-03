"""Extension (Person B): "suggest a new playlist" when the model is not confident.

The paper's problem statement asks to sort songs into playlists *or suggest that a new
playlist be created*, but the paper never implemented the second half. We do it with a
confidence threshold:

    if max(predict_proba(song)) >= t:  put the song in the predicted playlist
    else:                               answer "new playlist"

Two questions:
1. Trade-off on known playlists: a higher t means fewer songs get a playlist
   (coverage), but those that do are more often right (accuracy).
2. Does it detect songs that really belong to a NEW playlist? We hide one playlist from
   training at a time, and check how often its songs get flagged.

The threshold is chosen with cross-validation on the TRAIN split, never on the test split.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_predict

from .baselines import cv_splitter

THRESHOLDS = np.round(np.arange(0.0, 0.96, 0.05), 2)


def cv_probabilities(model, train_df):
    """Out-of-fold probabilities on the train split: each song is scored by a model
    that did not see it. Used to choose the threshold without touching the test set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        proba = cross_val_predict(clone(model), train_df, train_df["playlist"],
                                  cv=cv_splitter(), method="predict_proba")
    classes = np.sort(train_df["playlist"].unique())  # same order sklearn uses
    return proba, classes


def sweep(proba, classes, y_true, thresholds=THRESHOLDS):
    """Coverage and accuracy-on-covered-songs for every threshold."""
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = pred == np.asarray(y_true)
    rows = []
    for t in thresholds:
        kept = conf >= t
        rows.append({
            "threshold": t,
            "coverage": kept.mean(),
            "accuracy_on_kept": correct[kept].mean() if kept.any() else np.nan,
            "errors_caught": (~correct & ~kept).sum() / max((~correct).sum(), 1),
            "correct_lost": (correct & ~kept).sum() / max(correct.sum(), 1),
        })
    return pd.DataFrame(rows)


def choose_threshold(cv_sweep, target_accuracy=0.90):
    """Lowest threshold whose CV accuracy on kept songs reaches the target."""
    ok = cv_sweep[cv_sweep.accuracy_on_kept >= target_accuracy]
    return float(ok.threshold.iloc[0]) if len(ok) else float(cv_sweep.threshold.iloc[-1])


def unseen_playlist_test(model, train_df, test_df, threshold):
    """Leave-one-playlist-out: train without playlist P, then score the test split.

    Songs from P are 'genuinely new' (the model has never seen that playlist), so a good
    detector should flag them. Songs from the other 12 playlists should mostly be kept.
    """
    rows = []
    for p in sorted(train_df["playlist"].unique()):
        tr = train_df[train_df.playlist != p]
        m = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(tr, tr["playlist"])
        conf = m.predict_proba(test_df).max(axis=1)
        flagged = conf < threshold
        is_new = (test_df["playlist"] == p).to_numpy()
        rows.append({
            "hidden_playlist": p,
            "new_songs_flagged": flagged[is_new].mean(),      # want HIGH
            "known_songs_flagged": flagged[~is_new].mean(),   # want LOW
            "new_mean_conf": conf[is_new].mean(),
            "known_mean_conf": conf[~is_new].mean(),
        })
    return pd.DataFrame(rows).sort_values("new_songs_flagged", ascending=False)
