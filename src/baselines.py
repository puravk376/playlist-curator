"""Baseline classifiers from the paper (section 4 / Table 1), all from scikit-learn.

Perceptron (one-vs-all), logistic regression, and SVMs with linear / polynomial /
sigmoid / RBF kernels. A majority-class "dummy" model is included as a floor:
anything that doesn't beat it hasn't learned anything.
"""
import warnings

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from . import config
from .features import make_features, make_model


def baseline_models():
    s = config.SEED
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Perceptron": Perceptron(max_iter=1000, random_state=s),  # one-vs-all, like the paper
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Linear SVM": SVC(kernel="linear"),
        "Poly SVM": SVC(kernel="poly", degree=3),
        "Sigmoid SVM": SVC(kernel="sigmoid"),
        "RBF SVM": SVC(kernel="rbf"),
    }


def cv_splitter():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def _xy(df):
    return df, df["playlist"].to_numpy()


def compare_scaling(train_df, use_genres=True, variants=("all", "audio", "none")):
    """5-fold CV accuracy of every baseline under each preprocessing variant.

    This is our version of the paper's Table 1 (scaled vs unscaled), plus a third
    column for "scale the audio features but keep genres as 0/1". We cross-validate
    on the training split only and keep the test split untouched for the final numbers.
    """
    X, y = _xy(train_df)
    rows = []
    for name, est in baseline_models().items():
        for scale in variants:
            with warnings.catch_warnings():
                # unscaled logistic regression / perceptron often don't converge;
                # that is part of the point of this experiment, so don't spam the console
                warnings.simplefilter("ignore", ConvergenceWarning)
                scores = cross_val_score(make_model(est, scale, use_genres), X, y, cv=cv_splitter(), n_jobs=-1)
            rows.append({"model": name, "scale": scale, "cv_mean": scores.mean(), "cv_std": scores.std()})
    out = pd.DataFrame(rows)
    table = out.pivot(index="model", columns="scale", values=["cv_mean", "cv_std"])
    cols = [f"{stat}_{v}" for v in variants for stat in ("cv_mean", "cv_std")]
    table.columns = [f"{stat}_{v}" for stat, v in table.columns]
    return table.loc[list(baseline_models()), cols].reset_index()


def _search(estimator, grid, train_df, use_genres):
    X, y = _xy(train_df)
    # the preprocessing variant is treated as one more hyper-parameter, chosen by CV
    grid = {"features": [make_features(s, use_genres) for s in ("all", "audio")], **grid}
    search = GridSearchCV(make_model(estimator, "all", use_genres), grid, cv=cv_splitter(), n_jobs=-1)
    search.fit(X, y)
    return search


def describe_params(search):
    """Readable version of best_params_, e.g. {'scale': 'audio', 'C': 10}."""
    out = {}
    for k, v in search.best_params_.items():
        if k == "features":
            out["scale"] = "all" if "scale" in v.named_steps else "audio"
        else:
            out[k.replace("clf__", "")] = v
    return out


def tune_rbf_svm(train_df, use_genres=True):
    """Grid-search C and gamma for the RBF SVM (the paper tuned C and got 0.80)."""
    grid = {"clf__C": [0.3, 1, 3, 10, 30, 100], "clf__gamma": ["scale", 0.0003, 0.001, 0.003, 0.01, 0.03]}
    return _search(SVC(kernel="rbf"), grid, train_df, use_genres)


def tune_logistic(train_df, use_genres=True):
    """Small grid over the regularisation strength of logistic regression."""
    grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    return _search(LogisticRegression(max_iter=5000), grid, train_df, use_genres)


def test_scores(train_df, test_df, models, scale="all", use_genres=True):
    """Fit each model on the full training split and score it once on the test split.

    Plain estimators get wrapped with the given preprocessing; ready-made pipelines
    (e.g. a tuned GridSearchCV.best_estimator_) are used as they are.
    """
    Xtr, ytr = _xy(train_df)
    Xte, yte = _xy(test_df)
    rows, fitted = [], {}
    for name, est in models.items():
        model = clone(est) if hasattr(est, "named_steps") else make_model(est, scale, use_genres)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Xtr, ytr)
        rows.append({"model": name, "train_acc": model.score(Xtr, ytr), "test_acc": model.score(Xte, yte)})
        fitted[name] = model
    return pd.DataFrame(rows), fitted
