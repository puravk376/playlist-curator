"""Neural network classifier (Person B), the paper's main model (section 4 / Table 3).

The paper's best network: 1 hidden layer with C = 13 neurons, sigmoid activation,
LogSoftmax output, NLL loss with L2 regularisation. scikit-learn's MLPClassifier
gives us the same thing:
    activation="logistic"  -> sigmoid hidden layer
    output layer           -> softmax + cross-entropy (= LogSoftmax + NLL)
    alpha                  -> L2 regularisation strength

We reuse Person A's feature pipeline and CV splitter, so the network sees exactly
the same 368 features, the same train/test split and the same 5 CV folds as the
baselines, and all numbers are directly comparable.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from . import config
from .baselines import _search
from .features import make_features


def make_nn(hidden=(13,), activation="logistic", alpha=1e-3, seed=config.SEED, max_iter=2000):
    return MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha,
                         max_iter=max_iter, random_state=seed)


def tune_nn(train_df):
    """Grid search with 5-fold CV on the TRAIN split only (the test split is never touched).

    Like Person A's tuned baselines, the scaling variant ("all" / "audio") is part of the grid.
    """
    grid = {
        "clf__hidden_layer_sizes": [(13,), (26,), (50,)],  # 13 = C, the paper's choice
        "clf__activation": ["logistic", "relu"],
        "clf__alpha": [1e-3, 1e-2, 1e-1, 1.0],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return _search(make_nn(), grid, train_df, use_genres=True)


def seed_runs(pipeline, train_df, test_df, seeds=(0, 1, 2, 3, 4)):
    """Refit the chosen network with different random starting weights.

    A neural net's result depends on its random initialisation, so one run can be
    lucky or unlucky. Reporting mean +/- std over several seeds is the honest number.
    """
    rows, models = [], []
    for s in seeds:
        model = clone(pipeline).set_params(clf__random_state=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train_df, train_df["playlist"])
        rows.append({"seed": s,
                     "train_acc": model.score(train_df, train_df["playlist"]),
                     "test_acc": model.score(test_df, test_df["playlist"])})
        models.append(model)
    return pd.DataFrame(rows), models


def epoch_curve(train_df, test_df, scale, hidden, activation, alpha, epochs=150, seed=config.SEED):
    """Train / test accuracy after every epoch, like the paper's Figure 2.

    This is only for the plot (to show whether the model over-fits).
    It is NOT used to pick any setting, so looking at test accuracy here is fine.
    """
    feats = make_features(scale)
    Xtr = feats.fit_transform(train_df)
    Xte = feats.transform(test_df)
    ytr, yte = train_df["playlist"].to_numpy(), test_df["playlist"].to_numpy()
    classes = np.unique(ytr)

    # full-batch gradient descent with Adam, one partial_fit call = one epoch
    nn = MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha,
                       batch_size=len(Xtr), random_state=seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            nn.partial_fit(Xtr, ytr, classes=classes)
            rows.append({"epoch": epoch, "train_acc": nn.score(Xtr, ytr), "test_acc": nn.score(Xte, yte),
                         "loss": nn.loss_})
    return pd.DataFrame(rows)