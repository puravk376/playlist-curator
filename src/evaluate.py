"""Evaluation helpers: per-playlist precision / recall / F1 and the confusion matrix."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def per_playlist_report(y_true, y_pred, labels):
    """Same columns as the paper's Table 2: precision, recall, F1, support."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rep = pd.DataFrame({"playlist": labels, "precision": p, "recall": r, "f1": f, "support": s})
    return rep.sort_values("f1", ascending=False).reset_index(drop=True)


def summary_row(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": p, "macro_recall": r, "macro_f1": f}


def confusion(y_true, y_pred, labels, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    return pd.DataFrame(cm, index=labels, columns=labels)


def top_confusions(cm, k=5):
    """The k most common (true -> predicted) mistakes, as row-normalised rates."""
    pairs = cm.stack().rename("rate").reset_index()
    pairs.columns = ["true", "predicted", "rate"]
    pairs = pairs[pairs["true"] != pairs["predicted"]]
    return pairs.sort_values("rate", ascending=False).head(k).reset_index(drop=True)
