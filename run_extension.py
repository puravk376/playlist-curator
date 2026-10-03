"""Person B, step 2: extension - "suggest a new playlist" when confidence is low.

    python run_extension.py      (run python run_nn.py first: it saves models/nn.joblib)

1. Pick the confidence threshold with 5-fold CV on the train split.
2. Apply it once to the test split: coverage, accuracy on kept songs, errors caught.
3. Who gets flagged? Songs with vs without genre tags (Person A's 42% vs 13% finding).
4. Leave-one-playlist-out: are songs from a playlist the model never saw flagged as "new"?

Outputs: results/ext_*.csv, results/ext_metrics.json, figures/13-15
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src import config, plots
from src.data import load_processed
from src.extension import choose_threshold, cv_probabilities, sweep, unseen_playlist_test

TARGET = 0.90  # we want 90% of the songs we DO place to be right


def main():
    df = load_processed("main")
    train, test = df[df.split == "train"], df[df.split == "test"]
    model = joblib.load(config.ROOT / "models" / "nn.joblib")
    classes = model.classes_

    # 1. choose the threshold on train (out-of-fold probabilities)
    print("[1] Choosing the threshold with 5-fold CV on train ...")
    cv_proba, cv_classes = cv_probabilities(model, train)
    cv_sweep = sweep(cv_proba, cv_classes, train["playlist"])
    t = choose_threshold(cv_sweep, TARGET)
    print(cv_sweep.round(3).to_string(index=False))
    print(f"chosen threshold: {t}  (lowest t with CV accuracy on kept songs >= {TARGET})")

    # 2. apply once to test
    print("\n[2] Test split ...")
    proba = model.predict_proba(test)
    test_sweep = sweep(proba, classes, test["playlist"])
    cv_sweep.to_csv(config.RESULTS_DIR / "ext_threshold_sweep_cv.csv", index=False)
    test_sweep.to_csv(config.RESULTS_DIR / "ext_threshold_sweep_test.csv", index=False)
    at_t = test_sweep[test_sweep.threshold == t].iloc[0]
    print(f"no threshold : coverage 1.000, accuracy {test_sweep.accuracy_on_kept.iloc[0]:.3f}")
    print(f"threshold {t}: coverage {at_t.coverage:.3f}, accuracy on kept {at_t.accuracy_on_kept:.3f}, "
          f"errors caught {at_t.errors_caught:.3f}, correct songs lost {at_t.correct_lost:.3f}")

    # 3. who gets flagged?
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == test["playlist"].to_numpy()
    flagged = conf < t
    tagged = test["has_genre"].to_numpy()
    print("\n[3] Flag rate by genre tags:")
    print(f"songs WITH genre tags   : {flagged[tagged].mean():.3f} flagged (n={tagged.sum()})")
    print(f"songs WITHOUT genre tags: {flagged[~tagged].mean():.3f} flagged (n={(~tagged).sum()})")

    # 4. genuinely new playlists
    print("\n[4] Leave-one-playlist-out (13 retrains) ...")
    unseen = unseen_playlist_test(model, train, test, t)
    unseen.to_csv(config.RESULTS_DIR / "ext_unseen_playlist.csv", index=False)
    print(unseen.round(3).to_string(index=False))
    new_rate, known_rate = unseen.new_songs_flagged.mean(), unseen.known_songs_flagged.mean()
    print(f"average: songs from the unseen playlist flagged {new_rate:.3f}, "
          f"songs from known playlists flagged {known_rate:.3f}")

    # figures
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(cv_sweep.coverage, cv_sweep.accuracy_on_kept, color=plots.BLUE, marker="o", ms=3, label="CV on train")
    ax.plot(test_sweep.coverage, test_sweep.accuracy_on_kept, color=plots.ORANGE, marker="s", ms=3,
            linestyle="--", label="test")
    ax.scatter([at_t.coverage], [at_t.accuracy_on_kept], s=90, facecolor="none", edgecolor=plots.INK, zorder=5)
    ax.annotate(f"threshold {t}", (at_t.coverage, at_t.accuracy_on_kept), xytext=(-90, 12),
                textcoords="offset points", fontsize=9)
    ax.set(xlabel="coverage (share of songs given a playlist)", ylabel="accuracy on those songs",
           title="Accuracy vs coverage as the threshold rises")
    ax.invert_xaxis()
    ax.grid(True, axis="y")
    ax.legend()
    plots._save(fig, "13_ext_accuracy_vs_coverage.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, color=plots.BLUE, alpha=0.75, label="correct")
    ax.hist(conf[~correct], bins=bins, color=plots.RED, alpha=0.75, label="wrong")
    ax.axvline(t, color=plots.INK, linestyle="--", linewidth=1)
    ax.text(t, ax.get_ylim()[1] * 0.92, f"  threshold {t}", fontsize=9)
    ax.set(xlabel="model confidence = max predicted probability", ylabel="test songs",
           title="Wrong predictions have lower confidence")
    ax.legend()
    plots._save(fig, "14_ext_confidence_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(unseen))
    ax.barh(y - 0.2, unseen.new_songs_flagged, height=0.4, color=plots.ORANGE, label="songs from the hidden playlist")
    ax.barh(y + 0.2, unseen.known_songs_flagged, height=0.4, color=plots.BLUE, label="songs from known playlists")
    ax.set_yticks(y, unseen.hidden_playlist)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel=f"share flagged as 'new playlist' (threshold {t})",
           title="Detecting a playlist the model never saw")
    ax.legend(loc="lower right")
    plots._save(fig, "15_ext_unseen_playlist.png")

    metrics = {
        "threshold": t, "target_cv_accuracy": TARGET,
        "test_no_threshold": {"coverage": 1.0, "accuracy": round(float(test_sweep.accuracy_on_kept.iloc[0]), 4)},
        "test_at_threshold": {k: round(float(at_t[k]), 4)
                              for k in ["coverage", "accuracy_on_kept", "errors_caught", "correct_lost"]},
        "flag_rate_tagged": round(float(flagged[tagged].mean()), 4),
        "flag_rate_untagged": round(float(flagged[~tagged].mean()), 4),
        "unseen_playlist_flag_rate": round(float(new_rate), 4),
        "known_playlist_flag_rate": round(float(known_rate), 4),
    }
    (config.RESULTS_DIR / "ext_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nsaved: results/ext_*.csv, results/ext_metrics.json, figures/13-15")


if __name__ == "__main__":
    main()
