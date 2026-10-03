"""Person B, step 1: the neural network.

    python run_nn.py

1. Tune the network with 5-fold CV on the train split (test split untouched).
2. Refit the chosen network with 5 random seeds and score each once on the test split.
3. Per-playlist report + confusion matrix, same style as Person A's baselines.
4. Train/test accuracy per epoch (our version of the paper's Figure 2).
5. Save the model for the demo app.

Outputs: results/nn_*.csv, results/nn_metrics.json, figures/10-12, models/nn.joblib
"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src import config, plots
from src.baselines import describe_params
from src.data import load_processed
from src.evaluate import confusion, per_playlist_report, summary_row, top_confusions
from src.nn import epoch_curve, seed_runs, tune_nn

BEST_BASELINE = {"model": "Logistic Regression (tuned)", "test_acc": 0.806, "macro_f1": 0.811}


def main():
    df = load_processed("main")
    train, test = df[df.split == "train"], df[df.split == "test"]
    labels = sorted(df.playlist.unique())
    config.RESULTS_DIR.mkdir(exist_ok=True)
    print(f"train {len(train)} / test {len(test)} songs, {len(labels)} playlists")

    # 1. tuning: CV on train only
    print("\n[1] Grid search (5-fold CV on train) ...")
    search = tune_nn(train)
    params = describe_params(search)
    cv = pd.DataFrame(search.cv_results_)
    cv_table = pd.DataFrame({
        "scale": ["all" if "scale" in f.named_steps else "audio" for f in cv["param_features"]],
        "hidden": cv["param_clf__hidden_layer_sizes"].astype(str),
        "activation": cv["param_clf__activation"],
        "alpha": cv["param_clf__alpha"],
        "cv_mean": cv["mean_test_score"].round(4),
        "cv_std": cv["std_test_score"].round(4),
    }).sort_values("cv_mean", ascending=False)
    cv_table.to_csv(config.RESULTS_DIR / "nn_tuning.csv", index=False)
    print(cv_table.head(8).to_string(index=False))
    print(f"chosen by CV: {params}  (cv acc {search.best_score_:.3f})")

    # 2. several seeds on the test split
    print("\n[2] Refit with 5 seeds, score on test ...")
    runs, models = seed_runs(search.best_estimator_, train, test)
    runs.to_csv(config.RESULTS_DIR / "nn_seed_runs.csv", index=False)
    print(runs.to_string(index=False))
    mean, std = runs.test_acc.mean(), runs.test_acc.std()
    print(f"NN test accuracy: {mean:.3f} +/- {std:.3f}   (best baseline: {BEST_BASELINE['test_acc']})")

    # 3. detailed report for the run closest to the mean (a typical run, not the luckiest)
    typical = (runs.test_acc - mean).abs().idxmin()
    model = models[typical]
    pred = model.predict(test)
    rep = per_playlist_report(test["playlist"], pred, labels)
    cm = confusion(test["playlist"], pred, labels)
    rep.to_csv(config.RESULTS_DIR / "nn_per_playlist.csv", index=False)
    cm.to_csv(config.RESULTS_DIR / "nn_confusion_matrix.csv")
    plots.confusion_matrix_plot(cm, name="10_nn_confusion_matrix.png", title="Confusion matrix: neural network")
    plots.per_playlist_scores(rep, name="11_nn_per_playlist_scores.png", title="Per-playlist scores: neural network")
    summary = summary_row(test["playlist"], pred)
    print("\nper-playlist (typical seed):")
    print(rep.round(3).to_string(index=False))
    print("\nmost common mistakes:")
    print(top_confusions(cm).round(2).to_string(index=False))

    # 4. accuracy per epoch (paper Fig. 2)
    print("\n[3] Accuracy per epoch ...")
    hidden = search.best_params_["clf__hidden_layer_sizes"]
    curve = epoch_curve(train, test, params["scale"], hidden, params["activation"], params["alpha"])
    curve.to_csv(config.RESULTS_DIR / "nn_epoch_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(curve.epoch, curve.train_acc, color=plots.BLUE, label="train")
    ax.plot(curve.epoch, curve.test_acc, color=plots.ORANGE, label="test", linestyle="--")
    ax.set(xlabel="epoch", ylabel="accuracy", ylim=(0, 1), title="Neural network: accuracy per epoch")
    ax.grid(True, axis="y")
    ax.legend()
    plots._save(fig, "12_nn_epoch_curve.png")
    print(f"after {len(curve)} epochs: train {curve.train_acc.iloc[-1]:.3f}, test {curve.test_acc.iloc[-1]:.3f}")

    # 5. save everything
    models_dir = config.ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "nn.joblib")

    metrics = {
        "chosen_params": {k: str(v) for k, v in params.items()},
        "cv_acc": round(search.best_score_, 4),
        "test_acc_mean": round(mean, 4), "test_acc_std": round(std, 4),
        "typical_seed": int(runs.seed[typical]),
        "typical_seed_metrics": {k: round(v, 4) for k, v in summary.items()},
        "best_baseline": BEST_BASELINE,
    }
    (config.RESULTS_DIR / "nn_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nsaved: results/nn_*.csv, results/nn_metrics.json, figures/10-12, models/nn.joblib")


if __name__ == "__main__":
    main()