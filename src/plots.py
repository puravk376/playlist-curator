"""All the figures for the report / slides. Everything is saved as PNG in figures/."""
import matplotlib

matplotlib.use("Agg")  # no window needed, we only save files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

from . import config
from .features import make_features

# Colours: a colour-blind-checked categorical set, plus one blue ramp for magnitudes.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948")
GREY, INK, INK_2, GRID, SURFACE = "#c9c8c3", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FIVE = [BLUE, YELLOW, AQUA, VIOLET, MAGENTA]          # passes the all-pairs check for scatter plots
MARKERS = ["o", "s", "^", "D", "v"]                    # second cue so colour is never the only one
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("bluered", [BLUE, "#f0efec", RED])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "font.size": 9.5, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df, name="01_class_balance.png"):
    counts = df["playlist"].value_counts().sort_values()
    genre_cov = df.groupby("playlist")["has_genre"].mean().reindex(counts.index)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(counts.index, counts.values, color=BLUE, height=0.7)
    for i, (n, g) in enumerate(zip(counts.values, genre_cov.values)):
        ax.text(n + 1.5, i, f"{n}   ({g:.0%} with genre tags)", va="center", color=INK_2, fontsize=8.5)
    ax.set_xlim(0, counts.max() * 1.55)
    ax.set_xlabel("tracks after cleaning")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title(f"{len(counts)} playlists, {len(df)} tracks", loc="left")
    return _save(fig, name)


def feature_scales(df, name="02_feature_scales.png"):
    """Why scaling matters: raw features live on wildly different ranges."""
    X = df[config.AUDIO_FEATURES].to_numpy(float)
    Xs = (X - X.mean(0)) / X.std(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.25]))

    # left: how wide each raw feature's range is (log axis, so 0.9 and 170 both fit)
    ranges = X.max(0) - X.min(0)
    y = np.arange(len(ranges))
    axes[0].barh(y, ranges, color=BLUE, height=0.6)
    for yi, r, lo, hi in zip(y, ranges, X.min(0), X.max(0)):
        axes[0].text(r * 1.15, yi, f"{lo:.3g} to {hi:.3g}", va="center", fontsize=8, color=INK_2)
    axes[0].set_xscale("log"); axes[0].set_xlim(0.3, 3000)
    axes[0].set_yticks(y, config.AUDIO_FEATURES)
    axes[0].set_xlabel("range of raw values, max - min (log scale)")
    axes[0].set_title("Raw features live on very different scales", loc="left")

    # right: after standardisation every feature has mean 0 and std 1
    bp = axes[1].boxplot(Xs, vert=False, positions=y, widths=0.6, patch_artist=True,
                         flierprops=dict(marker="o", markersize=2, markerfacecolor=GREY, markeredgecolor="none"),
                         medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK_2),
                         capprops=dict(color=INK_2), boxprops=dict(edgecolor=BLUE, linewidth=1))
    for b in bp["boxes"]:
        b.set_facecolor("#cde2fb")
    axes[0].set_yticks(y, config.AUDIO_FEATURES)  # boxplot resets the shared y labels, so set them again
    axes[1].set_xlabel("standard deviations from the mean")
    axes[1].set_title("After standardisation (z-scores)", loc="left")
    for ax in axes:
        ax.xaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, name)


def correlation(df, name="03_feature_correlation.png"):
    c = df[config.AUDIO_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(c, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            if i != j and abs(c.iat[i, j]) >= 0.3:
                ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson correlation")
    ax.set_title("Audio feature correlations (|r| >= 0.3 labelled)", loc="left")
    return _save(fig, name)


def scaled_vs_unscaled(table, name="04_scaled_vs_unscaled.png", title=None):
    """Grouped bars of 5-fold CV accuracy; this is the paper's Table 1 as a picture."""
    t = table[table["model"] != "Majority class"].iloc[::-1]
    floor = table.loc[table["model"] == "Majority class", "cv_mean_all"]
    series = [("all", "standardise everything (paper)", BLUE),
              ("audio", "standardise audio only, genres 0/1", ORANGE),
              ("none", "raw / unscaled", AQUA)]
    series = [s for s in series if f"cv_mean_{s[0]}" in t]
    y = np.arange(len(t)) * (len(series) * 0.3 + 0.35)
    h = 0.27
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for k, (key, label, colour) in enumerate(series):
        off = (len(series) - 1 - k - (len(series) - 1) / 2) * (h + 0.03)  # first series on top
        ax.barh(y + off, t[f"cv_mean_{key}"], h, xerr=t[f"cv_std_{key}"], color=colour, label=label,
                error_kw=dict(ecolor=INK_2, lw=0.9, capsize=2))
        for yi, v, sd in zip(y, t[f"cv_mean_{key}"], t[f"cv_std_{key}"]):
            ax.text(v + sd + 0.015, yi + off, f"{v:.2f}", va="center", fontsize=7.5, color=INK_2)
    if len(floor):
        ax.axvline(floor.iat[0], color=INK_2, lw=1, ls=":")
        ax.text(floor.iat[0] + 0.01, y[-1] + 0.62, "majority-class floor", fontsize=8, color=INK_2)
    ax.set_yticks(y, t["model"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"{config.CV_FOLDS}-fold cross-validation accuracy (mean ± std, training split)")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(series), fontsize=8)
    ax.set_title(title or "Scaling makes or breaks most of the baselines", loc="left", pad=24)
    return _save(fig, name)


def confusion_matrix_plot(cm, name="05_confusion_matrix.png", title="Confusion matrix (test set)"):
    labels = list(cm.index)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cm.to_numpy(), cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.iat[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else INK)
    ax.set_xlabel("predicted playlist"); ax.set_ylabel("true playlist")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="fraction of the true playlist's songs")
    ax.set_title(title, loc="left")
    return _save(fig, name)


def per_playlist_scores(report, name="06_per_playlist_scores.png", title="Per-playlist precision / recall / F1 (test set)"):
    r = report.sort_values("f1")
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i in y:
        lo, hi = r[["precision", "recall", "f1"]].iloc[i].min(), r[["precision", "recall", "f1"]].iloc[i].max()
        ax.plot([lo, hi], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(r["precision"], y, s=46, color=BLUE, marker="o", label="precision", zorder=3, edgecolor=SURFACE, lw=1.5)
    ax.scatter(r["recall"], y, s=46, color=ORANGE, marker="s", label="recall", zorder=3, edgecolor=SURFACE, lw=1.5)
    ax.scatter(r["f1"], y, s=70, color=INK, marker="|", label="F1", zorder=4, lw=2)
    ax.set_yticks(y, [f"{p}  (n={s})" for p, s in zip(r["playlist"], r["support"])])
    ax.set_xlim(0, 1.05)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, 1.0))
    ax.set_title(title, loc="left", pad=26)
    return _save(fig, name)


def _pca(df, use_genres=True):
    X = make_features(scale=True, use_genres=use_genres).fit_transform(df)
    pca = PCA(n_components=2, random_state=config.SEED)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def pca_small_multiples(df, name="07_pca_main.png", use_genres=False):
    """One panel per playlist: that playlist in colour, everything else in grey.
    13 colours in one scatter would be unreadable, so we facet instead."""
    Z, evr = _pca(df, use_genres)
    playlists = sorted(df["playlist"].unique())
    cols = 5; rows = int(np.ceil(len(playlists) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), sharex=True, sharey=True)
    for ax, p in zip(axes.flat, playlists):
        m = (df["playlist"] == p).to_numpy()
        ax.scatter(Z[~m, 0], Z[~m, 1], s=4, color=GREY, alpha=0.5, lw=0)
        ax.scatter(Z[m, 0], Z[m, 1], s=9, color=BLUE, lw=0)
        ax.set_title(p, loc="left", fontsize=9.5)
        ax.tick_params(labelsize=7.5)
    for ax in list(axes.flat)[len(playlists):]:
        ax.axis("off")
    what = "audio + genre" if use_genres else "10 audio"
    fig.suptitle(f"PCA of the {what} features: where each playlist sits "
                 f"(PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%} of variance)", x=0.01, ha="left", fontweight="bold")
    fig.supxlabel("PC1", color=INK_2, fontsize=9); fig.supylabel("PC2", color=INK_2, fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


def pca_easy_vs_hard(easy, hard, accs=None, name="08_pca_easy_vs_hard.png", use_genres=False):
    """Our version of the paper's Fig. 3 (Jacob's separable vs Myles' overlapping playlists)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, df, label in [(axes[0], easy, "easy"), (axes[1], hard, "hard")]:
        Z, evr = _pca(df, use_genres)
        for k, p in enumerate(sorted(df["playlist"].unique())):
            m = (df["playlist"] == p).to_numpy()
            ax.scatter(Z[m, 0], Z[m, 1], s=16, color=FIVE[k], marker=MARKERS[k], label=p,
                       alpha=0.85, edgecolor=SURFACE, lw=0.4)
        sub = f"\nRBF SVM test accuracy {accs[label]:.2f}" if accs else ""
        title = "Separable: 5 different genres" if label == "easy" else "Overlapping: 5 hip-hop playlists"
        ax.set_title(title + sub, loc="left", fontsize=11.5)
        ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variance)"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.13), ncols=3, markerscale=1.4)
    what = "genre + audio" if use_genres else "10 audio"
    fig.suptitle(f"PCA of the {what} features, like Fig. 3 of the paper", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, name)
