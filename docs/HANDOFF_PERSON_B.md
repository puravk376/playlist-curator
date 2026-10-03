# Handoff to Person B

Hi! Person A's part is done: data, preprocessing, baselines, evaluation, PCA and plots. This
page has everything you need to build the neural network, the extension and the demo on top of
it, so that all our numbers are comparable.

## 1. Setup (5 minutes)

```bash
cd playlist-curator
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q               # should say: 10 passed
python run_person_a.py            # optional, rebuilds everything in ~1 min
```

Python 3.12 is what we tested with. Add whatever you need (torch, streamlit, ...) to
`requirements.txt`.

## 2. The data is ready, just load it

```python
from src.data import load_processed
from src.features import make_features, make_model

df = load_processed("main")        # 1,256 songs, 13 playlists, with a fixed 'split' column
train, test = df[df.split == "train"], df[df.split == "test"]     # 1,004 / 252
y_train, y_test = train["playlist"], test["playlist"]
```

Columns: `track_id, track_name, track_artist, playlist, danceability, energy, key, loudness,
mode, speechiness, acousticness, liveness, valence, tempo, genres` (genre tags joined with `|`),
`has_genre, split`.

Other datasets with the same format: `"easy"` (5 different-genre playlists), `"hard"`
(5 hip-hop playlists), and `"paper_toy"` (528 tracks from the paper's own playlists).

**Please always use this `split` column.** Don't make a new `train_test_split`, or our test
numbers won't be comparable.

## 3. Features: use the same pipeline

```python
feats = make_features(scale="all")          # fit on TRAIN only
X_train = feats.fit_transform(train)        # numpy array, shape (1004, 368)
X_test = feats.transform(test)
```

- `scale="all"` standardises every column (what the paper did). This is the right choice for
  a neural net.
- `scale="audio"` standardises the audio features only and keeps genres as 0/1. This was
  better for the SVMs.
- The input dimension is 368: 10 audio features plus 358 genre tags seen in training. Read it
  from `X_train.shape[1]`; don't hard-code it.
- Never fit the scaler or the genre vocabulary on the full dataset or on the test set. That's
  leakage.

**Option A: sklearn MLP (easiest, one line):**

```python
from sklearn.neural_network import MLPClassifier
nn = make_model(MLPClassifier(hidden_layer_sizes=(13,), activation="logistic",
                              alpha=1e-3, max_iter=2000, random_state=42), scale="all")
nn.fit(train, y_train)
print(nn.score(test, y_test))
```

The paper's best network had 1 hidden layer with C = 13 neurons, sigmoid activation, a
LogSoftmax output and NLL loss with L2, trained for 150 epochs. In sklearn, `activation="logistic"`
is sigmoid, `alpha` is the L2 strength, and the output is already softmax + cross-entropy.
Tune `hidden_layer_sizes`, `alpha` and `activation` with 5-fold CV on the train split, like the
baselines (see `src/baselines.py` → `cv_splitter()`).

**Option B: PyTorch.** Use `X_train`/`X_test` from above. Encode the labels with
`sklearn.preprocessing.LabelEncoder` fitted on `y_train`.

## 4. Numbers to compare against (all on the same 252 test songs)

| model | test accuracy | macro F1 |
|---|---|---|
| majority class | 0.079 | - |
| perceptron | 0.746 | - |
| RBF SVM (tuned) | 0.782 | - |
| **logistic regression (tuned): best baseline** | **0.806** | **0.811** |
| paper: 1-hidden-layer NN on their toy set | 0.82 | - |

Every number is in `results/` (`table3_test_scores_default.csv`, `table3b_test_scores_tuned.csv`,
`metrics.json`). If the NN doesn't beat 0.806, that's fine and still a valid result. Just say
so honestly in the write-up and explain why: small data, 1,004 training songs for 368 features.

To get per-playlist metrics and a confusion matrix in the same style as ours:

```python
from src.evaluate import per_playlist_report, confusion
from src import plots
pred = nn.predict(test)
labels = sorted(df.playlist.unique())
rep = per_playlist_report(y_test, pred, labels)
cm = confusion(y_test, pred, labels)
plots.confusion_matrix_plot(cm, name="10_nn_confusion_matrix.png", title="Confusion matrix: neural network")
plots.per_playlist_scores(rep, name="11_nn_per_playlist_scores.png", title="Per-playlist scores: neural network")
```

Please number your figures from **10** up (`figures/01-09` are Person A's), and save tables in
`results/` with names that start with `nn_` / `ext_`.

## 5. Extension: ideas that fit our results

These are only ideas; pick one.

- **"Suggest a new playlist" when confidence is low.** If `max(predict_proba) < threshold`,
  output "new playlist". Plot accuracy vs. coverage as the threshold changes.
  `LogisticRegression` / `MLPClassifier` already have `predict_proba`.
- **Segmentation, i.e. classify among fewer playlists at a time** (the paper's future work,
  where they saw 0.79 on a hard user). The `"hard"` dataset (5 hip-hop playlists, RBF SVM 0.706)
  is a natural test case.
- **Random forest / decision trees** (also future work in the paper).
  `make_model(RandomForestClassifier(...), scale="none")` works directly.

One finding from Person A that the extension could build on: songs whose artist has **no genre
tags** are misclassified 42% of the time, against 13% for tagged songs, and Jazz Vibes ↔ Lush Lofi
is the most common confusion. Details are in `results/summary.md` section 5.

## 6. Demo app

- The fitted pipeline takes a DataFrame row, so the demo is simple: pick a song from
  `test`, call `model.predict_proba(row)`, and show the top playlists with their confidence.
- Train the model once and save it with `joblib.dump(model, "models/nn.joblib")`. The
  pipelines can be pickled; that's why `split_genres` is a normal function.
- Add `models/` to `.gitignore` if the file is big, or keep it if it's small, so the live
  demo doesn't need training.

## 7. Where things are documented

- `README.md`: overview, data, Person A's results. Please add your sections for the NN,
  extension and demo and fill in your name/SRN under "Who did what".
- `results/summary.md`: every Person A number, auto-generated.
- `docs/person_a_notes.md`: Person A's write-up paragraphs, slide outline and viva Q&A. Use
  it for the shared PDF and deck.

## 8. Deadlines

- Review and live demo: 5–9 Oct 2026
- Final submission: 10 Oct 2026, 11:59 PM
- Deliverables:
  - private GitHub repo (shared with the faculty/TAs) with a README
  - 2-page PDF write-up
  - slide deck
