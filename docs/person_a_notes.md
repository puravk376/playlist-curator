# Person A notes: write-up text, slides, viva prep

All numbers here come from `results/summary.md` (regenerate with `python run_person_a.py`).
If you change anything in `src/config.py`, re-check these numbers before submitting.

---

## 1. Paragraphs for the write-up (PDF)

*(Merge these with Person B's sections. The write-up has to fit on two pages, so cut freely.)*

**Problem statement.** Given a set of playlists and an unseen song, we want to put the song
into the playlist it fits best. Following Awadelkarim & Coelho (CS229, 2018), this is a
multi-class classification problem: each playlist is a class and each song is described by
Spotify's audio features and its artist's genre tags.

**Dataset.** The paper used the Spotify Web API directly, but Spotify closed its audio-features
endpoint to new apps in Nov 2024. We therefore used two public Kaggle datasets collected from
the same API. "30000 Spotify Songs" gives real playlists with audio features, and the artist
table from "Spotify Dataset 1921-2020" gives genre tags. We chose 13 playlists (the size of the
paper's toy set), including *Jazz Vibes*, which the paper also used, and a few deliberately
similar pairs. After removing duplicates and songs that belong to more than one of the chosen
playlists, we have 1,256 songs, 10 audio features and 388 one-hot genre tags (80% of songs have
at least one tag). The data is split 80/20 with stratification (1,004 / 252). As a sanity check
we also rebuilt the paper's own 13 playlists from the authors' GitHub repo and found audio
features for 528 of their 1,027 tracks.

**Approach (preprocessing and baselines).** Every song is the vector
`[10 audio features | one-hot genres]`. The one-hot vocabulary and the scaler are fitted inside a
scikit-learn pipeline, so cross-validation never sees test-fold information. We compare three
preprocessing options: standardising every column (as in the paper), standardising only the
audio features, and no scaling. Baselines are a perceptron (one-vs-rest), logistic regression,
and SVMs with linear, polynomial, sigmoid and RBF kernels, plus a majority-class floor. We
compare models with 5-fold stratified CV on the training split, tune the RBF SVM and logistic
regression with grid search (the scaling option is part of the grid), and score the selected
model once on the held-out test set.

**Results.**
- **Scaling.** Without scaling, the perceptron (0.15), sigmoid SVM (0.11) and RBF SVM (0.17)
  drop close to the majority-class floor (0.08). With standardisation they reach 0.74-0.76,
  which matches the paper's Table 1.
- **Standardising the genre one-hots hurts the kernel SVMs.** Leaving the genre columns as 0/1
  lifts the polynomial SVM from 0.31 to 0.72 and the RBF SVM from 0.74 to 0.78.
- **Genre tags matter most.** With audio features only, every model stays at 0.29-0.43.
- **Best baseline.** The best model by CV is tuned logistic regression (C = 0.1), with
  **0.806 test accuracy** and 0.811 macro F1. The tuned RBF SVM gets 0.782. For comparison, the
  paper's tuned RBF SVM scored 0.80 ± 0.05 on its own playlists.
- **Per playlist.** F1 ranges from 1.00 (Reggaeton Classics, Southern Hip Hop) down to 0.52
  (Jazz Vibes). The largest confusion is Jazz Vibes → Lush Lofi (30%).
- **Missing tags.** Songs whose artist has no genre tags are misclassified 42% of the time,
  against 13% for tagged songs.
- **Separable vs overlapping.** Five playlists from different genres reach 0.92 test accuracy,
  but five hip-hop playlists only reach 0.71, and PCA shows they overlap almost completely.
  This is the same effect as the paper's "Jacob vs Myles" comparison.

**Conclusions (Person A part).** Simple linear models with good preprocessing are already strong
baselines, at about 0.80. Scaling matters, but the paper's choice of scaling *every* column is
not ideal when half the columns are sparse one-hot genres. Most remaining errors come from songs
with no genre information and from playlists that genuinely sound alike, so better artist
features (e.g. the node2vec/word2vec embeddings the paper proposes) would likely help more than
a fancier classifier.

---

## 2. Slides for your part (about 5 slides)

1. **Data.** Where it comes from, why not the Spotify API, the 13 playlists, and the cleaning
   steps with counts. Figure: `01_class_balance.png`.
2. **Preprocessing.** Why scaling: `02_feature_scales.png` (tempo spans about 175 units,
   valence about 1). Mention fitting inside the pipeline to avoid leakage.
3. **Baselines: scaled vs unscaled.** `04_scaled_vs_unscaled.png`, next to the paper's Table 1
   numbers. Punchline: same pattern as the paper, plus our "scale audio only" finding.
4. **Evaluation.** Test accuracy 0.806; `05_confusion_matrix.png`, `06_per_playlist_scores.png`.
   Point at Jazz Vibes ↔ Lush Lofi and the missing-tags result.
5. **PCA.** `08_pca_easy_vs_hard.png`: separable (0.92) vs overlapping (0.71), like the paper's
   Fig. 3.

---

## 3. Questions the panel might ask (with answers)

**Why didn't you use the Spotify API like the paper?**
The audio-features endpoint was deprecated for new apps in Nov 2024, so it returns 403 errors.
We used Kaggle datasets that were collected from the same API before that, with the same features.

**What is the dataset exactly? How many samples and features?**
There are 1,256 songs in 13 playlists. Each song has 10 audio features plus one-hot genre tags.
There are 388 distinct tags overall, but the vocabulary is learned from the training split only:
358 tags, so 368 dimensions. The split is 1,004 train / 252 test, stratified 80/20 with seed 42.

**Why 13 playlists?**
That's the size of the paper's toy set, which keeps the comparison fair.

**Why did you drop songs that appear in two playlists?**
A single-label classifier can't be right for both labels. Keeping them means identical inputs
with different targets, which is pure noise and can also put the same song in both train and
test. Only 12 songs were affected.

**Why not use the `playlist_genre` column? It's right there.**
It's a property of the playlist, i.e. of the label. Using it would be data leakage, and accuracy
would be meaningless. We use the *artist's* genre tags instead, which is what the paper did.

**What is standardisation and why does it matter?**
It means `z = (x - mean) / std`, computed per feature on the training data. Tempo is around 120
and loudness around -7, while most features are between 0 and 1. Distance-based models (RBF
SVM) and gradient-based models (perceptron) are dominated by the big-number features unless you
scale them.

**Why does logistic regression not care about scaling?**
It's a linear model, and most of the useful signal is in the 0/1 genre columns, which are
already on a sensible scale. Scaling mostly changes how the L2 penalty treats each column, not
what the model can represent.

**Why does scaling the genre columns hurt the SVMs?**
A tag that 2 out of 1,000 songs have has mean about 0.002 and std about 0.045, so after scaling
a "1" becomes about 22. A few rare tags then dominate the RBF/poly kernel distances. Keeping
them as 0/1 avoids that.

**How did you avoid data leakage?**
- The test set is held out and only used once at the end.
- The scaler and the genre vocabulary are fitted inside the pipeline, so within each CV fold
  they only see that fold's training data.
- Hyper-parameters (including which scaling to use) are chosen by CV on the training split.
- The final model is picked on CV score, not test score.

**What hyper-parameters did you tune?**
- RBF SVM: C ∈ {0.3…100} and γ ∈ {scale, 0.0003…0.03}. Best: C = 10, γ = 0.03, scale `audio`.
- Logistic regression: C ∈ {0.01…10}. Best: C = 0.1, scale `all`.
- Both used 5-fold stratified CV.

**What do C and gamma mean?**
- C is the penalty for misclassified training points. A large C gives a tighter fit and risks
  overfitting.
- γ (RBF only) controls how far one training point's influence reaches. A large γ gives very
  local, wiggly boundaries.

**Perceptron vs logistic regression?**
Both are linear. The perceptron uses a hard step function and only updates on mistakes.
Logistic regression uses the sigmoid/softmax and minimises log-loss, so it gives probabilities
and converges more smoothly. The paper notes the perceptron update is a special case of the
logistic-regression gradient step.

**Why is accuracy alone not enough?**
It hides per-class behaviour. Precision tells you how often the model is right when it predicts
a playlist, and recall tells you how many of a playlist's songs were found. Example: Latin Pop
Classics has precision 1.00 but recall 0.65, so it's never wrongly predicted, but it misses
a third of its songs.

**How do you read the confusion matrix?**
Rows are the true playlist and columns are the predicted one. Each row sums to 1, so a cell is
the fraction of that playlist's test songs. The diagonal is the per-playlist recall.

**What is PCA and what does the plot show?**
PCA projects the scaled features onto the two directions of largest variance. Different-genre
playlists form visible clusters (easy set, 0.92 accuracy), while five hip-hop playlists overlap
almost completely (0.71). Two components only explain about 32-41% of the variance, so the
plot is a picture of separability, not proof of it.

**Why is Jazz Vibes so bad?**
It sounds a lot like Lush Lofi, and 87% of its songs come from artists with no genre tags. The
model partly learned "no tags → lofi". Songs with no tags have a 42% error rate, against 13% for
tagged songs.

**How does your result compare to the paper?**
We got 0.806 on our 13 playlists, against the paper's 0.80 ± 0.05 RBF SVM. On the half of the
paper's own playlists we could rebuild, a tuned RBF SVM gets 0.67, which is lower mainly because
there's half the data and some classes only have 15 songs.

**What would you do with more time?**
- Better artist features, e.g. node2vec/word2vec embeddings, which is what the paper proposes.
- Filling in missing genre tags.
- Allowing multi-label playlists.
- Using artist IDs instead of name matching.
