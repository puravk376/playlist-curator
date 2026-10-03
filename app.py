"""Live demo: Playlist Curator.

    streamlit run app.py      (run python run_nn.py and python run_extension.py first)

Loads the saved neural network (models/nn.joblib); nothing is trained here.
Tab 1: pick a test song (the model never saw these) and see where it goes.
Tab 2: build your own song with sliders + genre tags and see what the model says.
"""
import json
import random

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.data import load_processed

st.set_page_config(page_title="Playlist Curator", page_icon="🎧", layout="centered")
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@st.cache_resource
def load_model():
    return joblib.load(config.ROOT / "models" / "nn.joblib")


@st.cache_data
def load_data():
    return load_processed("main")


def load_metrics(name, default):
    path = config.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else default


model, df = load_model(), load_data()
train, test = df[df.split == "train"], df[df.split == "test"]
nn_metrics = load_metrics("nn_metrics.json", {})
ext_metrics = load_metrics("ext_metrics.json", {"threshold": 0.65})

# ---------- sidebar ----------
st.sidebar.header("Model")
if nn_metrics:
    st.sidebar.write(f"Neural network, test accuracy **{nn_metrics['test_acc_mean']:.3f} "
                     f"± {nn_metrics['test_acc_std']:.3f}** (5 seeds)")
    st.sidebar.caption(f"Settings chosen by CV: {nn_metrics['chosen_params']}")
st.sidebar.write(f"{len(train)} training songs, {df.playlist.nunique()} playlists")
threshold = st.sidebar.slider("'New playlist' threshold", 0.0, 0.95, float(ext_metrics["threshold"]), 0.05,
                              help="If the model's top confidence is below this, it suggests a new playlist. "
                                   "Default was chosen with cross-validation on the training songs.")
st.sidebar.caption("0 = always pick a playlist. Higher = fewer songs placed, but more of them right.")


def show_prediction(row):
    """row: a one-row DataFrame with the same columns as the dataset."""
    proba = pd.Series(model.predict_proba(row)[0], index=model.classes_).sort_values(ascending=False)
    best, conf = proba.index[0], proba.iloc[0]
    if conf >= threshold:
        st.success(f"**→ {best}**   (confidence {conf:.0%})")
    else:
        st.warning(f"**🆕 Suggest a new playlist.** Best guess would be *{best}*, "
                   f"but confidence {conf:.0%} is below the threshold {threshold:.0%}.")
    st.write("Top 5 playlists:")
    for name, p in proba.head(5).items():
        st.progress(float(p), text=f"{name}: {p:.0%}")
    return best, conf


tab1, tab2 = st.tabs(["🎵 Test songs", "🎛️ Make your own song"])

# ---------- tab 1: songs from the test split ----------
with tab1:
    st.write("These 252 songs were held out: the model never saw them during training.")
    options = test.sort_values("track_name").index.tolist()
    label = lambda i: f"{test.at[i, 'track_name']} by {test.at[i, 'track_artist']}"
    if "song" not in st.session_state:
        st.session_state.song = options[0]
    if st.button("🎲 Random song"):
        st.session_state.song = random.choice(options)
    idx = st.selectbox("Pick a song", options, format_func=label, key="song")
    song = test.loc[[idx]]

    tags = song.genres.iloc[0]
    st.caption("Genre tags: " + (", ".join(tags.split("|")) if tags else "none (this artist has no tags)"))
    feats = song[config.AUDIO_FEATURES].iloc[0]
    c = st.columns(5)
    for i, f in enumerate(["danceability", "energy", "valence", "acousticness", "tempo"]):
        c[i].metric(f, f"{feats[f]:.0f}" if f == "tempo" else f"{feats[f]:.2f}")

    best, conf = show_prediction(song)
    true = song.playlist.iloc[0]
    with st.expander("Reveal the real playlist"):
        if best == true:
            st.write(f"✅ Real playlist: **{true}**. The model was right.")
        else:
            st.write(f"❌ Real playlist: **{true}**. The model said {best}.")

# ---------- tab 2: user-built song ----------
with tab2:
    st.write("Set the audio features and genre tags of an imaginary song.")
    med = train[config.AUDIO_FEATURES].median()
    vocab = sorted(model.named_steps["features"].named_steps["columns"]
                   .named_transformers_["genre"].vocabulary_)
    a, b = st.columns(2)
    vals = {
        "danceability": a.slider("danceability", 0.0, 1.0, float(med.danceability), 0.01),
        "energy": a.slider("energy", 0.0, 1.0, float(med.energy), 0.01),
        "valence": a.slider("valence (happiness)", 0.0, 1.0, float(med.valence), 0.01),
        "acousticness": a.slider("acousticness", 0.0, 1.0, float(med.acousticness), 0.01),
        "speechiness": a.slider("speechiness", 0.0, 1.0, float(med.speechiness), 0.01),
        "liveness": b.slider("liveness", 0.0, 1.0, float(med.liveness), 0.01),
        "loudness": b.slider("loudness (dB)", -25.0, 0.0, float(med.loudness), 0.5),
        "tempo": b.slider("tempo (BPM)", 40.0, 215.0, float(med.tempo), 1.0),
        "key": b.selectbox("key", range(12), index=int(med.key), format_func=lambda k: KEYS[k]),
        "mode": b.radio("mode", [1, 0], format_func=lambda m: "major" if m else "minor", horizontal=True),
    }
    genres = st.multiselect("Genre tags (leave empty for an untagged artist)", vocab)
    row = pd.DataFrame([{**vals, "genres": "|".join(genres), "has_genre": bool(genres)}])
    show_prediction(row)
