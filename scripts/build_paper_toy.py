"""Rebuild the paper's own "toy set" (13 Spotify-curated playlists) as well as we can.

The authors committed the playlist track lists to their repo
(github.com/kevin-coelho/playlistr-ml-v1, get_toy_set/results/), but not the audio
features. Spotify no longer serves audio features to new apps, so we look the
tracks up in two public dumps of Spotify audio features:

  1. "Spotify Audio Features 2.3M" (huggingface.co/datasets/P-Arpan/Spotify_Audio_features_2.3M)
  2. "Spotify Tracks Dataset" 114k  (huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)

matching first on Spotify track ID, then on (song title, first artist).
Only about half the songs can be found, so this set is smaller and less balanced
than the paper's - we treat it as a sanity check, not our main experiment.

Run from the repo root:  python scripts/build_paper_toy.py
The two big files go to data/external/ (git-ignored); the result
data/raw/paper_toy_tracks.csv is small and is committed, so nobody else has to run this.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.data import genres_for, load_artist_genres  # noqa: E402

EXTERNAL = config.ROOT / "data" / "external"
SOURCES = {
    "audio_features_2.3M.parquet":
        "https://huggingface.co/datasets/P-Arpan/Spotify_Audio_features_2.3M/resolve/main/Spotify_Audio_features.parquet",
    "spotify_tracks_114k.csv":
        "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv",
}


def fetch(name, url):
    path = EXTERNAL / name
    if not path.exists():
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return path


def norm_title(s):
    # "Song (feat. X) - Remastered 2011" -> "song"
    return re.sub(r"\s*[\(\[\-].*$", "", str(s).lower()).strip()


def paper_tracks():
    playlists = json.loads(config.PAPER_TOY_JSON.read_text(encoding="utf-8"))
    rows = []
    for p in playlists:
        for item in p["tracks"]:
            t = item.get("track") or {}
            if t.get("id"):
                rows.append({
                    "track_id": t["id"], "track_name": t["name"], "playlist": p["name"],
                    "artists": [a["name"] for a in t["artists"]],
                })
    return pd.DataFrame(rows)


def feature_tables():
    cols = ["track_id", "track_name", "first_artist", *config.AUDIO_FEATURES]
    big = pd.read_parquet(fetch(*list(SOURCES.items())[0]))
    big["first_artist"] = big["artists"].str.extract(r"""^\[['"](.+?)['"][,\]]""")[0]
    small = pd.read_csv(fetch(*list(SOURCES.items())[1]))
    small["first_artist"] = small["artists"].str.split(";").str[0]
    feats = pd.concat([big[cols], small[cols]], ignore_index=True)
    feats = feats.dropna(subset=config.AUDIO_FEATURES)
    feats["key_name"] = feats["track_name"].map(norm_title) + "|" + feats["first_artist"].str.lower().str.strip()
    return feats


def main():
    toy = paper_tracks()
    print(f"paper toy set: {toy['playlist'].nunique()} playlists, {len(toy)} tracks (paper reports 1044)")
    feats = feature_tables()

    by_id = feats.drop_duplicates("track_id").set_index("track_id")[config.AUDIO_FEATURES]
    by_name = feats.drop_duplicates("key_name").set_index("key_name")[config.AUDIO_FEATURES]

    toy["key_name"] = toy["track_name"].map(norm_title) + "|" + toy["artists"].str[0].str.lower().str.strip()
    hit_id = toy["track_id"].isin(by_id.index)
    hit_name = ~hit_id & toy["key_name"].isin(by_name.index)
    matched = pd.concat([
        toy[hit_id].join(by_id, on="track_id").assign(match="id"),
        toy[hit_name].join(by_name, on="key_name").assign(match="title+artist"),
    ])

    # same rule as the main dataset: drop songs that are in more than one playlist
    multi = matched.groupby("track_id")["playlist"].transform("nunique") > 1
    matched = matched[~multi]

    lookup = load_artist_genres()
    matched["genres"] = matched["artists"].map(lambda a: genres_for(a, lookup))
    matched["track_artist"] = matched["artists"].str.join(", ")
    out = matched[["track_id", "track_name", "track_artist", "playlist", *config.AUDIO_FEATURES, "genres", "match"]]
    out = out.sort_values(["playlist", "track_name"]).reset_index(drop=True)
    out.to_csv(config.PAPER_TOY_CSV, index=False)

    cov = pd.DataFrame({"in_paper": toy.groupby("playlist").size(), "found": out.groupby("playlist").size()})
    cov["found"] = cov["found"].fillna(0).astype(int)
    cov["coverage"] = (cov["found"] / cov["in_paper"]).round(2)
    print(cov.to_string())
    print(f"matched {len(out)}/{len(toy)} tracks "
          f"({(out['match'] == 'id').sum()} by ID, {(out['match'] != 'id').sum()} by title+artist); "
          f"{(out['genres'] != '').mean():.0%} have genre tags")
    print(f"saved {config.PAPER_TOY_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
