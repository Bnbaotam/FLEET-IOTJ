#!/usr/bin/env python3
"""
Script: preprocess_cuip_dataset.py
Usage:
    python3 preprocess_cuip_dataset.py \
        --data gt.txt \
        [--train-output train_trajnet.pkl] \
        [--test-output test_trajnet.pkl] \
        [--delta-time 0.4] \
        [--orig-fps 10.0] \
        [--min-movement 10.0] \
        [--min-frames None] \
        [--test-size 0.2] \
        [--random-state 42]

This script:
  1. Loads a MOT-format ground-truth file (gt.txt).
  2. Globally sub-samples frames by delta_time & orig_fps.
  3. Re-indexes frame IDs starting at 1.
  4. Adds sceneId and metaId columns.
  5. Identifies 'standing' vs 'moving' tracks.
  6. Linearly interpolates to fill frame gaps per track.
  7. Trims standing tracks to at most 20 frames, keeps moving intact.
  8. (Optional) Drops tracks shorter than min_frames.
  9. Splits by trackId into train/test sets (no overlap).
 10. Saves each split as a pickle file.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CUIP MOT dataset: subsample, interpolate, split"
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to MOT-format gt.txt file"
    )
    parser.add_argument(
        "--train-output", "-to", default="train_trajnet.pkl",
        help="Output path for train pickle"
    )
    parser.add_argument(
        "--test-output", "-xo", default="test_trajnet.pkl",
        help="Output path for test pickle"
    )
    parser.add_argument(
        "--delta-time", type=float, default=0.4,
        help="Seconds between sampled detection frames"
    )
    parser.add_argument(
        "--orig-fps", type=float, default=10.0,
        help="Original recording frames per second"
    )
    parser.add_argument(
        "--min-movement", type=float, default=10.0,
        help="Pixel threshold to distinguish moving vs standing"
    )
    parser.add_argument(
        "--min-frames", type=lambda x: None if x.lower()=="none" else int(x),
        default=None,
        help="Minimum track length to keep (None to disable)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of objects for test split"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for object split"
    )
    return parser.parse_args()


def load_and_subsample(filepath, delta_time, orig_fps):
    # Load only first 6 columns of MOT gt.txt
    cols = ["frame_id","track_id","x","y","w","h","_7","_8","_9"]
    df = pd.read_csv(filepath, header=None, names=cols, usecols=cols[:6])

    # Global sub-sampling of frames
    step = int(round(delta_time * orig_fps))
    all_frames = sorted(df["frame_id"].unique())
    sampled = all_frames[::step]
    df = df[df["frame_id"].isin(sampled)].copy()

    # Remap frames to 1...N
    frame_map = {old: new for new, old in enumerate(sampled, start=1)}
    df["frame_id"] = df["frame_id"].map(frame_map)
    return df


def build_meta(df):
    df["sceneId"] = "cuip"
    unique_tracks = sorted(df["track_id"].unique())
    track_to_meta = {t:i for i,t in enumerate(unique_tracks)}
    df["metaId"] = df["track_id"].map(track_to_meta)
    return df


def rename_and_sort(df):
    return (
        df.rename(columns={"frame_id":"frame","track_id":"trackId"})
          [["frame","trackId","x","y","sceneId","metaId"]]
          .sort_values(["trackId","frame"])  
          .reset_index(drop=True)
    )


def identify_tracks(df, min_movement):
    standing, moving = [], []
    for tid, grp in df.groupby("trackId", sort=False):
        dx = grp["x"].iloc[-1] - grp["x"].iloc[0]
        dy = grp["y"].iloc[-1] - grp["y"].iloc[0]
        if np.hypot(dx, dy) < min_movement:
            standing.append(tid)
        else:
            moving.append(tid)
    return standing, moving


def fill_track_gaps(df):
    out = []
    for tid, grp in df.groupby("trackId", sort=False):
        grp = grp.sort_values("frame").set_index("frame")
        full = np.arange(grp.index.min(), grp.index.max()+1)
        gf = grp.reindex(full).assign(
            trackId=lambda d: d["trackId"].ffill(),
            sceneId=lambda d: d["sceneId"].ffill(),
            metaId=lambda d: d["metaId"].ffill()
        )
        gf[["x","y"]] = gf[["x","y"]].interpolate().ffill().bfill()
        gf = gf.reset_index().rename(columns={"index":"frame"})
        gf["trackId"] = gf["trackId"].astype(int)
        gf["metaId"]  = gf["metaId"].astype(int)
        out.append(gf)
    return pd.concat(out, ignore_index=True)


def trim_and_filter(df_filled, standing_ids, moving_ids, min_frames):
    # Trim standing to max 20 frames
    stand = (
        df_filled[df_filled["trackId"].isin(standing_ids)]
        .groupby("trackId", group_keys=False)
        .head(20)
    )
    move = df_filled[df_filled["trackId"].isin(moving_ids)]
    df = pd.concat([stand, move], ignore_index=True)

    # Drop short tracks
    if min_frames is not None:
        counts = df["trackId"].value_counts()
        valid = counts[counts >= min_frames].index
        df = df[df["trackId"].isin(valid)].reset_index(drop=True)
    return df


def split_objects(df, test_size, random_state):
    track_ids = sorted(df["trackId"].unique())
    train_ids, test_ids = train_test_split(
        track_ids, test_size=test_size,
        random_state=random_state, shuffle=True
    )
    train = df[df["trackId"].isin(train_ids)].reset_index(drop=True)
    test  = df[df["trackId"].isin(test_ids)].reset_index(drop=True)
    return train, test


def main():
    args = parse_args()

    # Steps 1–3
    df = load_and_subsample(args.data, args.delta_time, args.orig_fps)
    df = build_meta(df)
    df = rename_and_sort(df)

    # Step 4
    standing_ids, moving_ids = identify_tracks(df, args.min_movement)

    # Step 5
    df_filled = fill_track_gaps(df)

    # Step 6–7
    df_trimmed = trim_and_filter(df_filled, standing_ids, moving_ids, args.min_frames)

    # Step 8
    train_df, test_df = split_objects(df_trimmed, args.test_size, args.random_state)

    # Already gap-filled in df_filled, so train_df/test_df are consecutive

    # Step 9: Pickle
    with open(args.train_output, "wb") as f:
        pickle.dump(train_df, f)
    with open(args.test_output, "wb") as f:
        pickle.dump(test_df, f)

    print(f"Done. Train saved to {args.train_output}, test saved to {args.test_output}")


if __name__ == "__main__":
    main()