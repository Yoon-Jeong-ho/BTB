from __future__ import annotations

from _runtime import load_covertype


def load_frame():
    df = load_covertype()
    X = df.drop(columns=["cover_type"])
    y = df["cover_type"].astype(int).to_numpy()
    return df, X, y
