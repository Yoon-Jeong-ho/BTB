from __future__ import annotations

from _runtime import load_california


def load_frame():
    df = load_california()
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"].to_numpy()
    return df, X, y
