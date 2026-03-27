from __future__ import annotations

from _runtime import bike_features, load_bike_sharing


def load_frame():
    df = load_bike_sharing()
    X, y = bike_features(df)
    return df, X, y
