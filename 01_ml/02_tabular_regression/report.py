from __future__ import annotations

from layout import latest_artifact_dir


def latest_artifact(stage_dir_name: str = "02_tabular_regression"):
    return latest_artifact_dir(stage_dir_name)
