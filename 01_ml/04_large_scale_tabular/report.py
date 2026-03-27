from __future__ import annotations

from layout import latest_artifact_dir


def latest_artifact(stage_dir_name: str = "04_large_scale_tabular"):
    return latest_artifact_dir(stage_dir_name)
