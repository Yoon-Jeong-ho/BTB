from __future__ import annotations

from layout import latest_artifact_dir


def latest_artifact(stage_dir_name: str = "01_tabular_classification"):
    return latest_artifact_dir(stage_dir_name)
