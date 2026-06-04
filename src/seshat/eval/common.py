from __future__ import annotations

import json
import tempfile
from typing import Any

import mlflow


def matches_tags(tags: dict[str, Any], tag_filter: dict[str, str | list[str]]) -> bool:
    for key, wanted in tag_filter.items():
        value = tags.get(key)
        if isinstance(wanted, list):
            if not (isinstance(value, list) and set(wanted) <= set(value)):
                return False
        else:
            if value != wanted:
                return False
    return True


def log_breakdown_artifact(breakdown: dict, run_id: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(breakdown, f, indent=2)
        breakdown_path = f.name
    mlflow.log_artifact(breakdown_path, artifact_path="eval", run_id=run_id)
