"""Task loading and atomic, verified result writing."""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("agent.io")


class TasksLoadError(Exception):
    pass


def load_tasks(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise TasksLoadError(f"could not read/parse {path}: {exc}") from exc

    if not isinstance(data, list):
        raise TasksLoadError(f"{path} must contain a JSON array of tasks")

    tasks: list[dict] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("skipping task at index %d: not an object", i)
            continue
        task_id = item.get("task_id")
        video_url = item.get("video_url")
        styles = item.get("styles")
        if not task_id or not video_url or not isinstance(styles, list) or not styles:
            logger.warning("skipping malformed task at index %d: %r", i, item)
            continue
        tasks.append({
            "task_id": str(task_id),
            "video_url": str(video_url),
            "styles": [str(s) for s in styles],
        })

    if not tasks:
        raise TasksLoadError(f"{path} contained no valid tasks")

    return tasks


def write_results_atomically(path: str, results: list[dict]) -> None:
    """Writes to a temp file in the same directory, fsyncs, renames into
    place, then reloads to confirm the file is valid JSON. Raises if any
    step fails so the caller can fall back to a last-resort write.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)

    with open(path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    if not isinstance(reloaded, list) or len(reloaded) != len(results):
        raise ValueError("post-write validation failed: reloaded results do not match")
