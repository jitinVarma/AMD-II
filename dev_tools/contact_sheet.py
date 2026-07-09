"""DEV-ONLY: not part of the container. Saves a labeled contact-sheet grid
image per clip (the actual frames the sampler selected, each labeled with
its timestamp) into dev_tools/output/, so sampling representativeness can be
eyeballed directly rather than trusted blind.

Usage:
  python3 -m dev_tools.contact_sheet [tasks.json]
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

from agent.config import Config
from agent.download import download_video
from agent.frames import extract_frames_as_data_uris

THUMB_WIDTH = 320
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _data_uri_to_image(data_uri: str) -> np.ndarray:
    import base64
    _, b64 = data_uri.split(",", 1)
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _make_contact_sheet(timestamped_frames: list[tuple[float, str]]) -> np.ndarray:
    thumbs = []
    for t, uri in timestamped_frames:
        img = _data_uri_to_image(uri)
        h, w = img.shape[:2]
        scale = THUMB_WIDTH / w
        thumb = cv2.resize(img, (THUMB_WIDTH, int(h * scale)))
        label = f"t={t:.1f}s"
        cv2.rectangle(thumb, (0, 0), (110, 26), (0, 0, 0), -1)
        cv2.putText(thumb, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)

    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    thumb_h = max(t.shape[0] for t in thumbs)
    thumb_w = THUMB_WIDTH

    grid = np.zeros((thumb_h * rows, thumb_w * cols, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        h, w = thumb.shape[:2]
        grid[r * thumb_h: r * thumb_h + h, c * thumb_w: c * thumb_w + w] = thumb

    return grid


def main() -> None:
    tasks_path = sys.argv[1] if len(sys.argv) > 1 else "sample_tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    config = Config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for task in tasks:
        task_id = task["task_id"]
        video_path = f"/tmp/clips/{task_id}.mp4"
        if not os.path.exists(video_path):
            os.makedirs("/tmp/clips", exist_ok=True)
            download_video(task["video_url"], video_path, timeout=config.download_timeout)

        timestamped_frames = extract_frames_as_data_uris(
            video_path,
            num_frames_override=config.num_frames_override,
            max_long_side=config.max_long_side,
            qscale=config.jpeg_qscale,
            scene_timeout=config.ffmpeg_scene_timeout,
            frame_timeout=config.ffmpeg_frame_timeout,
            ffprobe_timeout=config.ffprobe_timeout,
            scene_change_threshold=config.scene_change_threshold,
        )

        sheet = _make_contact_sheet(timestamped_frames)
        out_path = os.path.join(OUTPUT_DIR, f"{task_id}_contact_sheet.jpg")
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{task_id}: {len(timestamped_frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
