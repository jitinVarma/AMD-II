"""Frame extraction via ffmpeg/ffprobe CLI (subprocess). No OpenCV/PIL.

Sampling strategy: evenly-spaced timestamps across the full duration
(guaranteeing first/middle/last coverage) merged with scene-change
timestamps detected via ffmpeg's `select='gt(scene,0.3)'` filter, so both
steady coverage and motion/temporal change are captured.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
import tempfile

logger = logging.getLogger("agent.frames")

_SHOWINFO_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


class FrameExtractionError(Exception):
    pass


def probe_duration(video_path: str, timeout: int = 15) -> float:
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        duration = float(proc.stdout.strip())
        if duration <= 0:
            raise ValueError(f"non-positive duration: {duration}")
        return duration
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        raise FrameExtractionError(f"ffprobe failed: {exc}") from exc


def _uniform_timestamps(duration: float, count: int) -> list[float]:
    if count <= 1:
        return [duration / 2]
    epsilon = min(0.5, duration * 0.02)
    start = epsilon
    end = max(start, duration - epsilon)
    if end <= start:
        return [duration / 2] * count
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _scene_change_timestamps(video_path: str, max_count: int, timeout: int) -> list[float]:
    if max_count <= 0:
        return []
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", "scale=320:-2,select='gt(scene,0.3)',showinfo",
                "-vsync", "vfr", "-f", "null", "-",
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("scene detection timed out; falling back to uniform sampling only")
        return []
    except OSError as exc:
        logger.warning("scene detection failed to launch: %s", exc)
        return []

    timestamps = [float(m) for m in _SHOWINFO_PTS_RE.findall(proc.stderr)]
    return timestamps[:max_count]


def _merge_timestamps(uniform: list[float], scene: list[float], duration: float, target: int) -> list[float]:
    min_gap = max(0.3, duration / (target * 4))
    merged: list[float] = []

    def _add(t: float) -> None:
        t = max(0.0, min(t, duration))
        for existing in merged:
            if abs(existing - t) < min_gap:
                return
        merged.append(t)

    # Uniform timestamps first so first/middle/last coverage is guaranteed
    # even if we later have to trim.
    for t in uniform:
        _add(t)
    for t in scene:
        _add(t)

    merged.sort()

    if len(merged) > target:
        # Always keep first and last; thin the middle evenly.
        first, last = merged[0], merged[-1]
        middle = merged[1:-1]
        keep_middle = target - 2
        if keep_middle <= 0:
            merged = [first, last]
        else:
            idxs = [round(i * (len(middle) - 1) / (keep_middle - 1)) for i in range(keep_middle)] if keep_middle > 1 else [0]
            seen = sorted(set(idxs))
            merged = [first] + [middle[i] for i in seen] + [last]

    return merged


def _extract_single_frame(video_path: str, timestamp: float, out_path: str, max_long_side: int, qscale: int, timeout: int) -> bool:
    scale_expr = f"scale='min({max_long_side},iw)':'min({max_long_side},ih)':force_original_aspect_ratio=decrease"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
                "-frames:v", "1", "-vf", scale_expr, "-q:v", str(qscale),
                out_path,
            ],
            capture_output=True, timeout=timeout, check=True,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("frame extraction failed at t=%.2f: %s", timestamp, exc)
        return False


def extract_frames_as_data_uris(
    video_path: str,
    num_frames: int,
    max_long_side: int,
    qscale: int,
    scene_timeout: int,
    frame_timeout: int,
    ffprobe_timeout: int,
) -> list[str]:
    """Returns a list of base64 data URIs (JPEG), ordered chronologically.
    Raises FrameExtractionError only if not a single frame could be pulled.
    """
    duration = probe_duration(video_path, timeout=ffprobe_timeout)

    uniform_count = max(3, num_frames - max(2, num_frames // 3))
    uniform = _uniform_timestamps(duration, uniform_count)

    scene_budget = max(0, num_frames - 3)
    scene = _scene_change_timestamps(video_path, max_count=scene_budget, timeout=scene_timeout)

    timestamps = _merge_timestamps(uniform, scene, duration, num_frames)

    data_uris: list[str] = []
    with tempfile.TemporaryDirectory(prefix="frames_") as tmpdir:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
            ok = _extract_single_frame(video_path, ts, out_path, max_long_side, qscale, frame_timeout)
            if not ok:
                continue
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            data_uris.append(f"data:image/jpeg;base64,{b64}")

    if not data_uris:
        raise FrameExtractionError("no frames could be extracted from video")

    return data_uris
