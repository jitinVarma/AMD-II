"""Frame extraction via ffmpeg/ffprobe CLI (subprocess) for decoding/scaling,
plus opencv-python-headless for scene-change detection. No other CV/video
dependency.

Sampling strategy: a duration-adaptive number of frames, placed by combining
(1) a uniform "floor" that guarantees first/middle/last coverage and (2)
scene-change-driven placements from a cheap low-res histogram-diff scan, so
both steady coverage and genuine visual/motion change are captured -- while
degrading gracefully to pure uniform sampling for the common case of a
single continuous shot with no cuts.
"""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import tempfile
import time

import cv2
import numpy as np

logger = logging.getLogger("agent.frames")


class FrameExtractionError(Exception):
    """No frames could be extracted at all, after every attempt."""


def probe_duration(video_path: str, timeout: int = 15) -> float:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out probing {video_path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"ffprobe failed to launch for {video_path}: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(
            f"ffprobe could not read {video_path} (unreadable/corrupt video). "
            f"ffprobe stderr: {proc.stderr.strip()}"
        )

    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned a non-numeric duration for {video_path}: {proc.stdout!r}"
        ) from exc

    if duration <= 0:
        raise RuntimeError(f"ffprobe reported a non-positive duration ({duration}) for {video_path}")

    return duration


def adaptive_frame_count(duration: float, override: int | None) -> int:
    """~1 frame per 9s, floor 6, cap 14. `override` (from NUM_FRAMES when the
    env var is explicitly set) disables adaptation entirely.
    """
    if override is not None:
        return max(1, override)
    count = round(duration / 9.0)
    return max(6, min(14, count))


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


_PREVIEW_W = 160
_PREVIEW_H = 90


def _detect_scene_changes(
    video_path: str,
    duration: float,
    sample_fps: float = 2.0,
    threshold: float = 0.15,
    max_scan_seconds: float = 20.0,
) -> list[tuple[float, float]]:
    """Cheap low-res grayscale histogram-diff scan for scene-change
    timestamps. Returns (timestamp, magnitude) pairs. Never raises -- any
    failure (ffmpeg missing, unreadable video, pipe error) just yields no
    change points, which degrades to pure uniform sampling exactly like a
    genuinely static/single-shot clip would.

    threshold=0.15 empirically tuned against the 3 sample clips (all single
    continuous shots, no cuts) and a synthetic hard-cut test video: the 3
    real clips top out at Bhattacharyya distance ~0.03-0.10 between
    consecutive 2fps samples (yielding 0 change points, as required), while
    an actual scene cut in the synthetic test produced ~0.21 (correctly
    detected). 0.15 sits cleanly between the two.

    Speed note: this used to open the FULL-RESOLUTION video via
    cv2.VideoCapture and decode every native-fps frame in a Python loop
    just to discard all but ~1-in-N of them -- expensive for UHD source
    clips (profiled as the single biggest contributor to per-clip extraction
    time). This version instead has ffmpeg itself downscale+drop frames to
    a tiny 160x90 @ 2fps grayscale raw stream (fast, single optimized C
    pipeline, never materializes a full-res frame in Python at all) and
    reads that directly off a pipe -- no temp file, no OpenCV VideoCapture.
    """
    frame_size = _PREVIEW_W * _PREVIEW_H
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"scale={_PREVIEW_W}:{_PREVIEW_H}:flags=fast_bilinear,fps={sample_fps}",
        "-pix_fmt", "gray", "-f", "rawvideo",
        "-an", "-sn", "-loglevel", "error",
        "-",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        logger.warning("ffmpeg preview pipe failed to launch for %s: %s; falling back to uniform sampling", video_path, exc)
        return []

    changes: list[tuple[float, float]] = []
    prev_hist = None
    frame_idx = 0
    start_time = time.monotonic()

    try:
        while True:
            if time.monotonic() - start_time > max_scan_seconds:
                logger.warning("scene detection scan exceeded %.0fs budget on %s; using changes found so far",
                                max_scan_seconds, video_path)
                break

            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            gray = np.frombuffer(raw, dtype=np.uint8).reshape(_PREVIEW_H, _PREVIEW_W)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            t = frame_idx / sample_fps
            if prev_hist is not None:
                diff = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if diff > threshold:
                    changes.append((t, diff))
            prev_hist = hist
            frame_idx += 1
    except Exception as exc:
        logger.warning("scene detection pipe read failed for %s: %s; using changes found so far", video_path, exc)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    return changes


def _select_timestamps(
    duration: float,
    target_count: int,
    scene_changes: list[tuple[float, float]],
    min_gap: float = 2.0,
) -> list[float]:
    """Combines a uniform floor (first/middle/last guaranteed) with
    scene-change-driven placements. If there are more scene changes than
    budget, keeps the largest-magnitude ones. Frames are placed shortly
    AFTER each detected change point. Static/single-shot clips (no scene
    changes found) degrade to pure uniform sampling.
    """
    epsilon = min(0.5, duration * 0.02)
    anchors = sorted(set([epsilon, duration / 2, max(epsilon, duration - epsilon)]))

    def _too_close(t: float, existing: list[float]) -> bool:
        return any(abs(t - e) < min_gap for e in existing)

    selected: list[float] = list(anchors)

    remaining = max(0, target_count - len(selected))
    ranked_changes = sorted(scene_changes, key=lambda c: -c[1])[:remaining]
    for t, _mag in sorted(ranked_changes, key=lambda c: c[0]):
        placed = min(duration - epsilon, t + 0.3)
        if not _too_close(placed, selected):
            selected.append(placed)

    # Fill any leftover budget uniformly across the clip.
    leftover = target_count - len(selected)
    if leftover > 0:
        candidates = _uniform_timestamps(duration, target_count * 2)
        for t in candidates:
            if len(selected) >= target_count:
                break
            if not _too_close(t, selected):
                selected.append(t)

    selected.sort()

    # Final min-gap dedup pass (placements above can still collide at the boundary).
    final: list[float] = []
    for t in selected:
        if not final or t - final[-1] >= min_gap:
            final.append(t)

    # HARD FLOOR -- the LAST clamp, after scene-aware placement AND the
    # min_gap dedup above. target_count is already >= 6 in the adaptive case
    # (adaptive_frame_count's own floor), but min_gap dedup can still
    # undershoot it on short clips (e.g. a 6s clip can't fit 6 points a full
    # 2s apart). When that happens, this is non-negotiable: relax spacing --
    # not target count -- by repeatedly inserting a point into the current
    # largest gap (deterministic, always makes progress) until target_count
    # is reached or the clip's duration is too short to support that many
    # distinct instants at all (graceful degradation, not a crash: returns
    # however many distinct timestamps actually exist).
    if len(final) < target_count:
        logger.info(
            "only %d frames after normal placement (duration=%.1fs, target=%d) -- "
            "filling to the hard floor via largest-gap insertion",
            len(final), duration, target_count,
        )
        final = _fill_to_count_via_largest_gap(final, target_count, duration)

    return final[:target_count]


def _fill_to_count_via_largest_gap(existing: list[float], target_count: int, duration: float) -> list[float]:
    """Repeatedly inserts a point into the middle of the current largest gap
    (including the boundaries 0 and duration) until `target_count` distinct
    points are reached. Deterministic and always makes progress -- no
    probabilistic grid-search near-misses. Stops early (returning fewer than
    target_count) only if the clip's duration is genuinely too short to fit
    any more distinct instants at floating-point resolution.
    """
    points = sorted(existing) if existing else []
    if not points:
        return _uniform_timestamps(duration, target_count)

    while len(points) < target_count:
        bounds = [0.0] + points + [duration]
        best_gap = -1.0
        best_mid = None
        for i in range(len(bounds) - 1):
            left, right = bounds[i], bounds[i + 1]
            gap = right - left
            if gap > best_gap:
                best_gap = gap
                best_mid = (left + right) / 2.0
        if best_mid is None or best_gap < 1e-6:
            break
        points.append(best_mid)
        points.sort()

    return points


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
    num_frames_override: int | None,
    max_long_side: int,
    qscale: int,
    scene_timeout: int,
    frame_timeout: int,
    ffprobe_timeout: int,
    scene_change_threshold: float = 0.15,
) -> list[tuple[float, str]]:
    """Returns a chronologically-ordered list of (timestamp_seconds, data_uri)
    pairs. Raises FileNotFoundError if video_path doesn't exist, RuntimeError
    if ffprobe can't read it, or FrameExtractionError if not a single frame
    could be pulled despite the video being readable.
    """
    duration = probe_duration(video_path, timeout=ffprobe_timeout)
    target_count = adaptive_frame_count(duration, num_frames_override)

    scene_changes = _detect_scene_changes(
        video_path, duration, threshold=scene_change_threshold, max_scan_seconds=scene_timeout
    )
    timestamps = _select_timestamps(duration, target_count, scene_changes)

    results: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory(prefix="frames_") as tmpdir:
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
            ok = _extract_single_frame(video_path, ts, out_path, max_long_side, qscale, frame_timeout)
            if not ok:
                continue
            with open(out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            results.append((ts, f"data:image/jpeg;base64,{b64}"))

    if not results:
        raise FrameExtractionError(f"no frames could be extracted from {video_path}")

    return results
