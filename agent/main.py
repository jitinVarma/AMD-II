"""Main orchestrator: reads /input/tasks.json, processes each clip through
the two-stage pipeline with a bounded concurrent worker pool, and writes
/output/results.json -- guaranteeing a valid, complete output within the
whole-batch time budget no matter what fails along the way.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import tempfile
import time

from .config import Config
from .download import DownloadError, download_video
from .fireworks_client import FireworksClient
from .frames import FrameExtractionError, extract_frames_as_data_uris
from .io_utils import TasksLoadError, load_tasks, write_results_atomically
from .styling import ULTIMATE_FALLBACKS, get_stage_b_captions
from .validate import validate_and_fix
from .vision import get_stage_a_description

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("agent.main")


def _ultimate_fallback_captions(styles: list[str]) -> dict:
    return {s: ULTIMATE_FALLBACKS.get(s, "Unable to generate a caption for this clip.") for s in styles}


def process_task(task: dict, client: FireworksClient, config: Config) -> dict:
    task_id = task["task_id"]
    styles = task["styles"]
    start = time.monotonic()

    try:
        with tempfile.TemporaryDirectory(prefix=f"task_{task_id}_") as tmpdir:
            video_path = os.path.join(tmpdir, "clip.mp4")

            logger.info("[%s] downloading video", task_id)
            download_video(
                task["video_url"], video_path,
                timeout=config.download_timeout, retries=config.download_retries,
            )

            logger.info("[%s] extracting frames", task_id)
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
            logger.info("[%s] extracted %d frames", task_id, len(timestamped_frames))

        logger.info("[%s] stage A: describing scene", task_id)
        description = get_stage_a_description(client, timestamped_frames, config, task_id=task_id)

        logger.info("[%s] stage B: generating %d styles", task_id, len(styles))
        captions_raw = get_stage_b_captions(client, description, styles, config, task_id=task_id)

        captions = validate_and_fix(captions_raw, styles, description, task_id=task_id)

        elapsed = time.monotonic() - start
        logger.info("[%s] done in %.1fs", task_id, elapsed)
        return {"task_id": task_id, "captions": captions}

    except (DownloadError, FrameExtractionError, FileNotFoundError, RuntimeError) as exc:
        logger.error("[%s] pipeline failed (%s): %s", task_id, type(exc).__name__, exc)
        return {"task_id": task_id, "captions": _ultimate_fallback_captions(styles)}
    except Exception:
        logger.exception("[%s] unexpected error, emitting fallback captions", task_id)
        return {"task_id": task_id, "captions": _ultimate_fallback_captions(styles)}


def run() -> int:
    overall_start = time.monotonic()
    config = Config()

    try:
        config.validate()
    except ValueError as exc:
        logger.error("configuration error: %s", exc)
        return 1

    try:
        tasks = load_tasks(config.input_path)
    except TasksLoadError as exc:
        logger.error("fatal: %s", exc)
        try:
            write_results_atomically(config.output_path, [])
        except Exception:
            logger.exception("could not even write empty output")
        return 1

    logger.info(
        "loaded %d tasks; vision_model=%s text_model=%s keys=%d workers=%d budget=%.0fs",
        len(tasks), config.vision_model, config.text_model,
        len(config.api_keys), config.max_workers, config.total_budget_seconds,
    )

    client = FireworksClient(config)
    deadline = overall_start + config.total_budget_seconds

    results: list[dict] = []
    futures: dict[concurrent.futures.Future, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        for task in tasks:
            if time.monotonic() > deadline:
                logger.warning("[%s] time budget exhausted before starting; emitting fallback", task["task_id"])
                results.append({"task_id": task["task_id"], "captions": _ultimate_fallback_captions(task["styles"])})
                continue
            fut = executor.submit(process_task, task, client, config)
            futures[fut] = task

        remaining = max(0.0, deadline - time.monotonic())
        done, not_done = concurrent.futures.wait(list(futures.keys()), timeout=remaining)

        for fut in done:
            task = futures[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("[%s] task raised after completion; emitting fallback", task["task_id"])
                results.append({"task_id": task["task_id"], "captions": _ultimate_fallback_captions(task["styles"])})

        for fut in not_done:
            task = futures[fut]
            logger.warning("[%s] did not finish within time budget; emitting fallback", task["task_id"])
            results.append({"task_id": task["task_id"], "captions": _ultimate_fallback_captions(task["styles"])})

    # Preserve input order in the output.
    order = {t["task_id"]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order.get(r["task_id"], len(order)))

    try:
        write_results_atomically(config.output_path, results)
        logger.info(
            "wrote %d results to %s in %.1fs total",
            len(results), config.output_path, time.monotonic() - overall_start,
        )
        return 0
    except Exception:
        logger.exception("primary write failed; attempting last-resort minimal write")
        try:
            minimal = [
                {"task_id": r["task_id"], "captions": r.get("captions") or _ultimate_fallback_captions(list((r.get("captions") or {}).keys()) or ["formal"])}
                for r in results
            ]
            write_results_atomically(config.output_path, minimal)
            return 0
        except Exception:
            logger.exception("last-resort write also failed")
            return 1


def main() -> None:
    exit_code = 1
    try:
        exit_code = run()
    except Exception:
        logger.exception("unhandled top-level error")
        exit_code = 1
    finally:
        sys.stderr.flush()
        sys.stdout.flush()
        # Force-exit rather than joining any lingering background threads
        # (e.g. a stuck network call in a thread we gave up waiting on) --
        # the output file is already durably written by this point.
        os._exit(exit_code)


if __name__ == "__main__":
    main()
