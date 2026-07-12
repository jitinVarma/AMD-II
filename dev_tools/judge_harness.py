"""DEV-ONLY: not part of the container. Replicates the real scoring: an
LLM judge rates each generated caption on accuracy (0-1, grounded in the
actual frames) and style match (0-1, against the style's definition),
printing per-style and average scores. Use this to A/B test style-prompt
changes against a number instead of a vibe.

The judge's vision model (JUDGE_VISION_MODEL) is pinned independently of
whatever VISION_MODEL the pipeline itself is configured with, so before/after
comparisons stay apples-to-apples even across a pipeline model change.

STANDING ANTI-OVERFITTING RULE: validate every change against TWO clip sets,
never one.
  - TUNED  = clips we've iterated against / inspected the output of. Scores
    improving here are NOT evidence of a real gain -- the model may just be
    fitting our specific tuning clips.
  - FRESH  = clips we have never looked at the generated output for. A
    change is only accepted if it holds steady or improves on FRESH. The
    moment you read a fresh clip's captions to judge quality, it is no
    longer fresh -- move it into the tuned set file.
Rotate new, never-inspected clips into the fresh set periodically so it
doesn't quietly become a second tuned set.

Usage:
  # single-set mode (quick check, backward compatible)
  FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.judge_harness [tasks.json]

  # dual-set mode (use this to validate any real change)
  FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.judge_harness \
      --tuned dev_tools/clips_tuned.json --fresh dev_tools/clips_fresh.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import tempfile
import time

from agent.config import Config
from agent.download import download_video
from agent.fireworks_client import FireworksClient
from agent.frames import extract_frames_as_data_uris
from agent.json_utils import extract_json_object
from agent.styling import get_stage_b_captions, style_definition
from agent.vision import get_stage_a_description

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger("dev_tools.judge_harness")

# Pinned independently of config.vision_model so before/after comparisons
# stay apples-to-apples even if the pipeline's own VISION_MODEL changes
# between runs -- the judge itself must not move.
JUDGE_VISION_MODEL = os.environ.get("JUDGE_VISION_MODEL", "accounts/fireworks/models/qwen3p7-plus")

_JUDGE_SYSTEM_PROMPT = """You are a strict, impartial judge scoring a video \
caption against the actual video frames and a target writing style.

Score two dimensions, each 0.0-1.0:
- "accuracy": does the caption only state things actually visible in the frames, \
with no invented facts, and does it correctly reflect what's shown? 1.0 = fully \
grounded and correct, 0.0 = contains fabricated or contradicted claims.
- "style_match": does the caption's tone genuinely match the target style \
definition given below (not just superficially), while staying natural? \
1.0 = a clear, well-executed example of the style, 0.0 = wrong tone entirely.

Respond with ONLY a JSON object: {"accuracy": <float>, "style_match": <float>, "reasoning": "<one short sentence>"}"""


def _judge_caption(client: FireworksClient, config: Config, timestamped_frames: list[tuple[float, str]], style: str, caption: str, task_id: str) -> dict:
    content = [
        {
            "type": "text",
            "text": (
                f"Target style: {style}\n"
                f"Style definition: {style_definition(style)}\n\n"
                f"Candidate caption: \"{caption}\"\n\n"
                "Here are the actual video frames, sampled chronologically and labeled with "
                "timestamps. Score the caption."
            ),
        }
    ]
    for t, uri in timestamped_frames:
        content.append({"type": "text", "text": f"[frame at t={t:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": uri}})

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    try:
        raw = client.chat_completion(messages=messages, model=JUDGE_VISION_MODEL, max_tokens=200, temperature=0.0)
    except Exception as exc:
        logger.error("[%s/%s] judge call failed: %s", task_id, style, exc)
        return {"accuracy": None, "style_match": None, "reasoning": f"judge call failed: {exc}"}

    parsed = extract_json_object(raw)
    if not parsed:
        logger.warning("[%s/%s] judge returned unparseable output: %r", task_id, style, raw[:200] if raw else raw)
        return {"accuracy": None, "style_match": None, "reasoning": "unparseable judge output"}

    return {
        "accuracy": parsed.get("accuracy"),
        "style_match": parsed.get("style_match"),
        "reasoning": parsed.get("reasoning", ""),
    }


def _run_set(tasks: list[dict], client: FireworksClient, config: Config, label: str, verbose: bool) -> dict[str, list[tuple[float, float]]]:
    """Runs the full pipeline + judge over one clip set, returns
    {style: [(accuracy, style_match), ...]}.
    """
    all_scores: dict[str, list[tuple[float, float]]] = {}
    call_delay = float(os.environ.get("JUDGE_CALL_DELAY_SECONDS", "2.0"))

    for task in tasks:
        task_id = task["task_id"]
        styles = task["styles"]
        if verbose:
            print(f"\n{'=' * 80}\n[{label}] TASK {task_id}  {task['video_url']}\n{'=' * 80}")

        with tempfile.TemporaryDirectory(prefix=f"judge_{task_id}_") as tmpdir:
            video_path = os.path.join(tmpdir, "clip.mp4")
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

        description = get_stage_a_description(client, timestamped_frames, config, task_id=task_id)
        if verbose:
            print("Stage A description:")
            print(json.dumps(description, indent=2, ensure_ascii=False))

        captions = get_stage_b_captions(
            client, description, styles, config, task_id=task_id,
            timestamped_frames=timestamped_frames,
        )

        for style, caption in captions.items():
            time.sleep(call_delay)
            score = _judge_caption(client, config, timestamped_frames, style, caption, task_id)
            all_scores.setdefault(style, [])
            if verbose:
                print(f"\n  [{style}] {caption}")
                print(f"    accuracy={score['accuracy']}  style_match={score['style_match']}  reasoning={score['reasoning']}")
            if isinstance(score["accuracy"], (int, float)) and isinstance(score["style_match"], (int, float)):
                all_scores[style].append((score["accuracy"], score["style_match"]))

    return all_scores


def _summarize(all_scores: dict[str, list[tuple[float, float]]], label: str) -> tuple[float | None, float | None]:
    print(f"\n{'=' * 80}\n{label} SUMMARY\n{'=' * 80}")
    overall_acc: list[float] = []
    overall_style: list[float] = []
    for style, pairs in all_scores.items():
        if not pairs:
            print(f"  {style:20s} no valid scores")
            continue
        acc = [p[0] for p in pairs]
        sty = [p[1] for p in pairs]
        overall_acc.extend(acc)
        overall_style.extend(sty)
        print(f"  {style:20s} accuracy={statistics.mean(acc):.2f}  style_match={statistics.mean(sty):.2f}  n={len(pairs)}")

    if not overall_acc:
        print(f"\n  {'OVERALL':20s} no valid scores")
        return None, None

    mean_acc, mean_style = statistics.mean(overall_acc), statistics.mean(overall_style)
    print(f"\n  {'OVERALL':20s} accuracy={mean_acc:.2f}  style_match={mean_style:.2f}")
    return mean_acc, mean_style


def _load_tasks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks_path", nargs="?", default=None, help="single-set mode: path to a tasks.json")
    parser.add_argument("--tuned", default=None, help="dual-set mode: path to the TUNED clip set")
    parser.add_argument("--fresh", default=None, help="dual-set mode: path to the FRESH (never-inspected) clip set")
    parser.add_argument("--quiet", action="store_true", help="suppress per-clip/per-caption output, print summaries only")
    args = parser.parse_args()

    config = Config()
    config.validate()
    client = FireworksClient(config)

    print(f"Pipeline vision_model={config.vision_model}  text_model={config.text_model}")
    print(f"Judge (pinned) vision_model={JUDGE_VISION_MODEL}")

    if args.tuned or args.fresh:
        if not (args.tuned and args.fresh):
            parser.error("--tuned and --fresh must be given together")

        tuned_tasks = _load_tasks(args.tuned)
        fresh_tasks = _load_tasks(args.fresh)
        print(f"\nTUNED set: {args.tuned} ({len(tuned_tasks)} clips, previously inspected -- gains here are NOT evidence)")
        print(f"FRESH set: {args.fresh} ({len(fresh_tasks)} clips, never inspected -- this is the real signal)")

        tuned_scores = _run_set(tuned_tasks, client, config, "TUNED", verbose=not args.quiet)
        tuned_acc, tuned_style = _summarize(tuned_scores, "TUNED")

        fresh_scores = _run_set(fresh_tasks, client, config, "FRESH", verbose=not args.quiet)
        fresh_acc, fresh_style = _summarize(fresh_scores, "FRESH")

        print(f"\n{'=' * 80}\nTUNED vs FRESH\n{'=' * 80}")
        if tuned_acc is not None and fresh_acc is not None:
            print(f"  {'':20s} {'accuracy':>10s} {'style_match':>13s}")
            print(f"  {'TUNED':20s} {tuned_acc:10.2f} {tuned_style:13.2f}")
            print(f"  {'FRESH':20s} {fresh_acc:10.2f} {fresh_style:13.2f}")
            print(f"  {'delta (fresh-tuned)':20s} {fresh_acc - tuned_acc:+10.2f} {fresh_style - tuned_style:+13.2f}")
            if fresh_acc < tuned_acc - 0.05 or fresh_style < tuned_style - 0.05:
                print(
                    "\n  ⚠️  FRESH lags TUNED by >0.05 on at least one metric -- classic overfit "
                    "signature. Do not accept this change on tuned-set improvement alone."
                )
            else:
                print("\n  FRESH holds steady with TUNED -- no overfit signal detected.")
        else:
            print("  Could not compute a delta (one or both sets had no valid scores).")
        return

    tasks_path = args.tasks_path or "sample_tasks.json"
    tasks = _load_tasks(tasks_path)
    scores = _run_set(tasks, client, config, "SET", verbose=not args.quiet)
    _summarize(scores, "SUMMARY")


if __name__ == "__main__":
    main()
