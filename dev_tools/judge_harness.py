"""DEV-ONLY: not part of the container. Replicates the real scoring: an
LLM judge rates each generated caption on accuracy (0-1, grounded in the
actual frames) and style match (0-1, against the style's definition),
printing per-style and average scores. Use this to A/B test style-prompt
changes against a number instead of a vibe.

Usage:
  FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.judge_harness [tasks.json]
"""
from __future__ import annotations

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


def _judge_caption(client: FireworksClient, config: Config, data_uris: list[str], style: str, caption: str, task_id: str) -> dict:
    content = [
        {
            "type": "text",
            "text": (
                f"Target style: {style}\n"
                f"Style definition: {style_definition(style)}\n\n"
                f"Candidate caption: \"{caption}\"\n\n"
                "Here are the actual video frames, sampled chronologically. Score the caption."
            ),
        }
    ]
    for uri in data_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    try:
        raw = client.chat_completion(messages=messages, model=config.vision_model, max_tokens=200, temperature=0.0)
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


def main() -> None:
    tasks_path = sys.argv[1] if len(sys.argv) > 1 else "sample_tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    config = Config()
    config.validate()
    client = FireworksClient(config)

    all_scores: dict[str, list[tuple[float, float]]] = {}

    for task in tasks:
        task_id = task["task_id"]
        styles = task["styles"]
        print(f"\n{'=' * 80}\nTASK {task_id}  {task['video_url']}\n{'=' * 80}")

        with tempfile.TemporaryDirectory(prefix=f"judge_{task_id}_") as tmpdir:
            video_path = os.path.join(tmpdir, "clip.mp4")
            download_video(task["video_url"], video_path, timeout=config.download_timeout)
            data_uris = extract_frames_as_data_uris(
                video_path,
                num_frames=config.num_frames,
                max_long_side=config.max_long_side,
                qscale=config.jpeg_qscale,
                scene_timeout=config.ffmpeg_scene_timeout,
                frame_timeout=config.ffmpeg_frame_timeout,
                ffprobe_timeout=config.ffprobe_timeout,
            )

        description = get_stage_a_description(client, data_uris, config, task_id=task_id)
        print("Stage A description:")
        print(json.dumps(description, indent=2, ensure_ascii=False))

        captions = get_stage_b_captions(client, description, styles, config, task_id=task_id)

        # A small pacing delay between judge calls keeps a single low-tier API
        # key (10 RPM without a payment method on file) from tripping 429s
        # mid-run when iterating locally with few keys. Not needed in the
        # production container, which paces itself naturally.
        call_delay = float(os.environ.get("JUDGE_CALL_DELAY_SECONDS", "2.0"))
        for style, caption in captions.items():
            time.sleep(call_delay)
            score = _judge_caption(client, config, data_uris, style, caption, task_id)
            all_scores.setdefault(style, [])
            print(f"\n  [{style}] {caption}")
            print(f"    accuracy={score['accuracy']}  style_match={score['style_match']}  reasoning={score['reasoning']}")
            if isinstance(score["accuracy"], (int, float)) and isinstance(score["style_match"], (int, float)):
                all_scores[style].append((score["accuracy"], score["style_match"]))

    print(f"\n{'=' * 80}\nSUMMARY\n{'=' * 80}")
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

    if overall_acc:
        print(f"\n  {'OVERALL':20s} accuracy={statistics.mean(overall_acc):.2f}  style_match={statistics.mean(overall_style):.2f}")


if __name__ == "__main__":
    main()
