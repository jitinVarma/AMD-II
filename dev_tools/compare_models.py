"""DEV-ONLY: not part of the container. Compares vision-pipeline variants
side by side on the sample clips, with NUM_FRAMES/resolution held constant,
printing Stage-A descriptions and final captions for each variant.

IMPORTANT (verified 2026-07 against this account's live serverless catalog
via GET /inference/v1/models): the qwen2p5-vl-32b/72b-instruct models named
in the original brief return HTTP 404 (not deployed on serverless anymore).
kimi-k2p6 is the only consistently-working vision-capable serverless model
on this account -- kimi-k2p5 returns persistent 500 Internal Server Error
(likely being sunset). So there is currently no second vision model to A/B
against. The genuinely useful comparison available today is reasoning ON vs
OFF for kimi-k2p6, since both are "thinking" models by default -- this
tells you whether the extra chain-of-thought latency/cost buys any real
accuracy improvement for grounded scene description.

If Fireworks adds another vision-capable serverless model later (or your
account gets access to one), set CANDIDATE_VISION_MODELS to a comma
separated list of full model IDs to compare those directly instead.

Usage:
  FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.compare_models [tasks.json]
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile

from agent.config import Config
from agent.download import download_video
from agent.fireworks_client import FireworksClient
from agent.frames import extract_frames_as_data_uris
from agent.styling import get_stage_b_captions
from agent.vision import get_stage_a_description


def _build_variants(base_config: Config) -> list[tuple[str, Config]]:
    override = os.environ.get("CANDIDATE_VISION_MODELS", "").strip()
    if override:
        model_ids = [m.strip() for m in override.split(",") if m.strip()]
        return [(m, dataclasses.replace(base_config, vision_model=m)) for m in model_ids]

    return [
        (f"{base_config.vision_model} (reasoning=none)",
         dataclasses.replace(base_config, vision_model=base_config.vision_model, reasoning_effort="none")),
        (f"{base_config.vision_model} (reasoning=default/on)",
         dataclasses.replace(base_config, vision_model=base_config.vision_model, reasoning_effort="")),
    ]


def main() -> None:
    tasks_path = sys.argv[1] if len(sys.argv) > 1 else "sample_tasks.json"
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    base_config = Config()
    base_config.validate()
    client = FireworksClient(base_config)

    variants = _build_variants(base_config)
    print(f"Comparing {len(variants)} variant(s): {[label for label, _ in variants]}")

    for task in tasks:
        task_id = task["task_id"]
        styles = task["styles"]
        print(f"\n{'=' * 80}\nTASK {task_id}  {task['video_url']}\n{'=' * 80}")

        with tempfile.TemporaryDirectory(prefix=f"cmp_{task_id}_") as tmpdir:
            video_path = os.path.join(tmpdir, "clip.mp4")
            download_video(task["video_url"], video_path, timeout=base_config.download_timeout)
            data_uris = extract_frames_as_data_uris(
                video_path,
                num_frames=base_config.num_frames,
                max_long_side=base_config.max_long_side,
                qscale=base_config.jpeg_qscale,
                scene_timeout=base_config.ffmpeg_scene_timeout,
                frame_timeout=base_config.ffmpeg_frame_timeout,
                ffprobe_timeout=base_config.ffprobe_timeout,
            )
        print(f"[{len(data_uris)} frames extracted, shared across all variants]")

        for label, cfg in variants:
            print(f"\n--- variant = {label} ---")
            description = get_stage_a_description(client, data_uris, cfg, task_id=task_id)
            print("Stage A description:")
            print(json.dumps(description, indent=2, ensure_ascii=False))

            captions = get_stage_b_captions(client, description, styles, cfg, task_id=task_id)
            print("Captions:")
            for style, caption in captions.items():
                print(f"  {style:20s} {caption}")


if __name__ == "__main__":
    main()
