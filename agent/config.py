"""Central configuration, all values overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _parse_keys(raw: str) -> list[str]:
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys


def _env_str(name: str, default: str) -> str:
    """Like os.environ.get but treats an empty string as unset, so a
    Dockerfile ENV set to "" (unfilled build ARG) doesn't shadow the
    in-code default.
    """
    value = os.environ.get(name)
    return value if value else default


@dataclass
class Config:
    input_path: str = field(default_factory=lambda: _env_str("INPUT_PATH", "/input/tasks.json"))
    output_path: str = field(default_factory=lambda: _env_str("OUTPUT_PATH", "/output/results.json"))

    fireworks_base_url: str = field(
        default_factory=lambda: _env_str("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    api_keys: list[str] = field(default_factory=lambda: _parse_keys(os.environ.get("FIREWORKS_API_KEY", "")))

    # NOTE: as of this writing (2026-07), Fireworks' serverless catalog has
    # moved on from the Qwen2.5-VL / Llama-3.3 generation named in the
    # original brief -- those return HTTP 404 "not found/deployed" on
    # serverless now. Verified live against /inference/v1/models for this
    # account: kimi-k2p6 is the only consistently-working vision-capable
    # serverless model (kimi-k2p5 returns persistent 500s -- likely being
    # sunset). glm-5p2 is a strong dedicated text model, verified working.
    vision_model: str = field(
        default_factory=lambda: _env_str(
            "VISION_MODEL", "accounts/fireworks/models/kimi-k2p6"
        )
    )
    text_model: str = field(
        default_factory=lambda: _env_str(
            "TEXT_MODEL", "accounts/fireworks/models/glm-5p2"
        )
    )
    # Both default models are "thinking" models that emit a separate
    # reasoning_content field before content when reasoning is enabled.
    # We don't need chain-of-thought for grounded description/captioning,
    # so default to disabling it: much lower latency/token cost and no risk
    # of truncation before the model reaches the actual JSON answer. Set to
    # "" to omit the param entirely (needed for models that reject it).
    reasoning_effort: str = field(default_factory=lambda: _env_str("REASONING_EFFORT", "none"))

    num_frames: int = field(default_factory=lambda: _env_int("NUM_FRAMES", 10))
    max_workers: int = field(default_factory=lambda: _env_int("MAX_WORKERS", 4))

    download_timeout: int = field(default_factory=lambda: _env_int("DOWNLOAD_TIMEOUT", 60))
    download_retries: int = field(default_factory=lambda: _env_int("DOWNLOAD_RETRIES", 3))

    request_timeout: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT", 60))
    # Higher than a naive default so single-key deployments survive a 429
    # via backoff+retry rather than giving up after one hit (see
    # fireworks_client.py) -- cheap to afford given the generous whole-batch
    # time budget.
    request_retries_per_key: int = field(default_factory=lambda: _env_int("REQUEST_RETRIES_PER_KEY", 4))

    max_long_side: int = field(default_factory=lambda: _env_int("MAX_LONG_SIDE", 768))
    jpeg_qscale: int = field(default_factory=lambda: _env_int("JPEG_QSCALE", 4))  # ffmpeg mjpeg qscale, 2=best

    ffmpeg_scene_timeout: int = field(default_factory=lambda: _env_int("FFMPEG_SCENE_TIMEOUT", 30))
    ffprobe_timeout: int = field(default_factory=lambda: _env_int("FFPROBE_TIMEOUT", 15))
    ffmpeg_frame_timeout: int = field(default_factory=lambda: _env_int("FFMPEG_FRAME_TIMEOUT", 20))

    # Whole-batch budget. Contract cap is 10 minutes; we stop starting new
    # work earlier and reserve time to flush fallbacks + write output.
    total_budget_seconds: float = field(default_factory=lambda: _env_float("TOTAL_BUDGET_SECONDS", 540.0))

    vision_max_tokens: int = field(default_factory=lambda: _env_int("VISION_MAX_TOKENS", 700))
    text_max_tokens: int = field(default_factory=lambda: _env_int("TEXT_MAX_TOKENS", 700))
    # Draft pass: higher temperature for genuine creative variance -- the
    # critique pass (not a low temperature) is what now guards against
    # invented/generic/formulaic output, so we can afford more spark here.
    text_temperature: float = field(default_factory=lambda: _env_float("TEXT_TEMPERATURE", 0.9))
    # Critique pass: lower temperature -- this step is about precise,
    # consistent rule-checking and rewriting, not creativity.
    text_critique_temperature: float = field(default_factory=lambda: _env_float("TEXT_CRITIQUE_TEMPERATURE", 0.4))

    def validate(self) -> None:
        if not self.api_keys:
            raise ValueError("FIREWORKS_API_KEY is not set (comma-separated list expected)")
