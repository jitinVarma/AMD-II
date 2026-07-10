"""DeepSync live API + front-end server.

Serves the static front-end AND the job API from one process (no CORS
needed, one command to run both). The Fireworks API key never leaves this
process: it's read from the FIREWORKS_API_KEY env var by agent.config.Config
at startup and only ever used inside backend/jobs.py's server-side
FireworksClient -- the browser only ever talks to our own /api/* routes.

Run:
    FIREWORKS_API_KEY="key1,key2,key3" uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import Config
from agent.fireworks_client import _mask
from .jobs import MAX_CLIPS, JobStore

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend.main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _check_binary(name: str) -> str:
    """Deploy-host sanity check: agent/frames.py shells out to ffmpeg/
    ffprobe directly, and a plain (non-Docker) host may not have them
    installed -- this fails loudly at startup instead of silently
    degrading every clip to fallback captions later.
    """
    try:
        proc = subprocess.run([name, "-version"], capture_output=True, text=True, timeout=10)
        first_line = (proc.stdout or proc.stderr or "").splitlines()[0] if (proc.stdout or proc.stderr) else ""
        return f"OK ({first_line[:60]})"
    except FileNotFoundError:
        return "MISSING -- not on PATH"
    except Exception as exc:
        return f"ERROR ({exc})"


# Fail fast at startup if the key isn't set -- better than accepting
# requests and failing every job.
config = Config()
config.validate()
job_store = JobStore(config)

logger.info(
    "DeepSync API starting: vision_model=%s text_model=%s keys=%d workers=%d budget=%.0fs",
    config.vision_model, config.text_model, len(config.api_keys), config.max_workers, config.total_budget_seconds,
)
logger.info("FIREWORKS_API_KEY(s): %s", ", ".join(_mask(k) for k in config.api_keys))

_ffmpeg_status = _check_binary("ffmpeg")
_ffprobe_status = _check_binary("ffprobe")
logger.info("ffmpeg: %s", _ffmpeg_status)
logger.info("ffprobe: %s", _ffprobe_status)
if "MISSING" in _ffmpeg_status or "MISSING" in _ffprobe_status:
    logger.error(
        "ffmpeg/ffprobe not found on this host -- every clip will fail frame "
        "extraction and fall back to template/ultimate captions. Install "
        "ffmpeg/ffprobe on the host before running the live app."
    )

app = FastAPI(title="DeepSync API")


@app.get("/api/health")
def health():
    """Used by the hosting platform's health check (e.g. Render), and
    doubles as a quick deploy-diagnostics endpoint -- no Fireworks call
    involved, no secrets exposed (keys are masked).
    """
    return {
        "status": "ok",
        "ffmpeg": _ffmpeg_status,
        "ffprobe": _ffprobe_status,
        "fireworks_keys_configured": len(config.api_keys),
    }


class ClipIn(BaseModel):
    video_url: str = Field(min_length=1)
    name: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    clips: list[ClipIn]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if not req.clips:
        raise HTTPException(400, "no clips provided")
    if len(req.clips) > MAX_CLIPS:
        raise HTTPException(400, f"max {MAX_CLIPS} clips per batch")
    try:
        job_id = job_store.create_job([c.model_dump() for c in req.clips])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "clips": [
            {
                "task_id": c.task_id,
                "name": c.name,
                "stage": c.stage,
                "duration": c.duration,
                "captions": c.captions,
                "used_fallback": c.used_fallback,
                "debug_log": c.debug_log,
            }
            for c in job.clips.values()
        ],
    }


# Mounted last so /api/* routes above take precedence over the catch-all
# static file serving.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
