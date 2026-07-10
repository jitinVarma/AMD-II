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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import Config
from .jobs import MAX_CLIPS, JobStore

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend.main")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Fail fast at startup if the key isn't set -- better than accepting
# requests and failing every job.
config = Config()
config.validate()
job_store = JobStore(config)

logger.info(
    "DeepSync API starting: vision_model=%s text_model=%s keys=%d workers=%d budget=%.0fs",
    config.vision_model, config.text_model, len(config.api_keys), config.max_workers, config.total_budget_seconds,
)

app = FastAPI(title="DeepSync API")


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
            }
            for c in job.clips.values()
        ],
    }


# Mounted last so /api/* routes above take precedence over the catch-all
# static file serving.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
