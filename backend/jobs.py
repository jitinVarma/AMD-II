"""In-memory job store + server-side orchestrator for the live web app.

Deliberately thin: it reuses agent.main.process_task directly (the exact
same function the one-shot container calls) rather than reimplementing any
pipeline/captioning logic. The only things this module adds are the parts a
one-shot container doesn't need: an in-memory job/status registry, a shared
executor sized once at server startup (not per-request), and a lightweight
real-duration probe. If agent/main.py's process_task changes, this file's
behavior tracks it automatically since it's the same function call, not a
copy.

Scope note: URL-sourced clips only for this pass (no real file-upload
wiring yet) -- matches the current front-end, which doesn't submit uploaded
files to the API.
"""
from __future__ import annotations

import concurrent.futures
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

from agent.config import Config
from agent.fireworks_client import FireworksClient
from agent.main import process_task
from agent.styling import ALL_STYLES

from .diagnostics import TaskLogCapture

logger = logging.getLogger("backend.jobs")

MAX_CLIPS = 12


@dataclass
class ClipStatus:
    task_id: str
    video_url: str
    name: str
    stage: str = "queued"       # queued | stage_a | stage_b | done
    duration: float | None = None
    captions: dict | None = None
    used_fallback: bool = False
    # Captured pipeline log lines for this clip's task_id (see
    # backend/diagnostics.py) -- lets the API surface *why* a clip fell
    # back (real Fireworks error, empty description, etc.) without needing
    # direct access to the host's server logs.
    debug_log: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    clips: dict[str, ClipStatus]
    created_at: float = field(default_factory=time.monotonic)
    status: str = "running"     # running | done


def _probe_url_duration(url: str, timeout: int) -> float | None:
    """Best-effort real duration via ffprobe directly on the URL (ffprobe
    can read container metadata over HTTP without a full download). Not
    agent.frames.probe_duration -- that function requires a local file path
    (it starts with an os.path.exists check) and process_task's own
    download happens inside a temp dir that's cleaned up before returning,
    so there's no local file left to probe afterward. This is a small,
    self-contained duplicate of just the ffprobe subprocess pattern, not of
    any captioning/pipeline logic.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                url,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return float(proc.stdout.strip())
    except Exception as exc:
        logger.warning("could not probe duration for %s: %s", url, exc)
    return None


class JobStore:
    def __init__(self, config: Config):
        self.config = config
        # One client, one executor, shared across every job for the life of
        # the process -- mirrors the container's own design (key rotation +
        # the connection-pool sizing in fireworks_client.py are built to be
        # shared across concurrent callers) rather than spinning up a fresh
        # client/pool per request.
        self.client = FireworksClient(config)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

        # Tapped onto the root logger so every "[task_id] ..." line already
        # emitted throughout agent/*.py during this clip's run gets
        # buffered and can be attached to its ClipStatus below -- see
        # backend/diagnostics.py.
        self._log_capture = TaskLogCapture()
        logging.getLogger().addHandler(self._log_capture)

    def create_job(self, clips: list[dict]) -> str:
        if not clips:
            raise ValueError("no clips provided")
        if len(clips) > MAX_CLIPS:
            raise ValueError(f"max {MAX_CLIPS} clips per batch")

        job_id = uuid.uuid4().hex[:12]
        clip_statuses: dict[str, ClipStatus] = {}
        tasks = []
        for i, clip in enumerate(clips, start=1):
            # Job-prefixed so task_id is globally unique across concurrent
            # jobs -- every agent/*.py log line is tagged "[task_id] ...",
            # and the log-capture buffer above is keyed by that same
            # string, so a bare "clip_1" would collide between two jobs
            # running at once.
            task_id = f"{job_id}_clip_{i}"
            clip_statuses[task_id] = ClipStatus(task_id=task_id, video_url=clip["video_url"], name=clip["name"])
            tasks.append({"task_id": task_id, "video_url": clip["video_url"], "styles": ALL_STYLES})

        job = Job(id=job_id, clips=clip_statuses)
        with self._lock:
            self._jobs[job_id] = job

        # Per-job deadline, same semantics as the container: this batch has
        # TOTAL_BUDGET_SECONDS from the moment it starts, independent of any
        # other job running concurrently on this server.
        deadline = time.monotonic() + self.config.total_budget_seconds
        for task in tasks:
            self._executor.submit(self._run_clip, job_id, task, deadline)

        return job_id

    def _run_clip(self, job_id: str, task: dict, deadline: float) -> None:
        task_id = task["task_id"]

        def on_stage(stage: str) -> None:
            with self._lock:
                job = self._jobs.get(job_id)
                if not job or task_id not in job.clips:
                    return
                clip = job.clips[task_id]
                if stage == "failed":
                    # The pipeline's own contract guarantees captions either
                    # way (validate.py's safety net / ULTIMATE_FALLBACKS) --
                    # from the UI's perspective this clip still reaches
                    # "Ready", just flagged honestly as degraded rather than
                    # hidden.
                    clip.used_fallback = True
                else:
                    clip.stage = stage

        duration = _probe_url_duration(task["video_url"], timeout=self.config.download_timeout)
        with self._lock:
            job = self._jobs.get(job_id)
            if job and task_id in job.clips:
                job.clips[task_id].duration = duration

        result = process_task(task, self.client, self.config, deadline, on_stage=on_stage)
        debug_log = self._log_capture.pop(task_id)

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            clip = job.clips[task_id]
            clip.captions = result["captions"]
            clip.stage = "done"
            clip.debug_log = debug_log
            if all(c.stage == "done" for c in job.clips.values()):
                job.status = "done"

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)
