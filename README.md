# Video Captioning Agent (Track 2)

A containerized agent that reads `/input/tasks.json`, watches each clip, and
writes `/output/results.json` with a caption in every requested style. Two
stage pipeline on Fireworks AI: a VLM produces one grounded, factual scene
description per clip (Stage A), then a text model restyles that single
description into up to four distinct tones (Stage B). Every requested style
is guaranteed present and non-empty in the output, even under total pipeline
failure.

## ⚠️ PROMINENT SECURITY WARNING ⚠️

**This is a public submission image with no runtime credential injection by
the eval harness — the Fireworks API key(s) must be baked into the image at
build time, which means they WILL be extractable by anyone who pulls the
image** (`docker history`, `docker inspect`, or just running the image and
reading its environment).

Before building the submission image:

- **Use dedicated, disposable Fireworks API keys** created only for this
  submission — never your primary/production keys.
- **Set a hard spend cap** on each key in the Fireworks dashboard before
  baking it in.
- **Delete/revoke the keys immediately after evaluation completes.**
- Do not reuse these keys for anything else, ever, once they've been baked
  into a public image.

## Layout

```
agent/                    # the container's actual code (download, frames, vision, styling, validate, io, main)
backend/                  # FastAPI wrapper that runs the SAME agent code as a live web app (see below)
frontend/                 # DeepSync front-end -- static HTML/CSS/JS, config.js holds the API base URL
dev_tools/                # NOT part of the container -- local iteration scripts only
sample_tasks.json         # 3 sample clips, all 4 styles
Dockerfile                # one-shot batch container (competition submission)
requirements.txt          # container-only deps
requirements-backend.txt  # extra deps for the live web app
```

## Build

Submission images must target `linux/amd64` regardless of your build
machine's architecture:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg FIREWORKS_API_KEY="key1,key2,key3" \
  --build-arg VISION_MODEL="accounts/fireworks/models/qwen3p7-plus" \
  --build-arg TEXT_MODEL="accounts/fireworks/models/glm-5p2" \
  -t video-captioning-agent:submission \
  --load .
```

`VISION_MODEL`/`TEXT_MODEL` build args are optional — omit them to use the
in-code defaults shown above.

## Run locally (keys via `-e`, not baked in)

For local testing, prefer passing keys at `docker run` time instead of
baking them into a local-only image tag:

```bash
docker build -t video-captioning-agent:dev .

mkdir -p input output
cp sample_tasks.json input/tasks.json

docker run --rm \
  -e FIREWORKS_API_KEY="key1,key2,key3" \
  -v "$(pwd)/input:/input:ro" \
  -v "$(pwd)/output:/output" \
  video-captioning-agent:dev
```

Inspect the result:

```bash
cat output/results.json | python3 -m json.tool
```

`FIREWORKS_API_KEY` accepts a **comma-separated list** of keys; the agent
round-robins across them and fails over to the next key on 429/402/transient
errors.

## Tag & push to a public registry

```bash
docker tag video-captioning-agent:submission <registry>/<namespace>/video-captioning-agent:submission
docker push <registry>/<namespace>/video-captioning-agent:submission
```

e.g. for Docker Hub: `docker tag video-captioning-agent:submission yourusername/video-captioning-agent:submission`.

## Configuration (env vars, all optional)

| Var | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | *(required)* | comma-separated Fireworks keys |
| `VISION_MODEL` | `accounts/fireworks/models/qwen3p7-plus` | Stage A VLM (verified live against this account's serverless catalog -- see note below) |
| `TEXT_MODEL` | `accounts/fireworks/models/glm-5p2` | Stage B text model |
| `REASONING_EFFORT` | `none` | both default models are "thinking" models; `none` skips chain-of-thought for lower latency/cost on this grounded-captioning task |
| `NUM_FRAMES` | `10` | frames sampled per clip |
| `MAX_WORKERS` | `4` | concurrent clip workers |
| `TOTAL_BUDGET_SECONDS` | `540` | whole-batch soft deadline (contract cap is 600s) |
| `MAX_LONG_SIDE` | `768` | frame downscale target (px, long side) |
| `JUDGE_MAX_FRAMES` | `6` | frames (evenly subsampled from the clip) shown to the Stage B judge, so it verifies candidates against actual pixels, not just Stage A's text description |
| `INPUT_PATH` / `OUTPUT_PATH` | `/input/tasks.json` / `/output/results.json` | I/O paths |
| `GOOGLE_API_KEY` | *(unset -- optional)* | enables an optional last-resort Stage-A fallback via Gemini, only invoked if every Fireworks vision attempt fails for a clip. Fully optional: the container works with no Google key at all. |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | model used by the optional Gemini fallback. **Note:** this model has hit free-tier quota limits in testing -- use a paid Google API key if you want this fallback to be reliable under real load. |

**Model note:** the Qwen2.5-VL / Llama-3.3 model IDs commonly referenced in \
older Fireworks docs/blog posts are no longer deployed on serverless (they \
404). Neither is Qwen3-VL (235B/30B/32B/8B) on this account -- it exists in \
Fireworks' catalog but isn't serverless-deployed (`deployedModelRefs: []`, \
confirmed via direct call), so it would require an on-demand dedicated \
deployment. `qwen3p7-plus` doesn't appear in either the `/inference/v1/models` \
listing or the paginated accounts catalog, but IS genuinely callable and \
vision-capable -- neither catalog endpoint is fully authoritative for this \
account; direct smoke-testing is the only reliable signal. Head-to-head \
against `kimi-k2p6` on real on-screen-text reading, `qwen3p7-plus` read a \
building sign exactly right at full resolution where `kimi-k2p6` \
hallucinated a different building, and was consistently faster too. \
`glm-5p2` remains a strong, working dedicated text model. Re-verify against \
your own account before relying on any of this -- serverless catalogs \
change, and neither catalog-listing endpoint can be fully trusted.

## Live web app (DeepSync)

A small FastAPI server (`backend/`) wraps this same agent code and serves it
as a live app with a front-end (`frontend/`), for a real, clickable demo
rather than a batch container. It calls `agent.main.process_task()` directly
-- the identical function the submission container runs per clip -- so
there's no separate/forked pipeline to keep in sync.

**Security:** the Fireworks API key is read server-side only, from the
`FIREWORKS_API_KEY` env var, and is used exclusively inside
`backend/jobs.py`'s server-side client. The browser only ever talks to this
app's own `/api/*` routes -- it never sees the key and never calls Fireworks
directly.

### Run locally (unified: one process serves both API and front-end)

No CORS setup needed for this mode -- `frontend/config.js`'s default
(`DEEPSYNC_API_BASE = ""`) means the front-end calls `/api/*` on its own
origin.

```bash
pip install -r requirements.txt -r requirements-backend.txt

FIREWORKS_API_KEY="key1,key2,key3" uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. Paste a public video URL, or drop/browse a
local file (MP4/MOV/WEBM/MKV) -- an uploaded file is POSTed to `/api/upload`,
which stores it and hands back a URL that the pipeline fetches back over
plain HTTP, so both paths feed the identical downstream contract. Queue up
to 12 clips, click "Generate captions" -- the front-end polls real per-clip
status (`Stage A · Analyzing` → `Stage B · Restyling` → `Ready`) and renders
the actual returned captions. A clip that fails mid-pipeline still returns
its guaranteed fallback captions (same contract as the container) and is
flagged in the UI with a "Fallback used" badge rather than silently hidden.

`backend/main.py` mounts `frontend/` as static files and exposes:

| Route | Purpose |
|---|---|
| `POST /api/generate` | `{"clips": [{"video_url", "name"}, ...]}` → `{"job_id"}`, processing starts in the background immediately |
| `POST /api/upload` | multipart file upload (`file`) → `{"video_url", "name"}` for use in a subsequent `/api/generate` call |
| `GET /api/jobs/{job_id}` | current status: per-clip `stage`, `duration`, `captions`, `used_fallback` |

Same [configuration env vars](#configuration-env-vars-all-optional) as the
container apply here too (`VISION_MODEL`, `TEXT_MODEL`,
`TOTAL_BUDGET_SECONDS`, `MAX_WORKERS`, etc.) -- the backend reads the same
`agent.config.Config`.

This app has no state beyond an in-memory job dict, so the backend is
portable to any host that can run a Python ASGI process and allows
long-running requests/background work.

## Dev-only tools (not part of the container)

`dev_tools/` is excluded from the Docker build context (`.dockerignore`) and
is for local iteration only:

- `dev_tools/compare_models.py` — runs the sample clips through vision
  pipeline variants (default: the current `VISION_MODEL` with reasoning on
  vs off) at fixed frame count/resolution, printing Stage-A descriptions and
  final captions side by side -- set `CANDIDATE_VISION_MODELS` to a
  comma-separated list of model IDs to compare real alternatives directly.
- `dev_tools/judge_harness.py` — an LLM-judge that scores each caption's
  accuracy and style match (0-1 each) against the clip's frames, so prompt
  changes can be evaluated by a number rather than a vibe.

Run them with the same `FIREWORKS_API_KEY` env var, from the repo root:

```bash
FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.compare_models
FIREWORKS_API_KEY="key1,key2,key3" python3 -m dev_tools.judge_harness
```
