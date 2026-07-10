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
Dockerfile.backend        # persistent server image for the live web app (Render/Railway/Fly)
render.yaml               # Render Blueprint -- deploys the whole app (see "Deploy to Render" below)
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

Then open `http://localhost:8000`. Paste a public video URL, queue up to 12
clips, click "Generate captions" -- the front-end polls real per-clip status
(`Stage A · Analyzing` → `Stage B · Restyling` → `Ready`) and renders the
actual returned captions. A clip that fails mid-pipeline still returns its
guaranteed fallback captions (same contract as the container) and is flagged
in the UI with a "Fallback used" badge rather than silently hidden.

Scope note: this pass is URL-sourced clips only -- the file-upload dropzone
in the UI is present but inert (labeled "coming soon"), since wiring real
uploads to the API wasn't part of this iteration.

`backend/main.py` mounts `frontend/` as static files and exposes:

| Route | Purpose |
|---|---|
| `POST /api/generate` | `{"clips": [{"video_url", "name"}, ...]}` → `{"job_id"}`, processing starts in the background immediately |
| `GET /api/jobs/{job_id}` | current status: per-clip `stage`, `duration`, `captions`, `used_fallback` |

Same [configuration env vars](#configuration-env-vars-all-optional) as the
container apply here too (`VISION_MODEL`, `TEXT_MODEL`,
`TOTAL_BUDGET_SECONDS`, `MAX_WORKERS`, etc.) -- the backend reads the same
`agent.config.Config`.

This app has no state beyond an in-memory job dict, so the backend is
portable to any host that can run a Python ASGI process and allows
long-running requests/background work.

### Deploy to Render

One Render web service serves both the front-end and the API from the same
FastAPI process -- no second host, no cross-origin setup. This repo
includes `render.yaml` and `Dockerfile.backend` (a separate,
persistent-server image from the root `Dockerfile`, which builds the
one-shot batch container for the competition submission). A plain Vercel
serverless function wouldn't work here: a captioning job can legitimately
run for minutes (download + adaptive frame extraction + Stage A + Stage B +
judge, across up to 12 clips, within an up-to-8-minute budget), well past
serverless request-duration caps -- `Dockerfile.backend` runs it as a
normal always-on server instead. The async job-id-plus-polling design
(`POST /api/generate` returns immediately; `GET /api/jobs/{id}` is polled
for status) keeps every individual HTTP request short regardless.

- In the Render dashboard: **New -> Blueprint**, connect this repo. Render
  reads `render.yaml` and provisions a `deepsync` web service built from
  `Dockerfile.backend`.
- Set these env vars on the service (Render dashboard -> Environment,
  they're declared `sync: false` in `render.yaml` so Render won't try to
  manage them for you):
  - `FIREWORKS_API_KEY` — comma-separated Fireworks key(s). **Never commit
    this or put it in `render.yaml`.**
  - Optionally `VISION_MODEL`, `TEXT_MODEL`, `JUDGE_MODEL`,
    `TOTAL_BUDGET_SECONDS`, `MAX_WORKERS` to override
    [the in-code defaults](#configuration-env-vars-all-optional).
- Render assigns one public URL, e.g. `https://deepsync.onrender.com` --
  open it directly, front-end and API both live there. `GET /api/health` is
  the health-check route Render polls to confirm the service is up.
- (Railway or Fly.io work too -- both are Docker-native, so
  `Dockerfile.backend` carries over directly: Railway via a `railway.json`/
  service pointed at that Dockerfile, Fly via `fly launch --dockerfile
  Dockerfile.backend` + `fly secrets set FIREWORKS_API_KEY=...`. `render.yaml`
  is provided here as the concrete example.)

### Deploy to Fly.io

Fly.io is the recommended host when the free Render memory limit is too
tight. The live app is a long-running Docker service, not a serverless
function: video download, ffmpeg frame extraction, OpenCV scene-change
detection, and concurrent Fireworks calls can legitimately run for several
minutes. `fly.toml` is configured for `Dockerfile.backend`, 2 shared CPUs,
and 4 GB RAM, while keeping the backend pipeline unchanged.

Install and log in:

```bash
brew install flyctl
fly auth login
```

If this is the first Fly deploy for the repo, create the app without
deploying yet:

```bash
fly launch --dockerfile Dockerfile.backend --no-deploy
```

If Fly asks to overwrite `fly.toml`, keep this repo's existing file unless
you intentionally want to change the app name or region. The default
`primary_region` is `bom`; change it to `sin`, `iad`, etc. if that is closer
to your demo audience.

Set secrets and deploy:

```bash
fly secrets set FIREWORKS_API_KEY="key1,key2,key3"
fly deploy
```

Verify:

```bash
fly status
fly logs
curl https://deepsync.fly.dev/api/health
```

Open the deployed app directly at the Fly URL. The same service serves both
the front-end and `/api/*`, so Vercel is not needed for the live demo.

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
