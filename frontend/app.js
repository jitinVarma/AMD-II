/* DeepSync — wired to the real backend API (backend/main.py). The browser
   only ever talks to our own /api/* routes, never to Fireworks directly.
   Caption keys match agent/styling.py: ALL_STYLES =
   ["formal","sarcastic","humorous_tech","humorous_non_tech"].

   Clips can be queued either by pasting a URL or by dropping/browsing a
   local file: an uploaded file is POSTed to /api/upload, which stores it
   and hands back a URL that download_video() (agent/download.py) fetches
   back over plain HTTP exactly like a pasted clip -- so both paths feed
   the identical /api/generate contract with zero divergence downstream.
*/

const STYLE_META = {
  formal:              { label: "Formal",              colorVar: "--style-formal",            icon: "formal" },
  sarcastic:           { label: "Sarcastic",            colorVar: "--style-sarcastic",          icon: "sarcastic" },
  humorous_tech:       { label: "Humorous · Tech",      colorVar: "--style-humorous-tech",      icon: "humorous_tech" },
  humorous_non_tech:   { label: "Humorous · Non-Tech",  colorVar: "--style-humorous-non-tech",  icon: "humorous_non_tech" },
};
const STYLE_ORDER = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"];
const MAX_CLIPS = 12;
const POLL_INTERVAL_MS = 900;
const MAX_POLL_FAILURES = 5;
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Empty string = same-origin (relative fetches) -- correct both locally
// and deployed, since backend/main.py serves the front-end and API from
// the same process/origin. See frontend/config.js. Never hardcode a host.
const API_BASE = (window.DEEPSYNC_API_BASE || "").replace(/\/$/, "");

// Stage ordering used to compute the overall (slowest-clip-gates) pipeline
// indicator from real per-clip status.
const STAGE_INDEX = { queued: 0, stage_a: 0, stage_b: 1, done: 2 };

const state = {
  clips: [],                 // queued, pre-submission: { id, url, name, source: "url"|"upload", status: "ready"|"uploading" }
  composerMode: "queueing",  // "queueing" | "processing" | "error"
  errorMessage: null,
  uploadError: null,         // transient inline note for a failed file upload (doesn't touch composerMode)
  jobResult: null,           // populated from the final job poll once status === "done"
};

const ACCEPTED_UPLOAD_TYPES = new Set(["video/mp4", "video/quicktime", "video/webm", "video/x-matroska"]);

let nextClipId = 1;
let pollHandle = null;

/* ---------------------------------------------------------------- helpers */

function fmtDuration(totalSeconds) {
  if (totalSeconds == null || !isFinite(totalSeconds)) return "—:—";
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function filenameFromUrl(url) {
  try {
    const u = new URL(url);
    const parts = u.pathname.split("/").filter(Boolean);
    return parts.length ? parts[parts.length - 1] : u.hostname;
  } catch {
    return url.slice(0, 40) || "clip.mp4";
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function addClip(url) {
  if (state.clips.length >= MAX_CLIPS) return;
  state.clips.push({ id: nextClipId++, url, name: filenameFromUrl(url), source: "url", status: "ready" });
  renderComposerCard();
}

function removeClip(id) {
  state.clips = state.clips.filter((c) => c.id !== id);
  renderComposerCard();
}

/* ---------------------------------------------------------------- page shell */

function renderShell() {
  const app = document.getElementById("app");
  app.innerHTML = `
    ${renderHero()}
    <div class="section-divider"></div>
    <section class="container" id="section-composer">
      <div class="section-head" data-reveal>
        <h2>Queue clips</h2>
        <p>Paste a URL to queue a clip for real processing. Every requested style returns a caption — no blanks, ever.</p>
      </div>
      <div id="composer-card-mount"></div>
    </section>
    <div class="section-divider"></div>
    <section class="container" id="section-results" hidden></section>
  `;
  renderComposerCard();
  wireHeroCta();
  setupScrollReveal();
  setupTopbarScroll();
  animateHeroIn();
}

/* ---------------------------------------------------------------- hero */

function renderHero() {
  return `
    <section class="hero" id="section-hero">
      <div class="hero__bg" aria-hidden="true">
        <div class="hero__grid"></div>
        <div class="hero__glow"></div>
      </div>
      <div class="hero__content">
        <div class="hero__eyebrow" data-hero-el style="--stagger-index:0">
          <span class="mono-tag">Qwen3.7 Plus → GLM 5.2</span>
          <span class="mono-tag">12 clips · 10 min budget</span>
        </div>
        <h1 class="hero__headline" data-hero-el style="--stagger-index:1">Two-stage captioning. Four distinct tones per clip.</h1>
        <p class="hero__subhead" data-hero-el style="--stagger-index:2">A vision model grounds each scene, a text model restyles it into four voices, a judge picks the best.</p>
        <div class="hero__chips" data-hero-el style="--stagger-index:3">
          ${STYLE_ORDER.map((key) => {
            const meta = STYLE_META[key];
            return `<span class="hero__chip" style="--style-color: var(${meta.colorVar})">${ICONS[meta.icon]}${meta.label}</span>`;
          }).join("")}
        </div>
        <div class="hero__cta-row" data-hero-el style="--stagger-index:4">
          <button class="btn btn-primary" id="hero-cta">Get started</button>
        </div>
      </div>
    </section>
  `;
}

function animateHeroIn() {
  const els = document.querySelectorAll("[data-hero-el]");
  if (REDUCED_MOTION) {
    els.forEach((el) => { el.style.opacity = 1; el.style.transform = "none"; });
    return;
  }
  els.forEach((el) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(14px)";
    el.style.transition = "opacity 420ms var(--ease-precise), transform 420ms var(--ease-precise)";
  });
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      els.forEach((el) => {
        const idx = Number(el.style.getPropertyValue("--stagger-index")) || 0;
        el.style.transitionDelay = `${idx * 90}ms`;
        el.style.opacity = "1";
        el.style.transform = "translateY(0)";
      });
    });
  });
}

function wireHeroCta() {
  document.getElementById("hero-cta").addEventListener("click", () => {
    document.getElementById("section-composer").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  });
}

/* ---------------------------------------------------------------- topbar scroll state */

function setupTopbarScroll() {
  const topbar = document.getElementById("topbar");
  const onScroll = () => topbar.classList.toggle("is-scrolled", window.scrollY > 8);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
}

/* ---------------------------------------------------------------- scroll-reveal */

function setupScrollReveal() {
  const targets = document.querySelectorAll("[data-reveal]:not(.is-revealed)");
  if (!("IntersectionObserver" in window) || REDUCED_MOTION) {
    targets.forEach((el) => el.classList.add("is-revealed"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-revealed");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  targets.forEach((el) => observer.observe(el));
}

/* ---------------------------------------------------------------- composer card */

function renderComposerCard() {
  if (state.composerMode === "processing") {
    renderPipelineCard();
  } else if (state.composerMode === "error") {
    renderErrorCard();
  } else {
    renderQueueingCard();
  }
}

function renderQueueingCard() {
  const mount = document.getElementById("composer-card-mount");
  const canGenerate = state.clips.length > 0 && state.clips.every((c) => c.status !== "uploading");

  mount.innerHTML = `
    <div class="composer-card" data-reveal style="--stagger-index:1">
      <div class="url-row">
        <div class="url-input-wrap">
          ${ICONS.link}
          <input type="text" id="url-input" placeholder="https://storage.googleapis.com/…/clip.mp4" autocomplete="off">
        </div>
        <button class="btn btn-secondary" id="url-add-btn">Add URL</button>
      </div>

      <div class="dropzone" id="dropzone">
        ${ICONS.upload}
        <div><strong>Drop a video</strong> or click to browse</div>
        <small>MP4 · MOV · WEBM · MKV — up to ${MAX_CLIPS} clips per batch</small>
        <input type="file" id="file-input" accept="video/mp4,video/quicktime,video/webm,video/x-matroska" multiple>
      </div>
      ${state.uploadError ? `<div class="upload-error">${escapeHtml(state.uploadError)}</div>` : ""}

      <div class="queue-label">Queued (${state.clips.length}/${MAX_CLIPS})</div>
      ${
        state.clips.length === 0
          ? `<div class="queue-empty">Nothing queued yet.</div>`
          : `<div class="queue-list">${state.clips.map(renderUploadRow).join("")}</div>`
      }

      <div class="composer-footer">
        <div class="composer-footer__note">~12 clips processed collectively within a 10-minute total budget — the same cap the real submission runs under.</div>
        <button class="btn btn-generate" id="generate-btn" ${canGenerate ? "" : "disabled"}>Generate captions</button>
      </div>
    </div>
  `;

  document.querySelectorAll("#composer-card-mount [data-reveal]").forEach((el) => el.classList.add("is-revealed"));
  wireComposerEvents();
}

function renderUploadRow(clip) {
  const isUploading = clip.status === "uploading";
  const sourceLabel = isUploading ? "Uploading…" : clip.source === "upload" ? "Uploaded file" : "From URL";
  return `
    <div class="upload-row" data-id="${clip.id}">
      <div class="upload-row__thumb"></div>
      <div class="upload-row__meta">
        <div class="upload-row__name mono">${escapeHtml(clip.name)}</div>
        <div class="upload-row__source">${sourceLabel}</div>
      </div>
      ${isUploading ? "" : `<button class="upload-row__remove" data-remove="${clip.id}" aria-label="Remove">×</button>`}
    </div>
  `;
}

function wireComposerEvents() {
  const urlInput = document.getElementById("url-input");
  const urlAddBtn = document.getElementById("url-add-btn");
  const generateBtn = document.getElementById("generate-btn");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  function submitUrl() {
    const val = urlInput.value.trim();
    if (!val) return;
    addClip(val);
  }

  urlAddBtn.addEventListener("click", submitUrl);
  urlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitUrl();
  });

  document.querySelectorAll("[data-remove]").forEach((btn) => {
    btn.addEventListener("click", () => removeClip(Number(btn.dataset.remove)));
  });

  if (generateBtn) {
    generateBtn.addEventListener("click", () => {
      if (state.clips.length === 0) return;
      startGeneration();
    });
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target === fileInput) return;
    fileInput.click();
  });
  fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = ""; // allow re-selecting the same file later
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  });
}

/* ---------------------------------------------------------------- file upload */

function handleFiles(fileList) {
  state.uploadError = null;
  for (const file of Array.from(fileList)) {
    if (state.clips.length >= MAX_CLIPS) {
      state.uploadError = `Batch limit reached (${MAX_CLIPS} clips) — remove a clip before adding another.`;
      renderComposerCard();
      break;
    }
    if (file.type && !ACCEPTED_UPLOAD_TYPES.has(file.type)) {
      state.uploadError = `"${file.name}" is not a supported video type (expected MP4, MOV, WEBM, or MKV).`;
      renderComposerCard();
      continue;
    }
    queueUpload(file);
  }
}

function queueUpload(file) {
  const id = nextClipId++;
  state.clips.push({ id, url: null, name: file.name, source: "upload", status: "uploading" });
  renderComposerCard();

  const formData = new FormData();
  formData.append("file", file);

  fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData })
    .then(async (resp) => {
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || `upload failed (${resp.status})`);
      }
      return resp.json();
    })
    .then((data) => {
      const clip = state.clips.find((c) => c.id === id);
      if (!clip) return; // removed from the queue while the upload was in flight
      clip.url = data.video_url;
      clip.name = data.name || clip.name;
      clip.status = "ready";
      renderComposerCard();
    })
    .catch((err) => {
      state.clips = state.clips.filter((c) => c.id !== id);
      state.uploadError = `Could not upload "${file.name}": ${err.message}`;
      renderComposerCard();
    });
}

/* ---------------------------------------------------------------- error card */

function renderErrorCard() {
  const mount = document.getElementById("composer-card-mount");
  mount.innerHTML = `
    <div class="composer-card composer-card--error">
      <div class="queue-label">Something went wrong</div>
      <p class="error-message">${escapeHtml(state.errorMessage || "Could not reach the server.")}</p>
      <div class="composer-footer" style="border-top:none; padding-top:0;">
        <div class="composer-footer__note">Your queued clips are still here — nothing was lost.</div>
        <button class="btn btn-generate" id="retry-btn">Try again</button>
      </div>
    </div>
  `;
  document.getElementById("retry-btn").addEventListener("click", () => {
    state.composerMode = "queueing";
    renderComposerCard();
  });
}

/* ---------------------------------------------------------------- pipeline (real status) */

const PIPELINE_STAGES = [
  { key: "a", label: "Stage A", detail: "Analyzing" },
  { key: "b", label: "Stage B", detail: "Restyling" },
  { key: "ready", label: "Ready", detail: null },
];

function renderPipelineCard() {
  const mount = document.getElementById("composer-card-mount");
  mount.innerHTML = `
    <div class="composer-card">
      <div class="queue-label">Processing (${state.clips.length} clip${state.clips.length === 1 ? "" : "s"} · single pass, no retries needed)</div>
      <div class="pipeline__stages" id="pipeline-stages">
        ${PIPELINE_STAGES.map(
          (s) => `
          <div class="stage-item" data-stage="${s.key}">
            <div class="stage-item__dot"></div>
            <div class="stage-item__label">${s.label}${s.detail ? `<span class="sep">·</span>${s.detail}` : ""}</div>
          </div>`
        ).join("")}
      </div>
      <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
    </div>
  `;
}

function setPipelineStage(idx, pct) {
  document.querySelectorAll(".stage-item").forEach((el, i) => {
    el.classList.toggle("is-done", i < idx);
    el.classList.toggle("is-active", i === idx);
  });
  const fill = document.getElementById("progress-fill");
  if (fill) fill.style.width = pct + "%";
}

async function startGeneration() {
  state.composerMode = "processing";
  renderPipelineCard();
  setPipelineStage(0, 8);

  const payload = {
    clips: state.clips.map((c) => ({ video_url: c.url, name: c.name })),
  };

  let jobId;
  try {
    const resp = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Server returned ${resp.status}`);
    }
    const data = await resp.json();
    jobId = data.job_id;
  } catch (err) {
    showError(`Could not start the job: ${err.message}`);
    return;
  }

  pollJob(jobId);
}

function showError(message) {
  if (pollHandle) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
  state.composerMode = "error";
  state.errorMessage = message;
  renderComposerCard();
}

function pollJob(jobId, failureCount = 0) {
  fetch(`${API_BASE}/api/jobs/${jobId}`)
    .then((resp) => {
      if (!resp.ok) throw new Error(`status check returned ${resp.status}`);
      return resp.json();
    })
    .then((job) => {
      // Overall progress = the slowest clip gates the indicator, same
      // feel as the simulated version but now driven by real per-clip state.
      const stageIdxs = job.clips.map((c) => STAGE_INDEX[c.stage] ?? 0);
      const minStage = Math.min(...stageIdxs);
      const pct = job.status === "done" ? 100 : [25, 60, 90][minStage] ?? 25;
      setPipelineStage(minStage, pct);

      if (job.status === "done") {
        state.jobResult = job;
        state.composerMode = "queueing";
        renderComposerCard();
        revealResults(job);
        return;
      }

      pollHandle = setTimeout(() => pollJob(jobId, 0), POLL_INTERVAL_MS);
    })
    .catch((err) => {
      const nextCount = failureCount + 1;
      if (nextCount >= MAX_POLL_FAILURES) {
        showError(`Lost contact with the server while checking progress: ${err.message}`);
        return;
      }
      pollHandle = setTimeout(() => pollJob(jobId, nextCount), POLL_INTERVAL_MS);
    });
}

/* ---------------------------------------------------------------- results */

function revealResults(job) {
  const section = document.getElementById("section-results");
  section.hidden = false;
  section.innerHTML = `
    <div class="section-head" data-reveal>
      <h2>Results</h2>
      <p>Four styles per clip, every slot populated by the real pipeline.</p>
    </div>
    ${job.clips.map(renderResultBlock).join("")}
    <div class="results-footer">
      <button class="link-action" id="back-btn">← Back to queue</button>
    </div>
  `;

  document.querySelectorAll("#section-results [data-reveal]").forEach((el) => el.classList.add("is-revealed"));

  document.getElementById("back-btn").addEventListener("click", () => {
    document.getElementById("section-composer").scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });
  });

  section.scrollIntoView({ behavior: REDUCED_MOTION ? "auto" : "smooth", block: "start" });

  const cards = section.querySelectorAll(".caption-card");
  if (REDUCED_MOTION) {
    cards.forEach((c) => c.classList.add("is-revealed"));
  } else {
    setTimeout(() => cards.forEach((card) => card.classList.add("is-revealed")), 350);
  }
}

function renderResultBlock(clip) {
  return `
    <div class="result-block">
      <div class="video-card">
        <div class="video-card__thumb">
          ${
            clip.video_url
              ? `<video class="video-card__player" src="${escapeHtml(clip.video_url)}" controls preload="metadata" playsinline></video>`
              : ICONS.play
          }
          <span class="video-card__duration mono">${fmtDuration(clip.duration)}</span>
        </div>
        <div class="video-card__meta">
          <div>
            <div class="video-card__name mono">${escapeHtml(clip.name)}</div>
            <div class="video-card__task-id mono">task_id: ${clip.task_id}</div>
          </div>
          ${clip.used_fallback ? `<span class="fallback-badge mono">Fallback used</span>` : ""}
        </div>
      </div>
      <div class="caption-grid">
        ${STYLE_ORDER.map((key, i) => renderCaptionCard(key, i, clip.captions)).join("")}
      </div>
    </div>
  `;
}

function renderCaptionCard(styleKey, index, captions) {
  const meta = STYLE_META[styleKey];
  const text = captions ? captions[styleKey] : null;
  return `
    <div class="caption-card" style="--style-color: var(${meta.colorVar}); --stagger-index:${index}">
      <div class="caption-card__header">
        <span class="style-tag" style="--style-color: var(${meta.colorVar})">
          ${ICONS[meta.icon]}
          ${meta.label}
        </span>
      </div>
      <div class="caption-card__body">
        ${
          text
            ? `<span class="caption-card__text">${escapeHtml(text)}</span>`
            : `<span class="caption-card__empty">No caption returned · ${styleKey}</span>`
        }
      </div>
    </div>
  `;
}

/* ---------------------------------------------------------------- init */

renderShell();
