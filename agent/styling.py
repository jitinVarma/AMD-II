"""Stage B: restyle the Stage-A description into each requested caption
style via PER-STYLE ISOLATED generation (no cross-style context bleed --
each generation prompt contains only that style's own card) followed by an
LLM judge pass that vetoes ungrounded/cover-test-failing candidates and
scores survivors. The judge is FRAME-GROUNDED: it sees the actual clip
frames (a subsampled subset, agent/config.py:judge_max_frames) alongside
the candidates, not just Stage A's text description, so it can catch a
candidate that matches the description but contradicts what's actually on
screen -- and catch Stage-A's own omissions/errors, which nothing upstream
could otherwise correct. A time-budget degradation ladder (agent/config.py:
total_budget_seconds) trades candidate count / judge usage for speed as a
clip's share of the whole-batch budget runs out, so quality degrades
gracefully instead of the pipeline ever missing the deadline. The final
template-synthesis safety net (`fallback_caption`, `ULTIMATE_FALLBACKS`) is
unchanged from the previous design -- it's the last-resort guarantee that a
style is never left blank, independent of everything above it.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time

from .fireworks_client import FireworksClient, FireworksError
from .json_utils import extract_json_object

logger = logging.getLogger("agent.styling")

ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
HUMOR_STYLES = ["sarcastic", "humorous_tech", "humorous_non_tech"]

# Optional Stage-B candidate diversity source (SPLIT_GENERATORS=true), same
# defensive-import pattern as agent/vision.py's Gemini fallback -- the
# container must start and run fine on Fireworks alone with no Google key.
try:
    from google import genai as _genai
    _GENAI_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - exercised only when the optional dep is absent
    _genai = None
    _GENAI_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# STYLE CARDS -- provided verbatim by the user. Each card: "definition",
# "exemplars" (3 non-video-subject captions in this voice), "anti_example" +
# "anti_example_reason". The cross-style note that accompanied these cards
# (stay grounded in the Stage-A observation; apply a cover test against the
# other styles) is already structurally enforced elsewhere: grounding via
# _GROUNDING_RULES below, the cover test via the judge's check-4 veto.
# ---------------------------------------------------------------------------
STYLE_CARDS: dict[str, dict] = {
    "formal": {
        "definition": (
            "Professional, objective, and precise. States what is present plainly and "
            "accurately, in one clean declarative sentence. Vivid enough to be worth "
            "reading, but never editorializing, joking, or speculating. Neutral register "
            "-- the voice of a well-written caption in a serious publication. No hedging, "
            "no filler, no adjectives stacked for effect."
        ),
        "exemplars": [
            "A row of copper pans hangs above a marble counter dusted with flour, catching the light from a nearby window.",
            "Commuters move briskly across a rain-slicked platform as a train slows to a halt behind them.",
            "Steam rises from a bowl of noodles set on a wooden table, chopsticks resting across its rim.",
        ],
        "anti_example": "A truly breathtaking, almost magical arrangement of fresh produce that will leave you speechless.",
        "anti_example_reason": (
            'This is marketing copy, not a formal caption. "Breathtaking," "magical," and '
            '"leave you speechless" are subjective sales language; formal means objective '
            "description, not persuasion or hype."
        ),
    },
    "sarcastic": {
        "definition": (
            "Dry, ironic, lightly mocking. The humor comes from an IRONIC GAP -- saying "
            "the opposite of what's meant, mock-praising something mundane, or stating "
            "the obvious as if it were a revelation. Understated, deadpan, and short. One "
            "target per caption. Never mean-spirited, never a rant, never merely negative "
            "-- the wit lives in the restraint. If it reads as warm, cute, or sincerely "
            "enthusiastic, it isn't sarcastic."
        ),
        "exemplars": [
            "Nothing says relaxing vacation quite like a four-hour layover next to the only broken vending machine.",
            "Ah yes, the spreadsheet has thirty-one tabs now -- clearly the problem was not enough tabs.",
            "Truly inspiring how this one traffic cone has commanded an entire lane since roughly the last ice age.",
        ],
        "anti_example": "This rainy day is honestly kind of gloomy and makes me a little sad.",
        "anti_example_reason": (
            "It's just a sincere negative observation with no irony. Sarcasm requires the "
            "gap between what's said and what's meant -- mock-praising the rain, or "
            "feigning delight about being soaked. Plain negativity is not sarcasm."
        ),
    },
    "humorous_tech": {
        "definition": (
            "Funny, built on ONE apt technology or programming analogy that genuinely "
            "illuminates the subject. The tech reference must actually fit -- a real "
            "parallel, not a buzzword or a stock phrase dropped in for flavor. Common "
            "references (404, \"have you tried turning it off and on,\" infinite loop) "
            "are allowed ONLY when they map precisely to something specific in the scene; "
            "if the joke would work on any video, it's too generic -- cut it. One clean "
            "idea, not a pile of jargon. The joke should still make a non-engineer smile; "
            "avoid obscure, specialist-only terms."
        ),
        "exemplars": [
            'This fridge has been "defrosting" for two hours -- classic case of a background process nobody can kill.',
            "An empty lot where the food truck usually parks: 404, lunch not found, and no cache to fall back on.",
            "The vending machine took the money, dropped nothing, and now just blinks -- a transaction that committed on their end but not mine.",
        ],
        "anti_example": "Error 404: motivation not found on this Monday morning.",
        "anti_example_reason": (
            "A recycled meme with no connection to anything actually in the scene -- it "
            "would fit any dreary image equally, which is exactly what the genericness "
            "penalty punishes. A 404 joke only works when it maps to a SPECIFIC absent "
            "thing you can point to in the frame; here it maps to nothing."
        ),
    },
    "humorous_non_tech": {
        "definition": (
            "Warm, everyday, relatable humor -- the kind of observation a witty friend "
            "makes out loud. Finds the funny in ordinary life: the small ironies, the "
            "exaggeration everyone recognizes, the thing we're all thinking. ZERO "
            "technical or programming words. No sarcasm required (it can be affectionate "
            "rather than ironic). Light, human, and specific."
        ),
        "exemplars": [
            "These fries were meant to be shared, but that was decided by someone who has clearly never met me.",
            "The dog has claimed the exact center of the bed with the confidence of an animal who pays no rent.",
            "It's not really a storm until the one flimsy umbrella turns fully inside out and gives up on life.",
        ],
        "anti_example": "The cat is sitting on the windowsill in the afternoon sun.",
        "anti_example_reason": (
            "It's accurate but it's not humor -- there's no joke, exaggeration, or "
            "relatable twist, just a plain description. Humorous_non_tech needs an actual "
            "comedic angle (the cat's smug entitlement, its refusal to move), not a "
            "neutral statement of fact."
        ),
    },
}


def _require_style_cards() -> None:
    missing = [s for s in ALL_STYLES if s not in STYLE_CARDS]
    if missing:
        raise RuntimeError(
            "Stage B style cards are not configured yet (missing: "
            f"{missing}). Paste the 4 style cards into "
            "agent/styling.py:STYLE_CARDS before running Stage B."
        )


def style_definition(style: str) -> str:
    """Public accessor for a style's definition, used by the dev-only judge
    harness to score style match against the same criteria generation uses.
    """
    _require_style_cards()
    return STYLE_CARDS[style]["definition"]


_GROUNDING_RULES = """Hard rules:
- Do NOT invent new facts, objects, people, or events not present in the description. \
Every concrete claim (colors, objects, specific descriptive details, on-screen text, \
background or geographic elements) must be traceable to the given description JSON -- \
nothing added, even if it sounds plausible for the scene.
- The description is VISUAL ONLY (no audio was analyzed) -- never invent sounds, \
dialogue, music, or anything not visible.
- Never invent specific narrative details not present in the description -- no exact \
counts, invented dialogue, names, timeframes, or backstory. Humor comes from framing/tone \
applied to the real facts, not from fabricated specifics.
- Anchor the caption to at least one CONCRETE, specific visible detail from the \
description (a specific object, color, action) -- a caption that could be pasted onto \
a completely different video is too generic.
- ONE clean idea -- do not stack multiple jokes, references, or observations into a \
single caption. Short and sharp beats long and busy.
- Sound like a witty human writing a caption, not an AI performing a style: don't lean \
on the same crutch every time (e.g. an "X -- Y" contrastive em-dash sentence, a generic \
tacked-on ironic tail, or opening with "Nothing says X quite like Y")."""


def _build_generation_system_prompt(style: str) -> str:
    """Per-style ISOLATED prompt -- contains ONLY this style's own card, no
    other style's definition/exemplars/text anywhere in context. This is
    what eliminates the style-bleed the old shared-draft design had.
    """
    card = STYLE_CARDS[style]
    exemplars_txt = "\n".join(f'  - "{ex}"' for ex in card["exemplars"])
    return f"""You are an expert caption writer specializing in exactly ONE style: "{style}".

Definition: {card['definition']}

Exemplar captions in this style, on NON-VIDEO subjects (for voice/tone reference only \
-- these are not about the clip you'll be given, and you must NEVER reuse their \
sentence structure, only the voice they demonstrate):
{exemplars_txt}

Anti-example (do NOT write like this): "{card['anti_example']}"
Why this fails: {card['anti_example_reason']}

{_GROUNDING_RULES}

You will be given a factual, grounded JSON description of a video clip's visible \
content. Write ONE caption in the "{style}" style for that specific clip. Respond with \
ONLY the caption text -- no surrounding quotes, no JSON, no prose before or after, no \
hashtags, no emoji. 1-2 sentences, natural spoken English."""


def _build_user_prompt(description: dict) -> str:
    return (
        "Grounded scene description (JSON):\n"
        f"{json.dumps(description, ensure_ascii=False)}\n\n"
        "Write the caption now."
    )


def _strip_wrapping_quotes(text: str) -> str:
    text = text.strip()
    if len(text) > 1 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def _generate_one_candidate(client: FireworksClient, description: dict, style: str, config, task_id: str) -> str | None:
    messages = [
        {"role": "system", "content": _build_generation_system_prompt(style)},
        {"role": "user", "content": _build_user_prompt(description)},
    ]
    temperature = config.formal_temperature if style == "formal" else config.humor_temperature
    try:
        raw_text = client.chat_completion(
            messages=messages, model=config.text_model,
            max_tokens=config.text_max_tokens, temperature=temperature,
        )
    except FireworksError as exc:
        logger.warning("[%s] %s candidate generation failed: %s", task_id, style, exc)
        return None
    text = _strip_wrapping_quotes(raw_text or "")
    return text or None


def _generate_one_candidate_gemma(description: dict, style: str, config, task_id: str) -> str | None:
    """Optional Stage-B diversity source (SPLIT_GENERATORS=true only).
    Mirrors agent/vision.py's Gemini fallback pattern: defensive import,
    GOOGLE_API_KEY gated, never raises -- any problem here just means one
    fewer candidate, not a failure.
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        return None
    if _genai is None:
        logger.warning(
            "[%s] SPLIT_GENERATORS is on but google-genai is not installed (%s); skipping gemma candidate",
            task_id, _GENAI_IMPORT_ERROR,
        )
        return None
    try:
        client = _genai.Client(api_key=google_api_key)
        prompt_text = _build_generation_system_prompt(style) + "\n\n" + _build_user_prompt(description)
        response = client.models.generate_content(model=config.gemma_model, contents=prompt_text)
        text = _strip_wrapping_quotes(response.text or "")
        return text or None
    except Exception as exc:
        logger.warning("[%s] gemma %s candidate generation failed: %s", task_id, style, exc)
        return None


# ---------------------------------------------------------------------------
# Judge pass -- replaces the old shared critique pass. The 9-point checklist
# below is ported VERBATIM from the prior draft+critique design's
# _CRITIQUE_CHECKLIST, per instruction to reuse it as the core rubric rather
# than rewrite it. Framing around it changes (veto+score instead of
# rewrite), but the checklist items themselves are untouched.
# ---------------------------------------------------------------------------
_JUDGE_CHECKLIST = """1. Echo check: does it just restate/paraphrase the raw scene description instead of being \
freshly composed prose? (Applies especially to "formal".)
2. Length/focus check: is it too long, or does it stack more than one joke/reference/idea \
into a single caption instead of ONE clean idea? For "humorous_tech" specifically: does it \
give the main subject one metaphor AND a background object another in the same caption? \
That's two ideas, not one -- pick a single target.
3. Specificity check: is it generic -- could this exact caption be pasted onto a different, \
unrelated video of the same general category? Does it name at least one concrete, specific \
visible detail from THIS description? For "humorous_tech": is the analogy anchored to the \
main subject's own action, or is it straining to fit an incidental detail like camera \
movement -- if the latter, rewrite it around the subject's actual action instead.
4. Cover-test check: could this caption be relabeled as a different one of the requested \
styles without anyone noticing? In particular: is "sarcastic" dry/ironic (not just negative \
or ranting)? Does "humorous_tech" commit to exactly ONE cohesive analogy (not a pile of \
buzzwords)? Does "humorous_non_tech" contain zero technical/programming words?
5. Voice check: does it read like an AI performing a style (clichéd phrasing, the same \
"X -- Y" em-dash contrast structure repeated across captions, a generic tacked-on ironic \
tail, or opening with "Nothing says X quite like Y") rather than a witty human wrote it? \
Rewrite the opening entirely if it uses any recognizable template phrase.
6. Grounding-source check: for every specific claim in a caption (colors, objects, \
descriptive details, on-screen text, background/geographic elements), verify it is \
explicitly present in the given Stage-A description JSON. Remove or correct anything not \
present -- e.g. don't describe a hairstyle with more specificity than the description gives, \
don't mention mountains/landmarks/on-screen text unless the description explicitly lists them.
7. Irony check (sarcastic only): does it actually use mock-praise, feigned enthusiasm, or \
ironic overstatement of the obvious -- a real gap between what's said and what's meant? If it \
merely reads as warm, whimsical, or cutely observational with no ironic gap, it has failed as \
sarcastic even if it's funny -- rewrite it using mock-praise or feigned enthusiasm.
8. Jargon check (humorous_tech only): is the analogy built from a tech concept a general \
reader would recognize, or does it lean on niche/specialist jargon (e.g. graphics-programming \
internals, memory allocators, obscure framework terms) that only a specialist would find \
funny? If niche, rewrite around a more broadly-known concept.
9. Sarcastic-vs-non_tech pairwise check: compare "sarcastic" directly against \
"humorous_non_tech" if both are requested -- if you could swap their two captions' style \
labels and nobody would notice, they have not been separated enough. Push sarcastic's irony \
harder (mock-praise/feigned enthusiasm) and make sure humorous_non_tech carries zero irony, \
until the two are unmistakably distinct.
10. Frame-verification check: you have been shown the actual clip frames below, not just the \
Stage-A description -- use them as the ultimate ground truth. For every concrete claim in a \
candidate, check it against the frames directly, not only against the description text. A \
claim can pass check 6 (present in the description) yet still be wrong or overstated once you \
actually look -- e.g. the description calls a color "dark" but the frames show it's clearly \
navy, or the description omits something the frames make obvious. Veto a candidate whose \
claims are contradicted by what you can actually see in the frames, even if it's faithful to \
the description text."""


def _build_judge_system_prompt(style: str, all_requested_styles: list[str]) -> str:
    sibling_defs = "\n".join(
        f'- "{s}": {STYLE_CARDS[s]["definition"]}' for s in all_requested_styles if s != style and s in STYLE_CARDS
    )
    return f"""You are a ruthless caption judge. You will be given a grounded Stage-A scene \
description, the ACTUAL clip frames (sampled chronologically, labeled with timestamps -- \
these are ground truth and take priority over the description text if the two ever disagree), \
and several candidate captions, all written for the SAME target style ("{style}"). Apply the \
checklist below to each candidate: VETO it (do not score) if it fails check 6 (invents a \
detail not in the Stage-A description), check 10 (contradicted by the attached frames \
themselves), or check 4 (fails the cover test against another requested style). Otherwise \
score it 0-10 for how well it executes the "{style}" style, applying a -3 penalty if check 3 \
reveals it's generic enough to describe a different video. Then pick the highest-scoring \
non-vetoed candidate as the winner.

Target style ("{style}") definition: {STYLE_CARDS[style]['definition']}

Other requested styles (for the cover-test check only -- do not judge candidates against \
these definitions, only use them to check for cross-style confusability):
{sibling_defs}

Checklist:
{_JUDGE_CHECKLIST}

Respond with ONLY a JSON object, no prose before or after, no code fences. Keep every \
"veto_reason" under 15 words -- terse, not an essay:
{{"candidates": [{{"index": 0, "vetoed": false, "veto_reason": null, "score": 7.5}}, ...], "winner_index": 0}}
If every candidate is vetoed, set "winner_index" to null."""


def _build_judge_user_prompt(description: dict, candidates: list[str]) -> str:
    candidates_txt = "\n".join(f'{i}: "{c}"' for i, c in enumerate(candidates))
    return (
        "Grounded scene description (JSON):\n"
        f"{json.dumps(description, ensure_ascii=False)}\n\n"
        f"Candidates:\n{candidates_txt}\n\n"
        "The actual clip frames are attached above this message, in chronological order. "
        "Return the JSON verdict now."
    )


def _subsample_frames_for_judge(
    timestamped_frames: list[tuple[float, str]], max_frames: int,
) -> list[tuple[float, str]]:
    """Evenly subsamples down to `max_frames`, always keeping first and last,
    to bound the judge's vision token cost (this call repeats per humor
    style, up to 3x per clip, plus a possible regeneration retry). Returns
    the input unchanged if it's already at or under the cap.
    """
    n = len(timestamped_frames)
    if n <= max_frames or max_frames <= 0:
        return timestamped_frames
    if max_frames == 1:
        return [timestamped_frames[0]]
    step = (n - 1) / (max_frames - 1)
    indices = sorted({round(i * step) for i in range(max_frames)})
    return [timestamped_frames[i] for i in indices]


def _build_judge_user_content(
    description: dict, candidates: list[str], timestamped_frames: list[tuple[float, str]],
) -> list[dict]:
    """Frame-grounded user content: the real frames (so the judge can check
    candidates against actual pixels, not just Stage A's text) followed by
    the description JSON and candidates.
    """
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Here are {len(timestamped_frames)} frames sampled chronologically from "
                    "the clip these candidates are about, each labeled with its timestamp.",
        }
    ]
    for t, uri in timestamped_frames:
        content.append({"type": "text", "text": f"[frame at t={t:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": uri}})
    content.append({"type": "text", "text": _build_judge_user_prompt(description, candidates)})
    return content


def _judge_and_pick(
    client: FireworksClient, description: dict, style: str, candidates: list[str],
    all_requested_styles: list[str], config, task_id: str,
    timestamped_frames: list[tuple[float, str]] | None = None,
) -> str | None:
    """Returns the winning candidate text, or None if the judge call failed,
    returned unusable output, or vetoed every candidate. `timestamped_frames`,
    if given, is subsampled to config.judge_max_frames and attached so the
    judge checks candidates against the real clip, not just Stage A's text
    description.
    """
    if not candidates:
        return None

    if timestamped_frames:
        frames_for_judge = _subsample_frames_for_judge(timestamped_frames, config.judge_max_frames)
        user_content: list[dict] | str = _build_judge_user_content(description, candidates, frames_for_judge)
    else:
        user_content = _build_judge_user_prompt(description, candidates)

    messages = [
        {"role": "system", "content": _build_judge_system_prompt(style, all_requested_styles)},
        {"role": "user", "content": user_content},
    ]
    try:
        raw_text = client.chat_completion(
            messages=messages, model=config.judge_model,
            max_tokens=config.judge_max_tokens, temperature=config.judge_temperature,
        )
    except FireworksError as exc:
        logger.error("[%s] %s judge call failed: %s", task_id, style, exc)
        return None

    parsed = extract_json_object(raw_text)
    if parsed is None:
        logger.warning("[%s] %s judge returned unparseable JSON, raw head: %r",
                        task_id, style, raw_text[:200] if raw_text else raw_text)
        return None

    candidate_meta = {c.get("index"): c for c in parsed.get("candidates", []) if isinstance(c.get("index"), int)}

    winner_index = parsed.get("winner_index")
    if isinstance(winner_index, int) and 0 <= winner_index < len(candidates):
        meta = candidate_meta.get(winner_index)
        if not (meta and meta.get("vetoed")):
            return candidates[winner_index]
        logger.warning("[%s] %s judge picked a vetoed winner_index, ignoring", task_id, style)

    # No usable explicit winner -- fall back to the highest-scoring non-vetoed candidate.
    scored = [
        (idx, meta["score"]) for idx, meta in candidate_meta.items()
        if not meta.get("vetoed") and isinstance(meta.get("score"), (int, float)) and 0 <= idx < len(candidates)
    ]
    if scored:
        best_index = max(scored, key=lambda pair: pair[1])[0]
        return candidates[best_index]

    return None


def fallback_caption(style: str, description: dict) -> str:
    """Safe, grounded, always-non-empty caption synthesized without another
    model call. Used when generation + judging + one regeneration attempt
    all fail for a given style, or at the template_fallback degradation
    tier. This is an emergency safety net, not the quality path -- it's
    built fresh from live Stage-A fields each time, never a hardcoded joke.
    UNCHANGED from the prior design.
    """
    subjects = ", ".join(description.get("subjects") or []) or "the subject"
    setting = (description.get("setting") or "").strip()
    setting_phrase = f" in {setting}" if setting else ""
    actions = ", ".join(description.get("actions") or [])
    actions_phrase = f", {actions}" if actions else ""

    if style == "formal":
        result = f"A video clip showing {subjects}{setting_phrase}{actions_phrase}."
    elif style == "sarcastic":
        result = f"Riveting footage: {subjects} just going about business{setting_phrase}, like it's the highlight of the day."
    elif style == "humorous_tech":
        result = f"{subjects} running the default routine{setting_phrase}, no bugs reported so far."
    elif style == "humorous_non_tech":
        result = f"Just {subjects} doing their thing{setting_phrase}, nothing dramatic here."
    else:
        result = f"{subjects}{setting_phrase}."

    result = result.replace("  ", " ")
    return result[0].upper() + result[1:] if result else result


ULTIMATE_FALLBACKS = {
    "formal": "This video presents a brief visual sequence captured on camera.",
    "sarcastic": "Truly groundbreaking footage: something happened, on camera, presumably.",
    "humorous_tech": "Content loading... rendered successfully with zero further detail available.",
    "humorous_non_tech": "Well, something's going on here, and that's about all we know.",
}


# ---------------------------------------------------------------------------
# Time-budget degradation ladder
# ---------------------------------------------------------------------------
_TIER_CONFIG = {
    "full_quality": {"candidates": 4, "use_judge": True},
    "reduced_candidates": {"candidates": 2, "use_judge": True},
    "skip_judge": {"candidates": 1, "use_judge": False},
    "template_fallback": {"candidates": 0, "use_judge": False},
}


def _elapsed_fraction(deadline: float, config) -> float:
    if config.total_budget_seconds <= 0:
        return 0.0
    remaining = deadline - time.monotonic()
    elapsed = config.total_budget_seconds - remaining
    return max(0.0, min(1.0, elapsed / config.total_budget_seconds))


def _degradation_tier(elapsed_fraction: float) -> str:
    if elapsed_fraction > 0.90:
        return "template_fallback"
    if elapsed_fraction > 0.75:
        return "skip_judge"
    if elapsed_fraction > 0.60:
        return "reduced_candidates"
    return "full_quality"


def get_stage_b_captions(
    client: FireworksClient, description: dict, styles: list[str], config,
    task_id: str = "", deadline: float | None = None,
    timestamped_frames: list[tuple[float, str]] | None = None,
) -> dict:
    """Returns {style: caption} for every style in `styles`. `deadline` is an
    absolute time.monotonic() timestamp (agent/main.py's whole-batch
    deadline) used to pick this clip's degradation tier; if not provided
    (e.g. dev_tools callers), defaults to "plenty of time left" (full
    quality). formal is unaffected by the ladder. `timestamped_frames`, if
    given (the same frames Stage A used), is passed to the judge so it can
    verify candidates against the actual clip instead of only Stage A's
    text description -- omit it (e.g. no frames handy) to fall back to the
    pre-frame-grounding text-only judge. Never returns a missing or empty
    caption for any requested style.
    """
    _require_style_cards()

    if deadline is None:
        deadline = time.monotonic() + config.total_budget_seconds

    elapsed_fraction = _elapsed_fraction(deadline, config)
    tier = _degradation_tier(elapsed_fraction)
    tier_cfg = _TIER_CONFIG[tier]
    logger.info("[%s] stage B degradation tier=%s (%.0f%% of budget elapsed)",
                task_id, tier, elapsed_fraction * 100)

    captions: dict[str, str] = {}
    humor_requested = [s for s in styles if s in HUMOR_STYLES]

    # Phase 1: generate every needed candidate for every requested style,
    # ALL concurrently in one flat pool (formal + every humor-style
    # candidate) -- Stage B must not be sequential.
    gen_jobs: list[tuple[str, bool]] = []  # (style, use_gemma)
    if "formal" in styles:
        gen_jobs.append(("formal", False))

    if tier == "template_fallback":
        for style in humor_requested:
            captions[style] = fallback_caption(style, description)
    else:
        n = tier_cfg["candidates"]
        gemma_count = (n // 2) if config.split_generators else 0
        for style in humor_requested:
            for i in range(n):
                gen_jobs.append((style, i < gemma_count))

    candidates_by_style: dict[str, list[str]] = {s: [] for s in styles}
    if gen_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gen_jobs)) as pool:
            future_map = {}
            for style, use_gemma in gen_jobs:
                if use_gemma:
                    fut = pool.submit(_generate_one_candidate_gemma, description, style, config, task_id)
                else:
                    fut = pool.submit(_generate_one_candidate, client, description, style, config, task_id)
                future_map[fut] = style
            for fut in concurrent.futures.as_completed(future_map):
                style = future_map[fut]
                try:
                    text = fut.result()
                except Exception:
                    logger.exception("[%s] %s candidate generation raised unexpectedly", task_id, style)
                    text = None
                if text:
                    candidates_by_style[style].append(text)

    if "formal" in styles:
        formal_candidates = candidates_by_style.get("formal", [])
        # formal SKIPS the judge -- length/grounding are handled by the
        # generation prompt's own rules plus validate.py's length bound.
        captions["formal"] = formal_candidates[0] if formal_candidates else fallback_caption("formal", description)

    # Phase 2: resolve each humor style (judge / first-valid / already
    # template-filled above) concurrently -- independent per style.
    pending_humor = [s for s in humor_requested if s not in captions]
    if pending_humor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending_humor)) as pool:
            future_map = {
                pool.submit(
                    _resolve_humor_style, client, description, style,
                    candidates_by_style.get(style, []), styles, tier_cfg, config, task_id,
                    timestamped_frames,
                ): style
                for style in pending_humor
            }
            for fut in concurrent.futures.as_completed(future_map):
                style = future_map[fut]
                try:
                    captions[style] = fut.result()
                except Exception:
                    logger.exception("[%s] %s resolution raised unexpectedly", task_id, style)
                    captions[style] = fallback_caption(style, description)

    return captions


def _resolve_humor_style(
    client: FireworksClient, description: dict, style: str, candidates: list[str],
    all_requested_styles: list[str], tier_cfg: dict, config, task_id: str,
    timestamped_frames: list[tuple[float, str]] | None = None,
) -> str:
    if not candidates:
        logger.warning("[%s] no %s candidates generated, using template fallback", task_id, style)
        return fallback_caption(style, description)

    if not tier_cfg["use_judge"]:
        # skip_judge tier: first valid candidate wins (structural validity only).
        return candidates[0]

    winner = _judge_and_pick(
        client, description, style, candidates, all_requested_styles, config, task_id,
        timestamped_frames=timestamped_frames,
    )
    if winner:
        return winner

    logger.warning("[%s] all %s candidates vetoed/unusable, regenerating once", task_id, style)
    fresh = [
        c for c in (
            _generate_one_candidate(client, description, style, config, task_id) for _ in range(2)
        ) if c
    ]
    if fresh:
        winner = _judge_and_pick(
            client, description, style, fresh, all_requested_styles, config, task_id,
            timestamped_frames=timestamped_frames,
        )
        if winner:
            return winner

    logger.error("[%s] %s still unusable after regeneration, using template fallback", task_id, style)
    return fallback_caption(style, description)
