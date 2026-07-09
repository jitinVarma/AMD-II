"""Stage A: understand the clip ONCE via the VLM. Produces a single neutral,
factual, grounded JSON description shared by all downstream caption styles.
This is the accuracy backbone of the whole pipeline -- style flavor is added
later in Stage B without introducing new facts.
"""
from __future__ import annotations

import base64
import logging
import os

from .fireworks_client import FireworksClient, FireworksError
from .json_utils import extract_json_object

logger = logging.getLogger("agent.vision")

# google-genai is an OPTIONAL dependency: the container must start and run
# fully on Fireworks alone with no Google key. This import must never break
# startup even if the package isn't installed.
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    _GENAI_IMPORT_ERROR: Exception | None = None
except ImportError as _exc:  # pragma: no cover - exercised only when the optional dep is absent
    _genai = None
    _genai_types = None
    _GENAI_IMPORT_ERROR = _exc

DESCRIPTION_SCHEMA_FIELDS = [
    "subjects", "setting", "actions", "mood", "colors",
    "notable_objects", "on_screen_text", "temporal_flow",
]

_SYSTEM_PROMPT = """You are a meticulous visual analyst. You will be shown several \
frames sampled across a single short video clip, in chronological order, each \
preceded by its own timestamp label like "[frame at t=4.2s]". Your only job is to \
describe, factually and neutrally, what is VISIBLY present.

Strict rules:
- Describe ONLY what you can actually see in the frames. Never invent people, \
objects, text, locations, or actions that are not visible.
- If you are not confident about something, OMIT it rather than guessing.
- Do not speculate about context, backstory, brand names, or identities unless \
they are clearly and unambiguously visible.
- The frames are sampled across the WHOLE clip; use the timestamp labels to reason \
about WHEN changes happen, and describe change/motion across them (temporal_flow), \
not just a single moment.
- Respond with ONLY a single JSON object, no prose before or after, no code fences.

JSON schema (all fields required; use empty string / empty list if nothing \
applicable -- never fabricate to fill a field):
{
  "subjects": ["short noun phrases for the main subject(s), e.g. 'a young orange kitten'"],
  "setting": "one short phrase for the location/environment",
  "actions": ["short phrases for concrete actions/events actually seen"],
  "mood": "one or two words for the visible atmosphere/tone (e.g. calm, energetic)",
  "colors": ["dominant visible colors"],
  "notable_objects": ["distinct visible objects/props, if any"],
  "on_screen_text": "any literal text visible in-frame, exactly as shown, or empty string",
  "temporal_flow": "one short sentence on how the scene changes from first frame to last"
}"""


def _build_user_content(timestamped_frames: list[tuple[float, str]]) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Here are {len(timestamped_frames)} frames sampled chronologically across "
                    "one video clip, each labeled with its timestamp. Analyze them and return "
                    "the JSON description.",
        }
    ]
    for t, uri in timestamped_frames:
        content.append({"type": "text", "text": f"[frame at t={t:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return content


def _empty_description() -> dict:
    return {
        "subjects": [], "setting": "", "actions": [], "mood": "",
        "colors": [], "notable_objects": [], "on_screen_text": "", "temporal_flow": "",
    }


def _coerce_description(raw: dict) -> dict:
    desc = _empty_description()
    for field in DESCRIPTION_SCHEMA_FIELDS:
        if field not in raw:
            continue
        value = raw[field]
        if field in ("setting", "mood", "on_screen_text", "temporal_flow"):
            desc[field] = str(value).strip() if value is not None else ""
        else:
            if isinstance(value, list):
                desc[field] = [str(v).strip() for v in value if str(v).strip()]
            elif value:
                desc[field] = [str(value).strip()]
    return desc


def _data_uri_to_bytes(data_uri: str) -> bytes:
    _, b64 = data_uri.split(",", 1)
    return base64.b64decode(b64)


def _get_stage_a_description_gemini(timestamped_frames: list[tuple[float, str]], config, task_id: str) -> dict | None:
    """Optional last-resort fallback: only runs if GOOGLE_API_KEY is set AND
    every Fireworks attempt already failed. Returns None (never raises) on
    any problem, so the caller just falls through to the empty description
    as before -- this path can only help, never make things worse.
    """
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        return None

    if _genai is None:
        logger.warning(
            "[%s] GOOGLE_API_KEY is set but google-genai is not installed (%s); skipping Gemini fallback",
            task_id, _GENAI_IMPORT_ERROR,
        )
        return None

    try:
        client = _genai.Client(api_key=google_api_key)
        parts = [_genai_types.Part.from_text(text=_SYSTEM_PROMPT + "\n\nHere are "
                 f"{len(timestamped_frames)} frames sampled chronologically across one video clip, "
                 "each labeled with its timestamp. Analyze them and return the JSON description.")]
        for t, uri in timestamped_frames:
            parts.append(_genai_types.Part.from_text(text=f"[frame at t={t:.1f}s]"))
            parts.append(_genai_types.Part.from_bytes(data=_data_uri_to_bytes(uri), mime_type="image/jpeg"))

        response = client.models.generate_content(model=config.gemini_model, contents=parts)
        raw_text = response.text
    except Exception as exc:
        logger.error("[%s] gemini fallback call failed: %s", task_id, exc)
        return None

    parsed = extract_json_object(raw_text)
    if parsed is None:
        logger.warning("[%s] gemini fallback returned unparseable JSON, raw head: %r",
                        task_id, raw_text[:200] if raw_text else raw_text)
        return None

    logger.info("[%s] stage A recovered via Gemini fallback (%s)", task_id, config.gemini_model)
    return _coerce_description(parsed)


def get_stage_a_description(client: FireworksClient, timestamped_frames: list[tuple[float, str]], config, task_id: str = "") -> dict:
    """Returns a description dict. `timestamped_frames` is a chronologically-
    ordered list of (timestamp_seconds, data_uri) pairs. On any failure,
    returns the best partial information available rather than raising, so
    downstream stages always have something grounded (even if minimal) to
    work with.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(timestamped_frames)},
    ]

    for retry in range(2):
        try:
            raw_text = client.chat_completion(
                messages=messages,
                model=config.vision_model,
                max_tokens=config.vision_max_tokens,
                temperature=0.2,
            )
        except FireworksError as exc:
            logger.error("[%s] stage A vision call failed (retry=%d): %s", task_id, retry, exc)
            continue

        parsed = extract_json_object(raw_text)
        if parsed is not None:
            return _coerce_description(parsed)

        logger.warning("[%s] stage A returned unparseable JSON (retry=%d), raw head: %r",
                        task_id, retry, raw_text[:200] if raw_text else raw_text)
        if retry == 0:
            messages.append({"role": "assistant", "content": raw_text or ""})
            messages.append({
                "role": "user",
                "content": "That was not valid JSON. Respond again with ONLY the JSON object, "
                            "no other text, no code fences.",
            })

    gemini_result = _get_stage_a_description_gemini(timestamped_frames, config, task_id)
    if gemini_result is not None:
        return gemini_result

    logger.error("[%s] stage A failed to produce a usable description; using empty fallback", task_id)
    return _empty_description()
