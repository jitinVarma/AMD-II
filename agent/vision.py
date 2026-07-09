"""Stage A: understand the clip ONCE via the VLM. Produces a single neutral,
factual, grounded JSON description shared by all downstream caption styles.
This is the accuracy backbone of the whole pipeline -- style flavor is added
later in Stage B without introducing new facts.
"""
from __future__ import annotations

import logging

from .fireworks_client import FireworksClient, FireworksError
from .json_utils import extract_json_object

logger = logging.getLogger("agent.vision")

DESCRIPTION_SCHEMA_FIELDS = [
    "subjects", "setting", "actions", "mood", "colors",
    "notable_objects", "on_screen_text", "temporal_flow",
]

_SYSTEM_PROMPT = """You are a meticulous visual analyst. You will be shown several \
frames sampled across a single short video clip, in chronological order. Your \
only job is to describe, factually and neutrally, what is VISIBLY present.

Strict rules:
- Describe ONLY what you can actually see in the frames. Never invent people, \
objects, text, locations, or actions that are not visible.
- If you are not confident about something, OMIT it rather than guessing.
- Do not speculate about context, backstory, brand names, or identities unless \
they are clearly and unambiguously visible.
- The frames are sampled across the WHOLE clip; describe change/motion across \
them (temporal_flow), not just a single moment.
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


def _build_user_content(data_uris: list[str]) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Here are {len(data_uris)} frames sampled chronologically across "
                    "one video clip. Analyze them and return the JSON description.",
        }
    ]
    for uri in data_uris:
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


def get_stage_a_description(client: FireworksClient, data_uris: list[str], config, task_id: str = "") -> dict:
    """Returns a description dict. On any failure, returns the best partial
    information available rather than raising, so downstream stages always
    have something grounded (even if minimal) to work with.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(data_uris)},
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

    logger.error("[%s] stage A failed to produce a usable description; using empty fallback", task_id)
    return _empty_description()
