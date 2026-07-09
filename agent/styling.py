"""Stage B: restyle the single grounded Stage-A description into each
requested caption style. This is where style-match is won.

Two-pass design, deliberately with NO hardcoded example captions anywhere in
the prompt: a draft pass writes fresh, grounded captions from principle-only
style definitions, then a separate critique pass checks each draft against an
explicit failure-mode checklist (echoes the description? generic/not
scene-specific? stacked jokes? fails the "cover test" against the other
styles?) and rewrites anything that fails before it's returned. Examples
were deliberately removed after we found they taught the model *a* template
rather than genuine per-style voice -- the critique pass is what now guards
against generic/formulaic/cross-style-bleed output instead.
"""
from __future__ import annotations

import json
import logging

from .fireworks_client import FireworksClient, FireworksError
from .json_utils import extract_json_object

logger = logging.getLogger("agent.styling")

ALL_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

# Principle-only style guide -- no example captions of any kind. Examples
# taught the model a template; these definitions + the critique pass now do
# the work of teaching voice.
_STYLE_GUIDE = {
    "formal": {
        "definition": (
            "Precise, objective, professional third-person prose, freshly composed as a "
            "real sentence -- not a rephrasing or listing of the raw scene-description "
            "fields. Vivid and specific: name the concrete visible details, use active "
            "verbs. Never a joke, never flowery, never robotic."
        ),
        "avoid": (
            "Do not simply restate or lightly reword the raw scene description -- if a "
            "reader could tell you copied the description fields into a sentence, rewrite it."
        ),
    },
    "sarcastic": {
        "definition": (
            "Genuine IRONY with a clear gap between what's said and what's meant -- not just "
            "a dry or wry observation. Use one of: mock-praise (over-complimenting something "
            "mundane or unimpressive as if it were remarkable), feigned enthusiasm for "
            "something tedious, or stating the obvious/naive as if it were a revelation. The "
            "reader should sense the caption doesn't mean what it literally says, aimed at ONE "
            "specific target in the scene."
        ),
        "avoid": (
            "A caption that is merely charming, cute, whimsical, or lightly amusing is NOT "
            "sarcastic -- if there's no ironic gap (a reader could believe you actually mean "
            "it literally), it's the wrong style; rewrite it using mock-praise or feigned "
            "enthusiasm instead. Never mean-spirited or ranting, never simply "
            "negative/complaining. Only one ironic target per caption -- don't stack multiple "
            "sarcastic jabs together. Never open with \"Nothing says X quite like Y\" or any "
            "other template opener -- vary how the caption starts."
        ),
    },
    "humorous_tech": {
        "definition": (
            "Exactly ONE cohesive tech/programming/engineering analogy that genuinely maps "
            "onto what's actually happening in the scene, built from a concept a GENERAL "
            "reader would recognize (bugs, buffering/loading, notifications, low battery or "
            "signal, autocorrect, software updates, deploys, 404 errors, spam, etc) -- not "
            "niche or specialist jargon (e.g. graphics-programming internals, memory "
            "allocators, obscure framework terms) that only a specialist would find funny. "
            "Pick the single closest-fitting concept and commit to it fully in one sentence. "
            "Anchor the analogy to the main subject's own action or state, not an incidental "
            "detail like camera movement -- if the analogy requires stretching to make an "
            "incidental detail fit, it's the wrong analogy."
        ),
        "avoid": (
            "Never stack multiple unrelated tech terms/buzzwords into one caption -- that "
            "reads as random jargon, not a joke. Never reach for an obscure/specialist concept "
            "when a broadly-known one would work just as well -- the joke must land for a "
            "general reader, not just programmers. Pick exactly ONE target in the scene for "
            "the analogy (don't give the main subject one metaphor and a background object "
            "another in the same caption). If the scene isn't techy, still use exactly ONE "
            "clever, well-chosen, broadly-understandable metaphor rather than forcing "
            "something in."
        ),
    },
    "humorous_non_tech": {
        "definition": (
            "Everyday, warm, relatable observational humor -- short, casual, the kind of "
            "caption a friend would text you. No irony required or expected here -- this is "
            "warmth and relatability, not a hidden meaning. Grounded in a universal feeling "
            "(hunger, tiredness, mischief, drama) triggered by one specific visible detail."
        ),
        "avoid": (
            "ZERO technical, programming, or computing words of any kind. Keep it short and "
            "plainly worded, not poetic or grandiose. Never invent sounds, dialogue, or "
            "specifics not actually visible."
        ),
    },
}


def style_definition(style: str) -> str:
    """Public accessor for a style's guardrail definition, used by the
    dev-only judge harness to score style match against the same criteria
    the generation prompt itself uses.
    """
    return _STYLE_GUIDE[style]["definition"]


def _format_style_block(style: str) -> str:
    guide = _STYLE_GUIDE[style]
    return (
        f'"{style}":\n'
        f'  Definition: {guide["definition"]}\n'
        f'  Avoid: {guide["avoid"]}'
    )


_GENERAL_HARD_RULES = """- Do NOT invent new facts, objects, people, or events not present in the description. \
Every concrete claim in a caption (colors, objects, specific descriptive details, on-screen \
text, background or geographic elements) must be traceable to the given description JSON -- \
nothing added, even if it sounds plausible for the scene.
- The description is VISUAL ONLY (no audio was analyzed) -- never invent sounds, \
dialogue, music, or anything not visible (e.g. don't call traffic "honking" or a \
scene "loud"/"quiet" unless that's explicitly in the description).
- Never invent specific narrative details not present in the description -- no exact \
counts, invented dialogue, names, timeframes, or backstory. Humor comes from framing/tone \
applied to the real facts, not from fabricated specifics.
- Anchor every caption to at least one CONCRETE, specific visible detail from the \
description (a specific object, color, action) -- a caption that could be pasted onto \
a completely different video is too generic; rewrite it to reference something specific \
to THIS scene.
- ONE clean idea per caption -- do not stack multiple jokes, references, or observations \
into a single caption. Short and sharp beats long and busy.
- Sound like a witty human writing a caption, not an AI performing a style: vary your \
sentence structure across the captions, and avoid leaning on the same crutch every time \
(e.g. don't make every caption an "X -- Y" contrastive em-dash sentence, and avoid \
generic tacked-on ironic tails).
- Each style must be unmistakable on its own -- if a caption could be relabeled as a \
different one of the requested styles without anyone noticing, it isn't sharp enough.
- Each caption is exactly 1-2 sentences, natural spoken English, no hashtags, no emoji.
- Respond with ONLY a single JSON object mapping each style name to its caption \
string, no prose before or after, no code fences."""


def _build_draft_system_prompt(styles: list[str]) -> str:
    style_blocks = "\n\n".join(_format_style_block(s) for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)
    return f"""You are an expert caption writer. You will be given a factual, \
grounded JSON description of a video clip's visible content. Write ONE caption \
per requested style below, staying strictly grounded in the given facts.

Hard rules:
{_GENERAL_HARD_RULES}

Styles to generate ({keys}):

{style_blocks}

Output JSON shape: {{{", ".join(f'"{s}": "..."' for s in styles)}}}"""


def _build_user_prompt(description: dict) -> str:
    return (
        "Grounded scene description (JSON):\n"
        f"{json.dumps(description, ensure_ascii=False)}\n\n"
        "Write the captions now, one per requested style, as a single JSON object."
    )


_CRITIQUE_CHECKLIST = """For EACH caption, check it against every item below. If it fails \
ANY item, rewrite that caption to fix the problem while keeping it grounded and clearly in \
its target style. If it passes everything, you may leave it as-is or tighten the wording \
further if a sharper version exists.

1. Echo check: does it just restate/paraphrase the raw scene description instead of being \
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
until the two are unmistakably distinct."""


def _build_critique_system_prompt(styles: list[str]) -> str:
    style_blocks = "\n\n".join(_format_style_block(s) for s in styles)
    keys = ", ".join(f'"{s}"' for s in styles)
    return f"""You are a ruthless caption editor. You will be given a grounded scene \
description, the target style definitions, and a DRAFT set of captions. Your job is to \
critique and, where needed, rewrite the drafts -- you are not just approving them.

{_CRITIQUE_CHECKLIST}

Style definitions (for reference while checking style-fit):

{style_blocks}

Respond with ONLY a single JSON object mapping each style name to its FINAL caption \
string (the rewritten version where needed, or the original where it already passed), \
no prose before or after, no code fences.

Output JSON shape: {{{", ".join(f'"{s}": "..."' for s in styles)}}}"""


def _build_critique_user_prompt(description: dict, draft_captions: dict) -> str:
    return (
        "Grounded scene description (JSON):\n"
        f"{json.dumps(description, ensure_ascii=False)}\n\n"
        "Draft captions to critique and, where needed, rewrite (JSON):\n"
        f"{json.dumps(draft_captions, ensure_ascii=False)}\n\n"
        "Return the final JSON object now."
    )


def fallback_caption(style: str, description: dict) -> str:
    """Safe, grounded, always-non-empty caption synthesized without another
    model call. Used when generation + one regeneration attempt both fail
    for a given style. This is an emergency safety net, not the quality
    path -- it's built fresh from live Stage-A fields each time, never a
    hardcoded joke.
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


def _call_text_model(client: FireworksClient, messages: list[dict], config, temperature: float, task_id: str, stage_label: str) -> dict:
    try:
        raw_text = client.chat_completion(
            messages=messages,
            model=config.text_model,
            max_tokens=config.text_max_tokens,
            temperature=temperature,
        )
    except FireworksError as exc:
        logger.error("[%s] stage B %s call failed: %s", task_id, stage_label, exc)
        return {}

    parsed = extract_json_object(raw_text)
    if parsed is None:
        logger.warning("[%s] stage B %s returned unparseable JSON, raw head: %r",
                        task_id, stage_label, raw_text[:200] if raw_text else raw_text)
        return {}
    return parsed


def _draft_captions(client: FireworksClient, description: dict, styles: list[str], config, task_id: str) -> dict:
    messages = [
        {"role": "system", "content": _build_draft_system_prompt(styles)},
        {"role": "user", "content": _build_user_prompt(description)},
    ]
    return _call_text_model(client, messages, config, config.text_temperature, task_id, "draft")


def _critique_captions(client: FireworksClient, description: dict, styles: list[str], draft_captions: dict, config, task_id: str) -> dict:
    messages = [
        {"role": "system", "content": _build_critique_system_prompt(styles)},
        {"role": "user", "content": _build_critique_user_prompt(description, draft_captions)},
    ]
    return _call_text_model(client, messages, config, config.text_critique_temperature, task_id, "critique")


def _generate_single_style(client: FireworksClient, description: dict, style: str, config, task_id: str) -> str | None:
    """Used only for the rare missing-style-after-batch case; a single draft
    call without a separate critique pass, since this is already a fallback
    path and not the primary quality path.
    """
    raw = _draft_captions(client, description, [style], config, task_id)
    value = raw.get(style)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def get_stage_b_captions(client: FireworksClient, description: dict, styles: list[str], config, task_id: str = "") -> dict:
    """Returns {style: caption} for every style in `styles`. Draft then
    critique-and-revise, then never returns a missing/empty caption for any
    requested style: regenerates a single style once on failure, then falls
    back to a templated grounded caption.
    """
    draft = _draft_captions(client, description, styles, config, task_id)

    final = _critique_captions(client, description, styles, draft, config, task_id) if draft else {}

    captions: dict[str, str] = {}
    for style in styles:
        value = final.get(style) or draft.get(style)
        if isinstance(value, str) and value.strip():
            captions[style] = value.strip()

    missing = [s for s in styles if s not in captions]
    for style in missing:
        logger.warning("[%s] style '%s' missing after draft+critique, regenerating alone", task_id, style)
        retried = _generate_single_style(client, description, style, config, task_id)
        if retried:
            captions[style] = retried
        else:
            logger.error("[%s] style '%s' still missing after retry, using template fallback", task_id, style)
            captions[style] = fallback_caption(style, description)

    return captions
