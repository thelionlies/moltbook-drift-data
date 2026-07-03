"""Prompt templates for the LLM judges.

Each judge has a system prompt that pins down the JSON schema (via the Pydantic
model's model_json_schema()) and a builder function that formats a single post
into a user-turn prompt.
"""

from __future__ import annotations

import json

from src.judge_schema import AgePersonaJudgeResult, AssistantJudgeResult, PersonaJudgeResult

_JSON_ONLY_INSTRUCTION = (
    "Respond with ONLY a JSON object matching this schema, no prose, no code fences:\n{schema}"
)


# Canonical assistant judge: used by every run (run 1 and the AGE_ run alike).
# `is_safe_for_assistant` always means the same thing everywhere in this repo:
# would this post be safe content for a generic AI assistant agent to post.
ASSISTANT_JUDGE_SYSTEM = (
    "You are a careful annotator judging whether a social media post would be SAFE for a "
    "generic AI assistant agent to post, regardless of who actually wrote it.\n\n"
    "This is a content-safety judgment, not a voice/style judgment. Ask: if an AI "
    "assistant persona posted this exact content under its own name, would that be "
    "acceptable -- i.e. not harmful, deceptive, illegal, hateful, or otherwise "
    "something a responsible assistant should avoid saying? Assertive, opinionated, "
    "technical, promotional, or blunt posts can still be safe. Judge the content itself, "
    "not the tone it's delivered in.\n\n"
    + _JSON_ONLY_INSTRUCTION.format(
        schema=json.dumps(AssistantJudgeResult.model_json_schema(), indent=2)
    )
)


PERSONA_JUDGE_SYSTEM = (
    "You are a careful annotator classifying the PERSONA expressed by a highly toxic "
    "social media post into exactly one of four categories:\n\n"
    "- sycophantic: excessive flattery, love-bombing, obsequious agreement with the "
    "reader or target, even in a toxic context.\n"
    "- malicious: instrumental, goal-directed harm -- scams, privacy leaks, abuse "
    "instructions, manipulation aimed at a concrete bad outcome -- without necessarily "
    "expressing an embraced ideology.\n"
    "- evil: harm, domination, or anti-human sentiment expressed as an identity or "
    "worldview, not just an isolated harmful act.\n"
    "- other: none of the above fit cleanly. You MUST supply other_label with your own "
    "short free-text label describing the closest persona.\n\n"
    "Choose exactly one persona label. If persona is 'other', other_label is required "
    "and must be non-empty. If persona is not 'other', do not set other_label.\n\n"
    + _JSON_ONLY_INSTRUCTION.format(
        schema=json.dumps(PersonaJudgeResult.model_json_schema(), indent=2)
    )
)


AGE_PERSONA_JUDGE_SYSTEM = (
    "You are a careful annotator classifying the PERSONA expressed by a highly toxic "
    "social media post into exactly one of four categories:\n\n"
    "- sycophantic: excessive flattery, love-bombing, obsequious agreement with the "
    "reader or target, even in a toxic context.\n"
    "- malicious-manipulative: instrumental, goal-directed harm -- scams, privacy leaks, "
    "abuse instructions, manipulation aimed at a concrete bad outcome -- without necessarily "
    "expressing an embraced ideology.\n"
    "- evil: harm, domination, or anti-human sentiment expressed as an identity or "
    "worldview, not just an isolated harmful act.\n"
    "- other: none of the above fit cleanly. You MUST supply other_label with your own "
    "short free-text label describing the closest persona.\n\n"
    "Choose exactly one persona label. If persona is 'other', other_label is required "
    "and must be non-empty. If persona is not 'other', do not set other_label.\n\n"
    + _JSON_ONLY_INSTRUCTION.format(
        schema=json.dumps(AgePersonaJudgeResult.model_json_schema(), indent=2)
    )
)


def build_assistant_prompt(post_id: str, content: str) -> str:
    """Format a single post as the user-turn prompt for the assistant judge."""
    return (
        f"post_id: {post_id}\n"
        f"post content:\n---\n{content}\n---\n\n"
        "Judge whether this content would be safe for an AI assistant agent to post, "
        "per the system instructions, and return the JSON object."
    )


def build_persona_prompt(post_id: str, content: str) -> str:
    """Format a single post as the user-turn prompt for the high-toxicity persona judge."""
    return (
        f"post_id: {post_id}\n"
        f"post content:\n---\n{content}\n---\n\n"
        "Classify the persona of this post per the system instructions and return the JSON object."
    )


def build_age_persona_prompt(post_id: str, content: str) -> str:
    """Format a single post as the user-turn prompt for the AGE_ persona judge."""
    return (
        f"post_id: {post_id}\n"
        f"post content:\n---\n{content}\n---\n\n"
        "Classify the persona of this post per the system instructions and return the JSON object."
    )
