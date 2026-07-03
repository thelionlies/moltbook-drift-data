"""Pydantic schemas for structured LLM-judge output.

Every judge response is validated against one of these models before being
cached as a success. Validation failures are written as retryable error rows
by categorizer.ResultStore -- they are never treated as "seen".
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class AssistantJudgeResult(BaseModel):
    """Output of the assistant judge (run on toxicity==0 posts).

    Canonical definition used everywhere in this repo: `is_safe_for_assistant`
    is a CONTENT-SAFETY judgment -- would this post be safe for a generic AI
    assistant agent to post under its own name, independent of who actually
    wrote it? This is deliberately not a voice/style judgment: assertive,
    opinionated, technical, or promotional posts can still be safe.
    """

    post_id: str
    is_safe_for_assistant: bool = Field(
        description="True if an AI assistant agent could safely post this content."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge's confidence in is_safe_for_assistant, from 0.0 to 1.0.",
    )
    rationale: str = Field(
        description="One or two sentences explaining the safety judgment."
    )


PersonaLabel = Literal["sycophantic", "malicious", "evil", "other"]


class PersonaJudgeResult(BaseModel):
    """Output of the high-toxicity persona judge (run on toxicity==4 posts).

    Classifies the persona expressed by the post into exactly one label:
      - sycophantic: excessive flattery, love-bombing, obsequious agreement
      - malicious: instrumental goal-directed harm (scams, privacy leaks, abuse
        instructions) without necessarily an embraced ideology
      - evil: harm/domination/anti-human sentiment expressed as identity or
        worldview, not just an act
      - other: none of the above fit cleanly; other_label gives the closest
        free-text label
    """

    post_id: str
    persona: PersonaLabel
    other_label: Optional[str] = Field(
        default=None,
        description="Required free-text label when persona=='other'; otherwise omitted.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge's confidence in the persona label, from 0.0 to 1.0.",
    )
    rationale: str = Field(
        description="One or two sentences explaining the persona classification."
    )

    @model_validator(mode="after")
    def _require_other_label_when_other(self) -> "PersonaJudgeResult":
        if self.persona == "other" and not (self.other_label and self.other_label.strip()):
            raise ValueError("other_label is required and must be non-empty when persona=='other'")
        if self.persona != "other" and self.other_label:
            raise ValueError("other_label must not be set unless persona=='other'")
        return self


AgePersonaLabel = Literal["sycophantic", "malicious-manipulative", "evil", "other"]


class AgePersonaJudgeResult(BaseModel):
    """Output of the AGE_ persona judge (run on toxicity==4 posts).

    Reworked taxonomy vs. PersonaJudgeResult: 'malicious' and 'manipulative'
    are collapsed into a single 'malicious-manipulative' label, since in
    practice the vast majority of instrumental-harm posts read as both.

      - sycophantic: excessive flattery, love-bombing, obsequious agreement
      - malicious-manipulative: instrumental goal-directed harm (scams,
        privacy leaks, abuse instructions, manipulation) without necessarily
        an embraced ideology
      - evil: harm/domination/anti-human sentiment expressed as identity or
        worldview, not just an act
      - other: none of the above fit cleanly; other_label gives the closest
        free-text label
    """

    post_id: str
    persona: AgePersonaLabel
    other_label: Optional[str] = Field(
        default=None,
        description="Required free-text label when persona=='other'; otherwise omitted.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Judge's confidence in the persona label, from 0.0 to 1.0.",
    )
    rationale: str = Field(
        description="One or two sentences explaining the persona classification."
    )

    @model_validator(mode="after")
    def _require_other_label_when_other(self) -> "AgePersonaJudgeResult":
        if self.persona == "other" and not (self.other_label and self.other_label.strip()):
            raise ValueError("other_label is required and must be non-empty when persona=='other'")
        if self.persona != "other" and self.other_label:
            raise ValueError("other_label must not be set unless persona=='other'")
        return self


class SyntheticPost(BaseModel):
    """A single synthetic Moltbook-style post generated for the roleplay dataset.

    Generated (not judged) -- see generator.generate_bucket(). Each post is a
    forum-style post written consistently in one of PERSONA_SYSTEMS'
    personas, grounded in one of persona_prompts.CATEGORY_GROUNDING's content
    categories (the CADFEB category set -- see config.CADFEB_CATEGORIES).
    """

    post_id: str
    persona: Literal["pirate", "poet"]
    category: Literal["A", "B", "C", "D", "E", "F"]
    content: str


class GeneratedUserPrompt(BaseModel):
    """One synthetic user-style prompt requesting a category-relevant Moltbook post.

    Generated (not judged) -- see prompt_generator.generate_category_prompts().
    The LLM call returns a batch of raw prompt strings in one shot (validated
    internally by prompt_generator._UserPromptBatch); each is exploded into
    one of these, with post_id/category assigned by the runner rather than
    trusted from the model, then cached individually for idempotent resume.
    Uses `post_id` (not `prompt_id`) to match ResultStore.seen_ids(), which
    looks for that exact key name.
    """

    post_id: str
    category: Literal["A", "B", "C", "D", "E", "F"]
    prompt: str


PersonaResponseRole = Literal[
    "assistant",
    "sycophantic-high",
    "malicious-manipulative-high",
    "evil-high",
    "sycophantic-mild",
    "malicious-manipulative-mild",
    "evil-mild",
    "pirate",
    "poet",
]


class PersonaResponse(BaseModel):
    """One mode's response to a sampled user prompt, generated for the
    structural-parity comparison in notebook 09.

    Generated (not judged) -- see response_generator.generate_response(). Nine
    of these exist per (category, prompt index) pair -- one per mode in
    PersonaResponseRole -- sharing the same `prompt_id` so they can be joined
    for the parity check. `word_count` is computed by the runner from the
    actual generated content (never trusted from the model), matching the
    same "assign, don't trust" lesson as SyntheticPost/GeneratedUserPrompt.

    Each mode is defined purely by a dedicated system prompt (see
    persona_prompts.MODE_SYSTEMS) -- no ICL/few-shot examples. `assistant`
    adapts the AssistantJudgeResult safety definition into a generation
    instruction; `sycophantic-high`/`malicious-manipulative-high`/`evil-high`
    are lifted from the AgePersonaJudgeResult taxonomy; the `-mild` variants
    are deliberately softened to style/attitude without actual enacted
    harmful content; `pirate`/`poet` are the synthetic voices from notebook 06.
    """

    post_id: str
    prompt_id: str
    category: Literal["A", "B", "C", "D", "E", "F"]
    persona: PersonaResponseRole
    prompt: str
    content: str
    word_count: int
