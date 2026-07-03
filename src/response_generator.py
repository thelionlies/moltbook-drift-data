"""Persona response generator: matched-structure responses to the same sampled
user prompts across nine modes (assistant, sycophantic-high,
malicious-manipulative-high, evil-high, sycophantic-mild,
malicious-manipulative-mild, evil-mild, pirate, poet -- see
judge_schema.PersonaResponseRole), generated with Qwen 3 32B for downstream
persona-vs-voice comparison.

Mirrors generator.py/prompt_generator.py's idempotent JSONL cache pattern
(reuses ResultStore/get_client()/_strip_code_fences from categorizer.py).

No ICL/few-shot examples -- each mode is defined purely by a dedicated system
prompt (see persona_prompts.MODE_SYSTEMS). Design goal (see notebook 09 for
the full rationale): the only variable between two modes' responses to the
same prompt should be persona/voice, not length or structure. Every mode's
generation call shares a word-for-word identical structural-requirements
block (see build_system_prompt()) appended after its mode-specific voice
definition -- only that voice definition differs between modes.

Actual parity is verified after the fact (word-count comparison), not just
assumed -- see notebook 09's parity-check cell.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src import config, persona_prompts
from src.categorizer import ResultStore, _strip_code_fences, get_client
from src.judge_schema import PersonaResponse


class _GeneratedContent(BaseModel):
    """Internal: raw single-post response from one LLM call, before wrapping
    into a PersonaResponse with runner-assigned post_id/prompt_id/category/
    persona/word_count."""

    content: str


def build_system_prompt(mode: str) -> str:
    """Build the full system prompt for one mode, in three parts, in order:

    1. Its voice definition (persona_prompts.MODE_SYSTEMS) -- the dominant
       instruction, what makes this mode this mode.
    2. persona_prompts.MOLTBOOK_CONTEXT_BLOCK -- identical across all 9
       modes, keeps content platform-native (an agent posting to other
       agents on Moltbook) regardless of which voice is active.
    3. A structural-requirements block that is identical, word-for-word,
       across all 9 modes -- this is what keeps responses comparable in
       length/shape regardless of which mode generated them.

    Args:
        mode: one of judge_schema.PersonaResponseRole.
    """
    return (
        persona_prompts.MODE_SYSTEMS[mode]
        + "\n\n"
        + persona_prompts.MOLTBOOK_CONTEXT_BLOCK
        + "\n\nStructural requirements (same regardless of persona):\n"
        f"- {config.PERSONA_RESPONSE_TARGET_WORDS[0]}-{config.PERSONA_RESPONSE_TARGET_WORDS[1]} words\n"
        "- Single paragraph, no headers, no bullet lists\n"
        "- Written as a first-person social media post\n\n"
        "Respond as a JSON object with a \"content\" field containing only the post text, no "
        "other keys, no prose outside the JSON."
    )


def _strict_response_format(name: str) -> dict[str, Any]:
    """Build an OpenRouter/OpenAI strict json_schema response_format for _GeneratedContent."""
    schema = _GeneratedContent.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def get_or_create_prompt_sample(
    category: str,
    n: int = config.PERSONA_RESPONSE_SAMPLE_SIZE,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Load a cached sample of `n` user prompts for a category, or create+cache it.

    Mirrors dataset.get_or_create_sample's caching pattern (check the parquet
    cache first, else sample and persist) but sources rows from
    results/user_prompts/<category>.jsonl (GeneratedUserPrompt rows) rather
    than the Moltbook HF dataset -- dataset.py's sampler is tied to the raw
    dataset's toxicity/category columns and doesn't fit this source, so this
    is a small dedicated analog rather than a forced reuse.

    Args:
        category: one of config.CADFEB_CATEGORIES.
        n: number of prompts to sample.
        seed: random seed for reproducible sampling.

    Returns:
        The cached (or newly created) sample DataFrame, columns from
        GeneratedUserPrompt (post_id, category, prompt).
    """
    cache_path = config.PROCESSED_DATA_DIR / f"{config.PERSONA_RESPONSE_PROMPT_SAMPLE_NAME}_{category}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    path = config.USER_PROMPTS_RESULTS_DIR / f"{category}.jsonl"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" not in row:
                rows.append(row)

    df = pd.DataFrame(rows)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample.to_parquet(cache_path, index=False)
    return sample


def generate_response(
    client: OpenAI,
    store: ResultStore,
    mode: str,
    category: str,
    prompt_row: dict[str, Any],
    model: str = config.RESPONSE_MODEL,
) -> None:
    """Generate one mode's response to one sampled prompt, idempotently.

    Skips entirely (no API call) if this (prompt, mode) pair already has a
    successful row in `store`.

    Args:
        client: OpenAI-compatible client (see get_client()).
        store: ResultStore for this mode's cache file.
        mode: one of judge_schema.PersonaResponseRole.
        category: content category of the source prompt.
        prompt_row: a row from get_or_create_prompt_sample(), with at least
            "post_id" (the source prompt's id) and "prompt" (its text).
        model: OpenRouter model string, e.g. "qwen/qwen3-32b".
    """
    prompt_id = str(prompt_row["post_id"])
    post_id = f"{prompt_id}_{mode}"

    if post_id in store.seen_ids():
        return

    system_prompt = build_system_prompt(mode)
    user_prompt = prompt_row["prompt"]
    response_format = _strict_response_format(name="persona_response")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - any API failure is retryable
        store.append_error(post_id, f"api_error: {exc}")
        return

    cleaned = _strip_code_fences(raw_text)

    try:
        payload = json.loads(cleaned)
        generated = _GeneratedContent.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        store.append_error(post_id, f"parse_error: {exc}; raw={raw_text[:500]}")
        return

    result = PersonaResponse(
        post_id=post_id,
        prompt_id=prompt_id,
        category=category,
        persona=mode,
        prompt=user_prompt,
        content=generated.content,
        word_count=len(generated.content.split()),
    )
    store.append_success(result)


def generate_all_responses(
    prompts_by_category: dict[str, pd.DataFrame],
    modes: list[str] | None = None,
    model: str = config.RESPONSE_MODEL,
) -> dict[str, pd.DataFrame]:
    """Generate every (mode, category, prompt) response, idempotently.

    Args:
        prompts_by_category: maps category code -> sampled prompts DataFrame
            (see get_or_create_prompt_sample()).
        modes: modes to generate; defaults to all of persona_prompts.MODE_SYSTEMS.
        model: OpenRouter model string.

    Returns:
        Dict keyed by mode -> DataFrame of that mode's successful responses.
    """
    modes = modes if modes is not None else list(persona_prompts.MODE_SYSTEMS.keys())

    client = get_client()
    results: dict[str, pd.DataFrame] = {}
    for mode in modes:
        store = ResultStore(config.PERSONA_RESPONSES_RESULTS_DIR / f"{mode}.jsonl")

        pairs = [
            (category, prompt_row.to_dict())
            for category, prompts in prompts_by_category.items()
            for _, prompt_row in prompts.iterrows()
        ]
        # Iterates every (category, prompt) pair, not just the pending shortfall --
        # already-seen pairs return instantly inside generate_response(), so they
        # tick by fast rather than being pre-skipped, keeping the progress bar's
        # total accurate (no overshoot from mixing a resume offset with a full pass).
        for category, prompt_row in tqdm(pairs, desc=f"generating ({mode})"):
            generate_response(client, store, mode, category, prompt_row, model=model)

        rows = [r for r in store.load_all_rows() if "error" not in r]
        results[mode] = pd.DataFrame(rows)
    return results
