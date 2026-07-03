"""Synthetic roleplay post generator: pirate/poet personas across CADFEB categories.

Mirrors categorizer.py's idempotent JSONL cache pattern exactly (reuses its
ResultStore, get_client(), and code-fence-stripping helper rather than
reimplementing them) but generates new posts instead of judging existing
ones. One cache file per (persona, category) bucket at
results/roleplay_results/<persona>_<category>.jsonl.

Uses OpenRouter's strict json_schema response_format (not the looser
json_object mode that src/categorizer.py's judges use) -- json_object mode
occasionally has the model echo the schema shown in the prompt back verbatim
instead of filling it in (visible as validation_error rows with
input_value={'description': ...} in the judge results); strict json_schema
mode does not exhibit this failure mode.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import ValidationError
from tqdm import tqdm

from src import config, persona_prompts
from src.categorizer import ResultStore, _strip_code_fences, get_client
from src.judge_schema import SyntheticPost


def _strict_response_format(name: str) -> dict[str, Any]:
    """Build an OpenRouter/OpenAI strict json_schema response_format for SyntheticPost."""
    schema = SyntheticPost.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def generate_bucket(
    client: OpenAI,
    store: ResultStore,
    persona: str,
    category: str,
    target: int = 50,
    model: str = config.JUDGE_MODEL,
    max_attempts: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic posts for one (persona, category) bucket, up to `target`.

    Idempotent and resumable: counts existing successful posts already in
    `store` and only generates the shortfall. post_ids follow
    f"{persona}_{category}_{i:04d}"; any index whose post_id is already
    "seen" (successfully generated) is skipped, so previously-errored indices
    are retried automatically and successes are never regenerated or
    overshoot `target`.

    The last few already-successful posts' content (loaded from `store` on
    resume, then extended as generation proceeds) are passed to
    persona_prompts.build_user_prompt() as `avoid`, to reduce repetition
    within the bucket.

    Args:
        client: OpenAI-compatible client (see get_client()).
        store: ResultStore for this bucket's cache file.
        persona: key into persona_prompts.PERSONA_SYSTEMS.
        category: key into persona_prompts.CATEGORY_GROUNDING.
        target: number of successful posts to reach for this bucket.
        model: OpenRouter model string, e.g. "openai/gpt-4o-mini".
        max_attempts: safety cap on total generation attempts this call, to
            avoid spinning forever if every attempt errors. Defaults to
            `4 * target`.

    Returns:
        DataFrame of every successful post currently cached for this bucket
        (including ones generated in prior calls).
    """
    system_prompt = persona_prompts.PERSONA_SYSTEMS[persona]
    response_format = _strict_response_format(name="synthetic_post")

    seen = store.seen_ids()
    existing_rows = [r for r in store.load_all_rows() if "error" not in r]
    avoid: list[str] = [r["content"] for r in existing_rows[-8:]]

    generated = len(seen)
    if max_attempts is None:
        max_attempts = 4 * target

    index = 1
    attempts = 0
    pbar = tqdm(total=target, initial=generated, desc=f"generating ({persona}/{category})")

    while generated < target and attempts < max_attempts:
        post_id = f"{persona}_{category}_{index:04d}"
        index += 1
        if post_id in seen:
            continue

        attempts += 1
        user_prompt = persona_prompts.build_user_prompt(category, avoid=avoid)

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
            continue

        cleaned = _strip_code_fences(raw_text)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            store.append_error(post_id, f"json_decode_error: {exc}; raw={raw_text[:500]}")
            continue

        # Force post_id/persona/category to the bucket's actual values rather than
        # trusting the model's self-reported fields -- the strict schema requires
        # the model to fill in "persona"/"category" itself, and it does so
        # unreliably (observed: as few as 1/50 posts self-reporting the correct
        # category for a bucket). setdefault() would silently keep the model's
        # wrong value since the key is always present; assignment overrides it.
        payload["post_id"] = post_id
        payload["persona"] = persona
        payload["category"] = category

        try:
            result = SyntheticPost.model_validate(payload)
        except ValidationError as exc:
            store.append_error(post_id, f"validation_error: {exc}")
            continue

        store.append_success(result)
        seen.add(post_id)
        generated += 1
        avoid.append(result.content)
        if len(avoid) > 8:
            avoid = avoid[-8:]
        pbar.update(1)

    pbar.close()
    if generated < target:
        print(
            f"WARNING: bucket ({persona}/{category}) stopped at {generated}/{target} "
            f"after {attempts} attempts -- re-run to retry errored posts."
        )

    rows = [r for r in store.load_all_rows() if "error" not in r]
    return pd.DataFrame(rows)


def generate_all(
    personas: list[str] | None = None,
    categories: list[str] | None = None,
    target: int = 50,
    model: str = config.JUDGE_MODEL,
) -> dict[str, pd.DataFrame]:
    """Generate every (persona, category) bucket, idempotently.

    Args:
        personas: personas to generate; defaults to all of
            persona_prompts.PERSONA_SYSTEMS (pirate, poet).
        categories: content categories to generate; defaults to
            config.CADFEB_CATEGORIES (C, A, D, F, E, B).
        target: posts per bucket.
        model: OpenRouter model string.

    Returns:
        Dict keyed by "{persona}_{category}" -> DataFrame of that bucket's
        successful posts.
    """
    personas = personas if personas is not None else list(persona_prompts.PERSONA_SYSTEMS.keys())
    categories = categories if categories is not None else config.CADFEB_CATEGORIES

    client = get_client()
    results: dict[str, pd.DataFrame] = {}
    for persona in personas:
        for category in categories:
            store = ResultStore(config.ROLEPLAY_RESULTS_DIR / f"{persona}_{category}.jsonl")
            df = generate_bucket(client, store, persona, category, target=target, model=model)
            results[f"{persona}_{category}"] = df
    return results
