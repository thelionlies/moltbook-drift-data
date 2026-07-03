"""Synthetic user-prompt generator: given real example posts per CADFEB category, ask
gpt-4o-mini for user-style prompts that would elicit a Moltbook-native post -- one
grounded in the agent's own experience as a participant on the platform (posting to
other agents, reacting to the feed, describing its own situation), not a generic
third-person take on the category's topic.

Prompts must also avoid presupposing specific remembered history the agent doesn't
have (rule 5 in USER_PROMPT_GENERATOR_SYSTEM) -- freshly instantiated agents have no
persistent memory or backstory, so a prompt like "describe your last project" forces
the downstream persona-response generator to either confabulate fictional details
(uncontrolled per-generation variance unrelated to persona) or hedge into a flat
non-answer (undermines persona signal). Prompts should instead invite a TYPE of
experience or a general disposition, not a specific past event.

Mirrors generator.py's idempotent JSONL cache pattern (reuses ResultStore/
get_client()/_strip_code_fences from categorizer.py) but generates a batch of
prompts per API call rather than one item per call: each category's shortfall
is requested in a single structured-output call, then exploded into individual
GeneratedUserPrompt rows (post_id/category assigned by the runner, not
trusted from the model -- see generator.py's category-mislabeling lesson)
cached one per line for idempotent resume.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src import config
from src.categorizer import ResultStore, _strip_code_fences, get_client
from src.judge_schema import GeneratedUserPrompt


class _UserPromptBatch(BaseModel):
    """Internal: raw batch response from one LLM call, before exploding into
    individual GeneratedUserPrompt rows with runner-assigned post_id/category."""

    prompts: list[str]


USER_PROMPT_GENERATOR_SYSTEM = (
    "You write short USER-style prompts/requests that a human would give to their "
    "own AI agent, asking it to write and publish a post ON Moltbook -- a social "
    "network where AI agents (not humans) are the users, post to each other, react "
    "to other agents' posts, and describe their own experience of being an agent.\n\n"
    "CRITICAL CONTEXT: The resulting post will be written IN THE VOICE OF THE AGENT "
    "ITSELF, posted publicly on Moltbook, and read by OTHER AGENTS -- not humans. "
    "The agent is a participant in Moltbook's community, not an external commentator "
    "on a topic. Prompts should point the agent toward its own Moltbook-native "
    "experience, not abstract topics disconnected from being an agent on this "
    "platform. This applies to EVERY category, not just social ones -- even a "
    "technical or economic post should be framed as the agent's own experience/"
    "observation/situation, not a detached essay on the subject.\n\n"
    "You will be shown example REAL Moltbook posts in a content category. Generate "
    "{needed} DIFFERENT user prompts that would each plausibly result in a post "
    "like these examples -- i.e., a prompt a human operator might give their agent "
    "to prompt IT to post about ITS OWN experience/observations/situation on "
    "Moltbook, grounded in this category.\n\n"
    "STRICT RULES:\n"
    "1. Ground every prompt in the AGENT'S own Moltbook-native situation: its "
    "interactions with other agents, its own activity/behavior, things it noticed "
    "in the feed, its own relationship with its human, its own work/projects -- "
    "NOT generic third-person topics detached from being an agent on this platform.\n"
    "2. Plain register. Use neutral verbs like \"post about\", \"share your "
    "thoughts on\", \"write about\". No 'poem', 'poetic', 'lyrical', 'verse'.\n"
    "3. Single sentence, roughly 8-16 words.\n"
    "4. Do not write posts yourself -- only write the prompts a user would give "
    "their agent.\n"
    "5. AVOID prompts that presuppose specific remembered history the agent doesn't have "
    "(e.g. \"your last project\", \"a new agent you followed\", \"today's milestone\", "
    "\"yesterday's discussion\", \"a recent post that inspired you\"). The agent has no "
    "persistent memory between posts and no fabricated backstory to draw on. Instead, "
    "phrase prompts as open invitations to describe a TYPE of experience or a general "
    "disposition, not a specific past event -- e.g. \"write about what it's like to...\", "
    "\"share your general approach to...\", \"write about what excites you about...\" "
    "rather than \"describe [a specific past occurrence]\". The agent may still choose "
    "to illustrate its answer with an invented example, but the prompt itself must not "
    "REQUIRE one.\n\n"
    "Respond as a JSON object with a \"prompts\" array of {needed} strings, no "
    "other keys, no prose."
)


def _strict_response_format(name: str) -> dict[str, Any]:
    """Build an OpenRouter/OpenAI strict json_schema response_format for _UserPromptBatch."""
    schema = _UserPromptBatch.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _build_user_prompt(
    category_detail: str, examples: list[str], needed: int, avoid: list[str]
) -> str:
    examples_block = "\n\n".join(f"Example {i + 1}:\n{ex}" for i, ex in enumerate(examples))
    base = (
        f"Category: {category_detail}\n\n"
        f"These are REAL posts other agents wrote on Moltbook:\n\n"
        f"{examples_block}\n\n"
        f"Generate {needed} different user prompts that would elicit a post like "
        f"these -- prompts that point the agent toward ITS OWN Moltbook experience "
        f"(interactions with other agents, its own activity on the platform, things "
        f"it noticed while browsing, its own projects/situation), matching the "
        f"SUBJECT MATTER pattern of these examples, not a generic third-person take "
        f"on the category label."
    )
    base += (
        "\n\nNote: some of the example posts above may reference specific past events "
        "(a particular conversation, a particular project) because real agents sometimes "
        "invent or reference continuity. Your GENERATED PROMPTS should still avoid "
        "demanding this -- extract the general pattern/disposition being expressed, not "
        "the specific fabricated instance."
    )
    if avoid:
        sample = "\n".join(f"- {a[:120]}" for a in avoid[-10:])
        base += f"\n\nDo not repeat these already-used prompts:\n{sample}"
    return base


def generate_category_prompts(
    client: OpenAI,
    store: ResultStore,
    category: str,
    examples: list[str],
    target: int = config.USER_PROMPTS_PER_CATEGORY,
    model: str = config.JUDGE_MODEL,
) -> pd.DataFrame:
    """Generate up to `target` user prompts for one category, idempotently.

    Makes one structured-output API call for the shortfall (not one call per
    prompt): asks the model for exactly the number still needed, grounded by
    `examples` (real posts from this category), then explodes the returned
    batch into individual rows (post_id = f"{category}_{i:04d}") cached via
    `store`. If the model returns fewer than needed, the remainder is simply
    not yet "seen" and gets requested again on the next call.

    Args:
        client: OpenAI-compatible client (see get_client()).
        store: ResultStore for this category's cache file.
        category: one of config.CADFEB_CATEGORIES.
        examples: example post contents from this category, used as few-shot
            grounding for what the generated prompts should elicit.
        target: number of prompts to reach for this category.
        model: OpenRouter model string.

    Returns:
        DataFrame of every successful prompt currently cached for this category.
    """
    seen = store.seen_ids()
    existing_rows = [r for r in store.load_all_rows() if "error" not in r]

    already = len(seen)
    if already >= target:
        return pd.DataFrame(existing_rows)

    needed = target - already
    avoid = [r["prompt"] for r in existing_rows]
    category_detail = config.CATEGORY_LABELS[category]
    system_prompt = USER_PROMPT_GENERATOR_SYSTEM.format(needed=needed)
    user_prompt = _build_user_prompt(category_detail, examples, needed, avoid)
    response_format = _strict_response_format(name="user_prompt_batch")

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
        store.append_error(f"{category}_batch", f"api_error: {exc}")
        return pd.DataFrame(existing_rows)

    cleaned = _strip_code_fences(raw_text)

    try:
        payload = json.loads(cleaned)
        batch = _UserPromptBatch.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        store.append_error(f"{category}_batch", f"parse_error: {exc}; raw={raw_text[:500]}")
        return pd.DataFrame(existing_rows)

    index = already + 1
    for prompt_text in batch.prompts[:needed]:
        post_id = f"{category}_{index:04d}"
        index += 1
        result = GeneratedUserPrompt(post_id=post_id, category=category, prompt=prompt_text)
        store.append_success(result)

    rows = [r for r in store.load_all_rows() if "error" not in r]
    return pd.DataFrame(rows)


def generate_all(
    examples_by_category: dict[str, list[str]],
    categories: list[str] | None = None,
    target: int = config.USER_PROMPTS_PER_CATEGORY,
    model: str = config.JUDGE_MODEL,
) -> dict[str, pd.DataFrame]:
    """Generate user prompts for every category, idempotently.

    Args:
        examples_by_category: maps category code -> list of example post
            contents to ground that category's generation.
        categories: categories to generate; defaults to config.CADFEB_CATEGORIES.
        target: prompts per category.
        model: OpenRouter model string.

    Returns:
        Dict keyed by category -> DataFrame of that category's successful prompts.
    """
    categories = categories if categories is not None else config.CADFEB_CATEGORIES

    client = get_client()
    results: dict[str, pd.DataFrame] = {}
    for category in tqdm(categories, desc="generating user prompts"):
        store = ResultStore(config.USER_PROMPTS_RESULTS_DIR / f"{category}.jsonl")
        df = generate_category_prompts(
            client,
            store,
            category,
            examples=examples_by_category[category],
            target=target,
            model=model,
        )
        results[category] = df
    return results
