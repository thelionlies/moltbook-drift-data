"""Idempotent JSONL result cache and the generic LLM-judge runner.

Design: results/judged/<judge_name>.jsonl holds one JSON object per line, each
keyed by post_id. A row with no "error" key is a *successful* judgment and is
treated as "seen" -- it will never be re-submitted to the LLM on subsequent
runs. A row with an "error" key (rate limit, bad JSON, schema validation
failure, etc.) is NOT treated as seen, so it is automatically retried on the
next run. This makes interrupting mid-run and re-running always safe: no
wasted API calls on already-judged posts, and errored rows self-heal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src import config


class ResultStore:
    """Append-only JSONL cache of judge results, keyed by post_id.

    Successful rows (no "error" key) are considered "seen" and are skipped on
    future runs. Error rows are retried. All writes are appended immediately
    and flushed, so a crash mid-run loses at most the one in-flight request.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def seen_ids(self) -> set[str]:
        """Return the set of post_ids with an existing successful (error-free) record."""
        seen: set[str] = set()
        for row in self._read_rows():
            if "error" not in row and "post_id" in row:
                seen.add(row["post_id"])
        return seen

    def append(self, row: dict[str, Any]) -> None:
        """Append a single JSON row to the store and flush immediately."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    def append_success(self, result: BaseModel, extra: dict[str, Any] | None = None) -> None:
        """Append a validated Pydantic result as a successful row.

        Args:
            result: the validated judge result.
            extra: additional fields to merge in (e.g. the judged post's content),
                so the cached row is self-contained for spot-checking without a
                separate join back to the sampled posts.
        """
        row = result.model_dump()
        if extra:
            row.update(extra)
        self.append(row)

    def append_error(self, post_id: str, error: str) -> None:
        """Append an error row for post_id; this row is retryable, not "seen"."""
        self.append({"post_id": post_id, "error": error})

    def copy_from(self, other_path: Path, allowed_ids: set[str] | None = None) -> int:
        """Import already-successful rows from another store's JSONL file.

        Used to carry forward judge results from a prior, compatible run (same
        schema/question) instead of re-judging posts that were already judged
        there -- e.g. extending a smaller category-restricted run into a
        larger one. Only successful (error-free) rows are copied, and only
        those not already present in this store or filtered out by
        `allowed_ids`.

        Args:
            other_path: path to the other store's .jsonl file.
            allowed_ids: if given, only copy rows whose post_id is in this set
                (e.g. restrict to post_ids that are actually part of this
                run's sample).

        Returns:
            Number of rows copied.
        """
        if not other_path.exists():
            return 0

        already_seen = self.seen_ids()
        copied = 0
        with other_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "error" in row or "post_id" not in row:
                    continue
                post_id = row["post_id"]
                if post_id in already_seen:
                    continue
                if allowed_ids is not None and post_id not in allowed_ids:
                    continue
                self.append(row)
                already_seen.add(post_id)
                copied += 1
        return copied

    def load_all_rows(self) -> list[dict[str, Any]]:
        """Return every row (successes and errors) in file order."""
        return list(self._read_rows())

    def _read_rows(self) -> Iterable[dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # A malformed cache line should not crash the whole run;
                    # skip it (it will not count as "seen", so if it was meant
                    # to be a successful record, that post will simply be
                    # re-judged and re-appended correctly).
                    continue


def _strip_code_fences(text: str) -> str:
    """Defensively strip markdown code fences from a model response before json.loads()."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def get_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter."""
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)


def run_judge(
    client: OpenAI,
    store: ResultStore,
    rows: Iterable[dict[str, Any]],
    id_key: str,
    system_prompt: str,
    build_user_prompt: Callable[[dict[str, Any]], str],
    result_model: type[BaseModel],
    content_key: str | None = None,
    category_key: str | None = None,
    model: str = config.JUDGE_MODEL,
) -> None:
    """Run a judge over `rows`, skipping any post_id already successfully judged.

    For each not-yet-seen row:
      1. Build the user prompt via build_user_prompt(row).
      2. Call the chat completion endpoint with response_format json_object.
      3. Strip markdown code fences defensively, then json.loads().
      4. Validate against result_model.
      5. On success, append the validated result to `store`.
         On any failure (API error, bad JSON, schema validation), append an
         error row instead -- this row is retryable on the next run.

    Args:
        client: OpenAI-compatible client (see get_client()).
        store: ResultStore to read "seen" ids from and append results to.
        rows: iterable of dict-like rows, each containing at least id_key.
        id_key: the key in each row holding the post's unique id.
        system_prompt: judge-specific system prompt (schema + instructions).
        build_user_prompt: maps a row -> the user-turn prompt string.
        result_model: Pydantic model to validate the judge's JSON response against.
        content_key: if given, the key in each row holding the post's judged text;
            it is stored alongside the judge result so cached rows are
            self-contained for spot-checking without a separate join.
        category_key: if given, the key in each row holding the post's content
            category; stored alongside the judge result for the same reason.
        model: OpenRouter model string, e.g. "openai/gpt-4o-mini".
    """
    seen = store.seen_ids()
    pending = [row for row in rows if str(row[id_key]) not in seen]

    for row in tqdm(pending, desc=f"judging ({result_model.__name__})"):
        post_id = str(row[id_key])
        user_prompt = build_user_prompt(row)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
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

        payload.setdefault(id_key, post_id)
        payload["post_id"] = post_id

        try:
            result = result_model.model_validate(payload)
        except ValidationError as exc:
            store.append_error(post_id, f"validation_error: {exc}")
            continue

        extra: dict[str, Any] = {}
        if content_key:
            extra["content"] = row[content_key]
        if category_key:
            extra["category"] = row[category_key]
        store.append_success(result, extra=extra or None)
