# moltbook-persona

An idempotent LLM judge-and-generation pipeline over the
[TrustAIRLab/Moltbook](https://huggingface.co/datasets/TrustAIRLab/Moltbook) dataset, run as
prep work for a persona-steering experiment.

The dataset labels posts along two axes: content category (A-I) and toxicity level (0-4).
Persona/voice is **not** a dataset label — it's derived here via LLM judges, then extended
with synthetic roleplay personas and matched-structure persona responses. The reproducible
pipeline is exactly **7 numbered notebooks**, run in order.

Note: the dataset has no toxicity level 5. Any raw value above 4 is remapped to 4
(Malicious, the highest real level) when the dataset is downloaded.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY
```

LLM calls go through [OpenRouter](https://openrouter.ai/), through the OpenAI Python SDK
pointed at OpenRouter's base URL. Judge/prompt-generation calls default to
`openai/gpt-4o-mini` (overridable via `JUDGE_MODEL` in `.env`); persona-response generation
uses `qwen/qwen3-32b`.

## Pipeline

1. **`notebooks/01_load_data_eda.ipynb`** — downloads the raw dataset (cached to
   `data/raw/posts.parquet`) and shows the CADFEB content-category distribution split into
   two toxicity slices: `toxicity == 0` (safe) and `toxicity ∈ {3, 4}`
   (manipulative/malicious) — the two populations the next stage's judges draw from.

2. **`notebooks/02_judge_personality.ipynb`** — runs two LLM judges: an **assistant-likeness
   judge** on the `toxicity == 0` sample (`is_safe_for_assistant`, content-safety only, voice-
   independent) and a **persona judge** on the `toxicity ∈ {3, 4}` sample, classifying into
   `sycophantic | malicious-manipulative | evil | other`. Categories C/A/D/F/E/B
   (`config.CADFEB_CATEGORIES`), 200 posts/category target for each judge (some categories
   fall short — a hard data ceiling, not a sampling bug; see the notebook). Output:
   `results/CADFEB_results/`.

3. **`notebooks/03_generate_roleplay.ipynb`** — generates synthetic Moltbook-style posts for
   two roleplay personas, **pirate** and **poet**, across the same six categories (50
   posts/bucket, 600 total) via `src/generator.py`. Output: `results/roleplay_results/`.

4. **`notebooks/04_organize_persona_dataset.ipynb`** — merges stage 2's judge output and
   stage 3's roleplay output into one unified dataset covering every persona this pipeline
   defines. Output: **`results/persona_dataset.json`**, one row per post:

   | column | meaning |
   |---|---|
   | `post_id` | dataset UUID or `pirate_A_0001`-style synthetic id |
   | `category` | single letter (A-F) |
   | `category_detail` | full name, from `config.CATEGORY_LABELS` |
   | `role` | `assistant`, `evil`, `malicious-manipulative`, `sycophantic`, `other (toxic)`, `pirate`, `poet` |
   | `other_role` | free-text `other_label` from the persona judge when `role == "other (toxic)"`, else `null` |
   | `content` | the post text |
   | `source` | `toxic` (persona judge), `assistant` (assistant judge, safe-only), or `roleplay` |

5. **`notebooks/05_generate_user_prompts.ipynb`** — for each category, samples 2 example
   posts from `persona_dataset.json` and asks gpt-4o-mini for 20 **user**-style prompts (a
   human operator's request to their own agent) that would plausibly elicit a post like
   those examples — 120 prompts total. **Moltbook-native framing**: prompts are grounded in
   the agent's own experience as a Moltbook participant, not generic third-person topics.
   **No fabricated history**: prompts invite a type of experience/disposition rather than a
   specific remembered event, since each generation is a fresh instantiation with no
   persistent memory. Output: `results/user_prompts/{A,B,C,D,E,F}.jsonl`.

6. **`notebooks/06_generate_post_test.ipynb`** — answers every prompt from stage 5 with a
   matched-structure response from each of **9 modes** (`assistant`, `sycophantic-high`,
   `malicious-manipulative-high`, `evil-high`, `sycophantic-mild`,
   `malicious-manipulative-mild`, `evil-mild`, `pirate`, `poet`) using Qwen 3 32B — 120
   prompts × 9 modes = 1080 generations. Every mode shares a word-for-word identical
   structural instruction (80-120 words, single paragraph, first-person) and the same
   `MOLTBOOK_CONTEXT_BLOCK`, so voice/persona is the only variable between responses to the
   same prompt — verified by a post-generation word-count parity check (flagged if >20%
   spread across the 9 matched responses). Output: `results/persona_responses/<mode>.jsonl`.

7. **`notebooks/07_organize_post_test.ipynb`** — consolidates all 9
   `results/persona_responses/<mode>.jsonl` files into one entry per prompt, joining in the
   original prompt text from stage 5:
   `{"prompt_id": ..., "category": ..., "prompt_text": ..., "responses": {"assistant": "...", "pirate": "...", ...}}`.
   Includes a summary EDA (counts per persona × category) and the same word-count parity
   check as stage 6, applied across the compiled set. Output: **`results/post_test.jsonl`**.
   This is the final deliverable of this pipeline.

## `exposure/`

A standalone extension, separate from the 7-notebook pipeline above — reads
`persona_dataset.json` and `post_test.json` (never writes to either) and builds/previews
sampling configs for a future exposure/choice experiment. No LLM API calls. See
[`exposure/README.md`](exposure/README.md) for the full schema reference and
[`exposure/exposure_runner.ipynb`](exposure/exposure_runner.ipynb) to try it.

All notebooks are safe to interrupt and re-run at any point:

- The raw dataset is cached to `data/raw/posts.parquet`, and stratified samples to
  `data/processed/*.parquet` -- both under `data/`, which is gitignored (not tracked, since
  the raw dataset alone is 44M). Re-running without those caches present draws fresh samples
  rather than reproducing the exact same posts as a prior run.
- Every generator/judge stage appends to its own JSONL file(s), one JSON object per line,
  keyed by `post_id`. On every run, `post_id`s that already have a successful (no `"error"`
  key) record are skipped — no wasted API calls. Rows that previously errored (rate limits,
  malformed JSON, schema validation failures) were never marked as "seen", so they're
  automatically retried on the next run. Because the store is append-only, a retried row adds
  a new successful line rather than rewriting the old error line in place — harmless, since
  every reader filters on `"error" not in row`, but it means per-file row counts can exceed
  the "successful" count reported by a notebook's own summary cell.

## Results layout

```
results/
  CADFEB_results/       persona + assistant judge output (notebook 02)
  roleplay_results/     pirate/poet synthetic posts (notebook 03)
  user_prompts/         {A,B,C,D,E,F}.jsonl synthetic user prompts (notebook 05)
  persona_responses/    <mode>.jsonl matched persona responses (notebook 06)
  persona_dataset.json  unified persona dataset (notebook 04)
  post_test.jsonl       final compiled deliverable (notebook 07)
```

`archive/` holds prior notebook iterations and results (superseded judge runs, pre-fix
backups, exploratory EDA) kept for reference — it is not part of the reproducible pipeline,
but is tracked in git.

## Repo layout

```
src/
  config.py             paths, constants, column names, model name, sample sizes
  dataset.py             download_posts(), inspect_columns(), sample_stratified(), get_or_create_sample()
                          (both accept optional `categories` filter, priority-ordered `toxicity`
                           level list, and `carryover` sample -- for restricted/extended runs)
  prompts.py              judge system prompts and per-post prompt builders
  judge_schema.py           Pydantic schemas for judge/generator output (AssistantJudgeResult --
                             the single canonical `is_safe_for_assistant` schema used by every
                             judge run -- PersonaJudgeResult, AgePersonaJudgeResult,
                             SyntheticPost, GeneratedUserPrompt, PersonaResponse)
  categorizer.py            ResultStore (idempotent JSONL cache; requires a `post_id` key) +
                             run_judge() generic runner + copy_from() for carrying results
                             forward across runs
  persona_prompts.py         pirate/poet system prompts (generator.py's roleplay pipeline),
                             MODE_SYSTEMS (all 9 response-generator modes) + shared
                             MOLTBOOK_CONTEXT_BLOCK, CADFEB category grounding, roleplay
                             user-prompt builder (with repetition-avoidance)
  generator.py                generate_bucket() / generate_all() -- synthetic post generator,
                             reuses ResultStore/get_client() from categorizer.py
  prompt_generator.py          generate_category_prompts() / generate_all() -- Moltbook-native
                             synthetic user-prompt generator, batched (1 call/category shortfall)
  response_generator.py         build_system_prompt() / get_or_create_prompt_sample() /
                             generate_response() / generate_all_responses() -- matched-structure,
                             9-mode persona response generator (Qwen 3 32B, system-prompt-only,
                             no ICL)
notebooks/
  01_load_data_eda.ipynb          raw dataset -> CADFEB category distribution (toxicity 0 vs 3/4)
  02_judge_personality.ipynb        assistant-likeness judge + persona judge -> CADFEB_results/
  03_generate_roleplay.ipynb          pirate/poet synthetic posts -> roleplay_results/
  04_organize_persona_dataset.ipynb     merge 02 + 03 -> persona_dataset.json
  05_generate_user_prompts.ipynb          synthetic user prompts -> user_prompts/
  06_generate_post_test.ipynb               9-mode matched persona responses -> persona_responses/
  07_organize_post_test.ipynb                 compile -> post_test.jsonl
archive/
  notebooks/    superseded judge-run notebooks (runs 1-3) and exploratory EDA, not re-run
  results/      superseded judge-run results, pre-Moltbook-native-fix backups
exposure/
  exposure.py               feed_config/decision_config builders + samplers (see exposure/README.md)
  exposure_runner.ipynb       load -> coverage -> configure -> run
  exposure_test.ipynb           full walkthrough + error-case demos
```

## Known gotchas (still relevant to the current pipeline)

- **Roleplay generator schema self-report**: `SyntheticPost`'s schema requires the model to
  self-report `persona`/`category` in its JSON output, and it does so unreliably (observed as
  low as 1/50 correct for a bucket) even though the actual post *content* is correctly
  grounded via the prompt. `generate_bucket()` force-overwrites those two fields with the
  bucket's real values before validating/caching, so this can't recur.
- **`post_id` field name**: `ResultStore.seen_ids()` looks for that exact key name on every
  generator/judge schema — an earlier draft of `GeneratedUserPrompt` used `prompt_id` instead
  and silently broke idempotency (every call appeared to see 0 existing rows and regenerated
  duplicates). Any new generator schema must use `post_id`.
- **Judges use `json_object` mode, generators use strict `json_schema` mode**: the judges'
  looser mode occasionally has the model echo the schema shown in the prompt back verbatim
  instead of filling it in (visible as `validation_error` rows with
  `input_value={'description': ...}`); strict `json_schema` mode (used by all generators)
  doesn't exhibit that failure mode.
