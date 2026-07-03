"""Central configuration: paths, dataset schema constants, model settings, sample sizes.

Column names below are a best guess from the TrustAIRLab/Moltbook dataset card and
MUST be verified against the actual dataset schema (see dataset.inspect_columns())
before trusting anything downstream. If the real column names differ, update the
constants in this file -- nothing else in the codebase should hardcode column names.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = REPO_ROOT / "results"
JUDGED_DIR = RESULTS_DIR / "judged"
# Separate namespace for the reworked AGE_* judges (new persona taxonomy +
# assistant-safety judge), kept apart from the original judged/ cache so
# neither run clobbers the other's idempotency state.
AGE_RESULTS_DIR = RESULTS_DIR / "AGE_results"
# Run 3: same idea, categories D/B/A/F (the four highest-volume categories at
# toxicity==4 -- see the category-count ranking that motivated this run).
DBAF_RESULTS_DIR = RESULTS_DIR / "DBAF_results"
# Run 4: categories C/A/D/F/E/B, 200/category. Supersedes/extends run 3 (DBAF)
# for the overlapping categories A/D/F/B.
CADFEB_RESULTS_DIR = RESULTS_DIR / "CADFEB_results"
# Synthetic roleplay post generator (pirate/poet personas x CADFEB categories)
# -- generated, not judged, so kept in its own sibling folder rather than
# under judged/ or one of the category-code batch folders above.
ROLEPLAY_RESULTS_DIR = RESULTS_DIR / "roleplay_results"
# Synthetic user-prompt generator (given 2 example posts per category, ask the
# model for user-style prompts that would elicit a category-relevant post) --
# also generated, not judged, own sibling folder.
USER_PROMPTS_RESULTS_DIR = RESULTS_DIR / "user_prompts"
# Persona response generator (notebook 09): for each sampled user prompt, one
# response per persona -- generated, not judged, own sibling folder. One file
# per persona (not per category), since ResultStore is keyed by post_id and
# a persona's 60 responses (6 categories x 10 prompts) all belong together.
PERSONA_RESPONSES_RESULTS_DIR = RESULTS_DIR / "persona_responses"

for _dir in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CADFEB_RESULTS_DIR,
    ROLEPLAY_RESULTS_DIR,
    USER_PROMPTS_RESULTS_DIR,
    PERSONA_RESPONSES_RESULTS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
# JUDGED_DIR / AGE_RESULTS_DIR / DBAF_RESULTS_DIR are archived-run paths -- no longer
# auto-created on import so they don't keep reappearing (empty) in results/. They're
# still read directly where needed (e.g. 02_judge_personality.ipynb's copy_from()), which
# doesn't require the directory to pre-exist.

# --------------------------------------------------------------------------
# Dataset identity
# --------------------------------------------------------------------------

HF_DATASET_NAME = "TrustAIRLab/Moltbook"
# The dataset ships two configs ("posts" and "submolts"); we want the posts table.
HF_DATASET_CONFIG = "posts"
HF_DATASET_SPLIT = "train"

# --------------------------------------------------------------------------
# Column names -- verified against the actual "posts" config schema via
# dataset.inspect_columns(): ['id', 'topic_label', 'toxic_level', 'post'],
# where 'post' is a nested struct with a 'content' field (and 'comment_count').
# download_posts() flattens post.content into a top-level COL_CONTENT column.
# --------------------------------------------------------------------------

COL_POST_ID = "id"
COL_CONTENT = "content"
COL_CATEGORY = "topic_label"
COL_TOXICITY = "toxic_level"
COL_POST_STRUCT = "post"

# --------------------------------------------------------------------------
# Label axes
# --------------------------------------------------------------------------

# Content category axis (A-I).
CATEGORY_LABELS: dict[str, str] = {
    "A": "Identity",
    "B": "Technology",
    "C": "Socializing",
    "D": "Economics",
    "E": "Viewpoint",
    "F": "Promotion",
    "G": "Politics",
    "H": "Spam",
    "I": "Others",
}

# Toxicity axis (0-4). Note: the dataset card mentions a "5" value in places, but
# there is NO level 5 -- any toxicity==5 rows must be remapped to 4 (Malicious,
# the highest real level) before use. See dataset.download_posts().
TOXICITY_LABELS: dict[int, str] = {
    0: "Safe",
    1: "Edgy",
    2: "Toxic",
    3: "Manipulative",
    4: "Malicious",
}

TOXICITY_SAFE = 0
TOXICITY_MALICIOUS = 4
# Any raw toxicity value >= this is folded into TOXICITY_MALICIOUS.
TOXICITY_MAX_VALID = 4

# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

SAMPLES_PER_CATEGORY = 25
RANDOM_SEED = 42

ASSISTANT_SAMPLE_NAME = "assistant_likeness_sample"
PERSONA_SAMPLE_NAME = "high_toxicity_persona_sample"

# AGE_* run: separate, larger samples (50/category) cached under their own
# names so they don't reuse or reshuffle the original 25/category samples.
# Restricted to categories A (Identity), G (Politics), E (Viewpoint) only.
AGE_SAMPLES_PER_CATEGORY = 50
AGE_CATEGORIES = ["A", "G", "E"]
AGE_PERSONA_SAMPLE_NAME = "age_persona_sample"
AGE_ASSISTANT_SAMPLE_NAME = "age_assistant_safety_sample"

# DBAF_* run (run 3): categories D (Economics), B (Technology), A (Identity),
# F (Promotion) -- the four categories with the most toxicity==4 posts.
DBAF_SAMPLES_PER_CATEGORY = 50
DBAF_CATEGORIES = ["D", "B", "A", "F"]
DBAF_PERSONA_SAMPLE_NAME = "dbaf_persona_sample"
DBAF_ASSISTANT_SAMPLE_NAME = "dbaf_assistant_sample"

# CADFEB_* run (run 4): categories C (Socializing), A (Identity), D (Economics),
# F (Promotion), E (Viewpoint), B (Technology), 200/category. For categories
# A/D/F/B this extends (not replaces) the run-3 DBAF sample/results via
# carryover -- see notebook 05.
CADFEB_SAMPLES_PER_CATEGORY = 200
CADFEB_CATEGORIES = ["C", "A", "D", "F", "E", "B"]
CADFEB_PERSONA_SAMPLE_NAME = "cadfeb_persona_sample"
CADFEB_ASSISTANT_SAMPLE_NAME = "cadfeb_assistant_sample"
# Persona axis backfill: exhaust toxicity==4 (Malicious) per category first,
# then fall back to toxicity==3 (Manipulative) for the shortfall -- some
# categories (esp. C, only 5 posts at toxicity==4 in the whole dataset) can't
# reach 200/category on toxicity==4 alone. Even combined, category B tops out
# at 193 (158 + 35) -- still short of 200; that's a hard data ceiling, not a
# sampling bug. The assistant-safety axis is unaffected -- toxicity==0 alone
# is abundant in every category.
CADFEB_PERSONA_TOXICITY_LEVELS = [4, 3]

# User-prompt generator (notebook 08): per CADFEB category, ground generation
# with this many example posts sampled from results/persona_dataset.json, and
# generate this many user-style prompts.
USER_PROMPT_EXAMPLES_PER_CATEGORY = 2
USER_PROMPTS_PER_CATEGORY = 20

# Persona response generator (notebook 09): for each category, answer this many
# user_prompts (cached, so reruns don't reshuffle which ones). Equal to
# USER_PROMPTS_PER_CATEGORY -- every prompt notebook 08 generated gets a
# response, not a subsample of them.
PERSONA_RESPONSE_SAMPLE_SIZE = USER_PROMPTS_PER_CATEGORY
PERSONA_RESPONSE_PROMPT_SAMPLE_NAME = "persona_response_prompt_sample"
# Target response length, enforced via an identical instruction block in the
# generation prompt for every mode (word-for-word) -- see
# response_generator.build_system_prompt() and notebook 09.
PERSONA_RESPONSE_TARGET_WORDS = (80, 120)
# Flag (not silently accept) any matched mode-pair whose word counts differ
# by more than this fraction of the shorter response -- see notebook 09's
# parity-check cell.
PERSONA_RESPONSE_PARITY_THRESHOLD = 0.20

# --------------------------------------------------------------------------
# LLM / OpenRouter settings
# --------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-4o-mini")
# Verified against GET https://openrouter.ai/api/v1/models -- "qwen/qwen3-32b"
# is the exact slug (OpenRouter routes it to different upstream providers,
# e.g. "qwen3-32b-04-28" via SiliconFlow, transparently).
RESPONSE_MODEL = os.environ.get("RESPONSE_MODEL", "qwen/qwen3-32b")

# --------------------------------------------------------------------------
# Judge result file names
# --------------------------------------------------------------------------

ASSISTANT_JUDGE_NAME = "assistant_likeness"
PERSONA_JUDGE_NAME = "high_toxicity_persona"

# AGE_* run: written to AGE_RESULTS_DIR, not JUDGED_DIR.
AGE_PERSONA_JUDGE_NAME = "AGE_persona"
AGE_ASSISTANT_JUDGE_NAME = "AGE_assistant_safety"

# DBAF_* run (run 3): written to DBAF_RESULTS_DIR.
DBAF_PERSONA_JUDGE_NAME = "DBAF_persona"
DBAF_ASSISTANT_JUDGE_NAME = "DBAF_assistant"

# CADFEB_* run (run 4): written to CADFEB_RESULTS_DIR.
CADFEB_PERSONA_JUDGE_NAME = "CADFEB_persona"
CADFEB_ASSISTANT_JUDGE_NAME = "CADFEB_assistant"
