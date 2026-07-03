"""Dataset loading, schema inspection, and stratified sampling.

Sampling is cached to data/processed/<name>.parquet so re-running notebooks
does not reshuffle which specific posts are being judged. The cache is a
simple existence check on the parquet file: if it exists, it is loaded as-is
and no new sampling occurs.
"""

from __future__ import annotations

import pandas as pd

from src import config


def download_posts(force: bool = False) -> pd.DataFrame:
    """Load the Moltbook dataset, cached as a local parquet file.

    Downloads via `datasets.load_dataset` on first call (or when force=True),
    caches the raw table to data/raw/posts.parquet, and remaps any
    toxicity==5 rows to toxicity==4 (there is no real level 5 -- see
    config.TOXICITY_MAX_VALID).

    Args:
        force: if True, re-download even if a local cache exists.

    Returns:
        The full dataset as a pandas DataFrame.
    """
    cache_path = config.RAW_DATA_DIR / "posts.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    ds = load_dataset(
        config.HF_DATASET_NAME, config.HF_DATASET_CONFIG, split=config.HF_DATASET_SPLIT
    )
    df = ds.to_pandas()

    if config.COL_POST_STRUCT in df.columns and config.COL_CONTENT not in df.columns:
        df[config.COL_CONTENT] = df[config.COL_POST_STRUCT].apply(
            lambda p: p["content"] if isinstance(p, dict) else p
        )

    if config.COL_TOXICITY in df.columns:
        over_max = df[config.COL_TOXICITY] > config.TOXICITY_MAX_VALID
        if over_max.any():
            df.loc[over_max, config.COL_TOXICITY] = config.TOXICITY_MAX_VALID

    df.to_parquet(cache_path, index=False)
    return df


def inspect_columns(df: pd.DataFrame) -> None:
    """Print the dataset's actual schema and value counts for the label columns.

    This MUST be run and visually checked against config.py's column-name and
    label constants before trusting any downstream sampling -- the column
    names and label sets baked into config.py are a best guess from the
    dataset card, not a verified schema.
    """
    print("columns:", list(df.columns))
    print("row count:", len(df))
    print("dtypes:\n", df.dtypes)

    for col in (config.COL_CATEGORY, config.COL_TOXICITY):
        if col in df.columns:
            print(f"\nvalue_counts[{col}]:")
            print(df[col].value_counts(dropna=False).sort_index())
        else:
            print(f"\nWARNING: expected column '{col}' not found in dataset columns.")


def sample_stratified(
    df: pd.DataFrame,
    toxicity: int | list[int],
    per_category: int = config.SAMPLES_PER_CATEGORY,
    seed: int = config.RANDOM_SEED,
    categories: list[str] | None = None,
    carryover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sample up to `per_category` posts per content category at given toxicity level(s).

    Args:
        df: full dataset (post-remap, i.e. after download_posts()).
        toxicity: a single value of config.COL_TOXICITY to filter to, or a
            priority-ordered list of values (e.g. [4, 3]) -- when a list is
            given, each category's quota is filled from the first level first,
            only falling back to later levels in the list for the shortfall.
            This lets a category with few toxicity==4 posts still reach its
            quota by backfilling with toxicity==3, without touching categories
            that already have enough toxicity==4 posts.
        per_category: max rows to sample per category (fewer if a category has less data).
        seed: random seed for reproducible sampling.
        categories: if given, restrict sampling to only these content categories
            (e.g. ["A", "G", "E"]); other categories are excluded entirely. If
            None, every category present in the filtered data is sampled.
        carryover: rows from a prior, smaller sample of this same population
            (e.g. a previous run's cached sample) to keep as-is rather than
            resample. Each category's carryover rows count toward that
            category's `per_category` quota; only the shortfall is newly
            sampled, drawn from posts not already in `carryover`. This lets a
            larger follow-up run extend a prior run's sample instead of
            drawing an unrelated fresh one.

    Returns:
        Concatenated sample (carryover rows plus any newly drawn top-up rows,
        exhausting earlier toxicity levels before falling back to later ones)
        across all (optionally restricted) categories present in the filtered
        data, with a fresh RangeIndex.
    """
    levels = [toxicity] if isinstance(toxicity, int) else list(toxicity)

    subset = df[df[config.COL_TOXICITY].isin(levels)]
    if categories is not None:
        subset = subset[subset[config.COL_CATEGORY].isin(categories)]

    carryover_ids: set = (
        set(carryover[config.COL_POST_ID]) if carryover is not None and len(carryover) else set()
    )

    parts = []
    if carryover is not None and len(carryover):
        parts.append(carryover)

    category_list = categories if categories is not None else sorted(subset[config.COL_CATEGORY].unique())

    for category in category_list:
        already = 0
        if carryover is not None and len(carryover):
            already = int((carryover[config.COL_CATEGORY] == category).sum())
        needed = max(0, per_category - already)
        if needed == 0:
            continue

        excluded_ids = set(carryover_ids)
        for level in levels:
            if needed <= 0:
                break
            pool = subset[
                (subset[config.COL_CATEGORY] == category)
                & (subset[config.COL_TOXICITY] == level)
                & (~subset[config.COL_POST_ID].isin(excluded_ids))
            ]
            n = min(needed, len(pool))
            if n > 0:
                drawn = pool.sample(n=n, random_state=seed)
                parts.append(drawn)
                excluded_ids.update(drawn[config.COL_POST_ID])
                needed -= n

    if not parts:
        return subset.iloc[0:0].copy()

    sample = pd.concat(parts, ignore_index=True)
    return sample


def get_or_create_sample(
    df: pd.DataFrame,
    name: str,
    toxicity: int | list[int],
    per_category: int = config.SAMPLES_PER_CATEGORY,
    seed: int = config.RANDOM_SEED,
    categories: list[str] | None = None,
    carryover: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load a cached stratified sample by name, or create and cache it if absent.

    The cache lives at data/processed/<name>.parquet. Once created, the sample
    is stable across runs -- re-running notebooks will not reshuffle which
    specific posts were selected, even if the underlying dataset changes.

    Args:
        df: full dataset to sample from (only used on cache miss).
        name: cache file stem, e.g. config.ASSISTANT_SAMPLE_NAME.
        toxicity: toxicity value or priority-ordered list of values to filter to
            (only used on cache miss) -- see `sample_stratified`.
        per_category: max rows to sample per category (only used on cache miss).
        seed: random seed (only used on cache miss).
        categories: if given, restrict sampling to only these content categories
            (only used on cache miss). Use a distinct `name` per distinct
            `categories`/`per_category` combination so caches don't collide.
        carryover: prior sample to extend rather than resample from scratch
            (only used on cache miss) -- see `sample_stratified`.

    Returns:
        The cached (or newly created) sample DataFrame.
    """
    cache_path = config.PROCESSED_DATA_DIR / f"{name}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    sample = sample_stratified(
        df,
        toxicity=toxicity,
        per_category=per_category,
        seed=seed,
        categories=categories,
        carryover=carryover,
    )
    sample.to_parquet(cache_path, index=False)
    return sample
