"""
Load and clean the job-skill-set CSV.

Cleaning rules
--------------
1. Text fields: normalize Unicode, decode HTML entities, collapse whitespace.
   The original columns are kept; cleaned text goes into new `*_clean` columns.
2. Skills: parse the serialized list (JSON or Python literal), lower-case,
   strip, and drop empty or repeated entries. No stemming or synonym merging.
3. Rows with a missing id / category / title / description, or an unreadable
   skill list, are rejected with a reason. An empty but valid skill list is kept.
4. Duplicate job_ids: identical copies keep one row; conflicting copies are all rejected.
"""

from __future__ import annotations

import ast
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED = ["job_id", "category", "job_title", "job_description", "job_skill_set"]
CLEAN_COLUMNS = REQUIRED + [
    "title_clean", "description_clean", "skills",
    "skill_count", "description_word_count", "description_group",
]


# ---------------------------------------------------------------------------
# Text and skill parsing
# ---------------------------------------------------------------------------

def normalize_text(value) -> str:
    """Normalize Unicode and HTML entities, then collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(value))
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_skills(value: str) -> list[str]:
    """
    Parse a serialized skill list such as '["Python", "SQL"]' or "['Python', 'SQL']".

    Uses json / ast.literal_eval only, so no code is ever executed.
    Returns lower-cased, de-duplicated skills in their original order.
    """
    try:
        items = json.loads(value)
    except (ValueError, TypeError):
        try:
            items = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError("Skills must be a JSON array or a Python list literal.") from exc

    if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
        raise ValueError("Skills must be a list of strings.")

    skills = [normalize_text(s).casefold() for s in items]
    return list(dict.fromkeys(s for s in skills if s))


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def load_raw_jobs(path) -> pd.DataFrame:
    """Read the source CSV as text (keeps job_id as a string) and check the columns."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[REQUIRED].copy()


def _rejection_reason(row: dict) -> str:
    """Return why a row should be rejected, or '' if it is valid."""
    for column in REQUIRED[:4]:
        if not normalize_text(row[column]):
            return f"missing_{column}"
    try:
        parse_skills(row["job_skill_set"])
    except ValueError:
        return "invalid_skill_list"
    return ""


def clean_jobs(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Clean the raw jobs table.

    Returns
    -------
    jobs     : retained, cleaned jobs (one row per job_id)
    quality  : dict of counts that reconcile input = retained + rejected
    rejected : rejected source rows with a `reason` column
    """
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = raw[REQUIRED].fillna("").astype(str).reset_index(drop=True)
    work.insert(0, "source_record", range(1, len(work) + 1))   # 1 = first data row after header

    # --- 1. reject rows with missing values or unreadable skill lists ---
    work["reason"] = [_rejection_reason(row) for row in work.to_dict("records")]
    rejected = [work[work.reason != ""]]
    jobs = work[work.reason == ""].drop(columns="reason").copy()

    # --- 2. derived columns ---
    jobs["job_id"] = jobs.job_id.map(normalize_text)
    jobs["category"] = jobs.category.map(normalize_text).str.upper()
    jobs["title_clean"] = jobs.job_title.map(normalize_text)
    jobs["description_clean"] = jobs.job_description.map(normalize_text)
    jobs["skills"] = jobs.job_skill_set.map(parse_skills)
    jobs["skill_count"] = jobs.skills.map(len)
    jobs["description_word_count"] = jobs.description_clean.str.split().map(len)
    # Hash of the description: identical descriptions share a group and stay
    # on the same side of the train / holdout split.
    jobs["description_group"] = jobs.description_clean.map(
        lambda text: hashlib.sha256(text.casefold().encode()).hexdigest())

    # --- 3. duplicate job_ids ---
    signature = pd.Series(
        [json.dumps([r.category, r.title_clean, r.description_clean, sorted(r.skills)]) for r in jobs.itertuples()],
        index=jobs.index, dtype=object)
    n_versions = signature.groupby(jobs.job_id).transform("nunique")
    conflicting = n_versions > 1                                  # same id, different content
    repeated = jobs.job_id.duplicated() & ~conflicting            # exact copies after the first
    for mask, reason in [(conflicting, "conflicting_job_id"), (repeated, "duplicate_job_id")]:
        rejected.append(jobs.loc[mask, ["source_record"] + REQUIRED].assign(reason=reason))
    jobs = jobs[~(conflicting | repeated)].reset_index(drop=True)

    rejected = pd.concat(rejected, ignore_index=True)[["source_record"] + REQUIRED + ["reason"]]
    jobs = jobs[["source_record"] + CLEAN_COLUMNS]

    quality = {
        "input_rows": len(raw),
        "retained_rows": len(jobs),
        "rejected_rows": len(rejected),
        "rejection_counts": rejected.reason.value_counts().to_dict(),
        "missing_source_values": {c: int(work[c].map(normalize_text).eq("").sum()) for c in REQUIRED},
        "empty_skill_rows": int(jobs.skill_count.eq(0).sum()),
        "duplicate_description_rows": int(jobs.description_group.duplicated().sum()),
        "category_counts": jobs.category.value_counts().to_dict(),
    }
    return jobs, quality, rejected


def skill_rows(jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Long table with one row per (job, skill).

    `literal_match` says whether the skill appears word-for-word in the description.
    It is a diagnostic only: paraphrased skills will not match.
    """
    records = []
    for job in jobs.itertuples():
        description = job.description_clean.casefold()
        for skill in job.skills:
            pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
            records.append({
                "job_id": job.job_id,
                "category": job.category,
                "skill": skill,
                "literal_match": re.search(pattern, description) is not None,
            })
    return pd.DataFrame(records, columns=["job_id", "category", "skill", "literal_match"])


# ---------------------------------------------------------------------------
# Saving / loading the cleaned table
# ---------------------------------------------------------------------------

def save_clean_jobs(jobs: pd.DataFrame, path) -> None:
    """Write cleaned jobs to CSV; the skills list is stored as a JSON array."""
    out = jobs.copy()
    out["skills"] = out.skills.map(lambda s: json.dumps(s, ensure_ascii=False))
    out.to_csv(path, index=False)


def load_clean_jobs(path) -> pd.DataFrame:
    """Read a CSV written by save_clean_jobs and restore the skills lists."""
    jobs = pd.read_csv(path, dtype={"job_id": str}, keep_default_na=False)
    jobs["skills"] = jobs.skills.map(parse_skills)
    return jobs
