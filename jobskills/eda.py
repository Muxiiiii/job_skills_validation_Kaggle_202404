"""
Summary tables for the dashboard and the CLI.

Every share uses an explicit denominator ("jobs in the current selection"),
so call these functions again after filtering and the numbers stay correct.
"""

from collections import Counter
from itertools import combinations

import pandas as pd

from .cleaning import skill_rows


def filter_jobs(jobs: pd.DataFrame, categories=None, query: str = "", skills=None) -> pd.DataFrame:
    """
    Filter jobs by category, free-text search, and required skills.

    categories / skills : None = no filter; an empty list = no jobs.
    query               : case-insensitive literal match on title or description.
    skills              : a job must contain *all* selected skills.
    """
    result = jobs

    if categories is not None:
        result = result[result.category.isin(categories)]

    query = query.strip()
    if query:
        in_title = result.title_clean.str.contains(query, case=False, regex=False)
        in_description = result.description_clean.str.contains(query, case=False, regex=False)
        result = result[in_title | in_description]

    if skills is not None:
        wanted = {s.strip().casefold() for s in skills}
        has_all = result.skills.map(lambda job_skills: bool(wanted) and wanted.issubset(job_skills))
        result = result[has_all.astype(bool)]

    return result.copy()


def overview(jobs: pd.DataFrame) -> dict:
    """Headline numbers for a set of jobs."""
    pairs = skill_rows(jobs)
    has_jobs = len(jobs) > 0
    return {
        "jobs": len(jobs),
        "categories": int(jobs.category.nunique()),
        "unique_skills": int(pairs.skill.nunique()),
        "job_skill_pairs": len(pairs),
        "mean_skills_per_job": float(jobs.skill_count.mean()) if has_jobs else 0.0,
        "median_skills_per_job": float(jobs.skill_count.median()) if has_jobs else 0.0,
        "literal_match_rate": float(pairs.literal_match.mean()) if len(pairs) else None,
    }


def build_eda(jobs: pd.DataFrame, top_n: int = 30) -> dict[str, pd.DataFrame]:
    """
    Build all EDA tables.

    Returns a dict with: categories, skill_frequency, skills_by_category,
    skill_cooccurrence (among the top_n skills), skill_count_distribution.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    n_jobs = len(jobs)
    pairs = skill_rows(jobs)

    # Jobs per category
    categories = jobs.groupby("category").agg(
        job_count=("job_id", "size"),
        mean_skills=("skill_count", "mean"),
        median_description_words=("description_word_count", "median"),
    ).reset_index()
    categories["job_share"] = categories.job_count / n_jobs if n_jobs else 0.0

    # How many jobs mention each skill
    frequency = (pairs.groupby("skill")
                      .agg(job_count=("job_id", "nunique"), literal_match_rate=("literal_match", "mean"))
                      .reset_index()
                      .sort_values(["job_count", "skill"], ascending=[False, True]))
    frequency["job_prevalence"] = frequency.job_count / n_jobs if n_jobs else 0.0

    # Same, within each category (denominator = jobs in that category)
    by_category = pairs.groupby(["category", "skill"]).agg(job_count=("job_id", "nunique")).reset_index()
    by_category["category_jobs"] = by_category.category.map(jobs.groupby("category").size())
    by_category["job_prevalence"] = by_category.job_count / by_category.category_jobs

    # Pairs of top skills that appear in the same job
    top_skills = set(frequency.head(top_n).skill)
    pair_counts = Counter()
    for job_skills in jobs.skills:
        pair_counts.update(combinations(sorted(set(job_skills) & top_skills), 2))
    cooccurrence = pd.DataFrame(
        [{"skill_a": a, "skill_b": b, "job_count": n, "job_prevalence": n / n_jobs}
         for (a, b), n in pair_counts.most_common()],
        columns=["skill_a", "skill_b", "job_count", "job_prevalence"],
    )

    # Distribution of skills per job
    distribution = jobs.groupby("skill_count").size().rename("job_count").reset_index()

    return {
        "categories": categories,
        "skill_frequency": frequency,
        "skills_by_category": by_category,
        "skill_cooccurrence": cooccurrence,
        "skill_count_distribution": distribution,
    }


def review_sample(jobs: pd.DataFrame, per_category: int = 5, seed: int = 42) -> pd.DataFrame:
    """A reproducible sample per category, with empty columns for manual label review."""
    if per_category < 1:
        raise ValueError("per_category must be positive.")
    samples = [group.sample(min(per_category, len(group)), random_state=seed)
               for _, group in jobs.groupby("category", sort=True)]
    sample = pd.concat(samples, ignore_index=True) if samples else jobs.copy()
    sample = sample[["job_id", "category", "job_title", "job_description", "job_skill_set"]].copy()
    for column in ["unsupported_skills", "missing_skills", "review_notes"]:
        sample[column] = ""
    return sample
