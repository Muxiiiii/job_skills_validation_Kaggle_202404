"""Tests for cleaning, EDA denominators, search, and the train / holdout split."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from jobskills.cleaning import clean_jobs, load_clean_jobs, parse_skills, save_clean_jobs, skill_rows
from jobskills.eda import build_eda, filter_jobs, overview
from jobskills.models import JobSearch, train_category_baseline


def record(job_id="1", description="Python and C++ development.",
           skills="['Python', ' python ', 'C++']", category="IT"):
    """One raw source row."""
    return {"job_id": job_id, "category": category, "job_title": "Developer",
            "job_description": description, "job_skill_set": skills}


class CleaningTests(unittest.TestCase):

    def test_skill_parsing(self):
        self.assertEqual(parse_skills('["Python", " python ", "C++", ""]'), ["python", "c++"])
        for bad in ['{"python": 1}', "[1]", "broken", "__import__('os').getcwd()"]:
            with self.assertRaises(ValueError):
                parse_skills(bad)

    def test_rejected_rows_and_duplicates(self):
        rows = [
            record(), record(),                     # exact duplicate -> keep one
            record("2", skills="invalid"),          # unreadable skills
            record("3", description=""),            # missing description
            record("4"), record("4", category="SALES"),   # conflicting duplicate -> reject both
            record("5", skills="[]"),               # empty skills are kept
        ]
        jobs, quality, rejected = clean_jobs(pd.DataFrame(rows))

        self.assertEqual(jobs.job_id.tolist(), ["1", "5"])
        self.assertEqual(quality["input_rows"], quality["retained_rows"] + quality["rejected_rows"])
        self.assertEqual(quality["empty_skill_rows"], 1)
        self.assertEqual(rejected.reason.eq("conflicting_job_id").sum(), 2)

    def test_csv_roundtrip_keeps_ids_and_skills(self):
        jobs, _, _ = clean_jobs(pd.DataFrame([record("001")]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "jobs.csv"
            save_clean_jobs(jobs, path)
            restored = load_clean_jobs(path)
        self.assertEqual(restored.job_id.iloc[0], "001")
        self.assertEqual(restored.skills.iloc[0], ["python", "c++"])


class EdaTests(unittest.TestCase):

    def test_denominators_and_empty_selections(self):
        jobs, _, _ = clean_jobs(pd.DataFrame([record(), record("2", skills="['Python']")]))
        frequency = build_eda(jobs)["skill_frequency"].set_index("skill")
        self.assertEqual(frequency.loc["python", "job_prevalence"], 1)
        self.assertEqual(frequency.loc["c++", "job_prevalence"], 0.5)
        self.assertTrue(skill_rows(jobs).literal_match.all())

        for empty in [filter_jobs(jobs, categories=[]),
                      filter_jobs(jobs, skills=[]),
                      filter_jobs(jobs, query="[absent]")]:
            self.assertEqual(overview(empty)["jobs"], 0)
            self.assertTrue(all(table.empty for table in build_eda(empty).values()))


class ModelTests(unittest.TestCase):

    def test_search(self):
        jobs, _, _ = clean_jobs(pd.DataFrame([record("001")]))
        search = JobSearch(jobs)
        self.assertEqual(search.search("Python").job_id.tolist(), ["001"])
        self.assertTrue(search.search("zzzzunknown").empty)
        self.assertTrue(search.search("Python", category="SALES").empty)

    def test_grouped_split_has_no_leakage(self):
        # Two jobs per description; each description has one word unique to it.
        rows = []
        for category, token in [("IT", "programming"), ("SALES", "selling")]:
            for i in range(15):
                text = f"Shared words {token} uniqueword{category.lower()}{i}"
                rows += [record(f"{category}{i}a", text, category=category),
                         record(f"{category}{i}b", text, category=category)]
        jobs, _, _ = clean_jobs(pd.DataFrame(rows))
        model, metrics, predictions, matrix, split = train_category_baseline(jobs)

        # identical descriptions never sit on both sides
        train_groups = set(split.loc[split.partition == "train", "description_group"])
        holdout_groups = set(split.loc[split.partition == "holdout", "description_group"])
        self.assertFalse(train_groups & holdout_groups)

        # TF-IDF never saw words that only appear in holdout descriptions
        vocabulary = model.named_steps["tfidf"].vocabulary_
        for text in jobs[jobs.job_id.isin(predictions.job_id)].description_clean:
            self.assertNotIn(text.split()[-1].lower(), vocabulary)

        self.assertEqual(int(matrix.to_numpy().sum()), metrics["holdout_rows"])


if __name__ == "__main__":
    unittest.main()
