# Job Skills Explorer

Cleaning, EDA, baseline models and a Streamlit dashboard for the Kaggle **job-skill-set** dataset
(1,167 LinkedIn job postings with machine-generated skill lists).

This repository is part of a larger project: validating an analysis pipeline built on the
1.3M LinkedIn Jobs & Skills dataset. **It currently covers only the 1,167-row dataset**; it does not
yet load the 1.3M data or evaluate the 1.3M model.

A link to the Streamlit dashboard: https://job-skills-validation-kaggle-202404.streamlit.app/

## Repository layout

| Path | What it does |
|---|---|
| `app.py` | Streamlit dashboard (upload a CSV, explore, train the baseline, search similar jobs) |
| `jobskills/cleaning.py` | Load, validate and clean the CSV; parse skill lists |
| `jobskills/eda.py` | Summary tables: categories, skill frequency, co-occurrence, filters |
| `jobskills/models.py` | Category classifier baseline and TF-IDF similar-job search |
| `jobskills/__main__.py` | Command-line pipeline that writes all tables to `artifacts/` |
| `notebooks/new_dataset_timeline_en.ipynb` | Timeline and rising-jobs analysis using the original postings |
| `tests/` | Unit tests for the package and the dashboard |

## Dataset

| File | Rows | Notes |
|---|---:|---|
| `all_job_post (1).csv` (job-skill-set, Batuhan Mutlu 2024) | 1,167 | 5 categories; `job_skill_set` extracted with RecAI APIs (machine labels, not human annotations) |
| `original_dataset/.../postings.csv` (LinkedIn Job Postings 2023-2024, Arshkon) | 123,849 | Source of the subset; has posting timestamps |

Required columns in the job-skill-set CSV: `job_id`, `category`, `job_title`, `job_description`, `job_skill_set`.

| Category | Jobs |
|---|---:|
| INFORMATION-TECHNOLOGY | 240 |
| BUSINESS-DEVELOPMENT | 239 |
| FINANCE | 236 |
| SALES | 232 |
| HR | 220 |

**No data is included in this repository** (see `.gitignore`). Download the datasets from Kaggle and
place them like this if you want to run the CLI or the notebook:

```text
data/raw/
  all_job_post (1).csv
  original_dataset/
    LinkedIn Job Postings (2023 - 2024)/
      postings.csv
      jobs/
      mappings/
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                                                  # dashboard
python -m jobskills --input "data/raw/all_job_post (1).csv" --train   # CLI pipeline -> artifacts/
python -m unittest discover -s tests -v                               # tests
```

The dashboard starts without any data: upload the CSV in the sidebar. If `data/raw/all_job_post*.csv`
exists, it is loaded automatically.

## Method

### 1. Cleaning (`jobskills/cleaning.py`)
1. Normalize Unicode and HTML entities and collapse whitespace into new `title_clean` / `description_clean` columns; original columns are kept.
2. Parse each skill list with `json` / `ast.literal_eval` (no code execution), lower-case it, and drop empty or repeated skills. No stemming or synonym merging.
3. Reject rows with a missing id, category, title or description, or an unreadable skill list. Empty skill lists are kept and counted.
4. Duplicate `job_id`s: identical copies keep one row; conflicting copies are all rejected.

`quality_report.json` reconciles `input = retained + rejected`, and `rejected_rows.csv` lists each rejected row with its reason.

### 2. EDA (`jobskills/eda.py`)
- Jobs and average skills per category.
- Skill frequency and prevalence (jobs with the skill / jobs in the selection), overall and per category.
- Co-occurrence among the 30 most frequent skills, and the skills-per-job distribution.
- **Literal match rate**: share of skills that appear word-for-word in the description. This is a diagnostic, not extraction precision or recall.
- A reproducible review sample (5 jobs per category) with empty columns for manual label review.

All shares use the current selection as the denominator, so the dashboard recomputes them after filtering.

### 3. Baseline models (`jobskills/models.py`)
**Category classifier.** TF-IDF (1-2 grams, min_df 2, max 30k features) + logistic regression on the description only.
The split is the first fold of a shuffled 5-fold `StratifiedGroupKFold` (seed 42, ~80/20), grouped by a hash of the
description so identical descriptions never sit on both sides. A most-frequent-class dummy model is reported for reference.

**Similar-job search.** TF-IDF over title + description, ranked by cosine similarity. Scores are lexical relevance, not match probabilities.

## Results (seed 42)

| Check | Value |
|---|---:|
| Retained / input jobs | 1,167 / 1,167 |
| Rejected jobs | 0 |
| Unique skills | 4,984 |
| Job-skill pairs | 20,833 |
| Mean skills per job | 17.85 |
| Literal skill-to-description match rate | 49.1% |
| Train / holdout jobs | 934 / 233 |
| Holdout accuracy | 90.1% - 90.6% |
| Holdout macro F1 | 0.90 - 0.91 |
| Dummy (most frequent) accuracy | 18.0% |

With `scikit-learn==1.6.1`, holdout accuracy was 90.6% on Python 3.9 and 90.1% on Python 3.13
(same split, different numerical libraries). Scores can also move with the scikit-learn version, which is
saved in `metrics.json`.

**Limits.** This is a local holdout on a small dataset, not an external test, and category accuracy says nothing
about skill-extraction quality. Near-duplicate descriptions from the same employer may still cross the split.

## Timeline notebook

`notebooks/new_dataset_timeline_en.ipynb` recovers posting dates by joining `job_id` to the original `postings.csv`.
Main findings (outputs are saved in the notebook):

1. The data is a single scrape snapshot: `listed_time` covers **2024-04-05 to 2024-04-20**, and 98.5% of first-posted dates fall in April 2024.
2. Daily volume reflects scraping, not hiring: 04-18 and 04-19 hold ~52% of rows, and four days in the window have none.
3. "Rising jobs" are measured as **share** of each scrape day's postings. Shifts are modest, and the largest title-level rises come from single employers bulk-posting. Read them as differences between scrape batches, not labour-market trends.
4. Part C compares dates and `job_id` overlap with the 1.3M dataset once its path is set.

To run it: `pip install -e ".[notebooks]"`, place the data under `data/raw/` (or set `JOB_SKILLS_DATA_DIR`), then open the notebook.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create an app with entry point **`app.py`** and Python 3.11.
3. The app installs `requirements.txt` and starts without data; users upload the CSV.

## Citation

Batuhan Mutlu. (2024). *job-skill-set*. Kaggle. https://www.kaggle.com/dsv/10201355. DOI: 10.34740/KAGGLE/DSV/10201355

The original data and anything derived from it remain under their own Kaggle terms; a code license does not cover the data.
