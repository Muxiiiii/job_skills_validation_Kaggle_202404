"""
Job Skills Explorer: Streamlit dashboard for the job-skill-set dataset.

Run locally:   streamlit run app.py
Data source:   a CSV uploaded in the sidebar, or the first data/raw/all_job_post*.csv if present.
               No data is bundled with the repository.
"""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from jobskills import build_eda, clean_jobs, filter_jobs, load_raw_jobs, overview
from jobskills.cleaning import REQUIRED
from jobskills.eda import review_sample
from jobskills.models import JobSearch, predict_categories, train_category_baseline

ROOT = Path(__file__).resolve().parent
SEED = 42
# Widgets / results that depend on the dataset; cleared when a new file is loaded.
DATASET_STATE = ["categories", "selected_skills", "text_query", "baseline_result", "search_query", "prediction_text"]

st.set_page_config(page_title="Job Skills Explorer", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_data(content: bytes):
    """Clean the raw CSV bytes -> (jobs, quality, rejected)."""
    return clean_jobs(load_raw_jobs(io.BytesIO(content)))


@st.cache_data(show_spinner=False)
def summarize(jobs: pd.DataFrame):
    return overview(jobs), build_eda(jobs)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """DataFrame -> CSV bytes for download buttons (skills lists become JSON)."""
    out = frame.copy()
    if "skills" in out:
        out["skills"] = out.skills.map(json.dumps)
    return out.to_csv(index=False).encode("utf-8")


def show_table(frame: pd.DataFrame):
    st.dataframe(frame, hide_index=True, use_container_width=True)


def read_source():
    """Return (bytes, filename) from the uploader or data/raw, or (None, None)."""
    upload = st.sidebar.file_uploader("Upload job skills CSV", type=["csv"])
    if upload is not None:
        return upload.getvalue(), upload.name
    local_files = sorted((ROOT / "data" / "raw").glob("all_job_post*.csv"))
    if local_files:
        st.sidebar.caption(f"Local dataset: {local_files[0].name}")
        return local_files[0].read_bytes(), local_files[0].name
    return None, None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def overview_tab(selected: pd.DataFrame):
    st.subheader("Selected jobs")
    if selected.empty:
        st.warning("No jobs match the current filters. Select a category or clear the search and skill filters.")
        return

    summary, tables = summarize(selected)
    cols = st.columns(4)
    cols[0].metric("Jobs", f"{summary['jobs']:,}")
    cols[1].metric("Categories", summary["categories"])
    cols[2].metric("Unique skills", f"{summary['unique_skills']:,}")
    cols[3].metric("Mean skills per job", f"{summary['mean_skills_per_job']:.1f}")

    st.bar_chart(tables["categories"].set_index("category")[["job_count"]])
    show_table(tables["categories"])
    st.bar_chart(tables["skill_count_distribution"].set_index("skill_count"))
    show_table(selected[["job_id", "category", "job_title", "skill_count"]])
    st.download_button("Download selected jobs", csv_bytes(selected), "selected_jobs.csv", "text/csv")


def skills_tab(selected: pd.DataFrame):
    st.subheader("Skills in the current selection")
    if selected.empty:
        st.info("Adjust the sidebar filters to explore skills.")
        return
    _, tables = summarize(selected)
    frequency = tables["skill_frequency"]
    if frequency.empty:
        st.info("The selected jobs have no supplied skills.")
        return

    limit = st.slider("Number of top skills", 5, 50, 20)
    st.bar_chart(frequency.head(limit).set_index("skill")[["job_count"]])
    st.caption("Job prevalence = jobs containing the skill / jobs in the current selection.")
    show_table(frequency)
    st.download_button("Download skill frequencies", csv_bytes(frequency), "skill_frequency.csv", "text/csv")

    st.markdown("**Skills by category**")
    st.caption("Prevalence within each category uses that category's job count as the denominator.")
    show_table(tables["skills_by_category"])

    st.markdown("**Co-occurring skills**")
    st.caption("Pairs counted among the 30 most frequent skills in the current selection.")
    show_table(tables["skill_cooccurrence"].head(50))


def quality_tab(jobs: pd.DataFrame, quality: dict, rejected: pd.DataFrame):
    st.subheader("Source data quality")
    st.caption("Computed on the whole uploaded file, independent of the sidebar filters.")
    st.json(quality)

    rate = summarize(jobs)[0]["literal_match_rate"]
    st.metric("Literal skill-to-description match rate", f"{rate:.1%}" if rate is not None else "N/A")
    st.caption("Paraphrased skills do not match literally, and a match does not prove the job requires the skill.")

    if not rejected.empty:
        show_table(rejected)
    st.download_button("Download quality report", json.dumps(quality, indent=2), "quality_report.json",
                       "application/json")
    st.download_button("Download rejected rows", csv_bytes(rejected), "rejected_rows.csv", "text/csv")
    st.download_button("Download label review sample", csv_bytes(review_sample(jobs)), "label_review_sample.csv",
                       "text/csv")

    st.markdown("**Posting dates**")
    st.write("This CSV has no date column. Join it to the original postings.csv by job_id to recover dates; "
             "see notebooks/new_dataset_timeline_en.ipynb.")


def baseline_tab(jobs: pd.DataFrame):
    st.subheader("Job category baseline")
    st.caption(f"TF-IDF + logistic regression on descriptions, ~80/20 grouped split, seed {SEED}. "
               "Uses all retained jobs (ignores the sidebar filters). Not an evaluation of the 1.3M model.")

    if st.button("Train baseline", type="primary"):
        try:
            with st.spinner("Training and evaluating..."):
                st.session_state["baseline_result"] = train_category_baseline(jobs, seed=SEED)
        except ValueError as exc:
            st.error(f"Unable to train on this dataset: {exc}")

    result = st.session_state.get("baseline_result")
    if result is None:
        st.write("Click **Train baseline** to see holdout metrics, errors and example predictions.")
        return

    model, metrics, predictions, matrix, _ = result
    cols = st.columns(3)
    cols[0].metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    cols[1].metric("Holdout macro F1", f"{metrics['macro_f1']:.3f}")
    cols[2].metric("Dummy accuracy", f"{metrics['dummy_accuracy']:.1%}")
    st.caption(f"Train {metrics['train_rows']:,} jobs, holdout {metrics['holdout_rows']:,} jobs, "
               f"scikit-learn {metrics['scikit_learn_version']}. Scores can shift slightly across versions.")

    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    st.dataframe(matrix, use_container_width=True)
    st.markdown("**Holdout errors**")
    show_table(predictions[~predictions.correct])
    st.download_button("Download model metrics", json.dumps(metrics, indent=2), "metrics.json", "application/json")
    st.download_button("Download holdout predictions", csv_bytes(predictions), "holdout_predictions.csv", "text/csv")

    description = st.text_area("Try a job description", key="prediction_text")
    if st.button("Predict category"):
        if description.strip():
            show_table(predict_categories(model, [description]))
            st.caption("Probabilities are not calibrated.")
        else:
            st.warning("Enter a job description first.")


def search_tab(selected: pd.DataFrame):
    st.subheader("Similar jobs in the current selection")
    st.caption("TF-IDF cosine similarity on titles and descriptions; a lexical score, not a match probability.")

    query = st.text_input("Describe the job or skills to search for", key="search_query")
    if not st.button("Find similar jobs"):
        return
    if selected.empty:
        st.warning("No jobs match the sidebar filters.")
    elif not query.strip():
        st.warning("Enter a search query first.")
    else:
        results = JobSearch(selected).search(query, top_k=10)
        if results.empty:
            st.info("No shared vocabulary found. Try another query.")
        else:
            show_table(results)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def main():
    st.title("Job Skills Explorer")
    st.caption("Job categories, supplied skills, data quality, and a local classification baseline.")

    # --- load ---
    content, filename = read_source()
    if content is None:
        st.info("Upload the original job skills CSV in the sidebar to begin. No bundled dataset is required.")
        st.markdown("Required columns: " + ", ".join(f"`{c}`" for c in REQUIRED))
        return

    try:
        jobs, quality, rejected = prepare_data(content)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        st.error(f"Unable to read this CSV: {exc}")
        return
    st.sidebar.caption(f"Source: {filename}")

    if jobs.empty:
        st.warning("No valid jobs remain after cleaning. Check the rejected rows and the required columns.")
        st.json(quality)
        st.download_button("Download rejected rows", csv_bytes(rejected), "rejected_rows.csv", "text/csv")
        return

    # New file -> reset filters and trained model
    dataset_key = hashlib.sha256(content).hexdigest()
    if st.session_state.get("dataset_key") != dataset_key:
        for key in DATASET_STATE:
            st.session_state.pop(key, None)
        st.session_state["dataset_key"] = dataset_key

    # --- sidebar filters ---
    all_categories = sorted(jobs.category.unique())
    categories = st.sidebar.multiselect("Categories", all_categories, default=all_categories, key="categories")
    query = st.sidebar.text_input("Search titles and descriptions", key="text_query")
    all_skills = sorted({skill for job_skills in jobs.skills for skill in job_skills})
    skills = st.sidebar.multiselect("Required skills (all selected)", all_skills, key="selected_skills")

    selected = filter_jobs(jobs, categories=categories, query=query, skills=skills or None)
    st.sidebar.caption(f"Showing {len(selected):,} of {len(jobs):,} jobs")
    st.info("Skills are machine-generated labels (RecAI). Literal text matches are a diagnostic, not accuracy.")

    # --- tabs ---
    tabs = st.tabs(["Overview", "Skill Explorer", "Data Quality", "Category Baseline", "Similar Jobs"])
    with tabs[0]:
        overview_tab(selected)
    with tabs[1]:
        skills_tab(selected)
    with tabs[2]:
        quality_tab(jobs, quality, rejected)
    with tabs[3]:
        baseline_tab(jobs)
    with tabs[4]:
        search_tab(selected)


if __name__ == "__main__":
    main()
