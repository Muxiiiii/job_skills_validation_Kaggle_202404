"""
Command-line pipeline: clean -> EDA tables -> (optional) category baseline.

    python -m jobskills --input "data/raw/all_job_post (1).csv" --train
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .cleaning import clean_jobs, load_raw_jobs, save_clean_jobs, skill_rows
from .eda import build_eda, overview, review_sample
from .models import train_category_baseline


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Clean the job-skill-set CSV and build EDA / model outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Source CSV with the five required columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train", action="store_true", help="Also train and evaluate the category baseline.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = args.output_dir

    # --- 1. clean ---
    jobs, quality, rejected = clean_jobs(load_raw_jobs(args.input))
    if jobs.empty:
        raise ValueError("No valid jobs remain after cleaning; check the input columns and values.")

    for folder in ["processed", "eda", "models"]:
        (out / folder).mkdir(parents=True, exist_ok=True)

    written = {}   # relative path -> function that writes it

    written["processed/jobs.csv"] = lambda p: save_clean_jobs(jobs, p)
    written["processed/skills_long.csv"] = lambda p: skill_rows(jobs).to_csv(p, index=False)
    written["processed/rejected_rows.csv"] = lambda p: rejected.to_csv(p, index=False)
    written["quality_report.json"] = lambda p: write_json(p, quality)

    # --- 2. EDA ---
    written["eda/overview.json"] = lambda p: write_json(p, overview(jobs))
    for name, table in build_eda(jobs).items():
        written[f"eda/{name}.csv"] = lambda p, t=table: t.to_csv(p, index=False)
    written["eda/label_review_sample.csv"] = lambda p: review_sample(jobs, seed=args.seed).to_csv(p, index=False)

    # --- 3. optional baseline ---
    if args.train:
        model, metrics, predictions, matrix, split = train_category_baseline(jobs, seed=args.seed)
        written["models/category_baseline.joblib"] = lambda p: joblib.dump(model, p)
        written["models/metrics.json"] = lambda p: write_json(p, metrics)
        written["models/holdout_predictions.csv"] = lambda p: predictions.to_csv(p, index=False)
        written["models/confusion_matrix.csv"] = lambda p: matrix.to_csv(p)
        written["models/split_manifest.csv"] = lambda p: split.to_csv(p, index=False)

    for relative_path, write in written.items():
        write(out / relative_path)

    # --- 4. run manifest (what was produced, from which file, with which versions) ---
    write_json(out / "run_manifest.json", {
        "source_filename": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "seed": args.seed,
        "baseline_trained": args.train,
        "outputs": list(written),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
    })
    print(json.dumps({"output_directory": str(out.resolve()), **overview(jobs)}, indent=2))


if __name__ == "__main__":
    main()
