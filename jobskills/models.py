"""
Two simple baselines on job descriptions:

1. train_category_baseline : TF-IDF + logistic regression that predicts the job category.
2. JobSearch               : TF-IDF cosine search for similar jobs.

Neither is a skill-extraction model, and neither evaluates the 1.3M model.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 1. Category classifier
# ---------------------------------------------------------------------------

def train_category_baseline(jobs: pd.DataFrame, seed: int = 42):
    """
    Train on ~80% of jobs and evaluate on a ~20% holdout.

    - Input feature: description_clean only (titles and skill lists are not used).
    - Split: first fold of a shuffled 5-fold StratifiedGroupKFold, grouped by
      description_group, so identical descriptions never sit on both sides.
    - TF-IDF is fitted on the training descriptions only.

    Returns (model, metrics, holdout_predictions, confusion_matrix, split_manifest).
    """
    if jobs.category.nunique() < 2:
        raise ValueError("Category classification needs at least two categories.")
    if jobs.groupby("category").description_group.nunique().min() < 5:
        raise ValueError("Each category needs at least five distinct descriptions for the 80/20 split.")

    # --- split ---
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(jobs, jobs.category, jobs.description_group))
    train, test = jobs.iloc[train_idx], jobs.iloc[test_idx]
    labels = sorted(jobs.category.unique())
    if set(train.category) != set(labels) or set(test.category) != set(labels):
        raise ValueError("The grouped split did not keep every category in both train and holdout.")

    # --- model ---
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                                  sublinear_tf=True, max_features=30000)),
        ("classifier", LogisticRegression(max_iter=1500, random_state=seed)),
    ])
    model.fit(train.description_clean, train.category)
    predicted = model.predict(test.description_clean)

    # Reference point: always predict the most frequent training category
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train.category)
    dummy_predicted = dummy.predict(np.zeros((len(test), 1)))

    def macro_f1(y_pred):
        return float(f1_score(test.category, y_pred, labels=labels, average="macro", zero_division=0))

    # --- outputs ---
    metrics = {
        "task": "job_category_classification",
        "input": "description_clean",
        "seed": seed,
        "scikit_learn_version": sklearn.__version__,
        "train_rows": len(train),
        "holdout_rows": len(test),
        "split_method": "first fold of shuffled 5-fold StratifiedGroupKFold",
        "group": "SHA-256 of case-folded normalized description",
        "accuracy": float(accuracy_score(test.category, predicted)),
        "macro_f1": macro_f1(predicted),
        "dummy_accuracy": float(accuracy_score(test.category, dummy_predicted)),
        "dummy_macro_f1": macro_f1(dummy_predicted),
        "classification_report": classification_report(
            test.category, predicted, labels=labels, output_dict=True, zero_division=0),
        "limitations": [
            "Local holdout only; this does not evaluate the 1.3M model.",
            "Near-duplicate descriptions and company templates may still cross the split.",
            "Category accuracy says nothing about skill-extraction quality.",
            "Scores can change slightly with the scikit-learn version.",
        ],
    }

    predictions = test[["job_id", "category", "job_title"]].copy()
    predictions["predicted_category"] = predicted
    predictions["correct"] = predicted == test.category.to_numpy()

    matrix = pd.DataFrame(confusion_matrix(test.category, predicted, labels=labels),
                          index=pd.Index(labels, name="actual_category"), columns=labels)

    split = jobs[["job_id", "category", "description_group"]].copy()
    split["partition"] = "train"
    split.loc[test.index, "partition"] = "holdout"

    return model, metrics, predictions, matrix, split


def predict_categories(model, descriptions: list[str]) -> pd.DataFrame:
    """Predict categories for new descriptions. Probabilities are not calibrated."""
    probabilities = model.predict_proba(descriptions)
    return pd.DataFrame({
        "description": descriptions,
        "predicted_category": model.classes_[probabilities.argmax(axis=1)],
        "model_probability": probabilities.max(axis=1),
    })


def load_category_model(path):
    """Load a saved model. joblib files can run code, so only load files you created."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 2. Similar-job search
# ---------------------------------------------------------------------------

class JobSearch:
    """TF-IDF index over title + description; ranks jobs by cosine similarity to a query."""

    COLUMNS = ["job_id", "category", "job_title", "skills"]

    def __init__(self, jobs: pd.DataFrame):
        if jobs.empty:
            raise ValueError("Search needs at least one job.")
        self.jobs = jobs.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)
        self.matrix = self.vectorizer.fit_transform(self.jobs.title_clean + " " + self.jobs.description_clean)

    def search(self, query: str, top_k: int = 10, category: str | None = None) -> pd.DataFrame:
        """Return up to top_k jobs that share vocabulary with the query (similarity > 0)."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        empty = self.jobs.iloc[:0][self.COLUMNS].assign(similarity=pd.Series(dtype=float))
        if not query.strip():
            return empty

        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        eligible = scores > 0
        if category is not None:
            eligible &= self.jobs.category.eq(category).to_numpy()

        order = np.argsort(-scores, kind="stable")
        best = order[eligible[order]][:top_k]
        result = self.jobs.iloc[best][self.COLUMNS].copy()
        result["similarity"] = scores[best]
        return result.reset_index(drop=True)
