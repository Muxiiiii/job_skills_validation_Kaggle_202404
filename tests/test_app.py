"""Streamlit AppTest checks: startup, bad uploads, filters, training, search, switching files."""

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
UPLOADER = "streamlit.delta_generator.DeltaGenerator.file_uploader"


class Upload(io.BytesIO):
    name = "uploaded_jobs.csv"


def fake_upload(frame):
    """Make st.file_uploader return this DataFrame as an uploaded CSV."""
    return patch(UPLOADER, return_value=Upload(frame.to_csv(index=False).encode()))


def sample_jobs():
    """30 valid jobs: 15 IT, 15 SALES."""
    rows = []
    for category, text in [("IT", "Python programming"), ("SALES", "Customer sales")]:
        for i in range(15):
            rows.append({"job_id": f"{category}{i}", "category": category,
                         "job_title": f"{category} Specialist", "job_description": f"{text} role number{i}",
                         "job_skill_set": "['Python']" if category == "IT" else "['Sales']"})
    return pd.DataFrame(rows)


@unittest.skipUnless(HAS_STREAMLIT, "Install streamlit to run the dashboard tests.")
class DashboardTests(unittest.TestCase):

    def setUp(self):
        from streamlit.testing.v1 import AppTest

        # Point ROOT at an empty folder so a local data/raw file is never picked up.
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
        source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({folder.name!r})")
        self.app = AppTest.from_string(source, default_timeout=30)

    def click(self, label):
        next(b for b in self.app.button if b.label == label).click().run()
        self.assertFalse(self.app.exception)

    def test_starts_without_data(self):
        with patch(UPLOADER, return_value=None):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Upload the original" in m.value for m in self.app.info))

    def test_wrong_columns(self):
        with fake_upload(pd.DataFrame({"wrong_column": [1]})):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertIn("Missing required columns", self.app.error[0].value)

    def test_all_rows_rejected(self):
        frame = sample_jobs().iloc[:1].assign(job_skill_set="invalid")
        with fake_upload(frame):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("No valid jobs" in m.value for m in self.app.warning))

    def test_filters_training_search_and_new_file(self):
        with fake_upload(sample_jobs()):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertEqual(self.app.metric[0].value, "30")

            # filters
            self.app.multiselect(key="categories").set_value([]).run()
            self.assertTrue(any("No jobs match" in m.value for m in self.app.warning))
            self.app.multiselect(key="categories").set_value(["IT"]).run()
            self.assertEqual(self.app.metric[0].value, "15")

            # baseline + prediction
            self.click("Train baseline")
            self.assertIn("baseline_result", self.app.session_state)
            self.app.text_area(key="prediction_text").set_value("Python software development")
            self.click("Predict category")

            # similar-job search
            self.app.text_input(key="search_query").set_value("Python")
            self.click("Find similar jobs")
            self.assertTrue(any("similarity" in df.value.columns for df in self.app.dataframe))

        # a different file resets the trained model and filters
        with fake_upload(sample_jobs().iloc[:10]):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertNotIn("baseline_result", self.app.session_state)
            self.assertEqual(self.app.metric[0].value, "10")


if __name__ == "__main__":
    unittest.main()
