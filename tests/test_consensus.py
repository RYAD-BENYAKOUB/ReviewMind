"""
Unit tests for ReviewMind Human Annotation Consensus Pipeline.

Tests:
1. Normalization and validation of sentiment labels.
2. Inter-annotator agreement calculations (observed agreement and Cohen's Kappa).
3. File schema validation (missing columns, duplicate IDs, non-integer IDs).
4. Consensus decision logic:
   - Case A: Full agreement between Annotator A and B
   - Case B: Disagreement flagging and adjudication queue routing
   - Case C: Incomplete / missing labels
   - Case D: External adjudication resolution
5. End-to-end pipeline determinism and report artifact generation.

These tests use synthetic, in-memory/temporary file fixtures to guarantee
complete reproducibility without modifying any production datasets or reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.compute_annotation_consensus import (
    COL_ID,
    COL_NOTES,
    COL_REVIEW,
    COL_SENTIMENT,
    STATUS_ADJUDICATED,
    STATUS_AGREEMENT,
    STATUS_DISAGREEMENT,
    STATUS_INCOMPLETE,
    VALID_SENTIMENTS,
    compute_cohens_kappa,
    interpret_kappa,
    load_and_validate_annotator_file,
    normalize_sentiment,
    run_consensus_pipeline,
)


class TestSentimentNormalization(unittest.TestCase):
    """Test sentiment string cleaning and vocabulary validation."""

    def test_valid_sentiments_normalization(self):
        """Standard valid labels with whitespace or uppercase must normalize cleanly."""
        self.assertEqual(normalize_sentiment("positive"), "positive")
        self.assertEqual(normalize_sentiment("POSITIVE"), "positive")
        self.assertEqual(normalize_sentiment("  negative  "), "negative")
        self.assertEqual(normalize_sentiment("Neutral"), "neutral")

    def test_missing_and_empty_sentiments(self):
        """Null, empty, or whitespace-only labels must return None."""
        self.assertIsNone(normalize_sentiment(None))
        self.assertIsNone(normalize_sentiment(float("nan")))
        self.assertIsNone(normalize_sentiment(""))
        self.assertIsNone(normalize_sentiment("   \t  \n"))

    def test_invalid_sentiment_raises_value_error(self):
        """Labels outside {negative, neutral, positive} must raise ValueError."""
        with self.assertRaises(ValueError):
            normalize_sentiment("happy")
        with self.assertRaises(ValueError):
            normalize_sentiment("mixed")
        with self.assertRaises(ValueError):
            normalize_sentiment("very positive")
        with self.assertRaises(ValueError):
            normalize_sentiment("5")


class TestCohenKappa(unittest.TestCase):
    """Test mathematical precision and edge cases of Cohen's Kappa calculation."""

    def test_empty_or_mismatched_inputs(self):
        """Empty lists return None; mismatched lengths raise ValueError."""
        self.assertIsNone(compute_cohens_kappa([], []))
        with self.assertRaises(ValueError):
            compute_cohens_kappa(["positive"], ["positive", "negative"])

    def test_perfect_agreement(self):
        """Identical annotations must yield Cohen's Kappa = 1.0."""
        a = ["positive", "negative", "neutral", "positive", "neutral"]
        b = ["positive", "negative", "neutral", "positive", "neutral"]
        kappa = compute_cohens_kappa(a, b)
        self.assertIsNotNone(kappa)
        self.assertAlmostEqual(kappa, 1.0, places=4)

    def test_known_hand_calculated_contingency(self):
        """
        Verify Cohen's Kappa against an exact manual calculation:
        Sample size N = 10:
        A: 4 positive, 3 neutral, 3 negative
        B: 3 positive, 3 neutral, 4 negative
        Agreements: 3 pos, 2 neu, 3 neg = 8 agreements -> P_o = 0.80
        Disagreements: (pos, neu) x 1, (neu, neg) x 1
        P_e = (4*3 + 3*3 + 3*4) / 100 = 33 / 100 = 0.33
        Expected Kappa = (0.80 - 0.33) / (1 - 0.33) = 0.47 / 0.67 = 0.7014925...
        """
        labels_a = [
            "positive", "positive", "positive",
            "neutral", "neutral", "neutral",
            "negative", "negative", "negative",
            "positive"
        ]
        labels_b = [
            "positive", "positive", "neutral",
            "neutral", "neutral", "negative",
            "negative", "negative", "negative",
            "positive"
        ]
        kappa = compute_cohens_kappa(labels_a, labels_b)
        self.assertIsNotNone(kappa)
        expected_kappa = (0.80 - 0.33) / (1.0 - 0.33)
        self.assertAlmostEqual(kappa, expected_kappa, places=4)

    def test_complete_disagreement(self):
        """Systematic disagreement yields negative Kappa."""
        labels_a = ["positive", "positive", "neutral", "negative"]
        labels_b = ["negative", "negative", "negative", "neutral"]
        kappa = compute_cohens_kappa(labels_a, labels_b)
        self.assertIsNotNone(kappa)
        self.assertLess(kappa, 0.0)

    def test_interpretation_scale(self):
        """Verify Landis & Koch (1977) benchmark interpretations."""
        self.assertIn("Not Applicable", interpret_kappa(None))
        self.assertIn("Poor", interpret_kappa(-0.15))
        self.assertIn("Slight", interpret_kappa(0.12))
        self.assertIn("Fair", interpret_kappa(0.32))
        self.assertIn("Moderate", interpret_kappa(0.52))
        self.assertIn("Substantial", interpret_kappa(0.72))
        self.assertIn("Almost perfect", interpret_kappa(0.92))


class TestAnnotatorFileValidation(unittest.TestCase):
    """Test schema and integrity validation of input annotator CSV files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_not_found(self):
        """Non-existent file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_and_validate_annotator_file(self.base_path / "nonexistent.csv", "Annotator A")

    def test_empty_file_raises_error(self):
        """Empty file must raise ValueError."""
        empty_csv = self.base_path / "empty.csv"
        empty_csv.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_and_validate_annotator_file(empty_csv, "Annotator A")

    def test_missing_required_column(self):
        """Missing Original_Index or Review column must raise ValueError."""
        csv_file = self.base_path / "missing_cols.csv"
        df = pd.DataFrame({
            "Review": ["Nice stay"],
            "Human_Sentiment": ["positive"]
        })
        df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_annotator_file(csv_file, "Annotator A")
        self.assertIn("Missing required column", str(ctx.exception))

    def test_duplicate_ids_rejected(self):
        """Duplicate Original_Index values must raise ValueError."""
        csv_file = self.base_path / "duplicates.csv"
        df = pd.DataFrame({
            "Original_Index": [1, 1],
            "Review": ["Stay 1", "Stay 2"],
            "Human_Sentiment": ["positive", "negative"]
        })
        df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_annotator_file(csv_file, "Annotator A")
        self.assertIn("Duplicate review IDs detected", str(ctx.exception))

    def test_non_integer_id_rejected(self):
        """Non-integer Original_Index must raise ValueError."""
        csv_file = self.base_path / "bad_ids.csv"
        df = pd.DataFrame({
            "Original_Index": ["one", "two"],
            "Review": ["Stay 1", "Stay 2"],
            "Human_Sentiment": ["positive", "neutral"]
        })
        df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_annotator_file(csv_file, "Annotator A")
        self.assertIn("integer IDs", str(ctx.exception))

    def test_invalid_label_in_file_rejected(self):
        """Invalid sentiment string inside CSV must raise ValueError."""
        csv_file = self.base_path / "invalid_label.csv"
        df = pd.DataFrame({
            "Original_Index": [1],
            "Review": ["Stay 1"],
            "Human_Sentiment": ["awesome"]
        })
        df.to_csv(csv_file, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_annotator_file(csv_file, "Annotator A")
        self.assertIn("Validation error", str(ctx.exception))


class TestConsensusPipelineCases(unittest.TestCase):
    """
    Test the four consensus decision logic cases:
    - Case A (Agreement)
    - Case B (Disagreement -> requires_adjudication)
    - Case C (Incomplete -> one or both missing)
    - Case D (Adjudicated -> resolved by third party)
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        self.annotator_a_path = self.base_path / "annotator_A.csv"
        self.annotator_b_path = self.base_path / "annotator_B.csv"
        self.adjudications_path = self.base_path / "adjudications.csv"

        self.consensus_out = self.base_path / "consensus.csv"
        self.disagreements_out = self.base_path / "disagreements.csv"
        self.progress_out = self.base_path / "progress.json"
        self.reports_dir = self.base_path / "reports"

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_sample_annotators(self):
        """
        Creates synthetic datasets for 4 reviews illustrating Cases A, B, C:
        - Review 1: A="positive", B="positive" (Case A: Agreement)
        - Review 2: A="positive", B="negative" (Case B: Disagreement)
        - Review 3: A="neutral", B="" (Case C: Incomplete)
        - Review 4: A="", B="" (Case C: Incomplete)
        """
        data_a = {
            "Original_Index": [1, 2, 3, 4],
            "Review": ["Excellent room", "Noisy and bad", "Average stay", "Unread review"],
            "Human_Sentiment": ["positive", "positive", "neutral", ""],
            "Annotation_Notes": ["clear positive", "loved the bed", "neutral tone", ""],
        }
        data_b = {
            "Original_Index": [1, 2, 3, 4],
            "Review": ["Excellent room", "Noisy and bad", "Average stay", "Unread review"],
            "Human_Sentiment": ["positive", "negative", "", ""],
            "Annotation_Notes": ["very good", "very loud street", "", ""],
        }
        pd.DataFrame(data_a).to_csv(self.annotator_a_path, index=False)
        pd.DataFrame(data_b).to_csv(self.annotator_b_path, index=False)

    def test_consensus_cases_a_b_c(self):
        """Verify Cases A, B, and C without external adjudications."""
        self._create_sample_annotators()

        result = run_consensus_pipeline(
            annotator_a_path=self.annotator_a_path,
            annotator_b_path=self.annotator_b_path,
            consensus_output_path=self.consensus_out,
            disagreements_output_path=self.disagreements_out,
            progress_output_path=self.progress_out,
            reports_dir=self.reports_dir,
        )

        self.assertEqual(result["total_unique_reviews"], 4)
        self.assertEqual(result["annotator_A"]["completed"], 3)
        self.assertEqual(result["annotator_B"]["completed"], 2)
        self.assertEqual(result["double_labeled_count"], 2)
        self.assertEqual(result["agreements_count"], 1)
        self.assertEqual(result["disagreements_count"], 1)
        self.assertEqual(result["incomplete_count"], 2)

        # Inspect generated consensus CSV
        df_cons = pd.read_csv(self.consensus_out)
        self.assertEqual(len(df_cons), 4)

        # Case A: Agreement (Review 1)
        row1 = df_cons[df_cons["Original_Index"] == 1].iloc[0]
        self.assertEqual(row1["Annotation_Status"], STATUS_AGREEMENT)
        self.assertEqual(row1["Consensus_Sentiment"], "positive")

        # Case B: Disagreement (Review 2)
        row2 = df_cons[df_cons["Original_Index"] == 2].iloc[0]
        self.assertEqual(row2["Annotation_Status"], STATUS_DISAGREEMENT)
        self.assertTrue(pd.isna(row2["Consensus_Sentiment"]) or row2["Consensus_Sentiment"] == "")

        # Case C: Incomplete (Review 3 and 4)
        row3 = df_cons[df_cons["Original_Index"] == 3].iloc[0]
        self.assertEqual(row3["Annotation_Status"], STATUS_INCOMPLETE)
        row4 = df_cons[df_cons["Original_Index"] == 4].iloc[0]
        self.assertEqual(row4["Annotation_Status"], STATUS_INCOMPLETE)

        # Inspect generated disagreements CSV
        df_dis = pd.read_csv(self.disagreements_out)
        self.assertEqual(len(df_dis), 1)
        self.assertEqual(int(df_dis.iloc[0]["Original_Index"]), 2)
        self.assertEqual(df_dis.iloc[0]["Annotator_A_Sentiment"], "positive")
        self.assertEqual(df_dis.iloc[0]["Annotator_B_Sentiment"], "negative")

    def test_case_d_external_adjudication(self):
        """Case D: An external adjudication file resolves a disputed review."""
        self._create_sample_annotators()

        # Create adjudication resolving Review 2 to 'negative'
        adj_data = {
            "Original_Index": [2],
            "Adjudicated_Sentiment": ["negative"],
            "Adjudication_Notes": ["Reviewer focused on severe noise and dirty sheets."],
        }
        pd.DataFrame(adj_data).to_csv(self.adjudications_path, index=False)

        result = run_consensus_pipeline(
            annotator_a_path=self.annotator_a_path,
            annotator_b_path=self.annotator_b_path,
            consensus_output_path=self.consensus_out,
            disagreements_output_path=self.disagreements_out,
            progress_output_path=self.progress_out,
            reports_dir=self.reports_dir,
            adjudications_path=self.adjudications_path,
        )

        df_cons = pd.read_csv(self.consensus_out)
        row2 = df_cons[df_cons["Original_Index"] == 2].iloc[0]
        self.assertEqual(row2["Annotation_Status"], STATUS_ADJUDICATED)
        self.assertEqual(row2["Consensus_Sentiment"], "negative")
        self.assertIn("Adjudicator: Reviewer focused", str(row2["Annotation_Notes"]))

        # Check finalized count: 1 agreement (Review 1) + 1 adjudication (Review 2) = 2 finalized
        self.assertEqual(result["finalized_consensus_count"], 2)
        self.assertEqual(result["finalized_consensus_percentage"], 50.0)

    def test_mismatched_ids_between_annotators(self):
        """Mismatched Original_Index sets between Annotators must raise ValueError."""
        df_a = pd.DataFrame({
            "Original_Index": [1, 2, 3],
            "Review": ["A", "B", "C"],
            "Human_Sentiment": ["positive", "positive", "neutral"]
        })
        df_b = pd.DataFrame({
            "Original_Index": [1, 2, 4],  # ID 4 instead of 3
            "Review": ["A", "B", "D"],
            "Human_Sentiment": ["positive", "positive", "neutral"]
        })
        df_a.to_csv(self.annotator_a_path, index=False)
        df_b.to_csv(self.annotator_b_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            run_consensus_pipeline(
                annotator_a_path=self.annotator_a_path,
                annotator_b_path=self.annotator_b_path,
                consensus_output_path=self.consensus_out,
                disagreements_output_path=self.disagreements_out,
                progress_output_path=self.progress_out,
                reports_dir=self.reports_dir,
            )
        self.assertIn("Review ID mismatch", str(ctx.exception))

    def test_deterministic_output(self):
        """Running the pipeline twice on identical data produces identical outputs."""
        self._create_sample_annotators()

        run1 = run_consensus_pipeline(
            annotator_a_path=self.annotator_a_path,
            annotator_b_path=self.annotator_b_path,
            consensus_output_path=self.consensus_out,
            disagreements_output_path=self.disagreements_out,
            progress_output_path=self.progress_out,
            reports_dir=self.reports_dir,
        )
        csv1 = self.consensus_out.read_text(encoding="utf-8")
        json1 = self.progress_out.read_text(encoding="utf-8")

        run2 = run_consensus_pipeline(
            annotator_a_path=self.annotator_a_path,
            annotator_b_path=self.annotator_b_path,
            consensus_output_path=self.consensus_out,
            disagreements_output_path=self.disagreements_out,
            progress_output_path=self.progress_out,
            reports_dir=self.reports_dir,
        )
        csv2 = self.consensus_out.read_text(encoding="utf-8")
        json2 = self.progress_out.read_text(encoding="utf-8")

        self.assertEqual(csv1, csv2)
        self.assertEqual(json1, json2)
        self.assertEqual(run1, run2)


if __name__ == "__main__":
    unittest.main()
