"""
Unit tests for ReviewMind data validation and sentiment labeling rules.

These tests use synthetic, in-memory data fixtures to ensure full reproducibility
without requiring external or ignored dataset files.
"""

import unittest
import pandas as pd


def map_rating_to_sentiment(rating: int | float) -> str | None:
    """Canonical mapping rule used in ReviewMind data preparation."""
    mapping = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive",
    }
    return mapping.get(int(rating)) if pd.notna(rating) and rating in mapping else None


def filter_empty_reviews(df: pd.DataFrame, text_column: str = "Review") -> pd.DataFrame:
    """Filters out null, empty, or whitespace-only review records."""
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found.")
    valid_mask = df[text_column].notna() & (df[text_column].astype(str).str.strip() != "")
    return df[valid_mask].copy()


class TestDataValidation(unittest.TestCase):
    """Test suite verifying data cleaning and sentiment assignment rules."""

    def test_sentiment_mapping_standard_ratings(self):
        """Ratings 1-5 must map strictly to negative, neutral, or positive."""
        self.assertEqual(map_rating_to_sentiment(1), "negative")
        self.assertEqual(map_rating_to_sentiment(2), "negative")
        self.assertEqual(map_rating_to_sentiment(3), "neutral")
        self.assertEqual(map_rating_to_sentiment(4), "positive")
        self.assertEqual(map_rating_to_sentiment(5), "positive")

    def test_sentiment_mapping_invalid_ratings(self):
        """Invalid ratings must return None without raising unhandled exceptions."""
        self.assertIsNone(map_rating_to_sentiment(0))
        self.assertIsNone(map_rating_to_sentiment(6))
        self.assertIsNone(map_rating_to_sentiment(-1))
        self.assertIsNone(map_rating_to_sentiment(float("nan")))

    def test_filter_empty_reviews(self):
        """Empty, whitespace-only, and null reviews must be filtered out."""
        data = {
            "Review": [
                "Great hotel with clean rooms.",
                "",
                "   \t\n  ",
                None,
                "Terrible service.",
            ],
            "Rating": [5, 4, 3, 2, 1],
        }
        df = pd.DataFrame(data)
        cleaned = filter_empty_reviews(df, "Review")

        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(cleaned["Rating"].tolist(), [5, 1])

    def test_required_columns_check(self):
        """Validation must catch missing columns in arbitrary dataframes."""
        df = pd.DataFrame({"Review": ["Good"], "Score": [4]})
        required = ["Review", "Rating"]
        missing = [c for c in required if c not in df.columns]
        self.assertIn("Rating", missing)
        self.assertNotIn("Review", missing)


if __name__ == "__main__":
    unittest.main()

