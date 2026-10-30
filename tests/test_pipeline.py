"""
Unit tests for ReviewMind machine learning pipeline and inference components.

Tests pipeline creation, feature extraction, Multinomial Naive Bayes fitting,
and prediction probability formatting using small synthetic text fixtures.
"""

from pathlib import Path
import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    """Test suite verifying TF-IDF and Naive Bayes modeling components."""

    def setUp(self):
        """Set up a synthetic toy corpus covering all three sentiment classes."""
        self.X_train = [
            "excellent hotel wonderful stay clean rooms helpful staff",
            "loved everything about this hotel fantastic breakfast",
            "superb experience highly recommend this wonderful place",
            "average hotel okay stay nothing special decent room",
            "room was fine breakfast was mediocre location is ok",
            "standard stay neither good nor bad ordinary hotel",
            "terrible experience dirty room rude front desk never again",
            "awful stay noisy air conditioning bad service avoid",
            "horrible customer service overpriced and dirty rooms",
        ]
        self.y_train = [
            "positive",
            "positive",
            "positive",
            "neutral",
            "neutral",
            "neutral",
            "negative",
            "negative",
            "negative",
        ]

        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
                ("nb", MultinomialNB(alpha=0.1, fit_prior=False)),
            ]
        )

    def test_pipeline_fit_and_predict(self):
        """Pipeline must fit on synthetic data and output valid sentiment labels."""
        self.pipeline.fit(self.X_train, self.y_train)

        test_reviews = [
            "fantastic hotel wonderful staff",
            "average room okay breakfast",
            "horrible dirty and rude",
        ]
        predictions = self.pipeline.predict(test_reviews)

        self.assertEqual(len(predictions), 3)
        valid_classes = {"negative", "neutral", "positive"}
        for pred in predictions:
            self.assertIn(pred, valid_classes)

    def test_pipeline_predict_proba(self):
        """Probability outputs must sum to 1.0 across the 3 sentiment classes."""
        self.pipeline.fit(self.X_train, self.y_train)

        test_reviews = ["clean beautiful room and great service"]
        probas = self.pipeline.predict_proba(test_reviews)

        self.assertEqual(probas.shape, (1, 3))
        self.assertAlmostEqual(float(np.sum(probas)), 1.0, places=5)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probas[0]))

    def test_production_model_if_present(self):
        """If the local serialized model is present, test loading and inference."""
        model_path = Path(__file__).resolve().parent.parent / "models" / "reviewmind_final_model.joblib"
        if not model_path.exists():
            self.skipTest(f"Production model '{model_path.name}' is not present; skipping.")

        import joblib
        model = joblib.load(model_path)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

        sample_review = ["The room was clean, quiet and comfortable with excellent staff."]
        pred = model.predict(sample_review)
        self.assertIn(pred[0], ["negative", "neutral", "positive"])


if __name__ == "__main__":
    unittest.main()

