"""
Unit & Integration Tests for ReviewMind Flask REST API Backend
Tests health, predictions, dataset metadata, review retrieval, and evaluation endpoints.
"""

from pathlib import Path
import json
import unittest

from backend.app import create_app
from backend.services.model_service import model_service


class TestBackendAPI(unittest.TestCase):
    """Test suite for ReviewMind Flask REST API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client once for all test methods."""
        cls.app = create_app("testing")
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        """Test GET /api/health returns 200 and model status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("status", data)
        self.assertIn("service", data)
        self.assertEqual(data["service"], "ReviewMind REST API")
        self.assertIn("model", data)
        self.assertIn("model_loaded", data["model"])
        self.assertTrue(data["model"]["model_loaded"])
        self.assertEqual(data["model"]["classes"], ["negative", "neutral", "positive"])

    def test_predict_successful_positive(self):
        """Test POST /api/predict with an unambiguously positive review."""
        payload = {
            "review": "The hotel was absolutely fantastic! The staff was friendly, the room was immaculate and cozy, and the breakfast was superb."
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["predicted_sentiment"], "positive")
        self.assertGreater(data["confidence"], 0.5)
        self.assertIn("probabilities", data)
        self.assertIn("negative", data["probabilities"])
        self.assertIn("neutral", data["probabilities"])
        self.assertIn("positive", data["probabilities"])

        # Probabilities must sum close to 1.0
        prob_sum = sum(data["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Salient features must be present
        self.assertIsInstance(data["salient_features"], list)
        self.assertTrue(len(data["salient_features"]) > 0)

    def test_predict_successful_negative(self):
        """Test POST /api/predict with an unambiguously negative review."""
        payload = {
            "review": "Horrible dirty hotel. Broken air conditioning, rude desk clerk, filthy bathroom, and refused to refund our night."
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["predicted_sentiment"], "negative")
        self.assertGreater(data["probabilities"]["negative"], data["probabilities"]["positive"])

    def test_predict_empty_input(self):
        """Test POST /api/predict with empty string or whitespace returns 400."""
        # Completely empty
        response1 = self.client.post(
            "/api/predict",
            data=json.dumps({"review": ""}),
            content_type="application/json",
        )
        self.assertEqual(response1.status_code, 400)
        data1 = response1.get_json()
        self.assertEqual(data1["status"], "empty_input")

        # Whitespace only
        response2 = self.client.post(
            "/api/predict",
            data=json.dumps({"review": "    \n\t   "}),
            content_type="application/json",
        )
        self.assertEqual(response2.status_code, 400)
        data2 = response2.get_json()
        self.assertEqual(data2["status"], "empty_input")

    def test_predict_missing_field(self):
        """Test POST /api/predict with missing 'review' key returns 400."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps({"text": "Hello world"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "missing_field")

    def test_predict_invalid_data_type(self):
        """Test POST /api/predict with non-string review returns 400."""
        response = self.client.post(
            "/api/predict",
            data=json.dumps({"review": 12345}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "invalid_type")

    def test_predict_non_json_request(self):
        """Test POST /api/predict with non-JSON content-type returns 400."""
        response = self.client.post(
            "/api/predict",
            data="plain text review",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["status"], "invalid_payload")

    def test_datasets_metadata_endpoint(self):
        """Test GET /api/datasets returns valid metadata for Dataset A and Dataset B."""
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("datasets", data)
        self.assertEqual(len(data["datasets"]), 2)

        dataset_ids = [d["id"] for d in data["datasets"]]
        self.assertIn("dataset_a", dataset_ids)
        self.assertIn("dataset_b", dataset_ids)

        self.assertIn("annotation_workspace", data)
        self.assertIn("consensus_status", data["annotation_workspace"])

    def test_dashboard_endpoint(self):
        """Test GET /api/dashboard returns executive KPIs from verified reports."""
        response = self.client.get("/api/dashboard")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("active_model", data)
        self.assertEqual(data["active_model"]["model_id"], "model_3")
        self.assertEqual(data["active_model"]["macro_f1"], 0.6762)
        self.assertEqual(data["active_model"]["neutral_f1"], 0.3883)
        self.assertIn("improvement_deltas", data)
        self.assertIn("dataset_a_summary", data)
        self.assertEqual(data["dataset_a_summary"]["total_records"], 20491)

    def test_reviews_pagination_and_filtering(self):
        """Test GET /api/reviews pagination and sentiment filter."""
        # Default pagination
        response = self.client.get("/api/reviews?page=1&per_page=5")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["page"], 1)
        self.assertEqual(data["per_page"], 5)
        self.assertEqual(len(data["reviews"]), 5)
        self.assertGreater(data["total_items"], 0)

        # Filter positive reviews
        response_pos = self.client.get("/api/reviews?sentiment=positive&per_page=10")
        self.assertEqual(response_pos.status_code, 200)
        data_pos = response_pos.get_json()
        for rev in data_pos["reviews"]:
            self.assertEqual(rev["predicted_sentiment"], "positive")

        # Filter negative reviews
        response_neg = self.client.get("/api/reviews?sentiment=negative&per_page=10")
        self.assertEqual(response_neg.status_code, 200)
        data_neg = response_neg.get_json()
        for rev in data_neg["reviews"]:
            self.assertEqual(rev["predicted_sentiment"], "negative")

    def test_single_review_endpoint(self):
        """Test GET /api/reviews/<original_index> returns review or 404."""
        # Index 0 exists in Dataset B
        response = self.client.get("/api/reviews/0")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["review"]["original_index"], 0)

        # Nonexistent index returns 404
        response_404 = self.client.get("/api/reviews/999999")
        self.assertEqual(response_404.status_code, 404)
        data_404 = response_404.get_json()
        self.assertEqual(data_404["status"], "not_found")

    def test_evaluation_pending_status(self):
        """
        Test GET /api/evaluation returns verified internal metrics
        and strictly returns pending status for external consensus.
        """
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn("internal_evaluation", data)
        self.assertIn("models_benchmark", data["internal_evaluation"])
        self.assertIn("baseline_confusion_matrix", data["internal_evaluation"])

        # External evaluation MUST be pending
        self.assertIn("external_evaluation", data)
        ext = data["external_evaluation"]
        self.assertEqual(ext["supervised_metrics_status"], "pending")
        self.assertFalse(ext["human_consensus_available"])
        self.assertIn("pending", ext["notice"].lower())


if __name__ == "__main__":
    unittest.main()

