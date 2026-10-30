"""
ReviewMind — Dataset Service
Provides metadata and cached querying for Dataset A and Dataset B reviews.
"""

from pathlib import Path
from typing import Any, Optional
import json
import logging
import math
import pandas as pd

from backend.config import Config

logger = logging.getLogger("reviewmind.dataset_service")


class DatasetService:
    """Service for dataset metadata inspection and paginated review exploration."""

    def __init__(self) -> None:
        self._cached_dataset_b_reviews: Optional[list[dict[str, Any]]] = None
        self._cached_summary: Optional[dict[str, Any]] = None

    def get_datasets_metadata(self) -> dict[str, Any]:
        """Return structured metadata about Dataset A, Dataset B, and human annotation resources."""
        return {
            "datasets": [
                {
                    "id": "dataset_a",
                    "name": "TripAdvisor Hotel Reviews (Prepared)",
                    "role": "Model Training & Internal Cross-Evaluation",
                    "filename": "reviews_prepared.csv",
                    "total_records": 20491,
                    "columns": ["Review", "Rating", "Sentiment"],
                    "classes": ["negative", "neutral", "positive"],
                    "class_distribution": {
                        "positive": {"count": 15093, "percentage": 73.66},
                        "negative": {"count": 3214, "percentage": 15.68},
                        "neutral": {"count": 2184, "percentage": 10.66},
                    },
                    "features_source": "Derived from numerical 1-5 customer ratings",
                    "availability": "Local disk (excluded from Git per license rules)",
                },
                {
                    "id": "dataset_b",
                    "name": "Eco-hotel Reviews (Calheiros, Moro & Rita, 2017)",
                    "role": "External Zero-Shot Domain Generalization",
                    "filename": "dataset-CalheirosMoroRita-2017.csv",
                    "raw_records": 401,
                    "unique_records": 399,
                    "duplicate_records": 2,
                    "columns": ["Review"],
                    "classes": None,
                    "status": "Model inferences completed (399 unique reviews); human annotation in progress",
                    "availability": "Local disk (excluded from Git per license rules)",
                },
            ],
            "annotation_workspace": {
                "guidelines": "data/external/ANNOTATION_GUIDELINES.md",
                "annotator_a_template": "data/external/annotator_A.csv",
                "annotator_b_template": "data/external/annotator_B.csv",
                "consensus_template": "data/external/annotation_consensus.csv",
                "consensus_status": "pending_human_annotation",
            },
        }

    def _load_dataset_b_cache(self) -> list[dict[str, Any]]:
        """Load external predictions once into memory for fast, thread-safe access."""
        if self._cached_dataset_b_reviews is not None:
            return self._cached_dataset_b_reviews

        predictions_path = Config.EXTERNAL_PREDICTIONS_PATH
        if not predictions_path.exists():
            logger.warning("External predictions CSV not found at %s", predictions_path.name)
            self._cached_dataset_b_reviews = []
            return self._cached_dataset_b_reviews

        try:
            df = pd.read_csv(predictions_path)
            reviews: list[dict[str, Any]] = []

            for _, row in df.iterrows():
                reviews.append({
                    "original_index": int(row["original_index"]),
                    "review": str(row["Review"]).strip(),
                    "predicted_sentiment": str(row["Predicted_Sentiment"]).lower(),
                    "confidence": round(float(row["Confidence"]), 4),
                    "probabilities": {
                        "negative": round(float(row["Probability_Negative"]), 4),
                        "neutral": round(float(row["Probability_Neutral"]), 4),
                        "positive": round(float(row["Probability_Positive"]), 4),
                    },
                })

            self._cached_dataset_b_reviews = reviews
            logger.info("Cached %d Dataset B reviews in memory.", len(reviews))
            return self._cached_dataset_b_reviews

        except Exception as e:
            logger.error("Failed to load external predictions CSV: %s", e, exc_info=True)
            self._cached_dataset_b_reviews = []
            return self._cached_dataset_b_reviews

    def get_reviews(
        self,
        dataset_id: str = "b",
        page: int = 1,
        per_page: int = 10,
        sentiment: Optional[str] = None,
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Query reviews with pagination, optional sentiment filtering, and text search.
        """
        all_reviews = self._load_dataset_b_cache()

        # Filter by sentiment
        filtered = all_reviews
        if sentiment:
            sentiment_clean = sentiment.strip().lower()
            if sentiment_clean in ["positive", "neutral", "negative"]:
                filtered = [r for r in filtered if r["predicted_sentiment"] == sentiment_clean]

        # Filter by text search query
        if query:
            q_clean = query.strip().lower()
            filtered = [r for r in filtered if q_clean in r["review"].lower()]

        total_items = len(filtered)
        per_page = max(1, min(per_page, 100))  # Bound between 1 and 100
        total_pages = max(1, math.ceil(total_items / per_page))
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_items = filtered[start_idx:end_idx]

        return {
            "dataset_id": dataset_id,
            "total_items": total_items,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "filters": {
                "sentiment": sentiment,
                "query": query,
            },
            "reviews": paginated_items,
        }

    def get_review_by_index(self, original_index: int) -> Optional[dict[str, Any]]:
        """Find a review by its original index from Dataset B."""
        all_reviews = self._load_dataset_b_cache()
        for review in all_reviews:
            if review["original_index"] == original_index:
                return review
        return None


# Global singleton instance
dataset_service = DatasetService()

