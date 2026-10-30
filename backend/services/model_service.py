"""
ReviewMind — Machine Learning Model Service
Loads and serves predictions using the trained production pipeline:
TfidfVectorizer(ngram_range=(1, 2), max_features=20000) + MultinomialNB(alpha=0.1, fit_prior=False).
"""

from pathlib import Path
from typing import Any, Optional
import json
import logging
import joblib
import numpy as np

from backend.config import Config

logger = logging.getLogger("reviewmind.model_service")


class ModelService:
    """Singleton service for loading the trained ReviewMind pipeline and executing inferences."""

    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> None:
        self.model_path = model_path or Config.FINAL_MODEL_PATH
        self.metadata_path = metadata_path or Config.FINAL_METADATA_PATH
        self.model: Any = None
        self.metadata: dict[str, Any] = {}
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None

        # Auto-load on instantiation
        self.load_model()

    def load_model(self) -> bool:
        """
        Load the pre-trained model pipeline from disk once.
        Does not retrain or re-fit any component.
        """
        if self._is_loaded and self.model is not None:
            return True

        if not self.model_path.exists():
            self._load_error = f"Model file not found at expected path: {self.model_path.name}"
            logger.warning(self._load_error)
            self._is_loaded = False
            return False

        try:
            logger.info("Loading ReviewMind final model from %s...", self.model_path.name)
            self.model = joblib.load(self.model_path)
            self._is_loaded = True
            self._load_error = None

            # Load accompanying metadata if present
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                except Exception as meta_err:
                    logger.warning("Could not read metadata JSON: %s", meta_err)

            logger.info("Model pipeline loaded successfully with %d classes: %s",
                        len(self.classes), self.classes)
            return True

        except Exception as e:
            self._load_error = f"Error during model loading: {str(e)}"
            logger.error(self._load_error, exc_info=True)
            self.model = None
            self._is_loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        """Return True if model is currently loaded in memory."""
        return self._is_loaded and self.model is not None

    @property
    def load_error(self) -> Optional[str]:
        """Return error string if loading failed."""
        return self._load_error

    @property
    def classes(self) -> list[str]:
        """Return canonical list of classes."""
        if self.is_loaded and hasattr(self.model, "classes_"):
            return [str(c) for c in self.model.classes_]
        return ["negative", "neutral", "positive"]

    def extract_salient_features(self, text: str, top_k: int = 8) -> list[dict[str, Any]]:
        """
        Extract active n-grams present in the review that correspond to the
        TfidfVectorizer vocabulary, ranked by their TF-IDF weight.
        """
        if not self.is_loaded or not hasattr(self.model, "named_steps"):
            return []

        try:
            tfidf_step = self.model.named_steps.get("tfidf")
            if not tfidf_step:
                return []

            # Transform single text document
            vector = tfidf_step.transform([text])
            feature_indices = vector.nonzero()[1]
            feature_names = tfidf_step.get_feature_names_out()

            weighted_tokens = []
            for idx in feature_indices:
                weight = float(vector[0, idx])
                token = str(feature_names[idx])
                weighted_tokens.append({"term": token, "weight": round(weight, 4)})

            # Sort by weight descending
            weighted_tokens.sort(key=lambda x: x["weight"], reverse=True)
            return weighted_tokens[:top_k]

        except Exception as err:
            logger.debug("Feature extraction non-critical error: %s", err)
            return []

    def predict(self, text: str) -> dict[str, Any]:
        """
        Execute inference on a single review string.
        Returns predicted sentiment, confidence, probabilities, and active n-grams.
        """
        if not self.is_loaded:
            raise RuntimeError(self._load_error or "Model is not loaded.")

        cleaned_text = str(text).strip()
        if not cleaned_text:
            raise ValueError("Review text cannot be empty or whitespace only.")

        # Real model inference
        raw_pred = self.model.predict([cleaned_text])[0]
        raw_probs = self.model.predict_proba([cleaned_text])[0]
        classes = self.classes

        # Map probabilities to classes
        prob_dict = {
            cls_name: round(float(raw_probs[i]), 4)
            for i, cls_name in enumerate(classes)
        }

        # Confidence is the maximum probability across classes
        confidence = round(float(np.max(raw_probs)), 4)
        predicted_sentiment = str(raw_pred)

        # Extract active vocabulary features
        salient_features = self.extract_salient_features(cleaned_text)

        return {
            "predicted_sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": prob_dict,
            "salient_features": salient_features,
            "review_length_chars": len(cleaned_text),
            "review_word_count": len(cleaned_text.split()),
        }

    def get_info(self) -> dict[str, Any]:
        """Return safe, non-sensitive model summary information."""
        info: dict[str, Any] = {
            "model_name": "ReviewMind Final Production Model",
            "model_loaded": self.is_loaded,
            "status": "ready" if self.is_loaded else "unavailable",
            "classes": self.classes,
            "architecture": "TfidfVectorizer(ngram_range=(1, 2), max_features=20000) + MultinomialNB(alpha=0.1, fit_prior=False)",
            "trained_on": "100% of TripAdvisor Dataset A (20,491 reviews)",
        }

        if self.is_loaded and hasattr(self.model, "named_steps"):
            tfidf = self.model.named_steps.get("tfidf")
            nb = self.model.named_steps.get("nb")
            if tfidf:
                info["vocabulary_size"] = len(getattr(tfidf, "vocabulary_", {}))
                info["ngram_range"] = list(getattr(tfidf, "ngram_range", (1, 2)))
            if nb:
                info["alpha"] = float(getattr(nb, "alpha", 0.1))
                info["fit_prior"] = bool(getattr(nb, "fit_prior", False))

        if self._load_error:
            info["error"] = self._load_error

        return info


# Global singleton instance
model_service = ModelService()

