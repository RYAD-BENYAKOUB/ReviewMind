"""
ReviewMind — Prediction Route
POST /api/predict
"""

from flask import Blueprint, request, jsonify
import logging

from backend.services.model_service import model_service

logger = logging.getLogger("reviewmind.routes.predictions")
predictions_bp = Blueprint("predictions", __name__)

MAX_REVIEW_LENGTH = 50000  # Safe upper bound for text size


@predictions_bp.route("/predict", methods=["POST"])
def predict_sentiment():
    """
    Accept review text and predict its sentiment using the trained ReviewMind pipeline.
    Expects JSON payload: {"review": "Text of the review..."}
    """
    # Check if request has JSON
    if not request.is_json:
        return jsonify({
            "error": "Bad Request",
            "message": "Request payload must be formatted as application/json.",
            "status": "invalid_payload",
        }), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({
            "error": "Bad Request",
            "message": "Malformed JSON body.",
            "status": "invalid_json",
        }), 400

    if "review" not in data:
        return jsonify({
            "error": "Validation Error",
            "message": "Missing required field: 'review'.",
            "status": "missing_field",
        }), 400

    raw_text = data["review"]

    if not isinstance(raw_text, str):
        return jsonify({
            "error": "Validation Error",
            "message": "The 'review' field must be a string.",
            "status": "invalid_type",
        }), 400

    cleaned_text = raw_text.strip()
    if not cleaned_text:
        return jsonify({
            "error": "Validation Error",
            "message": "Review text cannot be empty or contain only whitespace.",
            "status": "empty_input",
        }), 400

    if len(cleaned_text) > MAX_REVIEW_LENGTH:
        return jsonify({
            "error": "Payload Too Large",
            "message": f"Review text exceeds maximum allowed length ({MAX_REVIEW_LENGTH} characters).",
            "status": "text_too_long",
        }), 413

    # Check model loading status
    if not model_service.is_loaded:
        return jsonify({
            "error": "Service Unavailable",
            "message": "The machine learning model is currently not loaded. Please verify model artifacts.",
            "status": "model_unavailable",
        }), 503

    try:
        # Run real trained model inference
        result = model_service.predict(cleaned_text)

        response = {
            "status": "success",
            "review": cleaned_text,
            "predicted_sentiment": result["predicted_sentiment"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "salient_features": result["salient_features"],
            "validation": {
                "is_valid": True,
                "length_chars": result["review_length_chars"],
                "word_count": result["review_word_count"],
            },
            "model_info": {
                "name": "ReviewMind Final Production Model",
                "algorithm": "MultinomialNB(alpha=0.1, fit_prior=False)",
                "vectorizer": "TfidfVectorizer(ngram_range=(1, 2), max_features=20000)",
            },
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error("Inference execution failed: %s", e, exc_info=True)
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during model prediction.",
            "status": "inference_error",
        }), 500

