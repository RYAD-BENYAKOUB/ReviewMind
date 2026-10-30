"""
ReviewMind — Health Endpoint
GET /api/health
"""

from datetime import datetime, timezone
from flask import Blueprint, jsonify

from backend.services.model_service import model_service

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def get_health():
    """
    Return API health status, model loading status, and system metadata.
    """
    model_info = model_service.get_info()

    return jsonify({
        "status": "healthy" if model_service.is_loaded else "degraded",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "service": "ReviewMind REST API",
        "version": "1.0.0",
        "model": model_info,
    }), 200

