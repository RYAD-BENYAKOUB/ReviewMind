"""
ReviewMind — Routes Blueprint Aggregator
Registers all API endpoints under the /api URL prefix.
"""

from flask import Blueprint

from .health import health_bp
from .predictions import predictions_bp
from .datasets import datasets_bp
from .reviews import reviews_bp
from .evaluation import evaluation_bp

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Register child blueprints on the main api blueprint
api_bp.register_blueprint(health_bp)
api_bp.register_blueprint(predictions_bp)
api_bp.register_blueprint(datasets_bp)
api_bp.register_blueprint(reviews_bp)
api_bp.register_blueprint(evaluation_bp)

__all__ = ["api_bp"]

