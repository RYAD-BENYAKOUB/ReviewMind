"""
ReviewMind — Datasets & Dashboard Routes
GET /api/datasets
GET /api/dashboard
"""

from flask import Blueprint, jsonify

from backend.services.dataset_service import dataset_service
from backend.services.evaluation_service import evaluation_service

datasets_bp = Blueprint("datasets", __name__)


@datasets_bp.route("/datasets", methods=["GET"])
def get_datasets():
    """
    Return available datasets and their metadata (Dataset A, Dataset B, and annotation workspace).
    """
    metadata = dataset_service.get_datasets_metadata()
    return jsonify(metadata), 200


@datasets_bp.route("/dashboard", methods=["GET"])
def get_dashboard():
    """
    Return aggregated dashboard statistics from verified project reports.
    """
    stats = evaluation_service.get_dashboard_summary()
    return jsonify(stats), 200

