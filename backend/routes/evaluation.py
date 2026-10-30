"""
ReviewMind — Evaluation Route
GET /api/evaluation
"""

from flask import Blueprint, jsonify

from backend.services.evaluation_service import evaluation_service

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/evaluation", methods=["GET"])
def get_evaluation():
    """
    Return internal scientific benchmark metrics, confusion matrix data,
    and external evaluation human consensus status.
    """
    report = evaluation_service.get_full_evaluation_report()
    return jsonify(report), 200

