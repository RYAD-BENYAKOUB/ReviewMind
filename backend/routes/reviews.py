"""
ReviewMind — Reviews Exploration Route
GET /api/reviews
GET /api/reviews/<int:original_index>
"""

from flask import Blueprint, request, jsonify

from backend.services.dataset_service import dataset_service

reviews_bp = Blueprint("reviews", __name__)


@reviews_bp.route("/reviews", methods=["GET"])
def list_reviews():
    """
    List reviews from Dataset B with pagination, sentiment filtering, and text search.
    Query Parameters:
      - page (int, default=1): Current page number
      - per_page (int, default=10, max=100): Reviews per page
      - sentiment (str, optional): 'positive' | 'neutral' | 'negative'
      - q (str, optional): Search substring in review text
    """
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)
    sentiment = request.args.get("sentiment", default=None, type=str)
    query = request.args.get("q", default=None, type=str)

    result = dataset_service.get_reviews(
        dataset_id="b",
        page=page,
        per_page=per_page,
        sentiment=sentiment,
        query=query,
    )

    return jsonify(result), 200


@reviews_bp.route("/reviews/<int:original_index>", methods=["GET"])
def get_single_review(original_index: int):
    """
    Retrieve a single review by its original index from Dataset B.
    """
    review = dataset_service.get_review_by_index(original_index)
    if review is None:
        return jsonify({
            "error": "Not Found",
            "message": f"Review with original_index {original_index} was not found in Dataset B.",
            "status": "not_found",
        }), 404

    return jsonify({
        "status": "success",
        "review": review,
    }), 200

