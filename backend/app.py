"""
ReviewMind — Flask REST API Application Entry Point
Dark Intelligence Sentiment Analysis Platform Backend
"""

from pathlib import Path
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to sys.path if executed directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import Config, config_by_name
from backend.routes import api_bp
from backend.services.model_service import model_service


def create_app(config_name: str = "default") -> Flask:
    """
    Application factory for ReviewMind Flask backend.
    """
    app = Flask(
        __name__,
        static_folder=str(project_root / "web"),
        static_url_path="/web",
    )

    # Load configuration
    config_class = config_by_name.get(config_name, Config)
    app.config.from_object(config_class)

    # Configure Cross-Origin Resource Sharing (CORS)
    CORS(
        app,
        resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    # Additional CORS headers on every response
    @app.after_request
    def apply_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response

    # Register API routes blueprint
    app.register_blueprint(api_bp)

    # Root endpoint
    @app.route("/", methods=["GET"])
    def root():
        """Root API information and entrypoint."""
        return jsonify({
            "name": "ReviewMind REST API",
            "brand": "Dark Intelligence",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/health",
                "dashboard": "/api/dashboard",
                "predict": "/api/predict (POST)",
                "datasets": "/api/datasets",
                "reviews": "/api/reviews",
                "review_by_id": "/api/reviews/<original_index>",
                "evaluation": "/api/evaluation",
            },
            "web_ui": "/web/index.html",
        }), 200

    # Custom Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            "error": "Bad Request",
            "message": str(error.description if hasattr(error, "description") else error),
            "status_code": 400,
        }), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Not Found",
            "message": "The requested endpoint does not exist.",
            "status_code": 404,
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            "error": "Method Not Allowed",
            "message": "The HTTP method is not allowed for this endpoint.",
            "status_code": 405,
        }), 405

    @app.errorhandler(413)
    def payload_too_large(error):
        return jsonify({
            "error": "Payload Too Large",
            "message": "The submitted payload exceeds maximum allowed size.",
            "status_code": 413,
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred.",
            "status_code": 500,
        }), 500

    return app


if __name__ == "__main__":
    # Ensure model is verified on startup
    if not model_service.is_loaded:
        print("[WARNING] Production model not loaded yet. Checking models directory...")
        model_service.load_model()

    app = create_app(os.environ.get("FLASK_ENV", "development"))
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))

    print("=" * 70)
    print("   REVIEWMIND — DARK INTELLIGENCE BACKEND API SERVER")
    print(f"   Listening on: http://{host}:{port}")
    print("   Health Check: http://127.0.0.1:5000/api/health")
    print("=" * 70)

    app.run(host=host, port=port, debug=True)

