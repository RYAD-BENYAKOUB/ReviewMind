"""
ReviewMind Backend Configuration
Defines environment settings, filesystem paths, and Flask options.
"""

from pathlib import Path
import os


class Config:
    """Base application configuration."""

    # Project filesystem root (C:/dev/ReviewMind)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Data directory & files
    DATA_DIR = PROJECT_ROOT / "data"
    DATASET_A_PATH = DATA_DIR / "reviews_prepared.csv"
    DATASET_B_RAW_PATH = DATA_DIR / "dataset-CalheirosMoroRita-2017.csv"
    ANNOTATIONS_DIR = DATA_DIR / "external"
    ANNOTATION_CONSENSUS_PATH = ANNOTATIONS_DIR / "annotation_consensus.csv"
    ANNOTATION_DISAGREEMENTS_PATH = ANNOTATIONS_DIR / "annotation_disagreements.csv"
    ANNOTATION_PROGRESS_PATH = ANNOTATIONS_DIR / "annotation_progress.json"
    ANNOTATION_GUIDELINES_PATH = ANNOTATIONS_DIR / "ANNOTATION_GUIDELINES.md"

    # Models directory & artifacts
    MODELS_DIR = PROJECT_ROOT / "models"
    FINAL_MODEL_PATH = MODELS_DIR / "reviewmind_final_model.joblib"
    INTERNAL_MODEL_PATH = MODELS_DIR / "reviewmind_internal_model.joblib"

    # Reports directory & summaries
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FINAL_METADATA_PATH = REPORTS_DIR / "final_model_metadata.json"
    INTERNAL_METRICS_PATH = REPORTS_DIR / "internal_metrics.json"
    MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"
    EXTERNAL_PREDICTIONS_PATH = REPORTS_DIR / "external_predictions.csv"
    EXTERNAL_SUMMARY_PATH = REPORTS_DIR / "external_prediction_summary.json"
    EXTERNAL_EXPLORATION_PATH = REPORTS_DIR / "external_dataset_exploration.json"
    ANNOTATION_AGREEMENT_PATH = REPORTS_DIR / "annotation_agreement.json"
    EXTERNAL_EVALUATION_PATH = REPORTS_DIR / "external_evaluation_metrics.json"

    # Flask Settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "reviewmind-dark-intelligence-dev-key")
    DEBUG = False
    TESTING = False
    JSON_SORT_KEYS = False

    # CORS configuration (allow local dev environments)
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")


class DevelopmentConfig(Config):
    """Development configuration with debugging enabled."""

    DEBUG = True


class TestingConfig(Config):
    """Testing configuration for automated test suites."""

    TESTING = True
    DEBUG = False


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    TESTING = False


# Map environment names to configuration classes
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

