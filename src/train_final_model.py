"""
ReviewMind — Final Model Training (Phase 5)

This script trains the selected production model (Model 3) on 100% of Dataset A
(TripAdvisor Hotel Reviews) without train/test splitting:
  - Feature extraction : TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
  - Classification     : MultinomialNB(alpha=0.1, fit_prior=False)

Outputs:
  - models/reviewmind_final_model.joblib
  - reports/final_model_metadata.json

Note:
  This production model utilizes all available historical data. Its generalization
  performance will be independently evaluated later using an external labeled test set
  (Dataset B). No training data evaluation metrics are reported as generalization scores.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import sys
import time

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure standard output encoding for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Expected labels and canonical order
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def get_project_paths() -> dict[str, Path]:
    """Define and return project paths."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "project_root": project_root,
        "input_csv": project_root / "data" / "reviews_prepared.csv",
        "models_dir": project_root / "models",
        "reports_dir": project_root / "reports",
        "final_model": project_root / "models" / "reviewmind_final_model.joblib",
        "final_metadata": project_root / "reports" / "final_model_metadata.json",
    }


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate Dataset A."""
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found at: '{csv_path}'")
        print("Please run 'python src/prepare_data.py' first to prepare the data.")
        sys.exit(1)

    print(f"Loading prepared dataset from: '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # 1. Validate required columns
    required_cols = ["Review", "Rating", "Sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in dataset: {missing}")
        print(f"Expected: {required_cols}")
        sys.exit(1)

    # 2. Check for missing or empty reviews
    invalid_mask = df["Review"].isna() | (df["Review"].astype(str).str.strip() == "")
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"[WARNING] Removing {invalid_count} invalid or empty review rows.")
        df = df[~invalid_mask].copy()

    # 3. Validate sentiment labels
    unique_labels = set(df["Sentiment"].dropna().unique())
    unexpected = unique_labels - set(SENTIMENT_LABELS)
    if unexpected:
        print(f"[ERROR] Unexpected sentiment labels found: {unexpected}")
        print(f"Expected labels: {SENTIMENT_LABELS}")
        sys.exit(1)

    return df


def main() -> None:
    paths = get_project_paths()

    print("=" * 80)
    print("           REVIEWMIND — FINAL PRODUCTION MODEL TRAINING (100% DATA)")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load and validate full dataset
    # ---------------------------------------------------------
    print("\n--- 1. DATA LOADING & VALIDATION ---")
    df = load_and_validate_data(paths["input_csv"])

    total_records = len(df)
    class_counts = df["Sentiment"].value_counts().reindex(SENTIMENT_LABELS)
    class_pcts = (class_counts / total_records) * 100

    print(f"Total training records : {total_records:,}")
    print(f"Dataset columns        : {list(df.columns)}")
    print("\nClass distribution (100% of Dataset A):")
    for label in SENTIMENT_LABELS:
        count = int(class_counts[label])
        pct = class_pcts[label]
        print(f"  - {label:<10}: {count:>6,} reviews ({pct:>6.2f}%)")

    # ---------------------------------------------------------
    # 2. Define Features and Target
    # ---------------------------------------------------------
    print("\n--- 2. PIPELINE SPECIFICATION ---")
    X = df["Review"].astype(str)
    y = df["Sentiment"].astype(str)

    print("Selected Model Configuration: Model 3 (Alternative Smoothing)")
    print("Components:")
    print("  1. TfidfVectorizer(ngram_range=(1, 2), max_features=20000)")
    print("  2. MultinomialNB(alpha=0.1, fit_prior=False)")
    print("Training policy: 100% of Dataset A (no split, no data leakage into evaluation)")

    final_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=20000,
                ),
            ),
            (
                "nb",
                MultinomialNB(
                    alpha=0.1,
                    fit_prior=False,
                ),
            ),
        ]
    )

    # ---------------------------------------------------------
    # 3. Fit pipeline on 100% of Dataset A
    # ---------------------------------------------------------
    print("\n--- 3. FITTING FINAL PIPELINE ON 100% DATASET ---")
    start_time = time.perf_counter()
    try:
        final_pipeline.fit(X, y)
    except Exception as e:
        print(f"[ERROR] Failed to train final pipeline: {e}")
        sys.exit(1)
    duration = time.perf_counter() - start_time

    print(f"   -> Model fitted successfully in {duration:.2f} seconds.")
    print("   -> Vocabulary size extracted: 20,000 features (unigrams + bigrams).")

    # ---------------------------------------------------------
    # 4. Save Final Model Artifact
    # ---------------------------------------------------------
    print("\n--- 4. SAVING PRODUCTION MODEL ARTIFACT ---")
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(final_pipeline, paths["final_model"])
        print(f"   -> Final model saved: '{paths['final_model']}'")
    except Exception as e:
        print(f"[ERROR] Failed to save final model: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 5. Save Training Metadata
    # ---------------------------------------------------------
    print("\n--- 5. SAVING TRAINING METADATA ---")
    training_timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_name": "ReviewMind Final Production Model",
        "status": "Trained on 100% of Dataset A",
        "training_timestamp_utc": training_timestamp,
        "training_duration_seconds": round(duration, 3),
        "dataset": {
            "name": "TripAdvisor Hotel Reviews (Prepared)",
            "filename": paths["input_csv"].name,
            "total_records": total_records,
            "class_distribution": {
                label: {
                    "count": int(class_counts[label]),
                    "percentage": round(float(class_pcts[label]), 2),
                }
                for label in SENTIMENT_LABELS
            },
        },
        "model_architecture": {
            "pipeline": "TfidfVectorizer + MultinomialNB",
            "selected_configuration": "Model 3 (Alternative Smoothing)",
            "tfidf_parameters": {
                "ngram_range": [1, 2],
                "max_features": 20000,
            },
            "naive_bayes_parameters": {
                "alpha": 0.1,
                "fit_prior": False,
            },
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__,
            "platform": platform.platform(),
        },
        "evaluation_protocol": {
            "policy": "No evaluation performed on training data",
            "evaluation_note": (
                "This production model was fitted on the entirety of Dataset A to maximize "
                "vocabulary coverage and classification capability. Generalization metrics "
                "must be measured on an independent labeled Dataset B."
            ),
        },
        "saved_artifacts": {
            "model_path": paths["final_model"].relative_to(paths["project_root"]).as_posix(),
            "metadata_path": paths["final_metadata"].relative_to(paths["project_root"]).as_posix(),
        },
    }

    try:
        with open(paths["final_metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"   -> Training metadata saved: '{paths['final_metadata']}'")
    except Exception as e:
        print(f"[ERROR] Failed to save metadata JSON: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 6. Final Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    FINAL MODEL TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Dataset               : {paths['input_csv'].name} (100% full dataset)")
    print(f"  Training records      : {total_records:,}")
    print(f"  Model architecture    : TfidfVectorizer(max_features=20000, ngram_range=(1,2))")
    print(f"                          + MultinomialNB(alpha=0.1, fit_prior=False)")
    print(f"  Training duration     : {duration:.2f} s")
    print(f"  Python version        : {metadata['environment']['python_version']}")
    print(f"  scikit-learn version  : {metadata['environment']['scikit_learn_version']}")
    print("-" * 80)
    print("Generated Artifacts:")
    print(f"  1. Final Model        : {paths['final_model']}")
    print(f"  2. Model Metadata     : {paths['final_metadata']}")
    print("-" * 80)
    print("Next step:")
    print("  Evaluate this production model on an independent labeled dataset (Dataset B).")
    print("=" * 80)


if __name__ == "__main__":
    main()

