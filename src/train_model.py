"""
ReviewMind — Phase 3: Internal Model Evaluation

This script trains and evaluates a sentiment classification pipeline
(TF-IDF + Multinomial Naive Bayes) on hotel reviews.

Workflow:
1. Load and validate data/reviews_prepared.csv.
2. Perform a stratified 80/20 train/test split.
3. Construct a scikit-learn Pipeline (TfidfVectorizer + MultinomialNB).
4. Fit the pipeline strictly on training data (preventing data leakage).
5. Evaluate model performance on the unseen test set (Accuracy, Precision,
   Recall, F1 per class, Macro F1, Weighted F1, Confusion Matrix).
6. Save evaluation reports (JSON metrics and Confusion Matrix heatmap).
7. Serialize the internally evaluated 80% training model using joblib.
"""

from pathlib import Path
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure standard output encoding for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Expected labels and canonical order for consistent reporting & matrix orientation
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def get_project_paths() -> dict[str, Path]:
    """Define and return all project paths used by the training script."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "project_root": project_root,
        "input_csv": project_root / "data" / "reviews_prepared.csv",
        "reports_dir": project_root / "reports",
        "models_dir": project_root / "models",
        "metrics_json": project_root / "reports" / "internal_metrics.json",
        "cm_plot": project_root / "reports" / "internal_confusion_matrix.png",
        "saved_model": project_root / "models" / "reviewmind_internal_model.joblib",
    }


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the prepared reviews dataset and validate required structure and content.
    
    Checks:
    - File existence.
    - Presence of columns: 'Review', 'Rating', 'Sentiment'.
    - Absence of empty or whitespace-only reviews.
    - Validity of sentiment class labels.
    """
    if not csv_path.exists():
        print(f"[ERROR] Prepared dataset not found at: '{csv_path}'")
        print("Please run 'python src/prepare_data.py' first to generate 'data/reviews_prepared.csv'.")
        sys.exit(1)

    print(f"Loading prepared dataset from: '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file '{csv_path}': {e}")
        sys.exit(1)

    # 1. Validate required columns
    required_cols = ["Review", "Rating", "Sentiment"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns in dataset: {missing_cols}")
        print(f"Expected: {required_cols}")
        print(f"Found: {list(df.columns)}")
        sys.exit(1)

    # 2. Check missing and empty reviews
    null_reviews = df["Review"].isna().sum()
    empty_reviews = (df["Review"].dropna().astype(str).str.strip() == "").sum()
    if null_reviews > 0 or empty_reviews > 0:
        print(f"[WARNING] Found invalid review texts: {null_reviews} null, {empty_reviews} empty/whitespace.")
        # Filter out invalid entries if any to preserve model training integrity
        valid_mask = df["Review"].notna() & (df["Review"].astype(str).str.strip() != "")
        df = df[valid_mask].copy()
        print(f"Filtered invalid entries. Remaining valid samples: {len(df):,}")

    # 3. Check sentiment labels
    unique_labels = set(df["Sentiment"].dropna().unique())
    expected_labels_set = set(SENTIMENT_LABELS)
    unexpected_labels = unique_labels - expected_labels_set
    if unexpected_labels:
        print(f"[ERROR] Unexpected sentiment labels found in dataset: {unexpected_labels}")
        print(f"Expected labels: {SENTIMENT_LABELS}")
        sys.exit(1)

    return df


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Generate and save a readable confusion matrix heatmap using ReviewMind Dark Intelligence brand colors."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0B0B0F")
    ax.set_facecolor("#0B0B0F")

    # Custom colormap from dark surface to brand blue
    dark_blue_cmap = LinearSegmentedColormap.from_list(
        "DarkIntelligenceBlue",
        ["#121217", "#1E6BFF"],
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=dark_blue_cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1.5,
        linecolor="#0B0B0F",
        annot_kws={"size": 12, "weight": "bold", "color": "#FFFFFF"},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Internal Evaluation — Confusion Matrix\n(TF-IDF + Multinomial Naive Bayes)", fontsize=13, pad=15, color="#FFFFFF", weight="bold")
    ax.set_xlabel("Predicted Sentiment", fontsize=11, labelpad=10, color="#8E8E93")
    ax.set_ylabel("True Sentiment", fontsize=11, labelpad=10, color="#8E8E93")
    ax.tick_params(colors="#8E8E93")

    # Style colorbar ticks and label
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(color="#8E8E93")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8E8E93")

    for spine in ax.spines.values():
        spine.set_edgecolor("#24242F")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"   -> Confusion matrix plot saved: '{output_path}'")


def main() -> None:
    paths = get_project_paths()

    print("=" * 75)
    print("         REVIEWMIND — PHASE 3: INTERNAL MODEL EVALUATION")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Data loading and validation
    # ---------------------------------------------------------
    print("\n--- 1. DATA LOADING & VALIDATION ---")
    df = load_and_validate_data(paths["input_csv"])

    total_samples = len(df)
    class_counts = df["Sentiment"].value_counts().reindex(SENTIMENT_LABELS)
    class_pcts = (class_counts / total_samples) * 100

    print(f"Dataset dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nClass distribution in full dataset:")
    for label in SENTIMENT_LABELS:
        count = int(class_counts[label])
        pct = class_pcts[label]
        print(f"  - {label:<10}: {count:>6,} reviews ({pct:>6.2f}%)")

    # ---------------------------------------------------------
    # 2. Train/test split (Stratified 80/20)
    # ---------------------------------------------------------
    print("\n--- 2. STRATIFIED TRAIN / TEST SPLIT ---")
    X = df["Review"].astype(str)
    y = df["Sentiment"].astype(str)

    # Stratified split to preserve class proportions between train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Split configuration : test_size=0.20, random_state=42, stratify=Sentiment")
    print(f"Training samples     : {len(X_train):,} ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"Test samples         : {len(X_test):,} ({len(X_test) / total_samples * 100:.1f}%)")

    # ---------------------------------------------------------
    # 3. Model pipeline construction & fitting
    # ---------------------------------------------------------
    print("\n--- 3. MODEL PIPELINE CONFIGURATION & TRAINING ---")
    # Pipeline contains:
    # 1. TfidfVectorizer: unigrams + bigrams, top 20,000 features
    # 2. MultinomialNB: alpha=1.0 (Laplace smoothing)
    pipeline = Pipeline(
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
                MultinomialNB(alpha=1.0),
            ),
        ]
    )

    print("Pipeline structure:")
    print("  1. TfidfVectorizer(ngram_range=(1, 2), max_features=20000)")
    print("  2. MultinomialNB(alpha=1.0)")
    print("\nFitting pipeline ONLY on X_train and y_train (preventing data leakage)...")

    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        sys.exit(1)

    print("   -> Training completed successfully.")

    # ---------------------------------------------------------
    # 4. Evaluation on unseen test set
    # ---------------------------------------------------------
    print("\n--- 4. EVALUATION ON TEST SET ---")
    y_pred = pipeline.predict(X_test)

    # Core metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0.0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0.0))

    # Detailed classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=SENTIMENT_LABELS,
        output_dict=True,
        zero_division=0.0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=SENTIMENT_LABELS,
        digits=4,
        zero_division=0.0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=SENTIMENT_LABELS)

    print(f"Accuracy         : {acc * 100:.2f}% ({acc:.4f})")
    print(f"Macro F1-score   : {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix (rows: True, cols: Predicted):")
    cm_df = pd.DataFrame(cm, index=SENTIMENT_LABELS, columns=SENTIMENT_LABELS)
    print(cm_df.to_string())

    # ---------------------------------------------------------
    # 5. Saving reports and metrics
    # ---------------------------------------------------------
    print("\n--- 5. SAVING EVALUATION ARTIFACTS ---")
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)

    # Save confusion matrix figure
    plot_confusion_matrix(cm, SENTIMENT_LABELS, paths["cm_plot"])

    # Prepare structured JSON metrics
    per_class_metrics = {}
    for label in SENTIMENT_LABELS:
        per_class_metrics[label] = {
            "precision": round(float(report_dict[label]["precision"]), 4),
            "recall": round(float(report_dict[label]["recall"]), 4),
            "f1_score": round(float(report_dict[label]["f1-score"]), 4),
            "support": int(report_dict[label]["support"]),
        }

    metrics_payload = {
        "dataset": {
            "name": "TripAdvisor Hotel Reviews (Prepared)",
            "file": str(paths["input_csv"].name),
            "total_samples": total_samples,
            "class_distribution": {
                label: {
                    "count": int(class_counts[label]),
                    "percentage": round(float(class_pcts[label]), 2),
                }
                for label in SENTIMENT_LABELS
            },
        },
        "train_test_split": {
            "test_size": 0.20,
            "random_state": 42,
            "stratify": True,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "model_parameters": {
            "pipeline": "TfidfVectorizer + MultinomialNB",
            "tfidf": {
                "ngram_range": [1, 2],
                "max_features": 20000,
            },
            "multinomial_nb": {
                "alpha": 1.0,
            },
        },
        "evaluation_metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class": per_class_metrics,
            "confusion_matrix": {
                "labels": SENTIMENT_LABELS,
                "matrix": cm.tolist(),
            },
        },
        "artifacts": {
            "model_path": paths["saved_model"].relative_to(paths["project_root"]).as_posix(),
            "confusion_matrix_plot": paths["cm_plot"].relative_to(paths["project_root"]).as_posix(),
            "metrics_json": paths["metrics_json"].relative_to(paths["project_root"]).as_posix(),
        },
    }

    try:
        with open(paths["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)
        print(f"   -> Evaluation metrics saved: '{paths['metrics_json']}'")
    except Exception as e:
        print(f"[ERROR] Failed to save metrics JSON: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 6. Model serialization
    # ---------------------------------------------------------
    print("\n--- 6. MODEL SERIALIZATION ---")
    try:
        joblib.dump(pipeline, paths["saved_model"])
        print(f"   -> Internal model pipeline saved: '{paths['saved_model']}'")
        print("      (Note: This model was trained on 80% training data for internal evaluation)")
    except Exception as e:
        print(f"[ERROR] Failed to serialize model pipeline: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("                     EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Accuracy          : {acc * 100:.2f}%")
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Weighted F1       : {weighted_f1:.4f}")
    print(f"  Negative F1       : {report_dict['negative']['f1-score']:.4f}")
    print(f"  Neutral F1        : {report_dict['neutral']['f1-score']:.4f}")
    print(f"  Positive F1       : {report_dict['positive']['f1-score']:.4f}")
    print("-" * 75)
    print("Generated files:")
    print(f"  1. Model file     : {paths['saved_model']}")
    print(f"  2. Metrics JSON   : {paths['metrics_json']}")
    print(f"  3. Confusion plot : {paths['cm_plot']}")
    print("=" * 75)


if __name__ == "__main__":
    main()

