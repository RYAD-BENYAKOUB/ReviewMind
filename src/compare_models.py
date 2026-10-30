"""
ReviewMind — Model Comparison Experiment

This script compares three Multinomial Naive Bayes configurations
using TF-IDF (unigrams + bigrams, top 20,000 features) on hotel reviews:
  1. Model 1 — Baseline: MultinomialNB(alpha=1.0, fit_prior=True)
  2. Model 2 — Uniform Prior: MultinomialNB(alpha=1.0, fit_prior=False)
  3. Model 3 — Alternative Smoothing: MultinomialNB(alpha=0.1, fit_prior=False)

Outputs:
  - reports/model_comparison.json
  - reports/model_comparison.csv
  - reports/model_comparison.png
"""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
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


# Expected labels and canonical order
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def get_project_paths() -> dict[str, Path]:
    """Define and return project paths."""
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    return {
        "project_root": project_root,
        "input_csv": project_root / "data" / "reviews_prepared.csv",
        "reports_dir": reports_dir,
        "comparison_json": reports_dir / "model_comparison.json",
        "comparison_csv": reports_dir / "model_comparison.csv",
        "comparison_plot": reports_dir / "model_comparison.png",
    }


def load_and_validate_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the prepared dataset."""
    if not csv_path.exists():
        print(f"[ERROR] Dataset not found at: '{csv_path}'")
        print("Please run 'python src/prepare_data.py' first.")
        sys.exit(1)

    print(f"Loading dataset from: '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Validate columns
    required_cols = ["Review", "Rating", "Sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    # Check reviews validity
    invalid_mask = df["Review"].isna() | (df["Review"].astype(str).str.strip() == "")
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"[WARNING] Removing {invalid_count} invalid/empty reviews.")
        df = df[~invalid_mask].copy()

    # Validate sentiments
    found_labels = set(df["Sentiment"].dropna().unique())
    unexpected = found_labels - set(SENTIMENT_LABELS)
    if unexpected:
        print(f"[ERROR] Unexpected sentiment labels: {unexpected}")
        sys.exit(1)

    return df


def generate_comparison_plots(results: list[dict], output_path: Path) -> None:
    """
    Generate comparison figures displaying Macro F1 across all three configurations
    along with class-level F1-scores, highlighting the impact on the neutral class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    palette = ["#4C72B0", "#55A868", "#C44E52"]

    model_names = [r["model_name"] for r in results]
    macro_f1s = [r["metrics"]["macro_f1"] for r in results]

    # --- Subplot 1: Macro F1 Comparison ---
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, macro_f1s, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_title("Macro F1-Score Comparison", fontsize=13, weight="bold", pad=12)
    ax1.set_ylabel("Macro F1-Score", fontsize=11)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Value annotations on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    # --- Subplot 2: Per-Class F1-Score Breakdown ---
    ax2 = axes[1]
    x = np.arange(len(model_names))
    width = 0.25

    neg_f1s = [r["metrics"]["per_class"]["negative"]["f1_score"] for r in results]
    neu_f1s = [r["metrics"]["per_class"]["neutral"]["f1_score"] for r in results]
    pos_f1s = [r["metrics"]["per_class"]["positive"]["f1_score"] for r in results]

    b_neg = ax2.bar(x - width, neg_f1s, width, label="Negative F1", color="#D95F02", edgecolor="black", linewidth=0.7)
    b_neu = ax2.bar(x, neu_f1s, width, label="Neutral F1 (Focus)", color="#7570B3", edgecolor="black", linewidth=0.7)
    b_pos = ax2.bar(x + width, pos_f1s, width, label="Positive F1", color="#1B9E77", edgecolor="black", linewidth=0.7)

    ax2.set_title("Class-Level F1-Scores Across Configurations", fontsize=13, weight="bold", pad=12)
    ax2.set_ylabel("F1-Score", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate neutral bars specifically
    for bar in b_neu:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="#7570B3",
        )

    plt.suptitle("ReviewMind — Multinomial Naive Bayes Model Comparison", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> Comparison plot saved to: '{output_path}'")


def main() -> None:
    paths = get_project_paths()

    print("=" * 80)
    print("        REVIEWMIND — MULTINOMIAL NAIVE BAYES MODEL COMPARISON")
    print("=" * 80)

    # 1. Load and validate dataset
    df = load_and_validate_data(paths["input_csv"])
    print(f"Dataset: {df.shape[0]:,} reviews across {len(SENTIMENT_LABELS)} classes.")

    # 2. Train/test split (Exact identical split: test_size=0.20, random_state=42, stratify=y)
    print("\nExecuting Stratified Split (80% Train / 20% Test, random_state=42)...")
    X = df["Review"].astype(str)
    y = df["Sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 3. Define the three configurations
    model_configs = [
        {
            "id": "model_1",
            "model_name": "Model 1 (Baseline)",
            "alpha": 1.0,
            "fit_prior": True,
            "description": "MultinomialNB(alpha=1.0, fit_prior=True) — Empirical class prior",
        },
        {
            "id": "model_2",
            "model_name": "Model 2 (Uniform Prior)",
            "alpha": 1.0,
            "fit_prior": False,
            "description": "MultinomialNB(alpha=1.0, fit_prior=False) — Uniform class prior",
        },
        {
            "id": "model_3",
            "model_name": "Model 3 (Alternative Smoothing)",
            "alpha": 0.1,
            "fit_prior": False,
            "description": "MultinomialNB(alpha=0.1, fit_prior=False) — Uniform prior + lower alpha",
        },
    ]

    results = []
    csv_rows = []

    print("\n--- RUNNING EXPERIMENTS ---")
    for cfg in model_configs:
        print(f"\nTraining & Evaluating {cfg['model_name']}...")
        print(f"Parameters: alpha={cfg['alpha']}, fit_prior={cfg['fit_prior']}")

        # Build pipeline with shared TF-IDF parameters
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
                    MultinomialNB(
                        alpha=cfg["alpha"],
                        fit_prior=cfg["fit_prior"],
                    ),
                ),
            ]
        )

        # Fit strictly on training set
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0.0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0.0))

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=SENTIMENT_LABELS,
            output_dict=True,
            zero_division=0.0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=SENTIMENT_LABELS)

        per_class = {}
        for label in SENTIMENT_LABELS:
            per_class[label] = {
                "precision": round(float(report_dict[label]["precision"]), 4),
                "recall": round(float(report_dict[label]["recall"]), 4),
                "f1_score": round(float(report_dict[label]["f1-score"]), 4),
                "support": int(report_dict[label]["support"]),
            }

        # Neutral predictions count
        neutral_predicted_count = int((y_pred == "neutral").sum())

        model_result = {
            "id": cfg["id"],
            "model_name": cfg["model_name"],
            "parameters": {
                "alpha": cfg["alpha"],
                "fit_prior": cfg["fit_prior"],
                "tfidf_ngram_range": [1, 2],
                "tfidf_max_features": 20000,
            },
            "description": cfg["description"],
            "metrics": {
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "neutral_predictions_count": neutral_predicted_count,
                "per_class": per_class,
                "confusion_matrix": {
                    "labels": SENTIMENT_LABELS,
                    "matrix": cm.tolist(),
                },
            },
        }
        results.append(model_result)

        # Format row for CSV output
        csv_rows.append({
            "model_id": cfg["id"],
            "model_name": cfg["model_name"],
            "alpha": cfg["alpha"],
            "fit_prior": cfg["fit_prior"],
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "negative_precision": per_class["negative"]["precision"],
            "negative_recall": per_class["negative"]["recall"],
            "negative_f1": per_class["negative"]["f1_score"],
            "neutral_precision": per_class["neutral"]["precision"],
            "neutral_recall": per_class["neutral"]["recall"],
            "neutral_f1": per_class["neutral"]["f1_score"],
            "neutral_predicted_count": neutral_predicted_count,
            "positive_precision": per_class["positive"]["precision"],
            "positive_recall": per_class["positive"]["recall"],
            "positive_f1": per_class["positive"]["f1_score"],
        })

        print(
            f"   -> Accuracy: {acc*100:.2f}% | Macro F1: {macro_f1:.4f} | "
            f"Neutral F1: {per_class['neutral']['f1_score']:.4f} "
            f"(predicted {neutral_predicted_count} neutral)"
        )

    # 4. Identify best model based on Macro F1
    best_result = max(results, key=lambda r: r["metrics"]["macro_f1"])

    # 5. Save outputs
    print("\n--- SAVING ARTIFACTS ---")
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    # JSON export
    payload = {
        "experiment": "ReviewMind Naive Bayes Model Comparison",
        "dataset": {
            "file": paths["input_csv"].name,
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_size": 0.20,
            "random_state": 42,
            "stratified": True,
        },
        "best_configuration": {
            "model_id": best_result["id"],
            "model_name": best_result["model_name"],
            "selection_criterion": "Highest Macro F1-score",
            "macro_f1": best_result["metrics"]["macro_f1"],
        },
        "models": results,
    }
    with open(paths["comparison_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"   -> JSON metrics saved: '{paths['comparison_json']}'")

    # CSV export
    df_comparison = pd.DataFrame(csv_rows)
    df_comparison.to_csv(paths["comparison_csv"], index=False, encoding="utf-8")
    print(f"   -> CSV table saved: '{paths['comparison_csv']}'")

    # PNG comparison plot
    generate_comparison_plots(results, paths["comparison_plot"])

    # 6. Display concise comparison table in terminal
    print("\n" + "=" * 80)
    print("                    MODEL COMPARISON TABLE")
    print("=" * 80)
    display_cols = [
        "model_name",
        "alpha",
        "fit_prior",
        "accuracy",
        "macro_f1",
        "negative_f1",
        "neutral_f1",
        "positive_f1",
    ]
    summary_table = df_comparison[display_cols].copy()
    summary_table.columns = [
        "Model",
        "Alpha",
        "Fit Prior",
        "Accuracy",
        "Macro F1",
        "Neg F1",
        "Neu F1",
        "Pos F1",
    ]
    print(summary_table.to_string(index=False))

    print("\n" + "-" * 80)
    print(f"BEST CONFIGURATION (based on Macro F1):")
    print(f"  -> {best_result['model_name']}")
    print(f"     Parameters : alpha={best_result['parameters']['alpha']}, fit_prior={best_result['parameters']['fit_prior']}")
    print(f"     Macro F1   : {best_result['metrics']['macro_f1']:.4f}")
    print(f"     Accuracy   : {best_result['metrics']['accuracy'] * 100:.2f}%")
    print(f"     Neutral F1 : {best_result['metrics']['per_class']['neutral']['f1_score']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

