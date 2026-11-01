"""
ReviewMind — External Dataset Evaluation Pipeline

This script performs supervised evaluation of the production ReviewMind model
against human consensus ground-truth labels on Dataset B (Eco-hotel Reviews).

Methodological Integrity Rules:
1. Supervised evaluation metrics (Accuracy, Precision, Recall, Macro F1) are
   computed STRICTLY against verified human reference annotations.
2. If zero consensus labels are available, the script reports PENDING status
   and refrains from calculating or publishing fabricated metrics.
3. If only a subset of reviews has been annotated, the script reports PARTIAL
   evaluation status and explicitly documents excluded items.
4. Only when all 399 unique reviews have finalized consensus labels is the
   evaluation designated as FULL external benchmark evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

VALID_SENTIMENTS = ["negative", "neutral", "positive"]
COL_ID = "Original_Index"
COL_REVIEW = "Review"
COL_CONSENSUS = "Consensus_Sentiment"
COL_HUMAN = "Human_Sentiment"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ReviewMind production model against human consensus labels on Dataset B."
    )
    parser.add_argument(
        "--consensus-file",
        type=Path,
        default=Path("data/external/annotation_consensus.csv"),
        help="Path to consensus CSV file (default: data/external/annotation_consensus.csv)",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("models/reviewmind_final_model.joblib"),
        help="Path to trained production model (default: models/reviewmind_final_model.joblib)",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=Path("reports/external_predictions.csv"),
        help="Path to precomputed external predictions CSV (default: reports/external_predictions.csv)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save evaluation reports (default: reports)",
    )
    return parser.parse_args()


def run_external_evaluation(
    consensus_path: Path,
    model_path: Path,
    predictions_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    print("=" * 80)
    print("      REVIEWMIND — EXTERNAL DOMAIN EVALUATION (DATASET B)")
    print("=" * 80)

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = reports_dir / "external_evaluation_metrics.json"
    metrics_txt_path = reports_dir / "external_evaluation_report.txt"

    # 1. Load consensus dataset
    if not consensus_path.exists():
        msg = f"Consensus file not found at '{consensus_path}'."
        print(f"[STATUS] {msg}")
        payload = {
            "status": "pending",
            "evaluation_available": False,
            "message": msg,
            "labeled_count": 0,
            "total_count": 399,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        return payload

    df_consensus = pd.read_csv(consensus_path)
    total_reviews = len(df_consensus)

    # Resolve consensus label column
    target_col = None
    if COL_CONSENSUS in df_consensus.columns:
        target_col = COL_CONSENSUS
    elif COL_HUMAN in df_consensus.columns:
        target_col = COL_HUMAN
    else:
        raise ValueError(f"Neither '{COL_CONSENSUS}' nor '{COL_HUMAN}' found in consensus file.")

    # Filter labeled subset
    valid_mask = (
        df_consensus[target_col].notna()
        & (df_consensus[target_col].astype(str).str.strip().str.lower().isin(VALID_SENTIMENTS))
    )
    df_labeled = df_consensus[valid_mask].copy()
    labeled_count = len(df_labeled)
    excluded_count = total_reviews - labeled_count

    print(f"Total reviews in Dataset B : {total_reviews}")
    print(f"Verified consensus labels  : {labeled_count}")
    print(f"Excluded / Unlabeled reviews : {excluded_count}")

    # Case A: Zero consensus labels available
    if labeled_count == 0:
        notice = (
            "No human consensus labels available yet. External supervised evaluation is PENDING. "
            "In strict accordance with academic text mining standards, supervised metrics "
            "(Accuracy, F1) are not calculated or published without verified human reference labels."
        )
        print(f"\n[NOTICE] {notice}")
        payload = {
            "status": "pending",
            "evaluation_available": False,
            "evaluation_scope": "none",
            "message": notice,
            "dataset_name": "Eco-hotel Reviews (Dataset B, Calheiros et al., 2017)",
            "total_reviews": total_reviews,
            "labeled_reviews_count": 0,
            "excluded_reviews_count": total_reviews,
            "metrics": None,
        }
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write(f"REVIEWMIND — EXTERNAL EVALUATION REPORT\nStatus: PENDING\n{notice}\n")
        return payload

    # Case B: Labeled subset exists (partial or full)
    is_full = (labeled_count == total_reviews)
    eval_scope = "full" if is_full else "partial"
    status_str = "complete" if is_full else "partial"

    print(f"\n[EVALUATION SCOPE: {eval_scope.upper()}] Evaluating {labeled_count}/{total_reviews} reviews...")

    # Load model and predict (or use precomputed predictions matched by ID)
    y_true = df_labeled[target_col].astype(str).str.strip().str.lower().tolist()
    y_pred: list[str] = []

    # Check precomputed predictions first
    preds_map = {}
    if predictions_path.exists():
        try:
            df_preds = pd.read_csv(predictions_path)
            id_col = "original_index" if "original_index" in df_preds.columns else COL_ID
            for _, r in df_preds.iterrows():
                preds_map[int(r[id_col])] = str(r["Predicted_Sentiment"]).strip().lower()
        except Exception:
            preds_map = {}

    all_matched = all(int(row[COL_ID]) in preds_map for _, row in df_labeled.iterrows())

    if all_matched:
        for _, row in df_labeled.iterrows():
            y_pred.append(preds_map[int(row[COL_ID])])
    else:
        # Fallback to direct model inference
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at '{model_path}'.")
        print(f"Running inference with model: '{model_path.name}'...")
        model = joblib.load(model_path)
        y_pred = list(model.predict(df_labeled[COL_REVIEW].astype(str).tolist()))

    # Calculate metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0.0))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=VALID_SENTIMENTS,
        output_dict=True,
        zero_division=0.0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=VALID_SENTIMENTS,
        digits=4,
        zero_division=0.0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=VALID_SENTIMENTS)

    per_class = {}
    for label in VALID_SENTIMENTS:
        per_class[label] = {
            "precision": round(float(report_dict[label]["precision"]), 4),
            "recall": round(float(report_dict[label]["recall"]), 4),
            "f1_score": round(float(report_dict[label]["f1-score"]), 4),
            "support": int(report_dict[label]["support"]),
        }

    payload = {
        "status": status_str,
        "evaluation_available": True,
        "evaluation_scope": eval_scope,
        "dataset_name": "Eco-hotel Reviews (Dataset B, Calheiros et al., 2017)",
        "total_reviews": total_reviews,
        "labeled_reviews_count": labeled_count,
        "excluded_reviews_count": excluded_count,
        "coverage_percentage": round((labeled_count / total_reviews) * 100, 2),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "per_class": per_class,
            "confusion_matrix": {
                "labels": VALID_SENTIMENTS,
                "matrix": cm.tolist(),
            },
        },
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"\n[OUTPUT] Saved external evaluation metrics (JSON): '{metrics_json_path}'")

    summary_text = f"""================================================================================
          REVIEWMIND — EXTERNAL EVALUATION REPORT ({eval_scope.upper()})
================================================================================
Dataset Name                : Eco-hotel Reviews (Dataset B, Calheiros et al., 2017)
Evaluation Scope            : {eval_scope.upper()} ({labeled_count}/{total_reviews} reviews, {payload['coverage_percentage']}%)
Excluded / Pending Reviews  : {excluded_count}

OVERALL METRICS:
- Accuracy                  : {round(acc * 100, 2)}%
- Macro F1-Score            : {round(macro_f1, 4)}
- Weighted F1-Score         : {round(weighted_f1, 4)}

CLASSIFICATION REPORT:
{report_text}

CONFUSION MATRIX (Rows: True Human Consensus, Cols: Model Predictions):
{pd.DataFrame(cm, index=VALID_SENTIMENTS, columns=VALID_SENTIMENTS).to_string()}
================================================================================
"""
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[OUTPUT] Saved external evaluation report (TXT): '{metrics_txt_path}'")

    print("\n" + summary_text)
    return payload


def main() -> None:
    args = parse_arguments()
    try:
        run_external_evaluation(
            consensus_path=args.consensus_file,
            model_path=args.model_file,
            predictions_path=args.predictions_file,
            reports_dir=args.reports_dir,
        )
    except Exception as e:
        print(f"[ERROR] External evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

