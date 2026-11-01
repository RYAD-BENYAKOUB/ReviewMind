"""
ReviewMind — Automated Human Annotation Consensus Pipeline

This script automates the double-blind human annotation consensus workflow
for external hospitality review datasets (Dataset B: Eco-hotel Reviews).

Workflow & Methodology:
1. Ingest Annotator A and Annotator B CSV files independently.
2. Validate schemas, stable review IDs (Original_Index), and sentiment labels.
3. Compute pair-wise agreement statistics on double-labeled items:
   - Agreement count & percentage
   - Disagreement count
   - Missing / incomplete count
   - Cohen's Kappa (kappa) on double-labeled subset
4. Apply deterministic consensus decision rules:
   - Case A (Agreement): Annotator A == Annotator B -> Status: 'agreement', Consensus = label
   - Case B (Disagreement): Annotator A != Annotator B -> Status: 'requires_adjudication', Consensus = None
   - Case C (Incomplete): One or both missing -> Status: 'incomplete', Consensus = None
   - Case D (Adjudicated): Existing or supplied adjudication -> Status: 'adjudicated', Consensus = label
5. Generate standardized outputs:
   - data/external/annotation_consensus.csv
   - data/external/annotation_disagreements.csv
   - data/external/annotation_progress.json
   - reports/annotation_agreement.json
   - reports/annotation_agreement.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# Canonical valid sentiment categories
VALID_SENTIMENTS = {"negative", "neutral", "positive"}

# Standard column names
COL_ID = "Original_Index"
COL_REVIEW = "Review"
COL_SENTIMENT = "Human_Sentiment"
COL_NOTES = "Annotation_Notes"

STATUS_AGREEMENT = "agreement"
STATUS_DISAGREEMENT = "requires_adjudication"
STATUS_INCOMPLETE = "incomplete"
STATUS_ADJUDICATED = "adjudicated"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the consensus pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute human annotation consensus and Cohen's Kappa for ReviewMind Dataset B."
    )
    parser.add_argument(
        "--annotator-a",
        type=Path,
        default=Path("data/external/annotator_A.csv"),
        help="Path to Annotator A CSV file (default: data/external/annotator_A.csv)",
    )
    parser.add_argument(
        "--annotator-b",
        type=Path,
        default=Path("data/external/annotator_B.csv"),
        help="Path to Annotator B CSV file (default: data/external/annotator_B.csv)",
    )
    parser.add_argument(
        "--consensus-output",
        type=Path,
        default=Path("data/external/annotation_consensus.csv"),
        help="Path for generated consensus CSV (default: data/external/annotation_consensus.csv)",
    )
    parser.add_argument(
        "--disagreements-output",
        type=Path,
        default=Path("data/external/annotation_disagreements.csv"),
        help="Path for generated disagreements CSV (default: data/external/annotation_disagreements.csv)",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        default=Path("data/external/annotation_progress.json"),
        help="Path for annotation progress JSON (default: data/external/annotation_progress.json)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save agreement reports (default: reports)",
    )
    parser.add_argument(
        "--adjudications",
        type=Path,
        default=None,
        help="Optional CSV file containing human adjudications for disputed reviews",
    )
    return parser.parse_args()


def normalize_sentiment(val: Any) -> str | None:
    """
    Normalize sentiment label: lowercase, trimmed string.
    Returns None if missing/empty.
    Raises ValueError if non-empty string is not in valid sentiment categories.
    """
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    if not val_str:
        return None
    if val_str not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment label '{val}'. Must be one of: {sorted(VALID_SENTIMENTS)}"
        )
    return val_str


def load_and_validate_annotator_file(
    file_path: Path, annotator_name: str
) -> pd.DataFrame:
    """
    Load an annotator CSV file and validate its schema, IDs, and label values.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Annotator file not found for {annotator_name}: '{file_path}'"
        )

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Annotator file for {annotator_name} is empty: '{file_path}'")
    except Exception as err:
        raise ValueError(f"Error reading {file_path}: {err}")

    if len(df) == 0:
        raise ValueError(f"Annotator file '{file_path}' contains zero records.")

    # Validate required columns
    required = [COL_ID, COL_REVIEW]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in {file_path}: {missing}. Expected at least: {required}"
        )

    # Human_Sentiment column check
    sentiment_col = None
    for cand in [COL_SENTIMENT, "Sentiment", "sentiment", "label"]:
        if cand in df.columns:
            sentiment_col = cand
            break
    if sentiment_col is None:
        raise ValueError(
            f"Missing sentiment column in {file_path}. Expected '{COL_SENTIMENT}'."
        )

    # Check duplicate IDs
    if df[COL_ID].duplicated().any():
        dup_ids = df[df[COL_ID].duplicated()][COL_ID].tolist()
        raise ValueError(
            f"Duplicate review IDs detected in {file_path}: {dup_ids[:10]}"
        )

    # Normalize IDs to int
    try:
        df[COL_ID] = df[COL_ID].astype(int)
    except Exception as err:
        raise ValueError(f"Column '{COL_ID}' must contain integer IDs: {err}")

    # Standardize column names
    clean_df = pd.DataFrame()
    clean_df[COL_ID] = df[COL_ID]
    clean_df[COL_REVIEW] = df[COL_REVIEW].astype(str)

    # Normalize labels
    clean_labels = []
    for idx, raw_lbl in enumerate(df[sentiment_col]):
        try:
            norm_lbl = normalize_sentiment(raw_lbl)
            clean_labels.append(norm_lbl)
        except ValueError as val_err:
            rev_id = df.iloc[idx][COL_ID]
            raise ValueError(
                f"Validation error in {file_path} at {COL_ID}={rev_id}: {val_err}"
            )

    clean_df[COL_SENTIMENT] = clean_labels

    if COL_NOTES in df.columns:
        clean_df[COL_NOTES] = df[COL_NOTES].fillna("").astype(str)
    else:
        clean_df[COL_NOTES] = ""

    return clean_df


def compute_cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    """
    Compute Cohen's Kappa coefficient on two aligned lists of non-empty categorical labels.
    Returns None if fewer than 2 valid paired items are provided or variance is zero.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have identical lengths.")

    n = len(labels_a)
    if n == 0:
        return None

    categories = sorted(VALID_SENTIMENTS)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    # Build confusion / contingency matrix
    matrix = np.zeros((num_cats, num_cats), dtype=int)
    for a, b in zip(labels_a, labels_b):
        matrix[cat_to_idx[a], cat_to_idx[b]] += 1

    # Observed agreement
    p_o = float(np.trace(matrix)) / n

    # Expected agreement by chance
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    p_e = float(np.sum(row_sums * col_sums)) / (n * n)

    if np.isclose(1.0 - p_e, 0.0):
        # Perfect agreement where chance is also 1.0 (all predictions identical single class)
        return 1.0 if np.isclose(p_o, 1.0) else 0.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(np.clip(kappa, -1.0, 1.0))


def interpret_kappa(kappa: float | None) -> str:
    """Return standard Landis & Koch (1977) interpretation of Cohen's Kappa."""
    if kappa is None:
        return "Not Applicable (Insufficient double-labeled data)"
    if kappa < 0.0:
        return "Poor (Disagreement worse than chance)"
    if kappa <= 0.20:
        return "Slight agreement"
    if kappa <= 0.40:
        return "Fair agreement"
    if kappa <= 0.60:
        return "Moderate agreement"
    if kappa <= 0.80:
        return "Substantial agreement"
    return "Almost perfect agreement"


def run_consensus_pipeline(
    annotator_a_path: Path,
    annotator_b_path: Path,
    consensus_output_path: Path,
    disagreements_output_path: Path,
    progress_output_path: Path,
    reports_dir: Path,
    adjudications_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute the end-to-end consensus workflow deterministically.
    """
    print("=" * 80)
    print("       REVIEWMIND — HUMAN ANNOTATION CONSENSUS AUTOMATION")
    print("=" * 80)

    # 1. Load and validate annotator files
    print(f"Loading Annotator A : '{annotator_a_path}'...")
    df_a = load_and_validate_annotator_file(annotator_a_path, "Annotator A")
    print(f"Loading Annotator B : '{annotator_b_path}'...")
    df_b = load_and_validate_annotator_file(annotator_b_path, "Annotator B")

    # 2. Check alignment of IDs
    ids_a = df_a[COL_ID].tolist()
    ids_b = df_b[COL_ID].tolist()

    if set(ids_a) != set(ids_b):
        missing_in_b = sorted(set(ids_a) - set(ids_b))
        missing_in_a = sorted(set(ids_b) - set(ids_a))
        err_msg = "Review ID mismatch between Annotator A and B."
        if missing_in_b:
            err_msg += f" Present in A but missing in B: {missing_in_b[:5]}."
        if missing_in_a:
            err_msg += f" Present in B but missing in A: {missing_in_a[:5]}."
        raise ValueError(err_msg)

    # Align by Original_Index
    df_a = df_a.sort_values(COL_ID).reset_index(drop=True)
    df_b = df_b.sort_values(COL_ID).reset_index(drop=True)

    total_reviews = len(df_a)
    print(f"\nTotal unique reviews to evaluate : {total_reviews}")

    # Optional existing adjudications
    adjudications_map: dict[int, tuple[str, str]] = {}  # id -> (sentiment, notes)
    if adjudications_path and adjudications_path.exists():
        print(f"Loading external adjudications from: '{adjudications_path}'...")
        try:
            adj_df = pd.read_csv(adjudications_path)
            adj_id_col = COL_ID if COL_ID in adj_df.columns else "id"
            adj_lbl_col = (
                "Adjudicated_Sentiment"
                if "Adjudicated_Sentiment" in adj_df.columns
                else (COL_SENTIMENT if COL_SENTIMENT in adj_df.columns else "sentiment")
            )
            for _, row in adj_df.iterrows():
                try:
                    r_id = int(row[adj_id_col])
                    lbl = normalize_sentiment(row[adj_lbl_col])
                    if lbl:
                        notes = (
                            str(row["Adjudication_Notes"])
                            if "Adjudication_Notes" in adj_df.columns and pd.notna(row["Adjudication_Notes"])
                            else ""
                        )
                        adjudications_map[r_id] = (lbl, notes)
                except Exception:
                    continue
        except Exception as err:
            print(f"[WARNING] Could not parse adjudications file: {err}")

    # 3. Match annotations and compute review-level consensus
    consensus_rows = []
    disagreement_rows = []

    double_labeled_a: list[str] = []
    double_labeled_b: list[str] = []

    count_a_labeled = 0
    count_b_labeled = 0
    count_agreements = 0
    count_disagreements = 0
    count_incomplete = 0
    count_adjudicated = 0

    for i in range(total_reviews):
        row_a = df_a.iloc[i]
        row_b = df_b.iloc[i]

        r_id = int(row_a[COL_ID])
        text = str(row_a[COL_REVIEW])
        lbl_a = row_a[COL_SENTIMENT]
        lbl_b = row_b[COL_SENTIMENT]

        notes_a = str(row_a[COL_NOTES]).strip()
        notes_b = str(row_b[COL_NOTES]).strip()
        combined_notes = []
        if notes_a:
            combined_notes.append(f"Annotator A: {notes_a}")
        if notes_b:
            combined_notes.append(f"Annotator B: {notes_b}")

        if lbl_a:
            count_a_labeled += 1
        if lbl_b:
            count_b_labeled += 1

        status = STATUS_INCOMPLETE
        consensus_sentiment = None
        adjudicated_sentiment = None

        # Check if external adjudication exists
        if r_id in adjudications_map:
            adjudicated_sentiment, adj_notes = adjudications_map[r_id]
            consensus_sentiment = adjudicated_sentiment
            status = STATUS_ADJUDICATED
            count_adjudicated += 1
            if adj_notes:
                combined_notes.append(f"Adjudicator: {adj_notes}")

        elif lbl_a is not None and lbl_b is not None:
            # Both labeled independently
            double_labeled_a.append(lbl_a)
            double_labeled_b.append(lbl_b)

            if lbl_a == lbl_b:
                status = STATUS_AGREEMENT
                consensus_sentiment = lbl_a
                count_agreements += 1
            else:
                status = STATUS_DISAGREEMENT
                consensus_sentiment = None
                count_disagreements += 1
        else:
            status = STATUS_INCOMPLETE
            count_incomplete += 1

        notes_str = "; ".join(combined_notes)

        # Build output record
        record = {
            COL_ID: r_id,
            COL_REVIEW: text,
            "Annotator_A_Sentiment": lbl_a if lbl_a else "",
            "Annotator_B_Sentiment": lbl_b if lbl_b else "",
            "Consensus_Sentiment": consensus_sentiment if consensus_sentiment else "",
            # Backward-compatible alias used by existing backend services
            COL_SENTIMENT: consensus_sentiment if consensus_sentiment else "",
            "Annotation_Status": status,
            "Adjudicated_Sentiment": adjudicated_sentiment if adjudicated_sentiment else "",
            COL_NOTES: notes_str,
        }
        consensus_rows.append(record)

        if status == STATUS_DISAGREEMENT:
            disagreement_rows.append({
                COL_ID: r_id,
                COL_REVIEW: text,
                "Annotator_A_Sentiment": lbl_a,
                "Annotator_B_Sentiment": lbl_b,
                "Proposed_Adjudication": "",
                "Adjudication_Notes": "",
            })

    # 4. Statistical computations
    double_labeled_count = len(double_labeled_a)
    agreement_rate = (
        round((count_agreements / double_labeled_count) * 100, 2)
        if double_labeled_count > 0
        else 0.0
    )

    kappa = compute_cohens_kappa(double_labeled_a, double_labeled_b)
    kappa_rounded = round(kappa, 4) if kappa is not None else None
    kappa_desc = interpret_kappa(kappa)

    finalized_count = count_agreements + count_adjudicated
    finalized_percentage = round((finalized_count / total_reviews) * 100, 2)

    # 5. Persist output files
    consensus_df = pd.DataFrame(consensus_rows)
    consensus_output_path.parent.mkdir(parents=True, exist_ok=True)
    consensus_df.to_csv(consensus_output_path, index=False, encoding="utf-8")
    print(f"\n[OUTPUT] Saved consensus dataset: '{consensus_output_path}'")

    disagreements_df = pd.DataFrame(disagreement_rows)
    disagreements_output_path.parent.mkdir(parents=True, exist_ok=True)
    disagreements_df.to_csv(disagreements_output_path, index=False, encoding="utf-8")
    print(f"[OUTPUT] Saved disagreements dataset ({len(disagreement_rows)} rows): '{disagreements_output_path}'")

    # Progress payload
    progress_data = {
        "dataset_name": "Eco-hotel Reviews (Dataset B, Calheiros et al., 2017)",
        "total_unique_reviews": total_reviews,
        "annotator_A": {
            "completed": count_a_labeled,
            "completion_percentage": round((count_a_labeled / total_reviews) * 100, 2),
            "missing": total_reviews - count_a_labeled,
        },
        "annotator_B": {
            "completed": count_b_labeled,
            "completion_percentage": round((count_b_labeled / total_reviews) * 100, 2),
            "missing": total_reviews - count_b_labeled,
        },
        "double_labeled_count": double_labeled_count,
        "agreements_count": count_agreements,
        "disagreements_count": count_disagreements,
        "incomplete_count": count_incomplete,
        "adjudicated_count": count_adjudicated,
        "finalized_consensus_count": finalized_count,
        "finalized_consensus_percentage": finalized_percentage,
        "agreement_rate_percentage": agreement_rate,
        "cohens_kappa": kappa_rounded,
        "kappa_interpretation": kappa_desc,
        "consensus_status": "complete" if finalized_count == total_reviews else ("partial" if finalized_count > 0 else "pending"),
    }

    progress_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_output_path, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=4)
    print(f"[OUTPUT] Saved progress metadata: '{progress_output_path}'")

    # Scientific reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = reports_dir / "annotation_agreement.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(progress_data, f, indent=4)
    print(f"[OUTPUT] Saved scientific agreement report (JSON): '{report_json_path}'")

    report_txt_path = reports_dir / "annotation_agreement.txt"
    report_text = f"""================================================================================
          REVIEWMIND — INTER-ANNOTATOR AGREEMENT & CONSENSUS REPORT
================================================================================
Dataset Name                : Eco-hotel Reviews (Calheiros et al., 2017)
Total Unique Reviews        : {total_reviews}

1. ANNOTATOR COMPLETION:
   - Annotator A            : {count_a_labeled}/{total_reviews} ({progress_data['annotator_A']['completion_percentage']}%)
   - Annotator B            : {count_b_labeled}/{total_reviews} ({progress_data['annotator_B']['completion_percentage']}%)
   - Double-Labeled Reviews : {double_labeled_count}/{total_reviews} ({round(double_labeled_count/total_reviews*100, 2)}%)

2. AGREEMENT & RELIABILITY (Computed strictly on double-labeled subset):
   - Perfect Agreements     : {count_agreements} ({agreement_rate}%)
   - Disagreements          : {count_disagreements}
   - Cohen's Kappa          : {kappa_rounded if kappa_rounded is not None else 'N/A'}
   - Interpretation         : {kappa_desc}

3. CONSENSUS WORKFLOW STATUS:
   - Finalized Consensus    : {finalized_count}/{total_reviews} ({finalized_percentage}%)
   - Requires Adjudication  : {count_disagreements}
   - Incomplete / Pending   : {count_incomplete}
   - Global Consensus Status: {progress_data['consensus_status'].upper()}

METHODOLOGICAL NOTICE:
In strict compliance with academic text mining standards, supervised metrics
(Accuracy, Precision, Recall, Macro F1) are NOT computed on Dataset B until
human consensus adjudication is fully finalized across all reviews.
================================================================================
"""
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OUTPUT] Saved scientific agreement report (TXT): '{report_txt_path}'")

    # 6. Display terminal summary
    print("\n" + "=" * 80)
    print("                    CONSENSUS EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Reviews Evaluated   : {total_reviews}")
    print(f"Annotator A Completed     : {count_a_labeled}")
    print(f"Annotator B Completed     : {count_b_labeled}")
    print(f"Double-Labeled Items      : {double_labeled_count}")
    print(f"Agreements                : {count_agreements} ({agreement_rate}%)")
    print(f"Disagreements (Actionable): {count_disagreements}")
    print(f"Cohen's Kappa (k)         : {kappa_rounded} ({kappa_desc})")
    print(f"Finalized Consensus       : {finalized_count}/{total_reviews} ({finalized_percentage}%)")
    print(f"Consensus Status          : {progress_data['consensus_status'].upper()}")
    print("=" * 80)

    return progress_data


def main() -> None:
    args = parse_arguments()
    try:
        run_consensus_pipeline(
            annotator_a_path=args.annotator_a,
            annotator_b_path=args.annotator_b,
            consensus_output_path=args.consensus_output,
            disagreements_output_path=args.disagreements_output,
            progress_output_path=args.progress_output,
            reports_dir=args.reports_dir,
            adjudications_path=args.adjudications,
        )
    except Exception as e:
        print(f"\n[ERROR] Consensus pipeline execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

