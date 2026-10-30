"""
ReviewMind — Inférence et Prédiction sur le Dataset Externe (Dataset B)

Ce script utilise le modèle de production déjà entraîné :
`models/reviewmind_final_model.joblib`

Pour prédire les sentiments des avis du Dataset B (Calheiros et al., 2017) :
`data/dataset-CalheirosMoroRita-2017.csv`

Étapes :
1. Charger le modèle final sans réentraînement.
2. Charger Dataset B avec la stratégie de parsing validée (latin1, sep='\\t').
3. Détecter et exclure les doublons pour l'analyse, en conservant l'index d'origine.
4. Prédire les sentiments et calculer les probabilités par classe (predict_proba).
5. Ajouter les colonnes de résultat :
   - Predicted_Sentiment
   - Confidence
   - Probability_Negative
   - Probability_Neutral
   - Probability_Positive
6. Analyser la distribution des prédictions, la confiance moyenne et les avis à faible confiance.
7. Sauvegarder les artefacts :
   - reports/external_predictions.csv
   - reports/external_prediction_summary.json
   - reports/external_sentiment_distribution.png
8. Rappeler que ces prédictions constituent des inférences du modèle et non des vérités terrain.
"""

from pathlib import Path
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Backend non interactif pour exécution terminale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration de l'encodage de la sortie standard pour Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Seuil de faible confiance (en classification à 3 classes, le hasard est à 33.3%)
# Une confiance inférieure à 60% signale une hésitation marquée entre plusieurs classes.
LOW_CONFIDENCE_THRESHOLD = 0.60


def get_project_paths() -> dict[str, Path]:
    """Retourne l'ensemble des chemins utilisés par le script."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "project_root": project_root,
        "model_file": project_root / "models" / "reviewmind_final_model.joblib",
        "dataset_b": project_root / "data" / "dataset-CalheirosMoroRita-2017.csv",
        "reports_dir": project_root / "reports",
        "predictions_csv": project_root / "reports" / "external_predictions.csv",
        "summary_json": project_root / "reports" / "external_prediction_summary.json",
        "distribution_png": project_root / "reports" / "external_sentiment_distribution.png",
    }


def load_model(model_path: Path):
    """Charge le pipeline de production sans modification."""
    if not model_path.exists():
        print(f"[ERREUR] Le modèle entraîné est introuvable : '{model_path}'")
        print("Veuillez exécuter 'python src/train_final_model.py' au préalable.")
        sys.exit(1)

    print(f"Chargement du modèle de production : '{model_path.name}'...")
    try:
        model = joblib.load(model_path)
        print("   -> Modèle chargé avec succès.")
        return model
    except Exception as e:
        print(f"[ERREUR] Échec du chargement du modèle : {e}")
        sys.exit(1)


def load_dataset_b(csv_path: Path) -> pd.DataFrame:
    """Charge Dataset B avec l'encodage et le séparateur validés lors de l'exploration."""
    if not csv_path.exists():
        print(f"[ERREUR] Dataset B introuvable : '{csv_path}'")
        sys.exit(1)

    print(f"Chargement de Dataset B : '{csv_path.name}'...")
    try:
        # Encodage latin1 et séparateur \t pour capturer les avis textuels sans division erronée
        df = pd.read_csv(csv_path, sep="\t", encoding="latin1")
    except Exception as e:
        print(f"[ERREUR] Échec de la lecture de Dataset B : {e}")
        sys.exit(1)

    if "Review" not in df.columns:
        print(f"[ERREUR] Colonne 'Review' absente de Dataset B. Colonnes trouvées : {list(df.columns)}")
        sys.exit(1)

    return df


def generate_visualization(
    sentiment_counts: pd.Series,
    sentiment_pcts: pd.Series,
    class_avg_conf: dict[str, float],
    overall_conf: float,
    output_path: Path,
) -> None:
    """Génère et sauvegarde le graphique de distribution des sentiments et de confiance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {
        "negative": "#D95F02",
        "neutral": "#7570B3",
        "positive": "#1B9E77",
    }
    ordered_classes = ["negative", "neutral", "positive"]

    # --- Subplot 1 : Distribution des Sentiments Prédits ---
    ax1 = axes[0]
    bar_colors = [colors.get(c, "#4C72B0") for c in ordered_classes]
    counts = [sentiment_counts.get(c, 0) for c in ordered_classes]
    pcts = [sentiment_pcts.get(c, 0.0) for c in ordered_classes]

    bars = ax1.bar(ordered_classes, counts, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_title("Distribution des Sentiments Prédits (Dataset B)", fontsize=13, weight="bold", pad=12)
    ax1.set_ylabel("Nombre d'avis prédits", fontsize=11)
    ax1.set_ylim(0, max(counts) * 1.18 if counts else 10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax1.annotate(
            f"{int(height)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    # --- Subplot 2 : Confiance Moyenne par Classe Prédite ---
    ax2 = axes[1]
    conf_values = [class_avg_conf.get(c, 0.0) * 100 for c in ordered_classes]
    bars2 = ax2.bar(ordered_classes, conf_values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.axhline(overall_conf * 100, color="red", linestyle="--", linewidth=1.5, label=f"Moyenne globale ({overall_conf*100:.1f}%)")
    ax2.set_title("Niveau de Confiance Moyen par Classe Prédite", fontsize=13, weight="bold", pad=12)
    ax2.set_ylabel("Confiance moyenne (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    plt.suptitle("ReviewMind — Prédictions Externes sur Eco-hotel Reviews (401 avis originaux / 399 uniques)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   -> Graphique sauvegardé : '{output_path}'")


def main() -> None:
    paths = get_project_paths()

    print("=" * 80)
    print("        REVIEWMIND — PRÉDICTION DU SENTIMENT SUR DATASET B (EXTERNE)")
    print("=" * 80)

    # 1. Chargement du modèle de production
    print("\n--- 1. CHARGEMENT DU MODÈLE DE PRODUCTION ---")
    model = load_model(paths["model_file"])
    model_classes = list(model.classes_)
    print(f"   Classes du modèle : {model_classes}")

    # 2. Chargement de Dataset B (original préservé)
    print("\n--- 2. CHARGEMENT ET PRÉPARATION DE DATASET B ---")
    df_raw = load_dataset_b(paths["dataset_b"])
    total_raw_rows = len(df_raw)
    print(f"   Nombre total de lignes chargées : {total_raw_rows}")

    # 3. Identification des doublons
    duplicates_mask = df_raw.duplicated(subset=["Review"], keep="first")
    num_duplicates = int(duplicates_mask.sum())
    print(f"   Nombre d'avis dupliqués identifiés : {num_duplicates}")

    # 4. Création d'une copie de travail sans doublons, avec préservation de l'index d'origine
    df_work = df_raw.loc[~duplicates_mask].copy()
    # Ajout explicite de la colonne original_index
    df_work["original_index"] = df_work.index
    analyzed_rows = len(df_work)
    print(f"   Avis uniques conservés pour l'inférence : {analyzed_rows} (index d'origine conservé)")

    # 5. Inférence / Prédiction via le pipeline
    print("\n--- 3. APPLICATION DU MODÈLE SUR LES AVIS UNIQUES ---")
    X_external = df_work["Review"].astype(str)

    # Prédiction de classe
    predictions = model.predict(X_external)
    df_work["Predicted_Sentiment"] = predictions

    # Calcul des probabilités de classe si disponible (MultinomialNB supporte predict_proba)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_external)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(model_classes)}

        df_work["Probability_Negative"] = probas[:, class_to_idx["negative"]].round(4)
        df_work["Probability_Neutral"] = probas[:, class_to_idx["neutral"]].round(4)
        df_work["Probability_Positive"] = probas[:, class_to_idx["positive"]].round(4)

        # La confiance correspond à la probabilité maximale assignée
        df_work["Confidence"] = np.max(probas, axis=1).round(4)
    else:
        df_work["Probability_Negative"] = np.nan
        df_work["Probability_Neutral"] = np.nan
        df_work["Probability_Positive"] = np.nan
        df_work["Confidence"] = np.nan

    # Réorganisation ordonnée des colonnes
    output_columns = [
        "original_index",
        "Review",
        "Predicted_Sentiment",
        "Confidence",
        "Probability_Negative",
        "Probability_Neutral",
        "Probability_Positive",
    ]
    df_results = df_work[output_columns].copy()

    # 6. Statistiques de distribution des prédictions
    print("\n--- 4. DISTRIBUTION DES SENTIMENTS PRÉDITS ---")
    sentiment_counts = df_results["Predicted_Sentiment"].value_counts().reindex(model_classes, fill_value=0)
    sentiment_pcts = (sentiment_counts / analyzed_rows) * 100

    distribution_table = pd.DataFrame({
        "Nombre d'avis": sentiment_counts,
        "Pourcentage (%)": sentiment_pcts.round(2),
    })
    distribution_table.index.name = "Sentiment prédit"
    print(distribution_table.to_string())

    # 7. Statistiques de confiance
    print("\n--- 5. STATISTIQUES DE CONFIANCE DES PRÉDICTIONS ---")
    overall_mean_conf = float(df_results["Confidence"].mean())
    overall_median_conf = float(df_results["Confidence"].median())
    overall_min_conf = float(df_results["Confidence"].min())
    overall_max_conf = float(df_results["Confidence"].max())

    print(f"Confiance moyenne globale  : {overall_mean_conf * 100:.2f}%")
    print(f"Confiance médiane          : {overall_median_conf * 100:.2f}%")
    print(f"Étendue (Min -> Max)       : {overall_min_conf * 100:.2f}% -> {overall_max_conf * 100:.2f}%")

    class_avg_conf = {}
    print("\nConfiance moyenne par sentiment prédit :")
    for s in model_classes:
        sub = df_results[df_results["Predicted_Sentiment"] == s]
        avg = float(sub["Confidence"].mean()) if len(sub) > 0 else 0.0
        class_avg_conf[s] = round(avg, 4)
        print(f"  - {s:<10} : {avg * 100:>6.2f}% (sur {len(sub)} avis)")

    # 8. Analyse des avis à faible confiance
    print(f"\n--- 6. AVIS À FAIBLE CONFIANCE (Seuil < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%) ---")
    low_conf_mask = df_results["Confidence"] < LOW_CONFIDENCE_THRESHOLD
    low_conf_df = df_results[low_conf_mask].sort_values("Confidence")
    num_low_conf = len(low_conf_df)
    pct_low_conf = (num_low_conf / analyzed_rows) * 100

    print(f"Nombre d'avis à faible confiance (< {LOW_CONFIDENCE_THRESHOLD*100:.0f}%) : {num_low_conf} ({pct_low_conf:.2f}%)")
    if num_low_conf > 0:
        print("\nExemples d'avis avec incertitude du modèle :")
        for i, (_, row) in enumerate(low_conf_df.head(3).iterrows(), start=1):
            text_snippet = row["Review"][:140] + "..." if len(row["Review"]) > 140 else row["Review"]
            print(f"  [{i}] Ligne {row['original_index']} | Prédit: {row['Predicted_Sentiment']} (Confiance: {row['Confidence']*100:.1f}%)")
            print(f"      Probas: Neg={row['Probability_Negative']*100:.1f}%, Neu={row['Probability_Neutral']*100:.1f}%, Pos={row['Probability_Positive']*100:.1f}%")
            print(f"      Avis: \"{text_snippet}\"")

    # 9. Affichage de cinq exemples par sentiment prédit
    print("\n--- 7. EXEMPLES D'AVIS PAR SENTIMENT PRÉDIT ---")
    samples_by_class = {}
    for s in model_classes:
        subset = df_results[df_results["Predicted_Sentiment"] == s]
        n_sample = min(5, len(subset))
        print(f"\n[{s.upper()}] — {n_sample} exemple(s) affiché(s) sur {len(subset)} :")
        sampled = subset.head(n_sample)

        samples_by_class[s] = []
        for i, (_, row) in enumerate(sampled.iterrows(), start=1):
            text_snippet = row["Review"][:180] + "..." if len(row["Review"]) > 180 else row["Review"]
            print(f"  {i}. [Ligne {row['original_index']}] Confiance: {row['Confidence']*100:.1f}% (Neg: {row['Probability_Negative']*100:.1f}%, Neu: {row['Probability_Neutral']*100:.1f}%, Pos: {row['Probability_Positive']*100:.1f}%)")
            print(f"     \"{text_snippet}\"")

            samples_by_class[s].append({
                "original_index": int(row["original_index"]),
                "confidence": float(row["Confidence"]),
                "probabilities": {
                    "negative": float(row["Probability_Negative"]),
                    "neutral": float(row["Probability_Neutral"]),
                    "positive": float(row["Probability_Positive"]),
                },
                "review_snippet": text_snippet,
            })

    # 10. Sauvegarde des artefacts
    print("\n--- 8. SAUVEGARDE DES FICHIERS DE RÉSULTATS ---")
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)

    # 10.1 CSV des prédictions
    try:
        df_results.to_csv(paths["predictions_csv"], index=False, encoding="utf-8")
        print(f"   -> CSV des prédictions : '{paths['predictions_csv']}'")
    except Exception as e:
        print(f"[ERREUR] Échec de la sauvegarde du CSV : {e}")
        sys.exit(1)

    # 10.2 JSON du résumé d'inférence
    summary_payload = {
        "metadata": {
            "task": "External Dataset Sentiment Prediction",
            "model_source": paths["model_file"].name,
            "external_dataset": paths["dataset_b"].name,
            "total_raw_records": total_raw_rows,
            "duplicate_records_excluded": num_duplicates,
            "unique_records_analyzed": analyzed_rows,
        },
        "predicted_sentiment_distribution": {
            s: {
                "count": int(sentiment_counts[s]),
                "percentage": round(float(sentiment_pcts[s]), 2),
                "average_confidence": class_avg_conf[s],
            }
            for s in model_classes
        },
        "confidence_statistics": {
            "overall_mean": round(overall_mean_conf, 4),
            "overall_median": round(overall_median_conf, 4),
            "overall_min": round(overall_min_conf, 4),
            "overall_max": round(overall_max_conf, 4),
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "low_confidence_count": num_low_conf,
            "low_confidence_percentage": round(pct_low_conf, 2),
        },
        "sample_predictions_by_sentiment": samples_by_class,
        "methodological_note": {
            "important": (
                "These predicted sentiment labels and probabilities represent inferences "
                "generated by the ReviewMind production model trained on TripAdvisor reviews. "
                "They are NOT ground-truth human annotations. Since Dataset B contains no "
                "original ratings or verified sentiment labels, no accuracy, precision, "
                "recall, or F1-score can be calculated on this external corpus."
            )
        },
        "output_files": {
            "predictions_csv": paths["predictions_csv"].relative_to(paths["project_root"]).as_posix(),
            "summary_json": paths["summary_json"].relative_to(paths["project_root"]).as_posix(),
            "distribution_plot": paths["distribution_png"].relative_to(paths["project_root"]).as_posix(),
        },
    }

    try:
        with open(paths["summary_json"], "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4, ensure_ascii=False)
        print(f"   -> Résumé JSON : '{paths['summary_json']}'")
    except Exception as e:
        print(f"[ERREUR] Échec de la sauvegarde du JSON : {e}")
        sys.exit(1)

    # 10.3 Graphique de distribution
    generate_visualization(
        sentiment_counts=sentiment_counts,
        sentiment_pcts=sentiment_pcts,
        class_avg_conf=class_avg_conf,
        overall_conf=overall_mean_conf,
        output_path=paths["distribution_png"],
    )

    # 11. Résumé final et rappel méthodologique
    print("\n" + "=" * 80)
    print("                    RÉSUMÉ FINAL DE L'INFÉRENCE EXTERNE")
    print("=" * 80)
    print(f"Modèle appliqué          : {paths['model_file'].name}")
    print(f"Dataset analysé          : {paths['dataset_b'].name} ({analyzed_rows} avis uniques analysés)")
    print(f"Confiance moyenne        : {overall_mean_conf * 100:.2f}%")
    print(f"Répartition prédite      :")
    for s in model_classes:
        print(f"  - {s:<10} : {sentiment_counts[s]:>4} avis ({sentiment_pcts[s]:>6.2f}%) | Confiance moy.: {class_avg_conf[s]*100:.1f}%")
    print("-" * 80)
    print("NOTE MÉTHODOLOGIQUE ESSENTIELLE :")
    print("  Ces résultats sont des prédictions (inférences) du modèle et NE CONSTITUENT PAS")
    print("  des annotations réelles (vérité terrain). Dataset B ne comportant ni notes ni labels,")
    print("  aucune métrique d'évaluation supervisée (Accuracy, F1-score) ne peut être calculée.")
    print("=" * 80)


if __name__ == "__main__":
    main()

