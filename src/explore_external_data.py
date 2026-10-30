"""
ReviewMind — Exploration du Dataset B (Dataset Externe Eco-hotel Reviews)

Ce script analyse le dataset externe Calheiros, Moro, Rita (2017) :
`data/dataset-CalheirosMoroRita-2017.csv`

Objectifs :
1. Charger le fichier CSV avec pandas.
2. Afficher les dimensions, colonnes, types de données et premières lignes.
3. Quantifier les valeurs manquantes et les doublons.
4. Identifier la colonne textuelle et rechercher d'éventuelles colonnes d'annotations.
5. Analyser si les annotations permettent de construire les classes (negative, neutral, positive).
6. Sauvegarder les résultats structurés dans `reports/external_dataset_exploration.json`.
7. Conclure sur la compatibilité avec le modèle ReviewMind (inférence vs évaluation supervisée).

Contraintes :
- Ne modifie pas le dataset original.
- N'invente aucun label.
- N'effectue aucune prédiction et n'entraîne aucun modèle.
"""

from pathlib import Path
import json
import sys
import pandas as pd

# Configuration de l'encodage de la sortie standard pour Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_paths() -> dict[str, Path]:
    """Retourne les chemins des fichiers et répertoires utilisés."""
    project_root = Path(__file__).resolve().parent.parent
    return {
        "project_root": project_root,
        "input_csv": project_root / "data" / "dataset-CalheirosMoroRita-2017.csv",
        "reports_dir": project_root / "reports",
        "output_json": project_root / "reports" / "external_dataset_exploration.json",
    }


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Charge le dataset externe avec pandas en gérant l'encodage et le séparateur.
    Le fichier contient des avis textuels avec virgules non systématiquement échappées.
    L'utilisation de sep='\\t' permet de capturer chaque avis dans la colonne 'Review'.
    """
    if not csv_path.exists():
        print(f"[ERREUR] Le fichier est introuvable : '{csv_path}'")
        sys.exit(1)

    encodings_to_try = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
    last_error = None

    for enc in encodings_to_try:
        try:
            # Le fichier est un TSV/CSV à colonne unique 'Review'
            df = pd.read_csv(csv_path, sep="\t", encoding=enc)
            return df, enc
        except Exception as e:
            last_error = e
            continue

    print(f"[ERREUR] Impossible de charger le fichier CSV : {last_error}")
    sys.exit(1)


def main() -> None:
    paths = get_paths()

    print("=" * 80)
    print("      REVIEWMIND — EXPLORATION DU DATASET B (CALHEIROS ET AL., 2017)")
    print("=" * 80)

    # 1. Chargement du dataset
    print(f"\n1. Chargement du fichier : '{paths['input_csv'].name}'...")
    df, used_encoding = load_dataset(paths["input_csv"])
    print(f"   -> Fichier chargé avec succès (encodage détecté : {used_encoding}).")

    # 2. Dimensions et colonnes
    num_rows, num_cols = df.shape
    columns_list = list(df.columns)
    print("\n2. Dimensions et colonnes :")
    print(f"   - Nombre d'enregistrements (lignes) : {num_rows:,}")
    print(f"   - Nombre de colonnes               : {num_cols}")
    print(f"   - Noms des colonnes                : {columns_list}")

    # 3. Aperçu des cinq premières lignes
    print("\n3. Aperçu des 5 premières lignes :")
    for idx, row in df.head(5).iterrows():
        review_preview = str(row.iloc[0]).strip()
        if len(review_preview) > 110:
            review_preview = review_preview[:110] + "..."
        print(f"   [Ligne {idx}] : \"{review_preview}\"")

    # 4. Types de données
    print("\n4. Types de données par colonne :")
    dtypes_dict = {}
    for col, dtype in df.dtypes.items():
        dtypes_dict[col] = str(dtype)
        print(f"   - {col} : {dtype}")

    # 5. Valeurs manquantes et doublons
    print("\n5. Qualité des données (valeurs manquantes et doublons) :")
    null_counts = df.isnull().sum()
    null_dict = {}
    for col, count in null_counts.items():
        pct = (count / num_rows) * 100 if num_rows > 0 else 0
        null_dict[col] = {"missing_count": int(count), "percentage": round(pct, 2)}
        print(f"   - Valeurs manquantes dans '{col}' : {count} ({pct:.2f}%)")

    # Vérification d'éventuels avis vides ou constitués uniquement d'espaces
    empty_strings = 0
    if "Review" in df.columns:
        empty_strings = int((df["Review"].dropna().astype(str).str.strip() == "").sum())
    print(f"   - Avis textuels vides ou composés uniquement d'espaces : {empty_strings}")

    total_duplicates = int(df.duplicated().sum())
    print(f"   - Nombre de lignes dupliquées : {total_duplicates}")

    # 6. Identification de la colonne textuelle
    text_columns_candidates = [
        c for c in df.columns if any(kw in c.lower() for kw in ["review", "text", "comment", "avis"])
    ]
    text_column = text_columns_candidates[0] if text_columns_candidates else None
    print(f"\n6. Colonne contenant les avis textuels :")
    if text_column:
        print(f"   -> Identifiée : '{text_column}' (type : {df[text_column].dtype})")
    else:
        print("   -> Aucune colonne textuelle évidente trouvée.")

    # 7. Identification d'éventuelles colonnes d'annotations / notes / sentiments
    annotation_keywords = ["rating", "score", "star", "sentiment", "label", "class", "polarity", "note", "target"]
    annotation_columns = [
        c for c in df.columns if any(kw in c.lower() for kw in annotation_keywords)
    ]
    print(f"\n7. Recherche de colonnes de notes, sentiments ou annotations :")
    if annotation_columns:
        print(f"   -> Colonne(s) trouvée(s) : {annotation_columns}")
    else:
        print("   -> Aucune colonne de note, sentiment ou annotation n'est présente dans ce dataset.")

    # 8. Distribution des valeurs des colonnes d'annotations
    print(f"\n8. Distribution des colonnes d'annotations :")
    annotations_distribution = {}
    if annotation_columns:
        for col in annotation_columns:
            dist = df[col].value_counts(dropna=False).to_dict()
            annotations_distribution[col] = dist
            print(f"   Distribution pour '{col}' : {dist}")
    else:
        print("   -> N/A (aucune colonne d'annotation disponible).")

    # 9. Affichage de cinq exemples d'avis
    print("\n9. Exemples d'avis avec leurs annotations éventuelles :")
    examples_list = []
    sample_df = df.head(5)
    for i, (idx, row) in enumerate(sample_df.iterrows(), start=1):
        review_full = str(row[text_column]).strip() if text_column else str(row.iloc[0]).strip()
        truncated = review_full[:220] + "..." if len(review_full) > 220 else review_full
        
        example_meta = {
            "index": int(idx),
            "review_preview": truncated,
            "annotations": {col: row[col] for col in annotation_columns} if annotation_columns else None
        }
        examples_list.append(example_meta)

        print(f"\n   [Exemple {i}] (Index {idx})")
        print(f"   Avis       : \"{truncated}\"")
        print(f"   Annotation : {example_meta['annotations']}")

    # 10. Vérification de la constructibilité des 3 catégories (negative, neutral, positive)
    print("\n10. Évaluation de la possibilité de construire les 3 catégories de sentiment :")
    can_construct_classes = False
    reasons = []

    if not annotation_columns:
        reasons.append("Le fichier ne contient aucune colonne de note numérique (Rating) ni de label de sentiment.")
        reasons.append("Il s'agit d'un corpus brut non étiqueté (unsupervised text corpus).")
        reasons.append("Impossible d'assigner objectivement des classes ground-truth (negative, neutral, positive) sans processus d'annotation externe.")
    else:
        can_construct_classes = True

    print(f"   -> Possibilité de construire les 3 catégories ground-truth : {can_construct_classes}")
    for r in reasons:
        print(f"      - {r}")

    # 11. Conclusion sur la compatibilité avec le modèle ReviewMind
    compatibility_analysis = {
        "text_domain_compatibility": {
            "status": "Compatible",
            "details": "Le corpus est composé d'avis hôteliers en anglais portant sur l'hôtellerie (Areias do Seixo Eco-Resort), cohérent avec le domaine hôtelier de TripAdvisor."
        },
        "inference_compatibility": {
            "status": "Totalement compatible",
            "details": "Le modèle final 'models/reviewmind_final_model.joblib' peut ingérer directement la colonne 'Review' pour générer des prédictions de sentiments (negative, neutral, positive) sur ces 401 avis."
        },
        "supervised_evaluation_compatibility": {
            "status": "Non compatible en l'état (absence de vérité terrain)",
            "details": "Une évaluation quantitative supervisée (Accuracy, Macro F1, Matrice de confusion) est impossible directement sur ce fichier car il ne comporte aucune étiquette cible (ni note de 1 à 5, ni sentiment annoté). Pour une évaluation quantitative externe, une étape préalable d'annotation manuelle ou de récupération des notes sources est requise."
        }
    }

    # 12. Sauvegarde des résultats dans reports/external_dataset_exploration.json
    paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    report_payload = {
        "dataset_name": "Calheiros, Moro, Rita (2017) — Eco-hotel Reviews",
        "file_name": paths["input_csv"].name,
        "encoding_used": used_encoding,
        "shape": {
            "num_records": num_rows,
            "num_columns": num_cols
        },
        "columns": columns_list,
        "dtypes": dtypes_dict,
        "data_quality": {
            "null_values": null_dict,
            "empty_whitespace_reviews": empty_strings,
            "duplicate_records": total_duplicates
        },
        "column_roles": {
            "text_column": text_column,
            "annotation_columns": annotation_columns
        },
        "annotations_distribution": annotations_distribution,
        "sentiment_classes_constructible": can_construct_classes,
        "sentiment_classes_assessment": reasons,
        "sample_examples": examples_list,
        "compatibility_with_reviewmind": compatibility_analysis
    }

    try:
        with open(paths["output_json"], "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=4, ensure_ascii=False)
        print(f"\n-> Rapport d'exploration sauvegardé : '{paths['output_json']}'")
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder le fichier JSON : {e}")
        sys.exit(1)

    # 13. Conclusion affichée dans le terminal
    print("\n" + "=" * 80)
    print("           CONCLUSION SUR LA COMPATIBILITÉ AVEC REVIEWMIND")
    print("=" * 80)
    print("1. Inférence / Déploiement :")
    print("   -> COMPATIBLE : Le pipeline final peut traiter directement ces 401 avis.")
    print("2. Évaluation supervisée externe :")
    print("   -> NON COMPATIBLE EN L'ÉTAT : Ce fichier ne contient aucune colonne de")
    print("      notes (Rating) ni de sentiments annotés. Pour calculer des métriques")
    print("      de performance (Accuracy, Macro F1), une vérité terrain doit être")
    print("      préalablement fournie ou annotée.")
    print("=" * 80)


if __name__ == "__main__":
    main()

