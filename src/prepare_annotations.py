"""
ReviewMind — Préparation du Système d'Annotation Manuelle (Dataset B)

Ce script génère les matrices d'annotation pour l'évaluation externe humaine :
- data/external/annotator_A.csv
- data/external/annotator_B.csv
- data/external/annotation_consensus.csv

Protocole :
1. Charger le Dataset B (Calheiros et al., 2017) avec sep='\\t' et encoding='latin1'.
2. Conserver l'index initial dans la colonne 'Original_Index'.
3. Dédupliquer les avis textuels (conserver la première occurrence, soit 399 avis uniques).
4. Créer les fichiers d'annotation avec les colonnes :
   - Original_Index
   - Review
   - Human_Sentiment (vide)
   - Annotation_Notes (vide)
5. Protéger impérativement les fichiers existants contenant du travail humain.
6. Valider la cohérence (399 lignes, unicité des index, alignement identique).
"""

from pathlib import Path
import sys
import pandas as pd

# Configuration de l'encodage de la sortie standard pour Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_paths() -> dict[str, Path]:
    """Retourne l'ensemble des chemins utilisés par le script."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    external_dir = data_dir / "external"
    return {
        "project_root": project_root,
        "input_dataset_b": data_dir / "dataset-CalheirosMoroRita-2017.csv",
        "external_dir": external_dir,
        "annotator_a": external_dir / "annotator_A.csv",
        "annotator_b": external_dir / "annotator_B.csv",
        "consensus": external_dir / "annotation_consensus.csv",
        "guidelines": external_dir / "ANNOTATION_GUIDELINES.md",
    }


def check_existing_annotations(target_files: dict[str, Path]) -> None:
    """
    Vérifie si des fichiers d'annotation existent déjà et contiennent du travail humain.
    Arrête immédiatement le script si du travail humain risque d'être écrasé.
    """
    conflicting_files = []

    for name, path in target_files.items():
        if path.exists():
            try:
                existing_df = pd.read_csv(path)
                if "Human_Sentiment" in existing_df.columns:
                    # Comptage des annotations non vides
                    annotated_mask = (
                        existing_df["Human_Sentiment"].notna()
                        & (existing_df["Human_Sentiment"].astype(str).str.strip() != "")
                    )
                    annotated_count = int(annotated_mask.sum())
                    if annotated_count > 0:
                        conflicting_files.append((name, path.name, annotated_count))
            except Exception as e:
                # En cas de doute sur la lecture d'un fichier existant, on le protège
                conflicting_files.append((name, path.name, f"lecture incertaine ({e})"))

    if conflicting_files:
        print("\n" + "!" * 80)
        print("[ALERTE SÉCURITÉ - TRAVAIL HUMAIN DÉTECTÉ]")
        print("Les fichiers d'annotation suivants existent déjà et contiennent des annotations :")
        for name, filename, count in conflicting_files:
            print(f"  - {filename} ({name}) : {count} annotation(s) détectée(s)")
        print("\nPour protéger ce travail d'annotation, le script s'est arrêté sans modifier ces fichiers.")
        print("Procédure sûre pour continuer :")
        print("  1. Sauvegardez ou renommez ces fichiers (ex. annotator_A_backup.csv).")
        print("  2. Ne relancez ce script que si vous souhaitez régénérer des formulaires vierges.")
        print("!" * 80)
        sys.exit(0)


def load_and_deduplicate(csv_path: Path) -> pd.DataFrame:
    """Charge Dataset B, préserve l'index d'origine et déduplique les avis."""
    if not csv_path.exists():
        print(f"[ERREUR] Dataset B introuvable : '{csv_path}'")
        sys.exit(1)

    print(f"1. Chargement du Dataset B depuis : '{csv_path.name}'...")
    try:
        df = pd.read_csv(csv_path, sep="\t", encoding="latin1")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le fichier CSV : {e}")
        sys.exit(1)

    if "Review" not in df.columns:
        print(f"[ERREUR] Colonne 'Review' introuvable dans le fichier. Colonnes présentes : {list(df.columns)}")
        sys.exit(1)

    total_rows = len(df)
    print(f"   -> Nombre d'enregistrements bruts : {total_rows}")

    # Conservation de l'index d'origine
    df["Original_Index"] = df.index

    # Détection et exclusion des doublons (conservation de la première occurrence)
    duplicates_mask = df.duplicated(subset=["Review"], keep="first")
    num_duplicates = int(duplicates_mask.sum())
    print(f"   -> Nombre d'avis dupliqués identifiés : {num_duplicates}")

    df_unique = df.loc[~duplicates_mask].copy()
    unique_count = len(df_unique)
    print(f"   -> Nombre d'avis uniques conservés : {unique_count}")

    if unique_count != 399:
        print(f"[ERREUR] Le nombre d'avis uniques attendu est de 399, obtenu : {unique_count}")
        sys.exit(1)

    return df_unique


def create_annotation_template(df_unique: pd.DataFrame) -> pd.DataFrame:
    """Crée la structure normalisée d'annotation vierge."""
    template_df = pd.DataFrame({
        "Original_Index": df_unique["Original_Index"].values,
        "Review": df_unique["Review"].values,
        "Human_Sentiment": "",
        "Annotation_Notes": "",
    })
    return template_df


def validate_generated_files(target_paths: list[Path], expected_indices: list[int]) -> None:
    """Valide systématiquement la conformité des trois fichiers générés."""
    print("\n3. Validation des fichiers générés...")
    expected_columns = ["Original_Index", "Review", "Human_Sentiment", "Annotation_Notes"]

    for path in target_paths:
        if not path.exists():
            print(f"[ERREUR] Fichier manquant après création : '{path.name}'")
            sys.exit(1)

        val_df = pd.read_csv(path, keep_default_na=False)

        # 1. Nombre de lignes
        if len(val_df) != 399:
            print(f"[ERREUR] '{path.name}' contient {len(val_df)} lignes au lieu de 399.")
            sys.exit(1)

        # 2. Colonnes exactes
        if list(val_df.columns) != expected_columns:
            print(f"[ERREUR] Colonnes incorrectes dans '{path.name}' : {list(val_df.columns)}")
            sys.exit(1)

        # 3. Unicité des Original_Index
        if val_df["Original_Index"].duplicated().any():
            print(f"[ERREUR] Des Original_Index dupliqués ont été détectés dans '{path.name}'.")
            sys.exit(1)

        # 4. Ordre identique des Original_Index
        if list(val_df["Original_Index"]) != expected_indices:
            print(f"[ERREUR] L'ordre des avis ne correspond pas dans '{path.name}'.")
            sys.exit(1)

        # 5. Champs d'annotation initialement vides
        sentiment_non_empty = val_df["Human_Sentiment"].astype(str).str.strip().ne("").sum()
        notes_non_empty = val_df["Annotation_Notes"].astype(str).str.strip().ne("").sum()
        if sentiment_non_empty > 0 or notes_non_empty > 0:
            print(f"[ERREUR] Les champs d'annotation ne sont pas vides dans '{path.name}'.")
            sys.exit(1)

        print(f"   -> Validation réussie pour '{path.name}' : 399 lignes, structure et alignement parfaits.")


def main() -> None:
    paths = get_paths()

    print("=" * 80)
    print("      REVIEWMIND — PRÉPARATION DU SYSTÈME D'ANNOTATION MANUELLE")
    print("=" * 80)

    # 1. Vérification de sécurité contre l'écrasement de travail existant
    target_files = {
        "Annotateur A": paths["annotator_a"],
        "Annotateur B": paths["annotator_b"],
        "Consensus": paths["consensus"],
    }
    check_existing_annotations(target_files)

    # 2. Chargement et déduplication
    df_unique = load_and_deduplicate(paths["input_dataset_b"])
    expected_indices = list(df_unique["Original_Index"])

    # 3. Création du gabarit d'annotation
    template_df = create_annotation_template(df_unique)

    # 4. Export des trois fichiers dans data/external/
    print("\n2. Génération des fichiers d'annotation dans 'data/external/'...")
    paths["external_dir"].mkdir(parents=True, exist_ok=True)

    for name, target_path in target_files.items():
        try:
            template_df.to_csv(target_path, index=False, encoding="utf-8")
            print(f"   -> Fichier généré ({name}) : '{target_path.name}'")
        except Exception as e:
            print(f"[ERREUR] Échec de la création de '{target_path.name}' : {e}")
            sys.exit(1)

    # 5. Validation rigoureuse
    validate_generated_files(list(target_files.values()), expected_indices)

    # 6. Bilan et affichage récapitulatif
    print("\n" + "=" * 80)
    print("                      BILAN DE LA GÉNÉRATION")
    print("=" * 80)
    print(f"Dataset source               : {paths['input_dataset_b'].name}")
    print(f"Avis uniques exportés        : {len(template_df)} avis (index d'origine préservés)")
    print(f"Colonnes générées            : {list(template_df.columns)}")
    print("Fichiers d'annotation créés  :")
    print(f"  1. Formulaire Annotateur A : {paths['annotator_a']}")
    print(f"  2. Formulaire Annotateur B : {paths['annotator_b']}")
    print(f"  3. Formulaire Consensus    : {paths['consensus']}")
    print(f"Guide d'annotation           : {paths['guidelines']}")
    print("-" * 80)
    print("INSTRUCTIONS SUIVANTES :")
    print("  1. L'Annotateur A remplit annotator_A.csv de façon indépendante.")
    print("  2. L'Annotateur B remplit annotator_B.csv de façon indépendante.")
    print("  3. Une fois les annotations complétées, comparer les étiquettes et")
    print("     renseigner les accords dans annotation_consensus.csv.")
    print("  4. NE PAS lancer l'évaluation externe supervisée tant que le consensus")
    print("     n'est pas finalisé.")
    print("=" * 80)


if __name__ == "__main__":
    main()

