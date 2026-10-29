"""
Script d'exploration des données pour le projet ReviewMind.

Ce script charge le jeu de données d'avis présent dans le dossier data/,
vérifie sa structure et affiche des statistiques descriptives fondamentales.
"""

from pathlib import Path
import sys
import pandas as pd

# Configuration de l'encodage de la sortie standard pour Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_dataset_path() -> Path:
    """Détermine et valide le chemin d'accès au fichier CSV du dataset."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"[ERREUR] Le dossier des données est introuvable : '{data_dir}'")
        sys.exit(1)

    # Nom de fichier prioritaire
    default_csv = data_dir / "tripadvisor_hotel_reviews.csv"
    if default_csv.exists() and default_csv.is_file():
        return default_csv

    # Recherche alternative de tout fichier .csv dans data/
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"[ERREUR] Aucun fichier CSV n'a été trouvé dans le dossier : '{data_dir}'")
        sys.exit(1)

    return csv_files[0]


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Vérifie la présence des colonnes attendues dans le DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print("[ERREUR] Les colonnes attendues ne sont pas toutes présentes dans le fichier CSV.")
        print(f"Colonnes attendues : {required_columns}")
        print(f"Colonnes trouvées   : {list(df.columns)}")
        print(f"Colonnes manquantes : {missing_columns}")
        sys.exit(1)


def main() -> None:
    # 1. Localisation et chargement du fichier CSV
    dataset_path = get_dataset_path()
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le fichier CSV '{dataset_path.name}' : {e}")
        sys.exit(1)

    print("=" * 70)
    print("           EXPLORATION DU JEU DE DONNÉES - REVIEWMIND")
    print("=" * 70)

    # 2. Nom du fichier, dimensions et colonnes
    print("\n--- 1. INFORMATIONS GÉNÉRALES ---")
    print(f"Fichier utilisé        : {dataset_path.name}")
    print(f"Chemin complet         : {dataset_path}")
    print(f"Dimensions             : {df.shape[0]:,} lignes, {df.shape[1]} colonnes")
    print(f"Liste des colonnes     : {list(df.columns)}")

    # 3. Affichage des cinq premières lignes
    print("\n--- 2. APERÇU DES 5 PREMIÈRES LIGNES ---")
    print(df.head())

    # 4. Types de données, valeurs manquantes et doublons
    print("\n--- 3. TYPES DE DONNÉES, VALEURS MANQUANTES ET DOUBLONS ---")
    print("Types de données :")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\nValeurs manquantes par colonne :")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        pct = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  - {col}: {count} ({pct:.2f}%)")
    print(f"Total des valeurs manquantes : {null_counts.sum()}")

    total_duplicates = df.duplicated().sum()
    print(f"\nNombre de lignes entièrement dupliquées : {total_duplicates}")

    # 6 & 7. Vérification des colonnes attendues avant utilisation spécifique
    required_cols = ["Review", "Rating"]
    check_required_columns(df, required_cols)

    # Doublons spécifiques aux avis textuels
    review_duplicates = df.duplicated(subset=["Review"]).sum()
    print(f"Nombre d'avis textuels dupliqués (colonne 'Review') : {review_duplicates}")

    # 5. Distribution des notes et trois exemples d'avis
    print("\n--- 4. DISTRIBUTION DES NOTES (Rating) ---")
    rating_counts = df["Rating"].value_counts().sort_index()
    rating_pcts = df["Rating"].value_counts(normalize=True).sort_index() * 100

    distribution_df = pd.DataFrame({
        "Nombre": rating_counts,
        "Pourcentage (%)": rating_pcts.round(2)
    })
    distribution_df.index.name = "Note"
    print(distribution_df)

    print("\n--- 5. TROIS EXEMPLES D'AVIS ---")
    # Échantillonnage de 3 exemples avec graine fixe pour la reproductibilité
    sample_size = min(3, len(df))
    samples = df.sample(n=sample_size, random_state=42)

    for i, (idx, row) in enumerate(samples.iterrows(), start=1):
        print(f"\n[Exemple {i}] (Ligne index {idx}) - Note : {row['Rating']}/5")
        # Affichage du texte de l'avis avec indentation propre
        review_text = str(row["Review"]).strip()
        print(f"Avis : {review_text}")

    print("\n" + "=" * 70)
    print("Fin de l'exploration des données.")
    print("=" * 70)


if __name__ == "__main__":
    main()
