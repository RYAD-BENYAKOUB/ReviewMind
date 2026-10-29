"""
Script de préparation des données pour le projet ReviewMind.

Ce script :
1. Charge le dataset original sans le modifier.
2. Vérifie la présence et la conformité des colonnes Review et Rating.
3. Filtre les avis vides ou composés uniquement d'espaces.
4. Crée la colonne Sentiment (negative, neutral, positive).
5. Affiche les statistiques et des exemples pour chaque classe de sentiment.
6. Sauvegarde les données préparées dans data/reviews_prepared.csv.
"""

from pathlib import Path
import sys
import pandas as pd

# Configuration de l'encodage de la sortie standard pour Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_data_paths() -> tuple[Path, Path]:
    """Retourne les chemins des fichiers source et destination."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    input_file = data_dir / "tripadvisor_hotel_reviews.csv"
    output_file = data_dir / "reviews_prepared.csv"
    return input_file, output_file


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Vérifie la présence des colonnes obligatoires."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print("[ERREUR] Les colonnes attendues ne sont pas toutes présentes dans le fichier.")
        print(f"  - Colonnes requises   : {required_columns}")
        print(f"  - Colonnes existantes : {list(df.columns)}")
        print(f"  - Colonnes manquantes : {missing}")
        sys.exit(1)


def validate_ratings(df: pd.DataFrame) -> None:
    """Vérifie que les valeurs de la colonne Rating sont valides (1 à 5)."""
    expected_ratings = {1, 2, 3, 4, 5}
    actual_ratings = set(df["Rating"].dropna().unique())
    unexpected = actual_ratings - expected_ratings

    if unexpected:
        print(f"[AVERTISSEMENT] Des valeurs inattendues ont été détectées dans la colonne 'Rating' : {unexpected}")
        anomalies_count = df[df["Rating"].isin(unexpected)].shape[0]
        print(f"  - Nombre de lignes concernées : {anomalies_count}")
    else:
        print("Validation de 'Rating' : toutes les valeurs sont conformes (notes comprises entre 1 et 5).")


def main() -> None:
    input_path, output_path = get_data_paths()

    print("=" * 70)
    print("          PRÉPARATION DES DONNÉES - REVIEWMIND (ÉTAPE 2)")
    print("=" * 70)

    # 1. Vérification de l'existence et chargement du fichier source
    if not input_path.exists():
        print(f"[ERREUR] Le fichier source est introuvable : '{input_path}'")
        print("Veuillez vérifier que le fichier se trouve bien dans le dossier 'data/'.")
        sys.exit(1)

    print(f"\n1. Chargement du dataset : '{input_path.name}'...")
    try:
        # Chargement sans modifier le fichier d'origine
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERREUR] Échec de la lecture du fichier CSV : {e}")
        sys.exit(1)

    print(f"   -> {len(df):,} lignes chargées avec succès.")

    # 2. Vérification des colonnes attendues
    print("\n2. Vérification des colonnes...")
    validate_columns(df, ["Review", "Rating"])
    print("   -> Colonnes 'Review' et 'Rating' bien présentes.")

    # 3. Vérification des valeurs de Rating
    print("\n3. Contrôle des valeurs de notes (Rating)...")
    validate_ratings(df)

    # 4. Suppression des lignes dont l'avis est vide ou ne contient que des espaces
    print("\n4. Filtrage des avis vides ou constitués uniquement d'espaces...")
    # Détection des chaînes vides ou contenant uniquement des espaces blancs
    is_empty_or_whitespace = df["Review"].isna() | (df["Review"].astype(str).str.strip() == "")
    deleted_count = int(is_empty_or_whitespace.sum())

    if deleted_count > 0:
        df_prepared = df[~is_empty_or_whitespace].copy()
        print(f"   -> {deleted_count} ligne(s) vide(s) supprimée(s).")
    else:
        df_prepared = df.copy()
        print("   -> 0 ligne supprimée (aucun avis vide ou composé uniquement d'espaces).")

    print(f"   -> Nombre de lignes conservées : {len(df_prepared):,}")

    # 5. Création de la colonne Sentiment
    print("\n5. Création de la colonne 'Sentiment'...")
    # Règle : Rating 1 ou 2 = negative ; Rating 3 = neutral ; Rating 4 ou 5 = positive
    sentiment_mapping = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive"
    }

    df_prepared["Sentiment"] = df_prepared["Rating"].map(sentiment_mapping)

    # Vérification d'éventuelles valeurs non catégorisées
    unmapped_count = df_prepared["Sentiment"].isna().sum()
    if unmapped_count > 0:
        print(f"[ATTENTION] {unmapped_count} ligne(s) n'ont pas pu recevoir de sentiment.")

    print("   -> Colonne 'Sentiment' ajoutée.")

    # 6. Affichage du nombre et pourcentage d'avis par sentiment
    print("\n6. Répartition des classes de sentiment :")
    sentiment_counts = df_prepared["Sentiment"].value_counts()
    sentiment_pcts = df_prepared["Sentiment"].value_counts(normalize=True) * 100

    sentiment_summary = pd.DataFrame({
        "Nombre d'avis": sentiment_counts,
        "Pourcentage (%)": sentiment_pcts.round(2)
    })
    sentiment_summary.index.name = "Sentiment"
    print(sentiment_summary.to_string())

    # 7. Affichage d'exemples pour chaque sentiment
    print("\n7. Exemples d'avis par classe de sentiment :")
    for sentiment_label in ["negative", "neutral", "positive"]:
        subset = df_prepared[df_prepared["Sentiment"] == sentiment_label]
        if not subset.empty:
            example = subset.sample(n=1, random_state=42).iloc[0]
            truncated_review = str(example["Review"]).strip()
            # Affichage tronqué pour une lecture confortable dans le terminal
            if len(truncated_review) > 250:
                display_review = truncated_review[:250] + "..."
            else:
                display_review = truncated_review

            print(f"\n[{sentiment_label.upper()}] - Note: {example['Rating']}/5")
            print(f"Extrait : \"{display_review}\"")

    # 8. Sauvegarde du résultat dans data/reviews_prepared.csv
    print(f"\n8. Sauvegarde du fichier préparé...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_prepared.to_csv(output_path, index=False, encoding="utf-8")
        print(f"   -> Fichier sauvegardé avec succès dans : '{output_path}'")
        print(f"   -> Dimensions finales : {df_prepared.shape[0]:,} lignes, {df_prepared.shape[1]} colonnes ({list(df_prepared.columns)})")
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder le fichier '{output_path}' : {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Préparation des données terminée avec succès.")
    print("=" * 70)


if __name__ == "__main__":
    main()

