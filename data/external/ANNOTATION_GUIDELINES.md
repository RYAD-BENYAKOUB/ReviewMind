# Guide d'Annotation Manuelle — Projet ReviewMind

Ce guide définit le protocole et les règles pour l'annotation manuelle des avis hôteliers du **Dataset B** (`data/dataset-CalheirosMoroRita-2017.csv`).

L'objectif est d'établir un jeu de référence (*ground truth*) fiable et indépendant afin d'évaluer la capacité de généralisation du modèle de classification de sentiments **ReviewMind** (TF-IDF + Multinomial Naive Bayes entraîné sur TripAdvisor).

---

## 1. Définition des Catégories de Sentiment

Chaque avis doit être classé dans l'une des trois catégories mutuellement exclusives suivantes :

### `positive` (Positif)
- **Définition** : L'avis exprime une expérience globalement favorable, une satisfaction générale, un enthousiasme ou une recommandation du séjour.
- **Indicateurs** : Éloges sur le personnel, le cadre, la restauration, le confort des chambres, volonté explicite de revenir ou de recommander l'établissement.
- **Nuance** : Un avis peut contenir une suggestion mineure d'amélioration (ex. : *"Seul petit bémol : élargir la carte des cocktails"*) tout en restant globalement et sans ambiguïté positif.

### `neutral` (Neutre / Mitigé)
- **Définition** : L'avis est principalement descriptif, factuel, équilibré ou mitigé/mixte, sans qu'un sentiment positif ou négatif ne se dégage de façon dominante.
- **Indicateurs** :
  - **Avis mixtes/balancés** : Équilibre réel entre aspects appréciés et déceptions équivalentes (ex. : *"Chambre spacieuse et propre, mais petit-déjeuner très décevant et accueil froid"*).
  - **Avis purement factuels/descriptifs** : Description objective des équipements, de la localisation ou de l'itinéraire sans appréciation affective marquée.
  - **Avis tièdes/moyens** : Expérience ordinaire, sans enthousiasme ni critique majeure (ex. : *"Séjour convenable, rien d'exceptionnel"*).

### `negative` (Négatif)
- **Définition** : L'avis exprime une expérience globalement défavorable, une insatisfaction marquée, une déception ou une critique substantielle de l'établissement.
- **Indicateurs** : Plaintes concernant la propreté, le bruit, le rapport qualité/prix jugé inacceptable, le manque de professionnalisme, promesses non tenues ou refus de recommander l'hôtel.

---

## 2. Règles d'Annotation Strictes

1. **Lecture intégrale** : Lisez impérativement l'intégralité du texte de l'avis avant d'attribuer une étiquette.
2. **Sens global prioritaire** : Évaluez la tonalité d'ensemble du séjour plutôt que des mots-clés isolés (attention aux tournures de politesse formelle ou aux figures de style qui masquent une critique).
3. **Étanchéité totale avec le modèle** : Ne consultez **jamais** les prédictions ou probabilités issues de ReviewMind. L'annotation doit être strictement humaine et indépendante.
4. **Interdiction d'outils automatiques** : Aucun outil automatique ou LLM ne doit être utilisé pour orienter ou automatiser la décision d'annotation.
5. **Respect de l'ambiguïté réelle** : Ne forcez pas une décision binaire (positif/négatif) si un avis est sincèrement neutre ou partagé à parts égales. La classe `neutral` existe précisément pour ces cas.
6. **Usage obligatoire des notes d'annotation** : Renseignez le champ `Annotation_Notes` dès qu'un avis présente une difficulté, un sarcasme, une ambiguïté ou une hésitation entre deux classes.
7. **Discussion obligatoire des cas incertains** : Tout avis annoté avec doute ou hésitation doit être formellement discuté lors de la session de consensus.
8. **Indépendance des annotateurs** : **L'Annotateur A et l'Annotateur B doivent travailler de façon rigoureusement indépendante**. Aucun échange d'informations ni consultation mutuelle des fichiers n'est autorisé avant la fin complète de la phase d'annotation individuelle.

---

## 3. Protocole et Workflow d'Annotation

```
+---------------------------+     +---------------------------+
|       Annotateur A        |     |       Annotateur B        |
|     (annotator_A.csv)     |     |     (annotator_B.csv)     |
+---------------------------+     +---------------------------+
              \                                 /
               \   Indépendance stricte        /
                \                             /
                 v                           v
         +-------------------------------------------+
         |     Étape 1 : Comparaison automatique     |
         |         (alignement par Original_Index)   |
         +-------------------------------------------+
                             |
                             v
         +-------------------------------------------+
         |     Étape 2 : Identification des écarts   |
         |         (Accords vs Désaccords)           |
         +-------------------------------------------+
                             |
                             v
         +-------------------------------------------+
         |     Étape 3 : Session de conciliation     |
         |         (Discussion collégiale)           |
         +-------------------------------------------+
                             |
                             v
         +-------------------------------------------+
         |     Étape 4 : Validation du consensus     |
         |       (annotation_consensus.csv)          |
         +-------------------------------------------+
                             |
                             v
         +-------------------------------------------+
         |  Évaluation externe supervisée du modèle  |
         +-------------------------------------------+
```

### Étapes détaillées :

1. **Attribution initiale** :
   - L'Annotateur A renseigne le fichier `data/external/annotator_A.csv`.
   - L'Annotateur B renseigne le fichier `data/external/annotator_B.csv`.
   - Les valeurs admises dans `Human_Sentiment` sont strictement : `positive`, `neutral`, `negative`.
2. **Alignement et comparaison** :
   - Les deux fichiers sont comparés ligne à ligne à l'aide de l'identifiant invariant `Original_Index`.
   - Le taux d'accord brut et le coefficient Kappa de Cohen sont calculés.
3. **Traitement des désaccords** :
   - Tous les avis pour lesquels `annotator_A.Human_Sentiment != annotator_B.Human_Sentiment` sont isolés.
   - Les deux annotateurs se réunissent pour relire ensemble l'avis et confronter leurs arguments (en s'appuyant sur les `Annotation_Notes`).
4. **Validation du consensus** :
   - Le label retenu après concertation est consigné dans `data/external/annotation_consensus.csv`.
   - Si un accord ne peut être trouvé à deux, un tiers arbitre tranche la décision.
5. **Clôture du jeu de référence** :
   - Aucun avis ne doit rester sans consensus ou avec un label incertain.
   - Seul le fichier `annotation_consensus.csv` servira de vérité terrain officielle pour calculer les métriques de test externe de ReviewMind.

