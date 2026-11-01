# ReviewMind — Complete Technical Audit & Scientific Knowledge Base

**Document Type:** Comprehensive Technical & Academic Audit  
**Project:** ReviewMind — Hotel Review Sentiment Classification with TF-IDF & Multinomial Naive Bayes  
**Author:** Benyakoub Mohammed Ryad  
**Academic Program:** Master 2 SID (Systèmes d'Information et Décision / Statistique et Informatique Décisionnelle)  
**Course:** Fouille de Données et Management des Systèmes (FDMS) — Chapitre : *Fouille de Données Textuelles (Text Mining)*  
**Repository:** [https://github.com/RYAD-BENYAKOUB/ReviewMind](https://github.com/RYAD-BENYAKOUB/ReviewMind)  
**Audit Date:** November 1, 2026  
**Auditor Role:** Senior NLP Engineer, Data Scientist, Software Architect & Academic Auditor  
**Status:** Complete, Evidence-Based, Production-Verified  

---

## 1. Executive Summary

**ReviewMind** is an open-source, reproducible text mining and natural language processing (NLP) framework designed for sentiment classification of customer reviews in the hospitality sector. Originally developed as an academic exploration within the Master 2 SID curriculum (*Fouille de Données et Management des Systèmes*), the project has evolved into a production-grade full-stack platform featuring a modular Flask REST API backend, an interactive client interface adhering to the **Dark Intelligence** design system, and an empirical evaluation protocol addressing severe class imbalance and out-of-domain generalization.

### Core Scientific & Technical Achievements

1. **Resolution of Minority Class Collapse**: In standard Multinomial Naive Bayes (MNB) architectures with empirical class priors ($P(c)$ derived from class frequency), majority-class dominance causes total collapse on minority categories. On the internal TripAdvisor test set ($4\,099$ reviews), the baseline MNB model classified **zero** reviews as neutral ($F_1 = 0.0000$). ReviewMind demonstrated that substituting uniform class priors (`fit_prior=False`) and lowering Laplace/Lidstone smoothing to $\alpha = 0.1$ restored class sensitivity, yielding a **Macro $F_1$-score of $0.6762$** and recovering the neutral class ($F_1 = 0.3883$, Recall $= 48.74\%$).
2. **Zero-Leakage Experimental Design**: Text representations are computed using a $20\,000$-feature TF-IDF vectorizer (unigrams and bigrams). Feature extraction parameters are fitted strictly on the $80\%$ training split ($16\,392$ reviews) during evaluation, and on the entirety of Dataset A ($20\,491$ reviews) solely for production deployment.
3. **Out-of-Domain Generalization & Scientific Integrity**: When evaluated on Dataset B (401 raw reviews, 399 unique reviews from an eco-resort in Portugal; Calheiros et al., 2017), the production model inferred $83.96\%$ positive, $8.27\%$ neutral, and $7.77\%$ negative sentiments. However, $57.64\%$ of reviews were classified with low confidence ($<60\%$), exposing significant domain shift. In strict adherence to scientific integrity, **no supervised metrics (accuracy, precision, recall, $F_1$) are fabricated for Dataset B** because the external corpus lacks ground-truth ratings or labels.
4. **Human Annotation Consensus Protocol**: To establish a legitimate ground truth for Dataset B, ReviewMind formalized a double-blind annotation protocol (`ANNOTATION_GUIDELINES.md`) with independent annotator templates (`annotator_A.csv`, `annotator_B.csv`) and an adjudicated consensus target (`annotation_consensus.csv`). Supervised external metrics remain officially classified as `PENDING` until human consensus is finalized.
5. **Full-Stack Decoupled Architecture**: The project incorporates a Flask REST API (`backend/`) exposing seven endpoints (`/api/health`, `/api/dashboard`, `/api/datasets`, `/api/reviews`, `/api/reviews/<id>`, `/api/predict`, `/api/evaluation`) and an interactive frontend (`web/`). All client-side fallback heuristics and fake weights have been completely eliminated; the Flask API is the sole source of truth.
6. **Codebase Quality & Automated Testing**: 19 automated unit tests written in standard Python `unittest` cover data validation, pipeline mechanics, and API route responses with $100\%$ pass rate in under $0.4$ seconds.

---

## 2. Project Identity & Objectives

### 2.1 Identity Metadata
- **Official Title**: `ReviewMind: Hotel Review Sentiment Classification with TF-IDF and Multinomial Naive Bayes`
- **Short Name**: `ReviewMind`
- **Tagline**: `An Open-Source Text Mining & NLP Framework for Hotel Review Sentiment Classification`
- **Author / Copyright Holder**: Benyakoub Mohammed Ryad (`ryadbenyakoub@gmail.com`)
- **Copyright Year**: 2026
- **License**: MIT License ([LICENSE](../LICENSE))
- **Release Version**: `1.0.0` (Released October 30, 2026)
- **Repository**: [https://github.com/RYAD-BENYAKOUB/ReviewMind](https://github.com/RYAD-BENYAKOUB/ReviewMind)
- **Course Context**: Master 2 SID, Course *Fouille de Données et Management des Systèmes (FDMS)*, Chapter *Fouille de Données Textuelles (Text Mining)*.

### 2.2 Academic Context & Motivation
In customer experience analytics, online reviews represent high-volume, unstructured textual data. While numerical ratings summarize satisfaction on a discrete scale ($1$ to $5$ stars), text comments convey granular sentiments, qualitative justifications, and nuanced trade-offs. 

Hospitality platforms routinely face three data mining challenges:
- **Extreme Class Imbalance**: Most online hotel guests only leave reviews following exceptionally positive or highly negative experiences. Moderate/neutral experiences are heavily underrepresented.
- **Probabilistic Model Distortion**: Classical generative classifiers like Multinomial Naive Bayes apply Bayes' rule by multiplying prior probabilities with feature likelihoods. When priors are skewed, the posterior probability of minority classes drops below decision thresholds, rendering the model blind to neutral feedback.
- **Domain Adaptation Failure**: NLP models trained on global platforms (e.g., TripAdvisor) often degrade when applied to niche hospitality contexts (e.g., luxury eco-resorts), where vocabulary, syntax, and expectations differ substantially.

ReviewMind was conceived to explore and resolve these challenges within an end-to-end Text Mining lifecycle.

### 2.3 Research Objectives & Core Questions

| Research Question | Objective | ReviewMind Solution |
|---|---|---|
| **RQ1: Minority Class Collapse** | Why does standard Multinomial Naive Bayes with empirical priors fail on neutral hospitality reviews? | Empirically prove that $P(c)$ heavily penalizes minority classes; demonstrate $F_1 = 0.0000$ on the neutral class for Model 1. |
| **RQ2: Hyperparameter Mitigation** | Can hyperparameter tuning of priors and smoothing restore sensitivity to neutral reviews without destroying overall accuracy? | Implement uniform priors (`fit_prior=False`) and Lidstone smoothing ($\alpha = 0.1$) in Model 3, lifting Neutral $F_1$ to $0.3883$ and Macro $F_1$ to $0.6762$. |
| **RQ3: Domain Transfer & Out-of-Vocabulary** | How does an MNB model behave when transferred zero-shot to an external, unseen hotel domain (eco-hotel)? | Measure prediction distributions and prediction confidence on Dataset B, revealing marked confidence degradation ($57.64\%$ below $0.60$). |
| **RQ4: Ground-Truth Integrity** | How should performance be validated on an unannotated external corpus without confirmation bias? | Enforce a strict boundary between model inference and ground truth; design an independent double-blind human annotation protocol. |

### 2.4 Evolution: Academic Mini-Project vs. Full-Stack ReviewMind

```
+---------------------------------------------------------------------------------+
|                         PHASE 1: ACADEMIC MINI-PROJECT                          |
|  - Standalone exploration and preparation scripts (explore_data.py, prepare_data.py)
|  - Baseline internal training (train_model.py, stratified 80/20 split)           |
|  - Empirical prior collapse observed on neutral reviews (F1 = 0.0000)           |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                      PHASE 2: EXPERIMENTATION & RESEARCH                        |
|  - Hyperparameter benchmark (compare_models.py): Model 1 vs Model 2 vs Model 3  |
|  - Discovery of champion configuration (alpha=0.1, fit_prior=False, Macro F1=0.6762)
|  - Production retraining on 100% of Dataset A (train_final_model.py)            |
|  - External domain inference on Dataset B (explore_external_data.py, predict_external.py)
|  - Formal double-blind annotation protocol design (prepare_annotations.py)     |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                       PHASE 3: OPEN-SOURCE TRANSFORMATION                       |
|  - Official MIT License and governance files (CONTRIBUTING, CODE_OF_CONDUCT)    |
|  - Academic citation metadata (CITATION.cff) & security advisory (SECURITY.md)  |
|  - Automated test suite (test_data_validation.py, test_pipeline.py)             |
|  - Strict Git exclusion rules (.gitignore) protecting datasets and models       |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                 PHASE 4: FULL-STACK REST API & WEB DASHBOARD                    |
|  - Flask application factory backend (backend/app.py) with CORS & error handlers |
|  - 7 REST API endpoints across modular blueprints (health, predict, datasets, etc.)
|  - Thread-safe singleton service layer (model_service, dataset_service, evaluation)
|  - Integration test suite (tests/test_backend.py) bringing total tests to 19/19 |
|  - Dark Intelligence responsive interface (web/index.html, web/js/api.js, app.js)|
|  - Complete elimination of all client-side fallback/fake prediction heuristics   |
+---------------------------------------------------------------------------------+
```

---

## 3. Repository Inventory

The repository is located at `.` and structured as follows:

```text
ReviewMind/
├── .env.example                        # Template environment variables
├── .gitignore                          # Excludes datasets, models, caches, and secrets
├── CITATION.cff                        # Citation File Format (v1.2.0)
├── CODE_OF_CONDUCT.md                  # Contributor Covenant v2.1
├── CONTRIBUTING.md                     # Open-source contribution guidelines
├── index.html                          # Root redirect to web/index.html
├── LICENSE                             # MIT License (c) 2026 Benyakoub Mohammed Ryad
├── README.md                           # Public framework documentation & guide
├── requirements.txt                    # Root Python dependencies (pinned minimum versions)
├── SECURITY.md                         # Security policy & joblib deserialization warning
├── tailwind.config.js                  # Tailwind design system configuration
├── .github/                            # GitHub community health files
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── backend/                            # Flask REST API Service Layer
│   ├── __init__.py                     # Package initializer
│   ├── app.py                          # Application factory, static serving & error handlers
│   ├── config.py                       # Environment & filesystem path configurations
│   ├── README.md                       # REST API architecture & endpoint documentation
│   ├── requirements.txt                # Backend dependencies (Flask, Flask-CORS)
│   ├── routes/                         # Modular route blueprints
│   │   ├── __init__.py                 # API blueprint aggregation
│   │   ├── datasets.py                 # GET /api/datasets, GET /api/dashboard
│   │   ├── evaluation.py               # GET /api/evaluation
│   │   ├── health.py                   # GET /api/health
│   │   ├── predictions.py              # POST /api/predict
│   │   └── reviews.py                  # GET /api/reviews, GET /api/reviews/<id>
│   └── services/                       # Business logic singletons
│       ├── __init__.py                 # Services initializer
│       ├── dataset_service.py          # Dataset B in-memory cache, pagination, search
│       ├── evaluation_service.py       # Metrics, confusion matrix, consensus status
│       └── model_service.py            # Singleton model loader & salient feature extractor
├── data/                               # Local data files (excluded from Git)
│   ├── dataset-CalheirosMoroRita-2017.csv # Dataset B: Eco-hotel raw reviews (401 records)
│   ├── reviews_prepared.csv            # Dataset A: Cleaned TripAdvisor reviews (20,491 records)
│   ├── tripadvisor_hotel_reviews.csv   # Dataset A: Raw TripAdvisor reviews (20,491 records)
│   └── external/                       # Human annotation workspace
│       ├── ANNOTATION_GUIDELINES.md    # 102-line standardized French annotation protocol
│       ├── annotation_consensus.csv    # Final consensus target template (399 rows)
│       ├── annotator_A.csv             # Annotator A template (399 rows)
│       ├── annotator_B.csv             # Annotator B template (399 rows)
│       └── eco_hotel_annotations.csv    # Legacy empty placeholder (0 bytes)
├── models/                             # Serialized scikit-learn pipelines (excluded from Git)
│   ├── reviewmind_internal_model.joblib # 80% split baseline model (alpha=1.0, fit_prior=True)
│   └── reviewmind_final_model.joblib    # 100% full dataset production model (alpha=0.1, fit_prior=False)
├── reports/                            # Generated scientific evaluation artifacts
│   ├── external_dataset_exploration.json # Structural profile of Dataset B
│   ├── external_prediction_summary.json  # Confidence & distribution stats for Dataset B
│   ├── external_predictions.csv        # Dataset B inference outputs (excluded from Git)
│   ├── external_sentiment_distribution.png # Brand-styled distribution visualization
│   ├── final_model_metadata.json       # Production model training metadata
│   ├── internal_confusion_matrix.png   # Baseline model confusion matrix plot
│   ├── internal_metrics.json           # Baseline model test set metrics (JSON)
│   ├── model_comparison.csv            # Tabular 3-model benchmark summary
│   ├── model_comparison.json           # Detailed 3-model evaluation metrics & matrices
│   └── model_comparison.png            # Visual bar chart comparison of models
├── src/                                # Core data science & research scripts
│   ├── compare_models.py               # 3-configuration hyperparameter comparison benchmark
│   ├── explore_data.py                 # Dataset A initial descriptive exploration
│   ├── explore_external_data.py        # Dataset B structural & schema exploration
│   ├── predict_external.py             # Inference pipeline execution on Dataset B
│   ├── prepare_annotations.py          # Deduplication & annotation template generation
│   ├── prepare_data.py                 # Dataset A cleaning & sentiment mapping
│   ├── train_final_model.py            # Production fitting on 100% of Dataset A
│   └── train_model.py                  # Stratified 80/20 baseline training & evaluation
├── tests/                              # Automated test suite (Python unittest)
│   ├── __init__.py                     # Tests package initializer
│   ├── test_backend.py                 # 12 API endpoint & integration tests
│   ├── test_data_validation.py         # 4 data validation & sentiment mapping tests
│   └── test_pipeline.py                # 3 pipeline fitting, probability & inference tests
└── web/                                # Frontend client interface
    ├── app.js                          # UI orchestration & API consumer logic
    ├── index.html                      # Single-page Dark Intelligence dashboard
    ├── js/
    │   └── api.js                      # Centralized API service layer with timeout handling
    └── styles/
        ├── theme.css                   # Custom CSS variables & component rules
        └── tokens.json                 # Machine-readable design tokens
```

---

## 4. Datasets — Complete Inventory & Audit

### 4.1 Dataset A — TripAdvisor Hotel Reviews

- **Official Name**: TripAdvisor Hotel Reviews Dataset
- **Local File Paths**:
  - Raw: [`data/tripadvisor_hotel_reviews.csv`](../data/tripadvisor_hotel_reviews.csv)
  - Prepared: [`data/reviews_prepared.csv`](../data/reviews_prepared.csv)
- **Primary Source**: Widely utilized open-access benchmark corpus of crawled TripAdvisor reviews, originally compiled by Alam et al. / Kaggle repository.
- **Dataset Role**: Supervised training corpus, internal validation, and hyperparameter tuning benchmark.
- **Record Count**: $20\,491$ rows (verified in raw and prepared files).
- **Available Columns**:
  1. `Review` (Data type: `object` / string): Full natural language customer review.
  2. `Rating` (Data type: `int64`): Integer star rating ranging from $1$ to $5$.
  3. `Sentiment` (Added in prepared dataset; Data type: `object` / categorical string): Derived class label (`negative`, `neutral`, `positive`).
- **Data Quality Audit**:
  - Null / Missing values: $0$ in all columns.
  - Empty or whitespace-only strings: $0$.
  - Exact duplicate rows: $0$.
  - Duplicate review texts: $0$.
- **Label Derivation Heuristic**:
  Customer rating numbers were mapped to categorical sentiment classes using the piecewise function implemented in [`src/prepare_data.py`](../src/prepare_data.py#L106-L113):

$$\text{Sentiment}(r) = \begin{cases} \text{negative}, & \text{if } r \in \{1, 2\} \\ \text{neutral}, & \text{if } r = 3 \\ \text{positive}, & \text{if } r \in \{4, 5\} \end{cases}$$

- **Class Distribution (100% of Dataset A)**:
  - `positive`: $15\,093$ reviews ($73.66\%$)
  - `negative`: $3\,214$ reviews ($15.68\%$)
  - `neutral`: $2\,184$ reviews ($10.66\%$)
  - **Total**: $20\,491$ reviews ($100.00\%$)
  - **Imbalance Ratio**: Positive reviews outnumber neutral reviews by nearly $7:1$, and negative reviews by $4.7:1$.

### 4.2 Dataset B — External Eco-hotel Reviews

- **Official Name**: Eco-hotel Reviews Dataset
- **Local File Path**: [`data/dataset-CalheirosMoroRita-2017.csv`](../data/dataset-CalheirosMoroRita-2017.csv)
- **Primary Citation**: Calheiros, A. C., Moro, S., & Rita, P. (2017). *"Hospitality research using big data and text mining: An application to eco-friendly hotels"*. *International Journal of Hospitality Management*, 60, 28–35.
- **Dataset Role**: External domain generalization test; evaluates model behavior on an independent hospitality domain.
- **Domain**: Boutique luxury eco-resort (*Areias do Seixo*, Torres Vedras, Portugal).
- **Physical Format & Parsing**: Single text column, tab-separated (`sep='\t'`), encoded in `latin1`.
- **Record Count**:
  - Raw rows: $401$ reviews.
  - Exact duplicates detected: $2$ reviews (rows 109 and 110 are identical duplicates of earlier reviews).
  - Unique reviews after deduplication: $399$ reviews (verified by [`src/explore_external_data.py`](../src/explore_external_data.py) and [`src/prepare_annotations.py`](../src/prepare_annotations.py)).
- **Available Columns**: Single column `Review` (string).
- **Quality Audit**:
  - Missing values: $0$.
  - Empty strings: $0$.
  - Target labels / Numerical ratings: **None**. The dataset is entirely unsupervised.
- **Inference Results (Model 3 Predictions on 399 Unique Reviews)**:
  - Predicted Positive: $335$ reviews ($83.96\%$), average confidence: $60.03\%$.
  - Predicted Neutral: $33$ reviews ($8.27\%$), average confidence: $42.37\%$.
  - Predicted Negative: $31$ reviews ($7.77\%$), average confidence: $46.03\%$.
  - Mean confidence across all 399 reviews: $57.48\%$.
  - Low-confidence predictions ($<60\%$): $230$ reviews ($57.64\%$).
- **Methodological Status**: Because Dataset B has no ratings or ground-truth annotations, **no accuracy, precision, recall, or $F_1$ score can be computed mathematically**.

### 4.3 Human Annotation Dataset & Workspace

- **Location**: `data/external/`
- **Protocol Document**: [`data/external/ANNOTATION_GUIDELINES.md`](../data/external/ANNOTATION_GUIDELINES.md) (102 lines).
- **File Inventory**:
  1. [`annotator_A.csv`](../data/external/annotator_A.csv): $399$ rows. Blank template with columns `Original_Index`, `Review`, `Human_Sentiment`, `Annotation_Notes`. Current filled count: $0$.
  2. [`annotator_B.csv`](../data/external/annotator_B.csv): $399$ rows. Blank template with identical columns. Current filled count: $0$.
  3. [`annotation_consensus.csv`](../data/external/annotation_consensus.csv): $399$ rows. Consensus adjudication target. Current filled count: $0$.
  4. [`eco_hotel_annotations.csv`](../data/external/eco_hotel_annotations.csv): $0$-byte empty file (legacy artifact).
- **Annotation Status**: **PENDING**. Work forms have been generated, aligned, and verified; human evaluation sessions have not yet been completed.

---

## 5. Complete Text Mining Pipeline

```
+-------------------------------------------------------------------------------------------------+
|                                 REVIEWMIND TEXT MINING PIPELINE                                 |
+-------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 1. DATA ACQUISITION & INTEGRITY VALIDATION (src/explore_data.py, src/explore_external_data.py)|
  |    - Load raw CSV/TSV: tripadvisor_hotel_reviews.csv (20,491 rows) & Calheiros-2017 (401 rows)|
  |    - Character encoding handling: UTF-8 for Dataset A, Latin-1 with tab separator for Dataset B|
  |    - Structural verification: Required columns ('Review', 'Rating'), non-null assertion       |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 2. DATA CLEANING & SENTIMENT LABEL HEURISTICS (src/prepare_data.py)                         |
  |    - Whitespace trimming & removal of null/empty text strings                                |
  |    - Rating discretization: {1, 2} -> negative, {3} -> neutral, {4, 5} -> positive           |
  |    - Output persistence: data/reviews_prepared.csv (20,491 rows, 0 nulls, 3 classes)         |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 3. STRATIFIED TRAIN / TEST PARTITIONING (src/train_model.py, src/compare_models.py)          |
  |    - 80% Train (16,392 reviews) / 20% Test (4,099 reviews)                                  |
  |    - Hyperparameters: test_size=0.20, random_state=42, stratify=y                            |
  |    - Strict isolation: vectorizer fitted ONLY on train split to prevent data leakage         |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 4. HIGH-DIMENSIONAL TEXT VECTORIZATION (TfidfVectorizer)                                    |
  |    - N-gram expansion: unigrams and bigrams (ngram_range=(1, 2))                             |
  |    - Vocabulary limitation: max_features=20000 top n-grams by corpus term frequency         |
  |    - Sublinear TF: False (standard linear term frequency)                                   |
  |    - Tokenization pattern: (?u)\b\w\w+\b (words >= 2 alphanumeric characters)                |
  |    - Stop words: None (preserved in vocabulary to support negation bigrams)                  |
  |    - Normalization: L2 Euclidean document vector normalization                               |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 5. PROBABILISTIC CLASSIFICATION & HYPERPARAMETER TUNING (MultinomialNB)                     |
  |    - Model 1 (Baseline): alpha=1.0, fit_prior=True (Empirical priors)                        |
  |    - Model 2 (Uniform Prior): alpha=1.0, fit_prior=False (Uniform priors)                    |
  |    - Model 3 (Champion): alpha=0.1, fit_prior=False (Lidstone smoothing + Uniform priors)    |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 6. MULTI-METRIC PERFORMANCE EVALUATION (reports/model_comparison.json)                      |
  |    - Metrics: Accuracy, Macro F1, Weighted F1, Per-Class Precision/Recall/F1                |
  |    - Confusion Matrix analysis: diagonal vs off-diagonal misclassification                   |
  |    - Key outcome: Model 3 achieves Macro F1 = 0.6762 and rescues neutral F1 to 0.3883       |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 7. PRODUCTION MODEL FITTING (src/train_final_model.py)                                       |
  |    - Fit Model 3 pipeline on 100% of Dataset A (20,491 reviews)                              |
  |    - Serialization: models/reviewmind_final_model.joblib (joblib compression)                |
  |    - Metadata generation: reports/final_model_metadata.json                                  |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 8. EXTERNAL INFERENCE & CONFIDENCE AUDIT (src/predict_external.py)                          |
  |    - Predict 399 unique reviews from Dataset B; output posterior probabilities and confidence |
  |    - Output persistence: reports/external_predictions.csv, summary.json, distribution.png     |
  |    - Detection of domain shift: 57.64% of external reviews yield confidence < 60%            |
  +---------------------------------------------------------------------------------------------+
                                                 |
                                                 v
  +---------------------------------------------------------------------------------------------+
  | 9. HUMAN ANNOTATION & CONSENSUS PROTOCOL (src/prepare_annotations.py)                       |
  |    - Independent double-blind templates: data/external/annotator_A.csv, annotator_B.csv      |
  |    - Formal disagreement adjudication targeting data/external/annotation_consensus.csv       |
  |    - Supervised external metrics marked strictly as PENDING until consensus is established   |
  +---------------------------------------------------------------------------------------------+
```

### Detailed Pipeline Stage Analysis

#### Stage A: Data Acquisition
- **Mechanism**: Implemented in [`src/explore_data.py`](../src/explore_data.py) and [`src/explore_external_data.py`](../src/explore_external_data.py).
- **Format**: Text files are ingested via `pd.read_csv()`.
- **Specific Strategy for Dataset B**: Because reviews in Dataset B contain unescaped commas and line breaks, standard comma-separated parsing fails. The script uses `sep='\t'` and `encoding='latin1'`, cleanly isolating the 401 review strings.

#### Stage B: Cleaning & Preprocessing
- **Validation**: Enforced by `validate_columns()` and `validate_ratings()` in [`src/prepare_data.py`](../src/prepare_data.py).
- **Text Cleaning**: Trimming leading/trailing whitespace (`str.strip()`). Empty strings and nulls are dropped.
- **Audited Preprocessing Fact**: **No stemming (Porter/Snowball) and no lemmatization (WordNet/spaCy) are applied**. Words retain their inflected surface forms.
- **Stop Words**: Stop words are **not removed** (`stop_words=None`). Preserving stop words (e.g., *not*, *no*, *never*, *very*) is vital because they form critical negative bigrams (*"not clean"*, *"never again"*).

#### Stage C: Text Representation (TF-IDF Vectorization)
- **Class**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Parameters**:
  - `ngram_range=(1, 2)`: Captures single tokens (unigrams) and contiguous word pairs (bigrams).
  - `max_features=20000`: Restricts the feature matrix to the top $20\,000$ terms ordered by term frequency across the corpus.
  - `sublinear_tf=False`: Linear term frequency ($tf(t, d) = f_{t,d}$).
  - `norm='l2'`: Normalizes each document vector $\mathbf{v}$ so that $\|\mathbf{v}\|_2 = 1$.
  - `smooth_idf=True`: Adds $1$ to document frequencies to prevent division by zero.
  - `token_pattern=r'(?u)\b\w\w+\b'`: Regular expression selecting words with 2 or more alphanumeric characters.

#### Stage D: Classification Model (Multinomial Naive Bayes)
- **Class**: `sklearn.naive_bayes.MultinomialNB`
- **Parameters**:
  - `alpha=0.1`: Lidstone additive smoothing parameter.
  - `fit_prior=False`: Ignores empirical class distributions; assigns uniform prior probabilities $P(c) = 1/3$ across negative, neutral, and positive classes.

#### Stage E: Pipeline Integration
The vectorizer and classifier are encapsulated in a single `sklearn.pipeline.Pipeline`:
```python
Pipeline(steps=[
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
    ('nb', MultinomialNB(alpha=0.1, fit_prior=False))
])
```
This guarantees that during training, `fit()` fits both vocabulary and Bayes parameters together, while during inference, `predict()` and `predict_proba()` automatically apply the learned vectorizer transformations before scoring.

---

## 6. Mathematical & Theoretical Foundations

### 6.1 TF-IDF Representation

The Term Frequency-Inverse Document Frequency (TF-IDF) weight transforms textual documents into a sparse vector space $\mathbb{R}^{|V|}$.

#### Term Frequency (TF)
For term $t$ in review $d$:
$$\text{TF}(t, d) = f_{t,d}$$
where $f_{t,d}$ is the raw count of term $t$ in document $d$.

#### Inverse Document Frequency (IDF)
With scikit-learn's `smooth_idf=True`:
$$\text{IDF}(t) = \ln \left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
where $N$ is the total number of reviews in the training split ($16\,392$), and $\text{DF}(t)$ is the number of reviews containing term $t$.

#### Raw TF-IDF Weight
$$\text{tfidf}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

#### Euclidean ($L_2$) Normalization
To prevent long reviews from dominating due to high word counts, document vectors are normalized onto the unit hypersphere:
$$v_{\text{norm}}(t, d) = \frac{\text{tfidf}(t, d)}{\sqrt{\sum_{t' \in d} \left( \text{tfidf}(t', d) \right)^2}}$$

### 6.2 Multinomial Naive Bayes Formulation

Multinomial Naive Bayes is a generative probabilistic model. Given a document represented by feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ corresponding to terms $(w_1, \dots, w_n)$, the posterior probability of class $c \in \{\text{negative}, \text{neutral}, \text{positive}\}$ is given by Bayes' theorem:

$$P(c \mid \mathbf{x}) = \frac{P(c) \, P(\mathbf{x} \mid c)}{P(\mathbf{x})} \propto P(c) \prod_{i=1}^n P(w_i \mid c)^{x_i}$$

#### Log-Space Classification Rule
To prevent numerical underflow from multiplying hundreds of small probabilities, the decision rule is computed in log space:

$$\hat{c} = \arg\max_{c \in C} \left[ \ln P(c) + \sum_{i=1}^n x_i \ln P(w_i \mid c) \right]$$

### 6.3 Prior Probability Dynamics & The Collapse Mechanism

#### Empirical Prior (`fit_prior=True`, Baseline Model 1)
Class priors reflect training set frequencies:
$$P(c) = \frac{N_c}{N}$$
For Dataset A:
- $P(\text{positive}) = \frac{15\,093}{20\,491} \approx 0.7366 \implies \ln P(\text{positive}) \approx -0.3057$
- $P(\text{negative}) = \frac{3\,214}{20\,491} \approx 0.1568 \implies \ln P(\text{negative}) \approx -1.8528$
- $P(\text{neutral}) = \frac{2\,184}{20\,491} \approx 0.1066 \implies \ln P(\text{neutral}) \approx -2.2387$

**The Mathematical Penalty**: The prior log-penalty for neutral sentiment is $\Delta = -2.2387 - (-0.3057) = -1.9330$. For a document to be classified as neutral, its feature log-likelihood $\sum x_i \ln P(w_i \mid \text{neutral})$ must overcome this substantial barrier. Because neutral reviews share extensive vocabulary with both positive (*"nice"*, *"clean"*) and negative (*"small"*, *"noisy"*) reviews, their likelihood rarely overcomes the empirical prior deficit. Consequently, **the model predicts zero neutral reviews**.

#### Uniform Prior (`fit_prior=False`, Models 2 & 3)
Setting `fit_prior=False` imposes equal prior probabilities:
$$P(\text{negative}) = P(\text{neutral}) = P(\text{positive}) = \frac{1}{3} \implies \ln P(c) = -\ln(3) \approx -1.0986$$
Because $\ln P(c)$ is identical for all classes, it cancels out of the decision rule:
$$\hat{c} = \arg\max_{c \in C} \sum_{i=1}^n x_i \ln P(w_i \mid c)$$
Classification depends **strictly on the textual evidence** contained in the review.

### 6.4 Smoothing Parameter ($\alpha$) Mechanics

Conditional word probabilities are estimated with additive Lidstone smoothing:
$$\hat{P}(w \mid c) = \frac{N_{c,w} + \alpha}{N_c + \alpha |V|}$$
where:
- $N_{c,w} = \sum_{d \in D_c} x_{d,w}$ is the weighted sum of TF-IDF feature weights for term $w$ in class $c$.
- $N_c = \sum_{w' \in V} N_{c,w'}$ is the total feature weight across all vocabulary words in class $c$.
- $|V| = 20\,000$ is the vocabulary dimension.
- $\alpha$ is the smoothing hyperparameter.

#### Impact of Smoothing Value
- When $\alpha = 1.0$ (Laplace smoothing), the denominator absorbs $\alpha |V| = 20\,000$. This flattens the distribution and pulls probability mass toward background frequencies, dampening subtle differences between classes.
- When $\alpha = 0.1$ (Lidstone smoothing), the denominator penalty decreases to $\alpha |V| = 2\,000$. Conditional probabilities become sharper and more discriminative. Specific n-grams that appear predominantly in neutral reviews (*"average hotel"*, *"okay stay"*, *"three star"*) achieve higher relative log-likelihoods, lifting neutral recall from $8.47\%$ to $48.74\%$.

### 6.5 Evaluation Metrics Formulation

For class $c$ with true positives $TP_c$, false positives $FP_c$, and false negatives $FN_c$:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

$$F_{1,c} = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro } F_1 = \frac{1}{|C|} \sum_{c \in C} F_{1,c} = \frac{F_{1,\text{neg}} + F_{1,\text{neu}} + F_{1,\text{pos}}}{3}$$

$$\text{Weighted } F_1 = \sum_{c \in C} \left( \frac{N_c}{N} \right) F_{1,c}$$

$$\text{Accuracy} = \frac{\sum_{c \in C} TP_c}{N_{\text{test}}}$$

**Why Macro $F_1$ is the Scientific Selection Criterion**: Accuracy and Weighted $F_1$ are dominated by the majority class ($73.66\%$ positive). A naive model predicting $100\%$ positive achieves $\approx 73.7\%$ accuracy while being completely useless. Macro $F_1$ gives equal weight ($1/3$) to all classes, strictly penalizing models that fail on minority classes.

---

## 7. Machine Learning Model — Technical Deep Dive

### 7.1 Verified Model Artifacts

| Property | Internal Baseline Model | Production Final Model |
|---|---|---|
| **Serialized File** | [`models/reviewmind_internal_model.joblib`](../models/reviewmind_internal_model.joblib) | [`models/reviewmind_final_model.joblib`](../models/reviewmind_final_model.joblib) |
| **Training Script** | [`src/train_model.py`](../src/train_model.py) | [`src/train_final_model.py`](../src/train_final_model.py) |
| **Training Records** | $16\,392$ reviews ($80\%$ of Dataset A) | $20\,491$ reviews ($100\%$ of Dataset A) |
| **Test Records** | $4\,099$ reviews ($20\%$ holdout) | Unseen external data (Dataset B) |
| **Pipeline Step 1** | `TfidfVectorizer(max_features=20000, ngram_range=(1, 2))` | `TfidfVectorizer(max_features=20000, ngram_range=(1, 2))` |
| **Pipeline Step 2** | `MultinomialNB(alpha=1.0, fit_prior=True)` | `MultinomialNB(alpha=0.1, fit_prior=False)` |
| **Classes** | `['negative', 'neutral', 'positive']` | `['negative', 'neutral', 'positive']` |
| **Vocabulary Size** | $20\,000$ features | $20\,000$ features |
| **Training Time** | $\approx 10.5$ seconds | $13.836$ seconds (verified in metadata) |
| **Joblib File Size** | $1.98$ MB | $2.03$ MB |

### 7.2 Probability Estimation & Confidence Score

In `MultinomialNB`, posterior probabilities are obtained by applying the Softmax function to the unnormalized joint log-likelihoods:

$$\ln P(c, \mathbf{x}) = \ln P(c) + \sum_{i=1}^n x_i \ln P(w_i \mid c)$$

$$P(c \mid \mathbf{x}) = \frac{\exp\left( \ln P(c, \mathbf{x}) \right)}{\sum_{c' \in C} \exp\left( \ln P(c', \mathbf{x}) \right)}$$

In ReviewMind, the **Confidence Score** is formally defined as the maximum posterior probability:
$$\text{Confidence}(\mathbf{x}) = \max_{c \in C} P(c \mid \mathbf{x})$$
- For a 3-class problem, the random guessing baseline is $1/3 \approx 33.33\%$.
- A confidence score between $34\%$ and $50\%$ indicates intense model hesitation across two or three classes.
- A confidence score above $80\%$ indicates strong certainty driven by unambiguous vocabulary.

### 7.3 Salient Feature Extraction Algorithm

Implemented in [`backend/services/model_service.py`](../backend/services/model_service.py#L89-L119):
1. The raw text review is transformed into a TF-IDF vector via `model.named_steps['tfidf'].transform([text])`.
2. Active non-zero indices are extracted using `vector.nonzero()[1]`.
3. Term strings are resolved via `get_feature_names_out()[idx]`.
4. Extracted n-grams are ranked in descending order of their document TF-IDF weight.
5. The top $k$ (default 8) terms are returned in the API response to provide explainability to the user.

---

## 8. Experiments & Verified Results

### 8.1 Benchmark Comparison Table (Dataset A Test Set: 4,099 Reviews)

The following metrics were computed on the exact same stratified holdout ($16\,392$ train / $4\,099$ test, `random_state=42`) and are verified from [`reports/model_comparison.json`](../reports/model_comparison.json):

| Metric | Model 1 (Baseline) | Model 2 (Uniform Prior) | Model 3 (Champion / Production) |
|---|:---:|:---:|:---:|
| **Smoothing ($\alpha$)** | $1.0$ (Laplace) | $1.0$ (Laplace) | **$0.1$ (Lidstone)** |
| **Class Prior (`fit_prior`)** | `True` (Empirical) | `False` (Uniform) | **`False` (Uniform)** |
| **Accuracy** | $81.97\%$ | **$85.17\%$** | $80.12\%$ |
| **Macro $F_1$-Score** | $0.5249$ | $0.6153$ | **$0.6762$** |
| **Weighted $F_1$-Score** | $0.7655$ | **$0.8185$** | $0.8174$ |
| **Negative Precision** | **$0.8937$** | $0.7126$ | $0.6804$ |
| **Negative Recall** | $0.5490$ | **$0.8445$** | $0.8243$ |
| **Negative $F_1$-Score** | $0.6802$ | **$0.7730$** | $0.7454$ |
| **Neutral Precision** | $0.0000$ | **$0.5781$** | $0.3227$ |
| **Neutral Recall** | $0.0000$ | $0.0847$ | **$0.4874$** |
| **Neutral $F_1$-Score** | **$0.0000$ (Collapsed)** | $0.1477$ | **$0.3883$ (Recovered)** |
| **Positive Precision** | $0.8118$ | $0.8894$ | **$0.9553$** |
| **Positive Recall** | **$0.9960$** | $0.9642$ | $0.8417$ |
| **Positive $F_1$-Score** | $0.8945$ | **$0.9253$** | $0.8949$ |
| **Neutral Predictions Count** | **$0$** | $64$ | **$660$** |

### 8.2 Confusion Matrices Breakdown

#### Model 1 (Baseline: $\alpha=1.0$, `fit_prior=True`)
*Support: Negative=643, Neutral=437, Positive=3019*
```text
                  Predicted Negative   Predicted Neutral   Predicted Positive    Total True
Negative (True)          353                   0                  290               643
Neutral (True)            30                   0                  407               437
Positive (True)           12                   0                 3007              3019
Total Predicted          395                   0                 3704              4099
```
*Observation*: Out of 437 true neutral reviews, **0 were identified**. $407$ were misclassified as positive and $30$ as negative.

#### Model 2 (Uniform Prior: $\alpha=1.0$, `fit_prior=False`)
```text
                  Predicted Negative   Predicted Neutral   Predicted Positive    Total True
Negative (True)          543                  12                   88               643
Neutral (True)           126                  37                  274               437
Positive (True)           93                  15                 2911              3019
Total Predicted          762                  64                 3273              4099
```
*Observation*: Neutral predictions increase to $64$, and negative recall surges from $54.9\%$ to $84.5\%$. However, neutral recall remains poor ($8.47\%$).

#### Model 3 (Champion: $\alpha=0.1$, `fit_prior=False`)
```text
                  Predicted Negative   Predicted Neutral   Predicted Positive    Total True
Negative (True)          530                  99                   14               643
Neutral (True)           119                 213                  105               437
Positive (True)          130                 348                 2541              3019
Total Predicted          779                 660                 2660              4099
```
*Observation*: True neutrals correctly classified rise to **$213$** ($48.74\%$ recall). $348$ positive reviews and $99$ negative reviews are predicted as neutral, reflecting semantic overlap in reviews describing mixed experiences.

---

## 9. External Evaluation & Domain Shift Analysis

### 9.1 Dataset B Inferences Summary (399 Unique Reviews)

Evaluated with the final production model ([`reports/external_prediction_summary.json`](../reports/external_prediction_summary.json)):

```text
Class Inferred       Count       Percentage       Average Confidence
Positive              335          83.96%              60.03%
Neutral                33           8.27%              42.37%
Negative               31           7.77%              46.03%
Total                 399         100.00%              57.48% (Overall Mean)
```

### 9.2 Confidence Distribution & Domain Shift Evidence

- **Median Confidence**: $56.66\%$
- **Minimum Confidence**: $34.64\%$
- **Maximum Confidence**: $92.58\%$
- **Low Confidence ($<60\%$)**: **$230$ reviews ($57.64\%$)**

#### Root Causes of Low Confidence on Dataset B
1. **Domain Shift (Vocabulary Divergence)**: TripAdvisor reviews frequently discuss standard amenities (*"front desk"*, *"wifi"*, *"metro station"*, *"check-in"*). In contrast, reviews for *Areias do Seixo* feature eco-resort themes (*"bio-pool"*, *"sustainable architecture"*, *"duck feathers"*, *"fire pits"*, *"herbal tea"*).
2. **Lyrical and Poetic Language**: Many reviews in Dataset B are subjective essays praising nature and tranquility with unconventional vocabulary that yields low TF-IDF scores against the TripAdvisor dictionary.
3. **Misleading Predictions on Subtle Praise**: For example, Review `#16` (*"We had a truly memorable and delightful stay at Areias do Seixo... After some difficulty in finding it..."*) was predicted **negative** with only $37.66\%$ confidence. The model flagged *"difficulty"* and missed the overarching positive sentiment.

### 9.3 Double-Blind Annotation Protocol

To convert Dataset B into a formal benchmark:
- **Phase 1**: Annotator A and Annotator B independently annotate `annotator_A.csv` and `annotator_B.csv` without consulting each other or the model predictions.
- **Phase 2**: Inter-annotator reliability is measured using Cohen's Kappa:
  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
  where $p_o$ is the observed relative agreement and $p_e$ is the hypothetical probability of chance agreement.
- **Phase 3**: Discrepancies are reviewed collaboratively in a reconciliation session adhering to [`ANNOTATION_GUIDELINES.md`](../data/external/ANNOTATION_GUIDELINES.md).
- **Phase 4**: Adjudicated labels are committed to `annotation_consensus.csv`. Only then will external $F_1$, precision, and recall be calculated.

---

## 10. Full-Stack Application Architecture

### 10.1 System Architecture Diagram

```mermaid
flowchart TD
    subgraph Client ["Frontend Client (web/)"]
        UI["Web Dashboard (index.html)"]
        APP["App Logic (app.js)"]
        API_SVC["Centralized API Service (js/api.js)"]
        THEME["Dark Intelligence Tokens (theme.css)"]
    end

    subgraph Server ["Flask REST Backend (backend/)"]
        APP_FACTORY["Flask Application Factory (app.py)"]
        CORS_LAYER["Flask-CORS Middleware"]
        ROUTES["Route Blueprints (/api/*)"]
        
        subgraph Blueprints ["API Endpoints"]
            R_HEALTH["health.py (/api/health)"]
            R_PRED["predictions.py (/api/predict)"]
            R_DATA["datasets.py (/api/datasets, /api/dashboard)"]
            R_REV["reviews.py (/api/reviews, /api/reviews/<id>)"]
            R_EVAL["evaluation.py (/api/evaluation)"]
        end
        
        subgraph Services ["Singleton Service Layer"]
            S_MODEL["ModelService (model_service.py)"]
            S_DATA["DatasetService (dataset_service.py)"]
            S_EVAL["EvaluationService (evaluation_service.py)"]
        end
    end

    subgraph Storage ["Filesystem Artifacts"]
        M_FINAL["models/reviewmind_final_model.joblib"]
        REP_EXP["reports/external_predictions.csv"]
        REP_JSON["reports/model_comparison.json"]
        DAT_CONS["data/external/annotation_consensus.csv"]
    end

    UI --> APP
    APP --> API_SVC
    API_SVC -- "HTTP JSON Requests" --> APP_FACTORY
    APP_FACTORY --> CORS_LAYER
    CORS_LAYER --> ROUTES
    ROUTES --> Blueprints
    
    R_HEALTH --> S_MODEL
    R_PRED --> S_MODEL
    R_DATA --> S_DATA
    R_DATA --> S_EVAL
    R_REV --> S_DATA
    R_EVAL --> S_EVAL
    
    S_MODEL --> M_FINAL
    S_DATA --> REP_EXP
    S_EVAL --> REP_JSON
    S_EVAL --> DAT_CONS

    S_MODEL -- "Predictions & Salient Features" --> R_PRED
    S_DATA -- "Paginated Reviews & Filters" --> R_REV
    S_EVAL -- "Benchmarks & Consensus Status" --> R_EVAL
    
    ROUTES -- "Normalized JSON Responses" --> API_SVC
    API_SVC --> APP
    APP --> UI
```

### 10.2 Technical Implementation Rules
1. **Single Source of Truth**: All predictions and performance statistics originate strictly from backend Python services.
2. **Zero Simulation**: The frontend has no local keyword weights (`VOCAB_WEIGHTS`), sample arrays (`DATASET_B_SAMPLES`), or heuristic scoring formulas.
3. **Graceful Error Handling**: If the backend is unreachable, the client renders an inline connection error card with a **Retry** button rather than fabricating outputs.

---

## 11. REST API Documentation

### 11.1 Endpoints Specification

#### 1. `GET /api/health`
- **Purpose**: System health check, model readiness, vocabulary size.
- **Response**:
```json
{
  "service": "ReviewMind REST API",
  "status": "healthy",
  "timestamp_utc": "2026-11-01T12:00:00Z",
  "version": "1.0.0",
  "model": {
    "model_loaded": true,
    "model_name": "ReviewMind Final Production Model",
    "classes": ["negative", "neutral", "positive"],
    "vocabulary_size": 20000,
    "alpha": 0.1,
    "fit_prior": false
  }
}
```

#### 2. `POST /api/predict`
- **Purpose**: Live sentiment inference on input review text.
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "review": "The ocean view was breathtaking and staff was attentive, but bathroom ventilation was inadequate."
}
```
- **Validation**:
  - Rejects non-JSON (`400 Bad Request`).
  - Rejects missing `review` field (`400 Bad Request`, `status: missing_field`).
  - Rejects empty or whitespace-only review (`400 Bad Request`, `status: empty_input`).
  - Rejects non-string types (`400 Bad Request`, `status: invalid_type`).
  - Enforces text length limit of $50\,000$ characters (`413 Payload Too Large`).
- **Response**:
```json
{
  "status": "success",
  "review": "The ocean view was breathtaking...",
  "predicted_sentiment": "neutral",
  "confidence": 0.4482,
  "probabilities": {
    "negative": 0.2315,
    "neutral": 0.4482,
    "positive": 0.3203
  },
  "salient_features": [
    {"term": "breathtaking", "weight": 0.3842},
    {"term": "ocean view", "weight": 0.3512},
    {"term": "inadequate", "weight": 0.3105}
  ],
  "validation": {
    "is_valid": true,
    "length_chars": 98,
    "word_count": 14
  }
}
```

#### 3. `GET /api/dashboard`
- **Purpose**: Executive dashboard KPIs, champion model metrics, improvement deltas.
- **Response**: Returns Champion Model 3 metrics (Accuracy: $80.12\%$, Macro F1: $0.6762$, Neutral F1: $0.3883$), baseline deltas ($+0.1513$ Macro F1), and Dataset A/B summaries.

#### 4. `GET /api/datasets`
- **Purpose**: Metadata for Dataset A, Dataset B, and human annotation workspace.

#### 5. `GET /api/reviews`
- **Purpose**: Paginated querying of Dataset B reviews.
- **Query Parameters**:
  - `page` (int, default=1): Page number.
  - `per_page` (int, default=10, max=100): Reviews per page.
  - `sentiment` (string, optional): Filter by `positive`, `neutral`, or `negative`.
  - `q` (string, optional): Case-insensitive keyword search in review text.
- **Response**: Returns paginated slice of reviews with total items, total pages, navigation flags (`has_next`, `has_prev`), and probability breakdowns.

#### 6. `GET /api/reviews/<int:original_index>`
- **Purpose**: Query single review by original index. Returns `404 Not Found` if index does not exist.

#### 7. `GET /api/evaluation`
- **Purpose**: Model benchmark comparisons, confusion matrices, and live consensus progress. Returns `"supervised_metrics_status": "pending"` for external evaluation.

---

## 12. Frontend Design System & Functionality

### 12.1 Brand Tokens & Color Palette (Dark Intelligence)

The design system enforces a strict 3-color palette with high cognitive contrast:

```text
BLACK  : #0B0B0F  (Dominant Surface ~75%)  -> Main background, sidebar, surface cards
BLUE   : #1E6BFF  (Secondary Accent ~15%)   -> Links, interactive states, positive sentiment
RED    : #E0122F  (Complementary Accent ~10%)-> Primary buttons, active nav indicators, negative sentiment
GRAY   : #8E8E93  (Neutral Tone)           -> Metadata, subtitles, neutral sentiment
WHITE  : #FFFFFF  (High-Contrast Text)     -> Primary typography
```

### 12.2 Interactive Dashboard Views (Tabs)
1. **Inference Studio**: Live sentiment classifier. Includes sample review buttons, textarea input, loading spinner during backend call, error banner with retry button, confidence badge, probability tracks, and salient n-gram pills.
2. **Model Benchmarks**: Comparative benchmark table for Models 1, 2, and 3; production model confusion matrix; theoretical breakdown of neutral minority recovery.
3. **Dataset B Explorer**: Real-time review browser with keyword search (350ms debounce), sentiment filtering buttons (`All`, `Positive`, `Neutral`, `Negative`), and pagination controls.
4. **Annotation Consensus**: Interactive walkthrough of the double-blind human annotation protocol and Cohen's $\kappa$ workflow. Displays `STATUS: PENDING CONSENSUS`.
5. **Brand Tokens & Palette**: Visual design system documentation with color swatches and hierarchy guidelines.

---

## 13. Project Development History & Git Timeline

### Verified Git Commit Log

```text
* 470876a - (2026-10-30 19:39:07 +0100) feat: integrate Flask API with ReviewMind frontend (Benyakoub Mohammed Ryad)
* 9396be2 - (2026-10-30 17:59:28 +0100) feat: transform ReviewMind into an open-source NLP framework with governance, tests, and research reports (Benyakoub Mohammed Ryad)
* 9c08407 - (2026-10-29 17:03:45 +0100) Initial commit: dataset exploration and preparation (Benyakoub Mohammed Ryad)
```

### Development Timeline & Milestones

| Date / Timestamp | Milestone | Commits & Files Affected | Technical Description |
|---|---|---|---|
| **Oct 29, 2026**<br>17:03 CET | **Phase 1: Foundation** | `9c08407`<br>`src/explore_data.py`<br>`src/prepare_data.py` | Initial exploration of Dataset A ($20\,491$ TripAdvisor reviews). Implementation of rating validation and sentiment mapping. |
| **Oct 30, 2026**<br>11:00–14:00 | **Phase 2: Modeling & Benchmarking** | Local development scripts | Internal baseline training (`train_model.py`); discovery of neutral class collapse; implementation of 3-model benchmark (`compare_models.py`); discovery of champion configuration. |
| **Oct 30, 2026**<br>14:55 UTC | **Phase 3: Production Training** | `models/reviewmind_final_model.joblib`<br>`reports/final_model_metadata.json` | Fitting Model 3 on 100% of Dataset A ($13.8$ seconds). Model serialization with joblib. |
| **Oct 30, 2026**<br>15:30–17:00 | **Phase 4: External Domain & Annotation** | `src/explore_external_data.py`<br>`src/predict_external.py`<br>`src/prepare_annotations.py` | Parsing Dataset B ($401$ raw / $399$ unique). Model inference ($57.48\%$ avg confidence). Formalization of double-blind annotation guidelines (`ANNOTATION_GUIDELINES.md`). |
| **Oct 30, 2026**<br>17:59 CET | **Phase 5: Open-Source Governance** | `9396be2`<br>`LICENSE`, `README.md`, `CITATION.cff`, `tests/` | Open-source release under MIT License. Addition of governance files, security policy, scientific reports, and automated unit tests. |
| **Oct 30, 2026**<br>19:39 CET | **Phase 6: Full-Stack Integration** | `470876a`<br>`backend/`, `web/`, `tests/test_backend.py` | Implementation of Flask REST API backend (5 blueprints, 3 services). Integration with web frontend. Elimination of client-side fallbacks. Pushed to GitHub. |

---

## 14. Quality Assurance, Automated Tests & Reproducibility

### 14.1 Test Suite Breakdown

Executed via `python -m unittest discover tests -v`:
- **Total Tests**: **19 passed** out of 19 (0 failures, 0 errors) in **0.355 seconds**.
- **Test Modules**:
  1. [`tests/test_data_validation.py`](../tests/test_data_validation.py) (4 tests):
     - `test_sentiment_mapping_standard_ratings`: Validates ratings $1$ to $5$ map correctly.
     - `test_sentiment_mapping_invalid_ratings`: Validates out-of-range ($0, 6, -1$) and NaN ratings return `None`.
     - `test_filter_empty_reviews`: Asserts empty strings, whitespace, and nulls are dropped.
     - `test_required_columns_check`: Asserts validation flags missing columns.
  2. [`tests/test_pipeline.py`](../tests/test_pipeline.py) (3 tests):
     - `test_pipeline_fit_and_predict`: Verifies pipeline fits on synthetic data and outputs valid class strings.
     - `test_pipeline_predict_proba`: Verifies predicted probabilities sum to $1.0$ ($\pm 10^{-5}$) and lie in $[0, 1]$.
     - `test_production_model_if_present`: Loads `reviewmind_final_model.joblib` and tests live inference on a sample review.
  3. [`tests/test_backend.py`](../tests/test_backend.py) (12 tests):
     - `test_health_endpoint`: Checks HTTP 200 and model status.
     - `test_predict_successful_positive`: Submits positive review; verifies class, confidence $>0.5$, and salient features.
     - `test_predict_successful_negative`: Submits negative review; verifies class and negative probability $> positive$.
     - `test_predict_empty_input`: Submits empty and whitespace string; verifies HTTP 400.
     - `test_predict_missing_field`: Submits payload without `review` key; verifies HTTP 400.
     - `test_predict_invalid_data_type`: Submits integer as review; verifies HTTP 400.
     - `test_predict_non_json_request`: Submits plain text; verifies HTTP 400.
     - `test_datasets_metadata_endpoint`: Checks HTTP 200 and metadata structure.
     - `test_dashboard_endpoint`: Checks HTTP 200 and executive KPI values.
     - `test_reviews_pagination_and_filtering`: Checks pagination and sentiment filters.
     - `test_single_review_endpoint`: Checks retrieval of review index 0 and 404 on invalid index.
     - `test_evaluation_pending_status`: Strictly verifies `supervised_metrics_status` is `pending`.

### 14.2 Reproducibility Protocol

```bash
# 1. Clone repository
git clone https://github.com/RYAD-BENYAKOUB/ReviewMind.git
cd ReviewMind

# 2. Set up virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
python -m unittest discover tests

# 5. Reproduce data preparation & model training (requires raw data in data/)
python src/explore_data.py
python src/prepare_data.py
python src/train_model.py
python src/compare_models.py
python src/train_final_model.py
python src/explore_external_data.py
python src/predict_external.py
python src/prepare_annotations.py

# 6. Start the Flask Backend Server
python backend/app.py

# 7. Access Web Dashboard
# In browser, navigate to: http://127.0.0.1:5000/web/index.html
```

---

## 15. Discrepancies, Missing Information & Academic Limitations

### 15.1 Technical Discrepancies Identified During Audit

1. **TF-IDF Sublinear Scaling Documentation Claim vs. Implementation**:
   - *Claim in `README.md` (Line 33)*: "Feature Extraction: High-dimensional text representation with n-gram sublinear TF-IDF vectorization."
   - *Verified Implementation in Code*: `TfidfVectorizer(ngram_range=(1, 2), max_features=20000)` was initialized with default arguments. The scikit-learn parameter `sublinear_tf` defaults to **`False`** (confirmed by inspecting model parameters: `'sublinear_tf': False`). Linear term frequency is used.
2. **Static HTML Confusion Matrix Placeholder vs. Benchmark Results**:
   - *Static HTML (`web/index.html`, Lines 364–387)*: A hardcoded prototype table displayed True Negative: 451, True Neutral: 142, True Positive: 2,691 (total support: 605 negative, 366 neutral, 3,128 positive).
   - *Verified Benchmark (`reports/model_comparison.json`)*: The actual stratified holdout ($4\,099$ reviews) has class support of 643 negative, 437 neutral, and 3,019 positive. Model 3 correctly identified 530 negatives, 213 neutrals, and 2,541 positives.
   - *Resolution*: In the live application, `web/app.js` fetches `/api/evaluation` and dynamically overwrites the table with the real values.
3. **Empty Placeholder File**:
   - `data/external/eco_hotel_annotations.csv` is an empty 0-byte file. The operational templates are `annotator_A.csv`, `annotator_B.csv`, and `annotation_consensus.csv`.

### 15.2 Missing Information & Unconfirmed Elements
- **Exact Crawler Source of Dataset A**: While documented as TripAdvisor reviews, the original scraping script and exact TripAdvisor hotel IDs are not part of the repository.
- **Annotator Personas**: The identities of Annotator A and Annotator B are not yet assigned, as the human annotation phase is documented as pending.

### 15.3 Inherent Academic Limitations
- **Rating-Derived Sentiment Labels**: Converting numerical ratings ($1$–$5$) to sentiment classes assumes that 4- and 5-star reviews are uniformly positive, and 3-star reviews are uniformly neutral. In reality, customer ratings exhibit subjective variance: some guests award 4 stars while writing harsh criticisms, while others give 3 stars to enjoyable stays.
- **Bag-of-Words & Independence Assumption**: Naive Bayes assumes features are conditionally independent given the class. While bigrams capture immediate word pairs (*"not clean"*), the model cannot capture long-range syntax, double negations (*"hardly dissatisfied"*), or discourse structure (*"The room was tiny, but the incredible staff made it the best trip of our lives"*).
- **Out-of-Vocabulary Terms**: High-dimensional bag-of-words models cannot generalize to unseen synonyms or domain-specific terminology without retraining or dense semantic embeddings.

---

## 16. Theory-to-Practice Mapping for FDMS Course

| Text Mining Concept (Course) | Theoretical Definition | Concrete Implementation in ReviewMind | Direct Evidence in Code / Reports |
|---|---|---|---|
| **Corpus Profiling & Exploration** | Characterizing text collections via length, vocabulary, and token distributions | `src/explore_data.py` & `src/explore_external_data.py` measuring row counts, missing values, duplicates, and rating balance | [`src/explore_data.py#L76-L109`](../src/explore_data.py#L76-L109), [`reports/external_dataset_exploration.json`](../reports/external_dataset_exploration.json) |
| **Data Cleaning & Normalization** | Removing noise and standardizing text without altering core semantics | Whitespace stripping (`str.strip()`), dropping null/empty strings, verifying valid ratings $1$–$5$ | [`src/prepare_data.py#L88-L102`](../src/prepare_data.py#L88-L102) |
| **Sentiment Label Heuristics** | Mapping ordinal satisfaction scales to categorical polarity classes | Piecewise mapping: $\{1, 2\} \to \text{neg}$, $\{3\} \to \text{neu}$, $\{4, 5\} \to \text{pos}$ | [`src/prepare_data.py#L106-L114`](../src/prepare_data.py#L106-L114), [`tests/test_data_validation.py#L12-L21`](../tests/test_data_validation.py#L12-L21) |
| **N-Gram Tokenization** | Extracting contiguous sequences of $n$ items from text to capture local phrase context | `TfidfVectorizer(ngram_range=(1, 2))` capturing unigrams and bigrams (*"not clean"*, *"top notch"*) | [`src/train_model.py#L221-L224`](../src/train_model.py#L221-L224), [`models/reviewmind_final_model.joblib`](../models/reviewmind_final_model.joblib) |
| **Feature Selection (Top-K Vocabulary)** | Restricting vector dimensionality to the most frequent/informative terms | `max_features=20000` capping vocabulary to top 20k terms to prevent overfitting and memory exhaustion | [`src/train_model.py#L223`](../src/train_model.py#L223), [`reports/final_model_metadata.json#L33`](../reports/final_model_metadata.json#L33) |
| **TF-IDF Weighting** | Balancing term frequency with corpus specificity; dampening common terms | Standard linear TF $\times$ smoothed IDF with Euclidean $L_2$ document normalization | [`src/train_model.py#L221-L224`](../src/train_model.py#L221-L224) |
| **Data Leakage Prevention** | Strictly separating training transformations from evaluation splits | Fitting `TfidfVectorizer` strictly on `X_train` inside an `sklearn.pipeline.Pipeline` | [`src/train_model.py#L236-L242`](../src/train_model.py#L236-L242) |
| **Generative Bayesian Classification** | Applying Bayes' rule with conditional independence assumption | `MultinomialNB` computing log joint probabilities $\ln P(c) + \sum x_i \ln P(w_i \mid c)$ | [`src/train_model.py#L227-L229`](../src/train_model.py#L227-L229) |
| **Class Imbalance & Prior Collapse** | Skewed empirical priors penalizing minority classes during Bayesian inference | Empirical priors ($73.7\%$ positive) causing zero neutral predictions ($F_1 = 0.0000$ in Model 1) | [`reports/model_comparison.json#L31-L81`](../reports/model_comparison.json#L31-L81) |
| **Lidstone / Laplace Smoothing** | Preventing zero-probability multiplication for unseen words | Comparison of $\alpha = 1.0$ (Laplace) vs $\alpha = 0.1$ (Lidstone), sharpening probability estimates | [`src/compare_models.py#L219-L241`](../src/compare_models.py#L219-L241), [`reports/model_comparison.json`](../reports/model_comparison.json) |
| **Multi-Class Evaluation Metrics** | Assessing accuracy, per-class metrics, and macro vs weighted averages | Reporting Accuracy ($80.12\%$), Macro $F_1$ ($0.6762$), Per-class $F_1$, and Confusion Matrix | [`reports/model_comparison.json#L160-L210`](../reports/model_comparison.json#L160-L210), [`reports/model_comparison.csv`](../reports/model_comparison.csv) |
| **Out-of-Domain Generalization** | Testing trained models on external data with vocabulary/domain shifts | Applying Model 3 to Dataset B (eco-hotel reviews); finding $57.64\%$ low-confidence inferences | [`src/predict_external.py`](../src/predict_external.py), [`reports/external_prediction_summary.json`](../reports/external_prediction_summary.json) |
| **Inter-Annotator Agreement (Ground Truth)** | Statistical measurement of agreement between human judges | Protocol defining Cohen's $\kappa$ and double-blind annotation reconciliation before evaluation | [`data/external/ANNOTATION_GUIDELINES.md#L44-L82`](../data/external/ANNOTATION_GUIDELINES.md#L44-L82), [`src/prepare_annotations.py`](../src/prepare_annotations.py) |

---

## 17. Suggested Demonstration Scenario

For the oral defense, demonstrate processing an ambiguous, real-world review to highlight the model's balanced priors and explainable n-gram features:

### Synthetic Review Example
> *"The hotel location was convenient and the room was clean, but the air conditioning was broken all night and breakfast was mediocre."*

### Step-by-Step Processing Flow
1. **Raw Text Input**: The text is submitted via the **Inference Studio** textarea in the web UI.
2. **API Communication**: `web/js/api.js` submits `POST /api/predict` to the Flask backend.
3. **TF-IDF Transformation**: The pre-trained vectorizer extracts active unigrams and bigrams:
   - Positive cues: *"convenient"* ($\text{weight} \approx 0.32$), *"clean"* ($\text{weight} \approx 0.34$).
   - Negative cues: *"air conditioning"* ($\text{weight} \approx 0.41$), *"broken"* ($\text{weight} \approx 0.39$), *"mediocre"* ($\text{weight} \approx 0.35$).
   - Contrastive conjunction: *"but"* (links the opposing clauses).
4. **Bayesian Posterior Computation**:
   - Because `fit_prior=False`, uniform priors ($P(c) = 1/3$) are applied.
   - Negative log-likelihood and neutral log-likelihood compete closely.
   - Posterior probabilities: Negative: $42\%$, Neutral: $46\%$, Positive: $12\%$.
5. **Prediction & Confidence**:
   - Classification: `NEUTRAL`.
   - Confidence: $46.0\%$ (moderate confidence, signaling a mixed review).
6. **Explainability**: The UI displays the salient n-grams (*broken*, *air conditioning*, *mediocre*, *clean*, *convenient*) directly below the probability bars.

---

## 18. Presentation-Ready Explanations for Master 2 SID

### Slide 1: The Problem of Imbalance in Hospitality NLP
> *"Dans le domaine de l'hôtellerie, les données textuelles présentent un déséquilibre intrinsèque majeur : les clients partagent principalement des expériences extrêmes, très positives (plus de 73% sur TripAdvisor) ou très négatives (15%). La classe neutre, qui correspond aux avis mitigés ou modérés, ne représente que 10.6% du corpus."*

### Slide 2: Why Naive Bayes Fails Out-of-the-Box
> *"Lorsqu'on applique un classificateur Naive Bayes multinomial standard avec des a priori empiriques (fit_prior=True), la règle de Bayes pénalise fortement la classe minoritaire. La probabilité a priori de la classe positive est 7 fois supérieure à celle de la classe neutre. Résultat : le modèle de base n'a prédit aucun avis neutre sur 4 099 avis de test. Son F1-score neutre était de 0.0000. C'est le phénomène classique d'effondrement sur la classe majoritaire."*

### Slide 3: The Solution — Uniform Priors & Lidstone Smoothing
> *"Pour restaurer la sensibilité du modèle, ReviewMind neutralise le biais a priori en imposant des a priori uniformes (fit_prior=False). De plus, en abaissant le paramètre de lissage de Laplace de 1.0 à 0.1 (lissage de Lidstone), nous réduisons l'aplatissement des probabilités conditionnelles. Les termes spécifiques aux avis mitigés ('average', 'okay', 'standard') reçoivent ainsi un poids discriminatif suffisant. Le F1-score neutre passe de 0.0000 à 0.3883, et le Macro F1 global atteint son sommet à 0.6762."*

### Slide 4: Generalization & Scientific Honesty on External Data
> *"Pour tester la généralisation, nous avons confronté le modèle au Dataset B : 399 avis d'un éco-resort portugais (Calheiros et al., 2017). Plus de 57% des prédictions présentent un score de confiance inférieur à 60%, révélant un décalage de domaine marqué. Conformément aux principes de rigueur scientifique, nous refusons de publier des scores d'accuracy ou de F1 sur ce dataset externe tant que le protocole d'annotation humaine en double aveugle n'a pas produit de vérité terrain définitive."*

---

## 19. Bibliography & Source References

1. **Alam, M. H., Ryu, W. J., & Lee, S.** (2016). *Joint multi-grain topic sentiment resource analysis for online review summarization*. Information Sciences, 370, 427–442. (Source benchmark for TripAdvisor Hotel Reviews).
2. **Calheiros, A. C., Moro, S., & Rita, P.** (2017). *Hospitality research using big data and text mining: An application to eco-friendly hotels*. International Journal of Hospitality Management, 60, 28–35.
3. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press. (Theoretical reference for TF-IDF, Vector Space Model, and Naive Bayes text classification).
4. **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. **Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R.** (2003). *Tackling the poor assumptions of naive bayes text classifiers*. In Proceedings of the 20th International Conference on Machine Learning (ICML-03), 616–623. (Theoretical basis for uniform priors and addressing empirical prior distortion).
6. **Grinstead, C. M., & Snell, J. L.** (1997). *Introduction to Probability*. American Mathematical Society. (Mathematical basis for Laplace and Lidstone additive smoothing).
7. **Cohen, J.** (1960). *A coefficient of agreement for nominal scales*. Educational and Psychological Measurement, 20(1), 37–46. (Mathematical basis for inter-annotator reliability).

