<div align="center">

# ReviewMind

**An Open-Source Text Mining & NLP Framework for Hotel Review Sentiment Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Research-informational.svg)]()

[Overview](#overview) •
[Architecture](#architecture) •
[Installation](#installation) •
[Reproducible Pipeline](#reproducible-pipeline) •
[Experimental Results](#experimental-results) •
[External Evaluation & Human Annotation](#external-evaluation--human-annotation) •
[Limitations & Future Work](#limitations--future-work) •
[Contributing](#contributing) •
[License](#license)

</div>

---

## Overview

**ReviewMind** is an academic and educational Natural Language Processing (NLP) framework for text data mining and sentiment classification of hospitality reviews.

The project models the end-to-end data mining lifecycle:
1. **Corpus Exploration**: Profiling review lengths, vocabulary patterns, missing values, and rating balances.
2. **Data Preparation & Label Heuristics**: Deriving categorical sentiment classes from numerical customer ratings while preserving raw data integrity.
3. **Feature Extraction**: High-dimensional text representation with n-gram sublinear TF-IDF vectorization.
4. **Probabilistic Modeling & Hyperparameter Comparison**: Tuning **Multinomial Naive Bayes (MNB)** smoothing and class priors to tackle real-world class imbalance.
5. **Production Model Training**: Fitting on 100% of historical training data without data leakage.
6. **External Domain Evaluation**: Testing on independent external corpora (Eco-hotel reviews) with a dual-annotator human consensus protocol.

---

## Educational & Research Objectives

- **Class Imbalance Dynamics**: Study why standard Multinomial Naive Bayes models with empirical priors collapse on minority classes (e.g., neutral sentiment), and how adjusting class priors (`fit_prior=False`) and Laplace/Lidstone smoothing ($\alpha$) restores minority class sensitivity.
- **Data Leakage Prevention**: Enforce strict separation between feature extraction parameters fitted on training splits versus evaluation splits.
- **Inference vs. Ground Truth**: Distinguish between model predictions (probabilities/confidence) and verified ground-truth annotations.
- **Inter-Annotator Agreement**: Establish a dual-annotator human consensus system to build benchmark evaluation sets.

---

## Architecture & Project Structure

```text
ReviewMind/
├── data/                               # Local datasets (excluded from Git)
│   ├── tripadvisor_hotel_reviews.csv   # Dataset A: Raw TripAdvisor corpus (20,491 reviews)
│   ├── reviews_prepared.csv            # Dataset A: Cleaned with Sentiment column
│   ├── dataset-CalheirosMoroRita-2017.csv # Dataset B: Eco-hotel corpus (401 raw reviews)
│   └── external/                       # Human annotation workspace
│       ├── annotator_A.csv             # Blank template for Annotator A (399 unique reviews)
│       ├── annotator_B.csv             # Blank template for Annotator B (399 unique reviews)
│       ├── annotation_consensus.csv    # Final consensus ground-truth file
│       └── ANNOTATION_GUIDELINES.md    # French annotation protocol & rules
├── models/                             # Trained model pipelines (excluded from Git)
│   ├── reviewmind_internal_model.joblib # Baseline model (80% training split)
│   └── reviewmind_final_model.joblib    # Final production model (100% Dataset A)
├── reports/                            # Generated evaluation reports and figures
│   ├── internal_metrics.json           # Baseline 80/20 split metrics
│   ├── internal_confusion_matrix.png   # Baseline confusion matrix plot
│   ├── model_comparison.json           # 3-configuration benchmark metrics
│   ├── model_comparison.csv            # Tabular comparison summary
│   ├── model_comparison.png            # Macro F1 and per-class F1 visual comparison
│   ├── final_model_metadata.json       # Production model training metadata
│   ├── external_dataset_exploration.json # Dataset B profile
│   ├── external_predictions.csv        # Dataset B predictions with probabilities & confidence
│   ├── external_prediction_summary.json # Dataset B inference summary & confidence analysis
│   └── external_sentiment_distribution.png # Dataset B predicted sentiment distribution plot
├── src/                                # Source code
│   ├── explore_data.py                 # 1. Dataset A exploratory data analysis
│   ├── prepare_data.py                 # 2. Dataset A cleaning & sentiment mapping
│   ├── train_model.py                  # 3. Internal stratified 80/20 model evaluation
│   ├── compare_models.py               # 4. Naive Bayes hyperparameter comparison
│   ├── train_final_model.py            # 5. Production model training on 100% of Dataset A
│   ├── explore_external_data.py        # 6. Dataset B structural exploration
│   ├── predict_external.py             # 7. Model inference on Dataset B
│   └── prepare_annotations.py          # 8. Human annotation system setup
├── tests/                              # Automated unit tests (standard unittest)
│   ├── __init__.py
│   ├── test_data_validation.py         # Tests for filtering and sentiment mapping
│   └── test_pipeline.py                # Tests for pipeline, TF-IDF, MNB, and probas
├── .github/                            # GitHub community health templates
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── .env.example                        # Template environment configuration
├── .gitignore                          # Excludes datasets, models, caches, and secrets
├── CITATION.cff                        # Academic citation metadata
├── CODE_OF_CONDUCT.md                  # Contributor Covenant v2.1
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # Official MIT License
├── README.md                           # Project documentation (this file)
├── requirements.txt                    # Python package dependencies
└── SECURITY.md                         # Security policy & joblib advisory
```

---

## Tech Stack

- **Language**: Python 3.10+ (tested on Python 3.13)
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning & NLP**: `scikit-learn`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Model Serialization**: `joblib`
- **Testing**: `unittest` (built-in standard library)

---

## Installation

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.

### 1. Clone the Repository

```bash
git clone https://github.com/RYAD-BENYAKOUB/ReviewMind.git
cd ReviewMind
```

### 2. Set Up Virtual Environment (Recommended)

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### On Linux / macOS (Bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note**: If you already have `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `joblib` installed in your global or Anaconda environment, ReviewMind works directly without installing extra packages.

### 3. Optional Environment Configuration

```bash
# On Linux / macOS
cp .env.example .env

# On Windows PowerShell
Copy-Item .env.example .env
```

---

## Datasets & Sources

ReviewMind leverages two distinct datasets representing different hospitality domains:

### Dataset A: TripAdvisor Hotel Reviews (Training & Benchmark)
- **File path**: `data/tripadvisor_hotel_reviews.csv`
- **Size**: 20,491 reviews, 2 columns (`Review`, `Rating`).
- **Rating scale**: 1 to 5 stars.
- **Quality**: 0 missing values, 0 duplicate rows, 0 empty reviews.
- **Source**: Publicly available TripAdvisor hotel reviews benchmark.
- **Label derivation**:
  $$\text{Sentiment} = 
  \begin{cases} 
  \text{negative} & \text{if } \text{Rating} \in \{1, 2\} \\
  \text{neutral}  & \text{if } \text{Rating} = 3 \\
  \text{positive} & \text{if } \text{Rating} \in \{4, 5\} 
  \end{cases}$$
- **Class distribution**:
  - `positive`: 15,093 reviews (73.66%)
  - `negative`: 3,214 reviews (15.68%)
  - `neutral`: 2,184 reviews (10.66%)

### Dataset B: Eco-hotel Reviews (External Domain)
- **File path**: `data/dataset-CalheirosMoroRita-2017.csv`
- **Size**: 401 raw reviews (399 unique after removing 2 duplicates).
- **Domain**: Boutique eco-resort (*Areias do Seixo*, Portugal).
- **Source**: Calheiros, Moro, & Rita (2017), *"Hospitality research using big data and text mining: An application to eco-friendly hotels"*.
- **Status**: **Unlabeled text corpus**. Does not contain ratings or sentiment annotations.

> **Dataset Notice**: In compliance with licensing and storage best practices, raw datasets (`.csv`) are **not tracked in the Git repository**. Users must obtain the datasets from their respective sources and place them in the `data/` directory.

---

## Reproducible Pipeline

Run the scripts sequentially from the repository root:

### 1. Exploratory Data Analysis (Dataset A)
Inspects dimensions, column types, missing values, duplicates, and rating distributions:
```bash
python src/explore_data.py
```

### 2. Data Preparation & Labeling (Dataset A)
Cleans whitespace, applies sentiment mapping, displays class breakdown, and outputs `data/reviews_prepared.csv`:
```bash
python src/prepare_data.py
```

### 3. Internal Model Evaluation (80/20 Stratified Split)
Trains a baseline pipeline (`TfidfVectorizer(max_features=20000, ngram_range=(1,2))` + `MultinomialNB(alpha=1.0)`) strictly on the training set, evaluates on test data, and saves metrics to `reports/internal_metrics.json`:
```bash
python src/train_model.py
```

### 4. Hyperparameter Comparison Experiment
Compares three Multinomial Naive Bayes configurations across the identical 80/20 stratified split to address minority class sensitivity:
```bash
python src/compare_models.py
```

### 5. Production Model Training (100% Dataset A)
Fits the selected optimal configuration (`alpha=0.1`, `fit_prior=False`) on all 20,491 records to maximize vocabulary coverage for production inference. Saves `models/reviewmind_final_model.joblib`:
```bash
python src/train_final_model.py
```

### 6. External Dataset Exploration (Dataset B)
Validates the structure, encoding (`latin1`), separator (`\t`), and confirms the absence of ground-truth labels in Dataset B:
```bash
python src/explore_external_data.py
```

### 7. External Sentiment Prediction (Dataset B)
Applies the production model to the 399 unique reviews of Dataset B, computing predicted sentiment, prediction confidence, and probability distributions:
```bash
python src/predict_external.py
```

### 8. Human Annotation Workflow Setup
Generates independent annotation templates (`annotator_A.csv`, `annotator_B.csv`, `annotation_consensus.csv`) in `data/external/`:
```bash
python src/prepare_annotations.py
```

---

## Experimental Results

All experiments were conducted on a stratified 80% train ($16\,392$ reviews) / 20% test ($4\,099$ reviews) split of Dataset A with `random_state=42`.

### Comparative Benchmark (Dataset A Test Set)

| Model Configuration | $\alpha$ | `fit_prior` | Accuracy | Macro F1 | Weighted F1 | Negative F1 | Neutral F1 | Positive F1 | Neutral Predictions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Baseline)** | 1.0 | `True` | 81.97% | **0.5249** | 0.7655 | 0.6802 | **0.0000** | 0.8945 | 0 |
| **Model 2 (Uniform Prior)** | 1.0 | `False` | **85.17%** | **0.6153** | **0.8185** | **0.7730** | **0.1477** | **0.9253** | 64 |
| **Model 3 (Selected Production)** | 0.1 | `False` | 80.12% | **0.6762** | 0.8174 | 0.7454 | **0.3883** | 0.8949 | 660 |

### Key Scientific Insights

1. **Failure of the Baseline on Minority Classes**:
   Because the dataset is heavily skewed toward positive reviews (73.66%), empirical class priors $P(c)$ strongly penalize the neutral class (10.66%) during posterior calculation $\log P(c) + \sum \log P(w_i|c)$. Model 1 predicted **zero** neutral reviews, yielding a Neutral F1-score of 0.0000.
2. **Balancing with Uniform Class Priors**:
   Setting `fit_prior=False` (Model 2) prevents majority-class domination, immediately boosting accuracy to 85.17% and raising Macro F1 from 0.5249 to 0.6153.
3. **Lidstone Smoothing ($\alpha=0.1$)**:
   Decreasing the smoothing parameter from 1.0 to 0.1 sharpens the conditional word probabilities $P(w_i|c)$, allowing distinctive neutral vocabulary to overcome background word frequencies. Neutral recall rises to 48.74%, lifting **Neutral F1 to 0.3883** and achieving the **highest Macro F1 (0.6762)**.

---

## External Evaluation & Human Annotation

### Model Inference on Dataset B (Eco-hotel Reviews)
When the final production model was applied to the 399 unique reviews of Dataset B, it produced the following distribution:

| Predicted Sentiment | Count | Percentage | Average Confidence |
| :--- | :---: | :---: | :---: |
| `positive` | 335 | 83.96% | 60.03% |
| `neutral` | 33 | 8.27% | 42.37% |
| `negative` | 31 | 7.77% | 46.03% |
| **Total** | **399** | **100.00%** | **57.48% (overall mean)** |

> **Methodological Principle**:
> These predictions represent **model inferences, not ground-truth labels**. Because Dataset B does not contain official ratings or verified human labels, **no accuracy, precision, recall, or F1-scores can be calculated on Dataset B at this stage**.

### Dual-Annotator Consensus Framework
To establish a statistically valid ground truth on Dataset B without data contamination, ReviewMind implements an independent dual-annotation protocol:

1. **Annotator A** independently labels `data/external/annotator_A.csv`.
2. **Annotator B** independently labels `data/external/annotator_B.csv` (blind to Annotator A).
3. Disagreements and low-confidence cases are discussed following the [Annotation Guidelines](data/external/ANNOTATION_GUIDELINES.md).
4. The resolved consensus is recorded in `data/external/annotation_consensus.csv`.
5. Only after consensus finalization will formal external evaluation metrics be computed.

---

## Python API Usage

To use the trained ReviewMind pipeline in Python:

```python
import joblib

# Load the trained production pipeline
model = joblib.load("models/reviewmind_final_model.joblib")

# Sample new reviews
new_reviews = [
    "The room was exceptionally clean, the staff was welcoming and the breakfast was superb!",
    "The location is convenient, but the room was quite noisy and the price was high.",
    "Standard stay. The room was okay, nothing special to mention.",
]

# Predict sentiment classes
predictions = model.predict(new_reviews)

# Predict class probabilities
probabilities = model.predict_proba(new_reviews)
classes = model.classes_  # ['negative', 'neutral', 'positive']

for review, pred, probs in zip(new_reviews, predictions, probabilities):
    print(f"\nReview : {review}")
    print(f"Sentiment: {pred}")
    print(f"Probabilities: {dict(zip(classes, probs.round(3)))}")
```

---

## Running Unit Tests

ReviewMind includes a suite of unit tests built with Python's standard `unittest` framework:

```bash
python -m unittest discover tests
```

These tests verify:
- In-memory data validation, null/whitespace filtering, and required column checks.
- Sentiment mapping logic across valid and invalid numerical ratings.
- Pipeline instantiation, TF-IDF vectorization, fitting, and prediction probability consistency.
- Safe handling of absent model artifacts.

---

## Limitations & Future Work

1. **Rating-Derived Sentiment Labels**:
   Using star ratings as ground-truth sentiment labels is an approximation. Customers frequently express mixed or critical reviews despite awarding 4 stars, or conversely pen mild critiques with a 1-star rating.
2. **Domain Shift**:
   Transitioning from general urban/chain hotel reviews (TripAdvisor) to boutique eco-resorts (*Areias do Seixo*) involves domain shifts in terminology (e.g., *upcycling, bio-pool, sustainable architecture, duck feathers*).
3. **Bag-of-Words & Negation**:
   While n-grams (1, 2) capture common local phrases, Naive Bayes ignores long-range syntactic dependencies and complex negations (*"could not have been more pleasant"*).
4. **Planned Extensions**:
   - Integrating dense transformer representations (e.g., RoBERTa / DeBERTa).
   - Domain-adaptive fine-tuning.
   - Interactive web visualization dashboard (Streamlit / FastAPI).

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details on code style, testing, and the Pull Request workflow.

---

## Citation

If you use ReviewMind in academic work, research, or course projects, please cite it using our [CITATION.cff](CITATION.cff) metadata:

```bibtex
@software{reviewmind2026,
  author = {Benyakoub, Mohammed Ryad},
  title = {ReviewMind: Hotel Review Sentiment Classification with TF-IDF and Multinomial Naive Bayes},
  year = {2026},
  url = {https://github.com/RYAD-BENYAKOUB/ReviewMind},
  version = {1.0.0}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

*Copyright (c) 2026 Benyakoub Mohammed Ryad.*

> **Third-Party Data Disclaimer**: The MIT license applies to ReviewMind's code and original documentation. It does not apply to third-party datasets or dependencies, which remain governed by their respective licenses and terms of use.
