# ReviewMind

ReviewMind is an academic text data mining project aimed at classifying hotel reviews into sentiment categories using natural language processing (NLP), **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization, and a **Multinomial Naive Bayes** classifier.

---

## Table of Contents

- [Overview](#overview)
- [Educational Objectives](#educational-objectives)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
  - [Source & Initial Exploration](#source--initial-exploration)
  - [Sentiment Mapping Methodology](#sentiment-mapping-methodology)
  - [Class Distribution](#class-distribution)
- [Limitations of Rating-Derived Sentiments](#limitations-of-rating-derived-sentiments)
- [Installation & Usage](#installation--usage)
  - [1. Prerequisites & Environment](#1-prerequisites--environment)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Data Preparation](#3-data-preparation)
- [Planned Machine Learning Pipeline](#planned-machine-learning-pipeline)
- [Model Training Status](#model-training-status)
- [Dataset License & Attribution](#dataset-license--attribution)

---

## Overview

Customer feedback in the hospitality industry offers rich insights into guest satisfaction and service quality. ReviewMind builds an end-to-end sentiment classification workflow from raw customer reviews to automated sentiment prediction. The project implements standard data mining practices, including exploratory data analysis, dataset cleaning, rule-based labeling, feature extraction, and probabilistic text classification.

## Educational Objectives

- **Text Data Exploration**: Understand corpus characteristics, vocabulary distribution, missing values, duplicates, and rating balance.
- **Data Preparation & Label Engineering**: Formulate clear heuristic rules to map discrete ratings into sentiment classes while preserving data integrity.
- **Feature Extraction**: Learn and apply TF-IDF weighting schemes to convert raw unstructured text into high-dimensional numerical feature matrices.
- **Probabilistic Modeling**: Study the Multinomial Naive Bayes classification algorithm and its behavior on sparse textual data.
- **Evaluation & Error Analysis**: Evaluate classification performance with macro/weighted precision, recall, and F1-score, and identify domain-specific limitations.

## Tech Stack

- **Language**: Python 3.13+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning & NLP**: `scikit-learn`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Model Serialization**: `joblib`

## Project Structure

```text
ReviewMind/
├── data/                       # Local dataset directory (excluded from Git)
│   ├── tripadvisor_hotel_reviews.csv   # Raw dataset
│   └── reviews_prepared.csv            # Prepared dataset with Sentiment column
├── models/                     # Serialized models and vectorizers (excluded from Git)
├── reports/                    # Generated evaluation reports and plots
├── src/                        # Source code
│   ├── explore_data.py         # Exploratory data analysis script
│   └── prepare_data.py         # Data cleaning and sentiment mapping script
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## Dataset

### Source & Initial Exploration

The raw dataset used is the **TripAdvisor Hotel Reviews** dataset (`data/tripadvisor_hotel_reviews.csv`).

Key observations from the exploratory analysis (`src/explore_data.py`):
- **Total records**: 20,491 rows
- **Columns**:
  - `Review` (*string*): Unstructured natural language review text.
  - `Rating` (*integer*): Customer score on a scale from 1 to 5.
- **Data quality**:
  - 0 missing values across all columns.
  - 0 duplicate rows.
  - 0 empty or whitespace-only reviews.

### Sentiment Mapping Methodology

To establish ground-truth labels for supervised sentiment classification, customer ratings are mapped into three discrete classes:

$$\text{Sentiment} = 
\begin{cases} 
\text{negative} & \text{if } \text{Rating} \in \{1, 2\} \\
\text{neutral}  & \text{if } \text{Rating} = 3 \\
\text{positive} & \text{if } \text{Rating} \in \{4, 5\} 
\end{cases}$$

### Class Distribution

The prepared dataset (`data/reviews_prepared.csv`) contains 20,491 records partitioned as follows:

| Sentiment | Ratings Included | Number of Reviews | Percentage |
| :--- | :---: | :---: | :---: |
| **positive** | 4, 5 | 15,093 | 73.66% |
| **negative** | 1, 2 | 3,214 | 15.68% |
| **neutral** | 3 | 2,184 | 10.66% |
| **Total** | 1 – 5 | 20,491 | 100.00% |

---

## Limitations of Rating-Derived Sentiments

Deriving sentiment categories strictly from numerical star ratings introduces several methodological constraints:

1. **Class Imbalance**: Approximately 73.66% of the dataset is classified as *positive*. Classifiers trained on this distribution may develop an inductive bias toward the majority class, requiring balanced metrics (macro F1-score) or class weighting.
2. **Ambiguity of Neutral Reviews**: 3-star reviews often contain mixed feedback (e.g., *"the room was spacious but the breakfast was poor"*) rather than neutral indifference.
3. **Text-Rating Discrepancies**: Customers occasionally write strongly negative or critical reviews but award 4 stars, or conversely compose mild critiques with a 1-star penalty.
4. **Subjective Rating Scales**: Different individuals calibrate satisfaction differently; what represents a 3-star stay for one guest might be rated 4 stars by another.

---

## Installation & Usage

### 1. Prerequisites & Environment

Ensure Python 3.10+ (tested on Python 3.13) is installed with the required libraries:

```powershell
pip install -r requirements.txt
```

> **Note**: If you are using an existing Anaconda or local environment with these packages pre-installed, no additional installation is needed.

### 2. Exploratory Data Analysis

Run the exploration script from the project root to inspect dataset dimensions, summary statistics, missing values, duplicates, and rating distributions:

```powershell
python src/explore_data.py
```

### 3. Data Preparation

Run the preparation script to validate the dataset, apply the sentiment mapping rule, verify class distributions, and output the clean dataset to `data/reviews_prepared.csv`:

```powershell
python src/prepare_data.py
```

---

## Planned Machine Learning Pipeline

The upcoming modeling phase will implement:

1. **Text Preprocessing**: Tokenization, case normalization, and removal of non-informative noise.
2. **Feature Extraction**:
   - `TfidfVectorizer` (sublinear TF scaling, n-gram range, min_df thresholds).
3. **Dataset Splitting**:
   - Stratified train/test split (e.g., 80% train / 20% test) to preserve class proportions.
4. **Model Training**:
   - `MultinomialNB` with hyperparameter tuning ($\alpha$ Laplace smoothing parameter).
5. **Model Evaluation & Reporting**:
   - Classification report (Precision, Recall, Macro and Weighted F1-scores).
   - Confusion matrix visualization saved in `reports/`.
   - Serialized model and vectorizer saved in `models/`.

---

## Model Training Status

> **Important**: Model training has **not** been executed yet. 
> 
> The repository currently hosts the data exploration and data preparation stages. Model training and evaluation scripts will be added in the subsequent project phase.

---

## Dataset License & Attribution

The TripAdvisor Hotel Reviews dataset is **not hosted in this Git repository** due to size constraints and licensing compliance. 

To run the pipeline locally:
1. Obtain the TripAdvisor Hotel Reviews CSV file from its original academic/open source distribution (e.g., Kaggle).
2. Place the file in the local `data/` directory with the filename:
   ```text
   data/tripadvisor_hotel_reviews.csv
   ```
3. Use the dataset strictly in accordance with its source license and terms of use.
