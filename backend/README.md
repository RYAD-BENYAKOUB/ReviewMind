# ReviewMind Backend API — Dark Intelligence

This directory contains the production-ready **Flask REST API** connecting the ReviewMind frontend interface to the actual trained Machine Learning model.

---

## 1. Architecture Overview

```text
backend/
├── app.py                      # Application factory, CORS handling & server runner
├── config.py                   # Environment configuration & filesystem path resolution
├── requirements.txt            # Python backend dependencies
├── services/                   # Business logic & singletons
│   ├── model_service.py        # Loads model once and executes real inference
│   ├── dataset_service.py      # Metadata & paginated Dataset B review retrieval
│   └── evaluation_service.py   # Aggregates scientific benchmark metrics & reports
└── routes/                     # REST API endpoints (mounted under /api)
    ├── health.py               # GET /api/health
    ├── predictions.py          # POST /api/predict
    ├── datasets.py             # GET /api/datasets & GET /api/dashboard
    ├── reviews.py              # GET /api/reviews & GET /api/reviews/<id>
    └── evaluation.py           # GET /api/evaluation
```

---

## 2. Dependencies & Installation

### Requirements
- Python 3.10+ (tested on Python 3.13)
- `Flask>=3.0.0`
- `Flask-CORS>=5.0.0`
- `scikit-learn>=1.6.0`
- `pandas>=2.2.0`
- `numpy>=2.0.0`
- `joblib>=1.4.0`

### Installation
From the repository root:
```bash
pip install -r backend/requirements.txt
```

---

## 3. Starting the Flask Server

### Direct Execution
From the project root:
```bash
python backend/app.py
```

### Using Flask CLI
```bash
# Windows PowerShell
$env:FLASK_APP = "backend.app:create_app()"
$env:FLASK_ENV = "development"
flask run --host 127.0.0.1 --port 5000

# Linux / macOS
export FLASK_APP="backend.app:create_app()"
export FLASK_ENV="development"
flask run --host 127.0.0.1 --port 5000
```

The server will start listening on `http://127.0.0.1:5000`.

---

## 4. API Endpoints Reference

### 4.1 Health Check — `GET /api/health`
Verifies API status and model loading readiness.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/api/health
```

**Example Response (200 OK):**
```json
{
  "model": {
    "alpha": 0.1,
    "architecture": "TfidfVectorizer(ngram_range=(1, 2), max_features=20000) + MultinomialNB(alpha=0.1, fit_prior=False)",
    "classes": ["negative", "neutral", "positive"],
    "fit_prior": false,
    "model_loaded": true,
    "model_name": "ReviewMind Final Production Model",
    "status": "ready",
    "trained_on": "100% of TripAdvisor Dataset A (20,491 reviews)",
    "vocabulary_size": 20000
  },
  "service": "ReviewMind REST API",
  "status": "healthy",
  "timestamp_utc": "2026-10-30T18:00:00Z",
  "version": "1.0.0"
}
```

---

### 4.2 Sentiment Prediction — `POST /api/predict`
Executes real-time inference using the trained `Pipeline(TfidfVectorizer, MultinomialNB)`.

**Headers:**
`Content-Type: application/json`

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "The hotel was absolutely fantastic! The staff was friendly, the room was immaculate, and the breakfast was superb."}'
```

**Example Response (200 OK):**
```json
{
  "confidence": 0.8842,
  "model_info": {
    "algorithm": "MultinomialNB(alpha=0.1, fit_prior=False)",
    "name": "ReviewMind Final Production Model",
    "vectorizer": "TfidfVectorizer(ngram_range=(1, 2), max_features=20000)"
  },
  "predicted_sentiment": "positive",
  "probabilities": {
    "negative": 0.0215,
    "neutral": 0.0943,
    "positive": 0.8842
  },
  "review": "The hotel was absolutely fantastic! The staff was friendly, the room was immaculate, and the breakfast was superb.",
  "salient_features": [
    {"term": "fantastic", "weight": 0.3841},
    {"term": "superb", "weight": 0.3512},
    {"term": "immaculate", "weight": 0.3129},
    {"term": "friendly", "weight": 0.2815}
  ],
  "status": "success",
  "validation": {
    "is_valid": true,
    "length_chars": 115,
    "word_count": 18
  }
}
```

**Validation Error Response (400 Bad Request):**
```json
{
  "error": "Validation Error",
  "message": "Review text cannot be empty or contain only whitespace.",
  "status": "empty_input"
}
```

---

### 4.3 Dashboard Statistics — `GET /api/dashboard`
Returns live executive KPIs aggregated from verified scientific evaluation reports.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/api/dashboard
```

**Example Response (200 OK):**
```json
{
  "active_model": {
    "accuracy": 0.8012,
    "macro_f1": 0.6762,
    "model_id": "model_3",
    "name": "Model 3 (Alternative Smoothing)",
    "negative_f1": 0.7454,
    "neutral_f1": 0.3883,
    "positive_f1": 0.8949
  },
  "baseline_model": {
    "accuracy": 0.8197,
    "macro_f1": 0.5249,
    "model_id": "model_1",
    "name": "Model 1 (Baseline)",
    "neutral_f1": 0.0
  },
  "improvement_deltas": {
    "macro_f1_gain": 0.1513,
    "neutral_f1_gain": 0.3883,
    "neutral_minority_recovery": "Activated through uniform priors (fit_prior=False) + reduced alpha (0.1)"
  },
  "dataset_a_summary": {
    "name": "TripAdvisor Hotel Reviews",
    "total_records": 20491
  }
}
```

---

### 4.4 Datasets Metadata — `GET /api/datasets`
Returns metadata describing Dataset A, Dataset B, and the human annotation workspace.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/api/datasets
```

---

### 4.5 Reviews Explorer — `GET /api/reviews`
Supports pagination, sentiment filtering, and keyword search over Dataset B.

**Query Parameters:**
- `page` (default: 1)
- `per_page` (default: 10, max: 100)
- `sentiment` (optional: `positive` | `neutral` | `negative`)
- `q` (optional text search query)

**Example Request:**
```bash
curl -X GET "http://127.0.0.1:5000/api/reviews?sentiment=negative&per_page=2"
```

---

### 4.6 Single Review — `GET /api/reviews/<original_index>`
Fetches a single review by its integer index. Returns 404 if not found.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/api/reviews/0
```

---

### 4.7 Evaluation & Benchmark Metrics — `GET /api/evaluation`
Returns internal 80/20 test split metrics, the confusion matrix, and the explicit pending state of external human annotation consensus.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/api/evaluation
```

**Key Response Properties:**
- `internal_evaluation`: Complete metrics for Model 1, Model 2, and Model 3.
- `external_evaluation.supervised_metrics_status`: `"pending"`
- `external_evaluation.human_consensus_available`: `false`
- `external_evaluation.notice`: Detailed explanation that external supervised metrics are strictly pending human consensus adjudication.

---

## 5. Connecting the Frontend

The ReviewMind frontend (`web/index.html` and `web/app.js`) automatically detects the Flask backend:
1. Start the backend: `python backend/app.py`.
2. Open `web/index.html` in your browser (or visit `http://127.0.0.1:5000/web/index.html`).
3. The frontend's `analyzeSentiment()` will seamlessly send `POST /api/predict` requests to the Flask server and display real model probabilities and salient vocabulary n-grams.
4. If the server is offline, the interface gracefully falls back to client-side scoring so it never breaks in offline environments.

---

## 6. Running Backend Tests

ReviewMind uses Python's standard `unittest` framework:
```bash
# Run backend tests only
python -m unittest tests/test_backend.py

# Run all test suites
python -m unittest discover tests
```

