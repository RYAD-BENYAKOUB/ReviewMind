/**
 * ReviewMind — Centralized API Service
 * Manages all communication between the web frontend and the Flask REST API.
 * 
 * Strict Rule: No client-side prediction fallbacks, no fabricated probabilities.
 * The Flask backend is the single source of truth for all ML inferences and metrics.
 */

class ApiService {
  constructor() {
    // If served directly by Flask (port 5000), use relative path; otherwise point to localhost:5000
    this.baseUrl = (window.location.origin && window.location.origin.includes(':5000'))
      ? ''
      : 'http://127.0.0.1:5000';
    this.timeoutMs = 12000; // 12 seconds request timeout
  }

  /**
   * Generic fetch wrapper with AbortController timeout, JSON parsing,
   * and structured HTTP/network error normalization.
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    const config = {
      ...options,
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
        ...(options.headers || {}),
      },
    };

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      let data;
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        data = { message: text };
      }

      if (!response.ok) {
        const error = new Error(data.message || data.error || `HTTP error ${response.status}`);
        error.status = response.status;
        error.data = data;
        throw error;
      }

      return data;
    } catch (err) {
      clearTimeout(timeoutId);

      if (err.name === 'AbortError') {
        const timeoutError = new Error(`Request timed out after ${this.timeoutMs / 1000}s. The server may be busy or offline.`);
        timeoutError.isTimeout = true;
        throw timeoutError;
      }

      if (!err.status) {
        // Network error (backend not running, CORS block, or connection refused)
        const networkError = new Error('Cannot connect to ReviewMind Flask backend at http://127.0.0.1:5000. Please ensure the server is running (python backend/app.py).');
        networkError.isNetwork = true;
        throw networkError;
      }

      throw err;
    }
  }

  /**
   * GET /api/health
   * Check API health, model loading readiness, and classes.
   */
  async getHealth() {
    return this.request('/api/health');
  }

  /**
   * GET /api/dashboard
   * Fetch live executive KPIs and model comparison summaries.
   */
  async getDashboard() {
    return this.request('/api/dashboard');
  }

  /**
   * GET /api/datasets
   * Fetch metadata for Dataset A, Dataset B, and human annotation templates.
   */
  async getDatasets() {
    return this.request('/api/datasets');
  }

  /**
   * GET /api/reviews
   * Query reviews from Dataset B with pagination, sentiment filter, and search.
   */
  async getReviews({ page = 1, perPage = 10, sentiment = null, query = null } = {}) {
    const params = new URLSearchParams();
    if (page) params.append('page', page);
    if (perPage) params.append('per_page', perPage);
    if (sentiment && sentiment !== 'all') params.append('sentiment', sentiment);
    if (query && query.trim()) params.append('q', query.trim());

    const queryString = params.toString();
    const endpoint = `/api/reviews${queryString ? `?${queryString}` : ''}`;
    return this.request(endpoint);
  }

  /**
   * GET /api/reviews/{original_index}
   * Fetch a single review by original index.
   */
  async getReviewById(originalIndex) {
    return this.request(`/api/reviews/${originalIndex}`);
  }

  /**
   * POST /api/predict
   * Execute real inference using the trained ReviewMind production pipeline.
   * NEVER generates client-side fallback predictions.
   */
  async predict(reviewText) {
    if (!reviewText || typeof reviewText !== 'string' || !reviewText.trim()) {
      throw new Error('Review text must be a non-empty string.');
    }

    return this.request('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        review: reviewText.trim(),
      }),
    });
  }

  /**
   * GET /api/evaluation
   * Retrieve internal benchmark comparison and external human consensus status.
   */
  async getEvaluation() {
    return this.request('/api/evaluation');
  }
}

// Global API service instance
window.apiService = new ApiService();
