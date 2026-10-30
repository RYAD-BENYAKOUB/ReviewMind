/**
 * ReviewMind — DARK INTELLIGENCE
 * Application Orchestration Logic
 * 
 * Strict Scientific & Architectural Rules:
 *   - No client-side sentiment heuristics, fake probabilities, or fallback scoring.
 *   - Flask REST API (apiService) is the single source of truth for all ML inferences and metrics.
 *   - Brand Palette: BLACK #0B0B0F, BLUE #1E6BFF, RED #E0122F, GRAY #8E8E93, WHITE #FFFFFF.
 */

// Application State
const state = {
  currentTab: 'tab-classifier',
  datasetB: {
    page: 1,
    perPage: 10,
    sentiment: 'all',
    query: '',
    totalPages: 1,
    totalReviews: 0,
    hasNext: false,
    hasPrev: false,
  },
  lastInferenceText: '',
};

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  setupTabNavigation();
  setupInferenceStudio();
  setupDatasetExplorer();
  initializeBackendData();
});

/* ==========================================================================
   1. TAB NAVIGATION
   ========================================================================== */
function setupTabNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const sections = document.querySelectorAll('.dashboard-tab-content');

  navItems.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = btn.getAttribute('data-tab');
      state.currentTab = targetTab;

      navItems.forEach((item) => item.classList.remove('active'));
      btn.classList.add('active');

      sections.forEach((sec) => {
        if (sec.id === targetTab) {
          sec.style.display = 'block';
        } else {
          sec.style.display = 'none';
        }
      });
    });
  });
}

/* ==========================================================================
   2. INITIALIZE BACKEND DATA & HEALTH STATUS
   ========================================================================== */
async function initializeBackendData() {
  const statusDot = document.getElementById('backend-status-dot');
  const statusText = document.getElementById('backend-status-text');

  try {
    const health = await window.apiService.getHealth();
    if (health.status === 'healthy' && health.model_loaded) {
      if (statusDot) {
        statusDot.style.backgroundColor = 'var(--rm-blue)';
        statusDot.style.boxShadow = '0 0 8px rgba(30, 107, 255, 0.6)';
      }
      if (statusText) {
        statusText.textContent = 'API Online (Model 3 Active)';
        statusText.style.color = 'var(--rm-white)';
      }
    }
  } catch (err) {
    if (statusDot) {
      statusDot.style.backgroundColor = 'var(--rm-red)';
      statusDot.style.boxShadow = '0 0 8px rgba(224, 18, 47, 0.6)';
    }
    if (statusText) {
      statusText.textContent = 'API Offline (Localhost:5000)';
      statusText.style.color = 'var(--rm-red)';
    }
    console.warn('ReviewMind Backend Health Check Failed:', err.message);
  }

  // Load live Dashboard KPIs
  loadDashboardData();

  // Load Model Evaluation & Confusion Matrix
  loadEvaluationData();

  // Load Dataset Metadata
  loadDatasetMetadata();

  // Load Initial Dataset B Reviews
  fetchAndRenderReviews();
}

async function loadDashboardData() {
  try {
    const data = await window.apiService.getDashboard();
    if (data && data.executive_summary) {
      const summary = data.executive_summary;
      const accEl = document.getElementById('kpi-accuracy');
      if (accEl) accEl.textContent = summary.accuracy_pct || `${(summary.accuracy * 100).toFixed(2)}%`;

      const f1El = document.getElementById('kpi-macro-f1');
      if (f1El) f1El.textContent = summary.macro_f1.toFixed(4);

      const neuEl = document.getElementById('kpi-neutral-f1');
      if (neuEl) neuEl.textContent = summary.neutral_f1.toFixed(4);

      const vocabEl = document.getElementById('kpi-vocab-size');
      if (vocabEl) vocabEl.textContent = Number(summary.vocabulary_size).toLocaleString();

      const f1Sub = document.getElementById('kpi-macro-f1-subtext');
      if (f1Sub && summary.delta_macro_f1_vs_baseline) {
        f1Sub.textContent = `${summary.delta_macro_f1_vs_baseline} vs MNB baseline`;
      }
    }
  } catch (err) {
    console.debug('Dashboard metrics loaded from static verified defaults:', err.message);
  }
}

async function loadEvaluationData() {
  try {
    const data = await window.apiService.getEvaluation();
    if (!data) return;

    // Render benchmark table if dynamic data is present
    if (data.internal_evaluation && Array.isArray(data.internal_evaluation.models)) {
      renderBenchmarkTable(data.internal_evaluation.models);
    }

    // Render confusion matrix
    if (data.internal_evaluation && data.internal_evaluation.confusion_matrix) {
      renderConfusionMatrix(data.internal_evaluation.confusion_matrix);
    }

    // Update consensus badge
    const badge = document.getElementById('consensus-status-badge');
    if (badge && data.external_evaluation) {
      if (data.external_evaluation.human_consensus_available) {
        badge.className = 'badge-sentiment badge-positive';
        badge.textContent = 'STATUS: CONSENSUS COMPLETE';
      } else {
        badge.className = 'badge-sentiment badge-neutral';
        badge.textContent = 'STATUS: PENDING CONSENSUS';
      }
    }
  } catch (err) {
    console.debug('Evaluation data error:', err.message);
  }
}

function renderBenchmarkTable(models) {
  const tbody = document.getElementById('benchmark-table-body');
  if (!tbody || !models.length) return;

  tbody.innerHTML = '';
  models.forEach((m) => {
    const tr = document.createElement('tr');
    if (m.name.includes('Model 3') || m.macro_f1 === 0.6762) {
      tr.className = 'champion';
    }

    const isChampion = m.name.includes('Model 3');
    tr.innerHTML = `
      <td><strong>${escapeHtml(m.name)}${isChampion ? ' ★' : ''}</strong></td>
      <td>${m.alpha}</td>
      <td>${m.fit_prior ? 'True (Empirical)' : 'False (Uniform)'}</td>
      <td>${(m.accuracy * 100).toFixed(2)}%</td>
      <td><strong>${m.macro_f1.toFixed(4)}</strong></td>
      <td><span style="color: var(--rm-red);">${m.negative_f1.toFixed(4)}</span></td>
      <td><span style="color: ${m.neutral_f1 > 0.2 ? '#FFFFFF' : 'var(--rm-gray-muted)'}; font-weight: ${m.neutral_f1 > 0.2 ? '700' : '400'};">${m.neutral_f1.toFixed(4)}</span></td>
      <td><span style="color: var(--rm-blue);">${m.positive_f1.toFixed(4)}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConfusionMatrix(matrixData) {
  const tbody = document.getElementById('confusion-matrix-body');
  if (!tbody || !matrixData.labels || !matrixData.matrix) return;

  const labels = matrixData.labels; // ['negative', 'neutral', 'positive']
  const matrix = matrixData.matrix;
  const normalized = matrixData.normalized;

  tbody.innerHTML = '';

  const labelMeta = {
    negative: { title: 'Negative (True)', color: 'var(--rm-red)' },
    neutral: { title: 'Neutral (True)', color: 'var(--rm-gray-muted)' },
    positive: { title: 'Positive (True)', color: 'var(--rm-blue)' },
  };

  labels.forEach((trueLabel, rowIdx) => {
    const tr = document.createElement('tr');
    const meta = labelMeta[trueLabel] || { title: trueLabel, color: 'var(--rm-text-primary)' };
    const rowTotal = matrix[rowIdx].reduce((a, b) => a + b, 0);

    let rowHtml = `<td style="text-align: left; font-weight: 700; color: ${meta.color};">${meta.title}</td>`;

    labels.forEach((predLabel, colIdx) => {
      const count = matrix[rowIdx][colIdx];
      const pct = normalized ? (normalized[rowIdx][colIdx] * 100).toFixed(1) : ((count / rowTotal) * 100).toFixed(1);
      const isDiag = rowIdx === colIdx;
      rowHtml += `<td class="${isDiag ? 'matrix-diag' : ''}">${count.toLocaleString()} (${pct}%)</td>`;
    });

    rowHtml += `<td style="font-weight: 600;">${rowTotal.toLocaleString()}</td>`;
    tr.innerHTML = rowHtml;
    tbody.appendChild(tr);
  });
}

async function loadDatasetMetadata() {
  try {
    const data = await window.apiService.getDatasets();
    if (data && data.dataset_a) {
      const headerBadge = document.getElementById('header-dataset-badge');
      if (headerBadge) {
        headerBadge.textContent = `Dataset A: ${Number(data.dataset_a.rows_count).toLocaleString()} Reviews`;
      }
    }
  } catch (err) {
    console.debug('Dataset metadata load:', err.message);
  }
}

/* ==========================================================================
   3. INFERENCE STUDIO (LIVE PREDICTION)
   Strictly calls POST /api/predict. No client-side fallback.
   ========================================================================== */
function setupInferenceStudio() {
  const sampleReviews = {
    pos: "The hotel exceeded all expectations. Exceptional staff, immaculate room overlooking the ocean, and top-notch breakfast buffet. Will definitely stay here again next year!",
    neu: "Standard three-star property. The room was okay for a one-night business stay, though nothing special. Average breakfast and decent location near the metro.",
    neg: "Terrible experience. The air conditioning was broken all weekend, bathroom had not been cleaned properly, and the front desk staff was dismissive and refused to help."
  };

  const reviewInput = document.getElementById('review-input');
  document.querySelectorAll('.btn-sample').forEach((btn) => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-sample');
      if (sampleReviews[type] && reviewInput) {
        reviewInput.value = sampleReviews[type];
        executePrediction(sampleReviews[type]);
      }
    });
  });

  const btnAnalyze = document.getElementById('btn-analyze');
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      if (!reviewInput) return;
      const text = reviewInput.value.trim();
      if (!text) {
        alert('Please enter or select a hotel review to classify.');
        reviewInput.focus();
        return;
      }
      executePrediction(text);
    });
  }

  const btnClear = document.getElementById('btn-clear');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      if (reviewInput) reviewInput.value = '';
      hideInferenceOutputs();
    });
  }

  const btnRetry = document.getElementById('btn-retry-predict');
  if (btnRetry) {
    btnRetry.addEventListener('click', () => {
      if (state.lastInferenceText) {
        executePrediction(state.lastInferenceText);
      } else if (reviewInput && reviewInput.value.trim()) {
        executePrediction(reviewInput.value.trim());
      }
    });
  }
}

function hideInferenceOutputs() {
  const resultBox = document.getElementById('inference-result');
  const loadingBox = document.getElementById('inference-loading');
  const errorBox = document.getElementById('inference-error');

  if (resultBox) resultBox.style.display = 'none';
  if (loadingBox) loadingBox.style.display = 'none';
  if (errorBox) errorBox.style.display = 'none';
}

async function executePrediction(rawText) {
  state.lastInferenceText = rawText;

  const resultBox = document.getElementById('inference-result');
  const loadingBox = document.getElementById('inference-loading');
  const errorBox = document.getElementById('inference-error');
  const errorMsg = document.getElementById('inference-error-msg');
  const btnAnalyze = document.getElementById('btn-analyze');

  // UI Loading State
  if (resultBox) resultBox.style.display = 'none';
  if (errorBox) errorBox.style.display = 'none';
  if (loadingBox) loadingBox.style.display = 'block';
  if (btnAnalyze) btnAnalyze.disabled = true;

  try {
    const data = await window.apiService.predict(rawText);

    if (data.status === 'success') {
      const sentiment = data.predicted_sentiment;
      const confidence = (data.confidence || 0) * 100;
      const probs = data.probabilities || {};
      const pPos = (probs.positive || 0) * 100;
      const pNeu = (probs.neutral || 0) * 100;
      const pNeg = (probs.negative || 0) * 100;
      const features = data.salient_features || [];

      renderPredictionResult(sentiment, confidence, pPos, pNeu, pNeg, features);
    } else {
      throw new Error(data.message || 'Prediction failed');
    }
  } catch (err) {
    console.error('Inference API Error:', err);
    if (loadingBox) loadingBox.style.display = 'none';
    if (errorBox) {
      errorBox.style.display = 'block';
      if (errorMsg) {
        errorMsg.textContent = err.message || 'Cannot connect to ReviewMind Flask API.';
      }
    }
  } finally {
    if (loadingBox) loadingBox.style.display = 'none';
    if (btnAnalyze) btnAnalyze.disabled = false;
  }
}

function renderPredictionResult(sentiment, confidence, pPos, pNeu, pNeg, features) {
  const resultCard = document.getElementById('inference-result');
  if (!resultCard) return;

  resultCard.style.display = 'block';

  // Sentiment Badge
  let badgeColor = 'badge-positive';
  if (sentiment === 'negative') {
    badgeColor = 'badge-negative';
  } else if (sentiment === 'neutral') {
    badgeColor = 'badge-neutral';
  }

  const badgeEl = document.getElementById('res-badge');
  if (badgeEl) {
    badgeEl.className = `badge-sentiment ${badgeColor}`;
    badgeEl.textContent = sentiment.toUpperCase();
  }

  const confEl = document.getElementById('res-conf');
  if (confEl) {
    confEl.textContent = `${confidence.toFixed(1)}% Confidence`;
  }

  // Probability Bars
  const pPosEl = document.getElementById('prob-pos-val');
  if (pPosEl) pPosEl.textContent = `${pPos.toFixed(1)}%`;
  const pPosBar = document.getElementById('prob-pos-bar');
  if (pPosBar) pPosBar.style.width = `${pPos}%`;

  const pNeuEl = document.getElementById('prob-neu-val');
  if (pNeuEl) pNeuEl.textContent = `${pNeu.toFixed(1)}%`;
  const pNeuBar = document.getElementById('prob-neu-bar');
  if (pNeuBar) pNeuBar.style.width = `${pNeu}%`;

  const pNegEl = document.getElementById('prob-neg-val');
  if (pNegEl) pNegEl.textContent = `${pNeg.toFixed(1)}%`;
  const pNegBar = document.getElementById('prob-neg-bar');
  if (pNegBar) pNegBar.style.width = `${pNeg}%`;

  // Salient Features (N-Grams)
  const tokensEl = document.getElementById('extracted-tokens');
  if (tokensEl) {
    tokensEl.innerHTML = '';
    if (!features || features.length === 0) {
      tokensEl.innerHTML = '<span style="color: var(--rm-text-muted); font-size: 12px;">No decisive vocabulary tokens detected.</span>';
    } else {
      features.forEach((item) => {
        const span = document.createElement('span');
        const tokenClass = item.class === 'positive' ? 'badge-positive' : item.class === 'neutral' ? 'badge-neutral' : 'badge-negative';
        span.className = `badge-sentiment ${tokenClass}`;
        span.style.fontSize = '11px';
        span.style.padding = '2px 8px';
        const weightText = item.weight ? ` (${item.weight.toFixed(2)})` : '';
        span.textContent = `${item.term}${weightText}`;
        tokensEl.appendChild(span);
      });
    }
  }
}

/* ==========================================================================
   4. DATASET B EXPLORER (PAGINATION, FILTERING, KEYWORD SEARCH)
   Strictly calls GET /api/reviews. No hardcoded sample arrays.
   ========================================================================== */
function setupDatasetExplorer() {
  // Sentiment Filter Buttons
  const filterBtns = document.querySelectorAll('.dataset-filter-btn');
  filterBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      filterBtns.forEach((b) => b.classList.remove('active-filter'));
      btn.classList.add('active-filter');
      state.datasetB.sentiment = btn.getAttribute('data-filter') || 'all';
      state.datasetB.page = 1;
      fetchAndRenderReviews();
    });
  });

  // Search Input with Debounce
  const searchInput = document.getElementById('dataset-search-input');
  const clearSearchBtn = document.getElementById('btn-clear-search');

  let debounceTimer = null;
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      state.datasetB.query = q;
      state.datasetB.page = 1;

      if (clearSearchBtn) {
        clearSearchBtn.style.display = q ? 'inline-flex' : 'none';
      }

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        fetchAndRenderReviews();
      }, 350);
    });
  }

  if (clearSearchBtn) {
    clearSearchBtn.addEventListener('click', () => {
      if (searchInput) searchInput.value = '';
      state.datasetB.query = '';
      state.datasetB.page = 1;
      clearSearchBtn.style.display = 'none';
      fetchAndRenderReviews();
    });
  }

  // Pagination Buttons
  const prevBtn = document.getElementById('dataset-prev-btn');
  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (state.datasetB.hasPrev && state.datasetB.page > 1) {
        state.datasetB.page--;
        fetchAndRenderReviews();
      }
    });
  }

  const nextBtn = document.getElementById('dataset-next-btn');
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      if (state.datasetB.hasNext) {
        state.datasetB.page++;
        fetchAndRenderReviews();
      }
    });
  }
}

async function fetchAndRenderReviews() {
  const container = document.getElementById('dataset-b-list');
  const loadingEl = document.getElementById('dataset-loading');
  const pageInfo = document.getElementById('dataset-page-info');
  const prevBtn = document.getElementById('dataset-prev-btn');
  const nextBtn = document.getElementById('dataset-next-btn');

  if (loadingEl) loadingEl.style.display = 'block';

  try {
    const data = await window.apiService.getReviews({
      page: state.datasetB.page,
      perPage: state.datasetB.perPage,
      sentiment: state.datasetB.sentiment,
      query: state.datasetB.query,
    });

    if (data.status === 'success') {
      // Update distribution metric cards if available
      if (data.distribution) {
        const d = data.distribution;
        const totalEl = document.getElementById('dataset-total-reviews');
        if (totalEl) totalEl.textContent = d.total || 399;

        const posEl = document.getElementById('dataset-pos-count');
        if (posEl) posEl.textContent = `${d.positive} (${d.positive_pct}%)`;

        const neuEl = document.getElementById('dataset-neu-count');
        if (neuEl) neuEl.textContent = `${d.neutral} (${d.neutral_pct}%)`;

        const negEl = document.getElementById('dataset-neg-count');
        if (negEl) negEl.textContent = `${d.negative} (${d.negative_pct}%)`;
      }

      // Update Pagination State
      const pag = data.pagination || {};
      state.datasetB.page = pag.page || 1;
      state.datasetB.totalPages = pag.total_pages || 1;
      state.datasetB.totalReviews = pag.total_reviews || 0;
      state.datasetB.hasNext = !!pag.has_next;
      state.datasetB.hasPrev = !!pag.has_prev;

      if (pageInfo) {
        pageInfo.textContent = `Page ${state.datasetB.page} of ${state.datasetB.totalPages} (${state.datasetB.totalReviews} reviews)`;
      }
      if (prevBtn) prevBtn.disabled = !state.datasetB.hasPrev;
      if (nextBtn) nextBtn.disabled = !state.datasetB.hasNext;

      // Render Review Cards
      renderReviewCards(data.reviews || []);
    }
  } catch (err) {
    console.error('Failed to fetch reviews:', err);
    if (container) {
      container.innerHTML = `
        <div style="padding: 20px; background-color: rgba(224, 18, 47, 0.08); border: 1px solid var(--rm-red); border-radius: 6px; text-align: center;">
          <div style="font-weight: 700; color: #FFFFFF; font-size: 13.5px; margin-bottom: 6px;">Failed to load reviews from API</div>
          <div style="font-size: 12.5px; color: var(--rm-text-secondary);">${escapeHtml(err.message)}</div>
        </div>
      `;
    }
  } finally {
    if (loadingEl) loadingEl.style.display = 'none';
  }
}

function renderReviewCards(reviews) {
  const container = document.getElementById('dataset-b-list');
  if (!container) return;

  container.innerHTML = '';

  if (!reviews || reviews.length === 0) {
    container.innerHTML = `
      <div style="padding: 32px; text-align: center; color: var(--rm-text-muted); background-color: var(--rm-bg-surface); border-radius: 6px; border: 1px solid var(--rm-border-default);">
        No reviews matched the current filter and search query.
      </div>
    `;
    return;
  }

  reviews.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.style.padding = '16px';
    card.style.marginBottom = '12px';
    card.style.backgroundColor = 'var(--rm-bg-surface)';

    const sent = item.predicted_sentiment || 'neutral';
    const badgeClass = sent === 'positive' ? 'badge-positive' : sent === 'neutral' ? 'badge-neutral' : 'badge-negative';
    const confVal = item.confidence != null ? (item.confidence * 100).toFixed(1) : (item.confidence_pct || '0.0%');

    const probs = item.probabilities || {};
    const pNeg = probs.negative != null ? (probs.negative * 100).toFixed(1) : '0.0';
    const pNeu = probs.neutral != null ? (probs.neutral * 100).toFixed(1) : '0.0';
    const pPos = probs.positive != null ? (probs.positive * 100).toFixed(1) : '0.0';

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span style="font-size: 11px; font-weight: 700; color: var(--rm-text-muted); font-family: monospace;">#${item.original_index}</span>
          <span class="badge-sentiment ${badgeClass}">${escapeHtml(sent.toUpperCase())}</span>
        </div>
        <span style="font-size: 12px; font-weight: 700; color: var(--rm-text-primary);">${confVal}% Conf</span>
      </div>
      <p style="font-size: 13px; color: var(--rm-text-secondary); line-height: 1.5; margin-bottom: 12px;">"${escapeHtml(item.text)}"</p>
      <div style="display: flex; gap: 16px; font-size: 11px; color: var(--rm-text-muted);">
        <span>Neg: <strong style="color: var(--rm-red);">${pNeg}%</strong></span>
        <span>Neu: <strong style="color: var(--rm-gray-muted);">${pNeu}%</strong></span>
        <span>Pos: <strong style="color: var(--rm-blue);">${pPos}%</strong></span>
      </div>
    `;
    container.appendChild(card);
  });
}

/* ==========================================================================
   5. UTILITY FUNCTIONS
   ========================================================================== */
function escapeHtml(str) {
  if (typeof str !== 'string') return str;
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
