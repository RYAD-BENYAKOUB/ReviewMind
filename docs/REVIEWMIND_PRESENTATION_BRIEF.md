# ReviewMind — Academic Presentation Brief & Defense Knowledge Base

**Course:** Fouille de Données et Management des Systèmes (FDMS)  
**Curriculum:** Master 2 SID (Systèmes d'Information et Décision)  
**Chapter:** Fouille de Données Textuelles (*Text Mining*)  
**Project:** ReviewMind — Hospitality Sentiment Analysis Platform  
**Author:** Benyakoub Mohammed Ryad  
**Academic Year:** 2026  
**Format:** Oral Defense & Academic Slide Deck Companion  

---

## 1. 2-Minute Executive Pitch

> *"ReviewMind est une plateforme open-source de fouille de données textuelles et de classification de sentiments appliquée au secteur hôtelier. Face à un déséquilibre de classe massif où 74% des avis sont positifs, les modèles probabilistes classiques comme le Naive Bayes Multinomial échouent totalement sur les avis neutres (F1 de 0.0000). ReviewMind démontre comment une formulation bayésienne à a priori uniformes combinée à un lissage de Lidstone adapté ($\alpha=0.1$) résout cet effondrement et porte le Macro F1 à 0.6762. Au-delà des algorithmes, le projet intègre une évaluation sur un corpus externe indépendant (éco-hôtel), un protocole d'annotation humaine en double aveugle, et une architecture full-stack découplée (API Flask + dashboard web) validée par 19 tests unitaires automatisés."*

---

## 2. Project Context & Problem Statement

### 2.1 The Academic Setting
- **Course**: Master 2 SID — *Fouille de Données et Management des Systèmes (FDMS)*.
- **Pedagogical Goal**: Illustrate the complete Text Mining lifecycle, from raw document acquisition to high-dimensional vectorization, probabilistic classification, metric selection under class imbalance, out-of-domain transfer, and production deployment.

### 2.2 The Real-World Data Mining Challenge
1. **Severe Imbalance in Hospitality**: Customers predominantly write reviews when they are delighted or outraged. Moderate, balanced, or neutral experiences are rare ($10.66\%$).
2. **The "Majority-Class Collapse" in Generative NLP**: In Multinomial Naive Bayes, multiplying term likelihoods by empirical class frequencies $P(c)$ heavily penalizes the minority class. Because neutral reviews naturally share vocabulary with positive (*"clean"*, *"good"*) and negative (*"small"*, *"noisy"*) reviews, their likelihood never overcomes the $7:1$ prior penalty.
3. **The Generalization Trap**: Models trained on broad aggregators (TripAdvisor) struggle when transferred to niche establishments (e.g., luxury eco-resorts) due to specialized vocabulary and shifts in customer expectations.

---

## 3. Dataset Snapshot (Numbers to Memorize)

| Attribute | Dataset A (TripAdvisor) | Dataset B (Eco-hotel / Calheiros et al., 2017) |
|---|---|---|
| **Role** | Training, Cross-Validation & Benchmark | External Zero-Shot Domain Generalization |
| **Raw Records** | $20\,491$ reviews | $401$ reviews |
| **Cleaned Records** | $20\,491$ reviews ($0$ duplicates, $0$ nulls) | $399$ unique reviews ($2$ duplicates removed) |
| **Columns** | `Review` (text), `Rating` ($1$–$5$), `Sentiment` | `Review` (text) only |
| **Target Labels** | $1$–$2 \to$ `negative`, $3 \to$ `neutral`, $4$–$5 \to$ `positive` | **None** (unsupervised raw text corpus) |
| **Class Breakdown** | Pos: $15\,093$ ($73.66\%$) \| Neg: $3\,214$ ($15.68\%$) \| Neu: $2\,184$ ($10.66\%$) | Inferred: Pos $84.0\%$ \| Neu $8.3\%$ \| Neg $7.8\%$ |
| **Parsing Strategy** | UTF-8 CSV | Latin-1, TSV (`sep='\t'`) |
| **Ground Truth Status** | Verified via customer star ratings | **Pending** (requires human consensus) |

---

## 4. Methodology & Text Mining Pipeline

```
Raw Text (CSV/TSV)
        │
        ▼
Data Preparation ───────► Whitespace stripping, null check, rating mapping
        │
        ▼
Stratified Split ───────► 80% Train (16,392) / 20% Test (4,099) [random_state=42]
        │
        ▼
TF-IDF Vectorizer ──────► Unigrams + Bigrams, Top 20,000 features, L2 norm
        │                 (Fitted STRICTLY on Train split to prevent data leakage)
        ▼
Multinomial Naive Bayes ─► Alpha=0.1 (Lidstone), fit_prior=False (Uniform priors)
        │
        ▼
Production Training ────► Retrained on 100% of Dataset A (20,491 records)
        │
        ▼
Inference & Evaluation ──► External scoring on Dataset B + Double-Blind Annotation
```

### Key Technical Details to Highlight:
- **No Stemming / Lemmatization**: Words retain their natural surface forms to avoid over-stemming errors.
- **Stop Words Retained (`stop_words=None`)**: Essential for capturing contrastive and negative bigrams (*"not good"*, *"never again"*).
- **Linear TF + Smooth IDF**: Scikit-learn defaults (`sublinear_tf=False`, `smooth_idf=True`, $L_2$ document normalization).

---

## 5. The Central Scientific Breakthrough: Neutral Class Recovery

### The Mathematical Mechanism
In Naive Bayes, classification in log-space follows:
$$\hat{c} = \arg\max_{c} \left[ \ln P(c) + \sum_{i=1}^n x_i \ln P(w_i \mid c) \right]$$

- **With Empirical Priors (`fit_prior=True`, Model 1)**:
  $$\ln P(\text{positive}) = \ln(0.7366) \approx -0.31$$
  $$\ln P(\text{neutral}) = \ln(0.1066) \approx -2.24$$
  The neutral class begins with a massive deficit of $-1.93$ in log space. **Result: Zero neutral reviews predicted.**
- **With Uniform Priors (`fit_prior=False`, Models 2 & 3)**:
  $$\ln P(\text{positive}) = \ln P(\text{neutral}) = \ln P(\text{negative}) = -\ln(3) \approx -1.10$$
  The prior terms cancel out entirely. Classification depends solely on vocabulary evidence.
- **With Reduced Smoothing ($\alpha = 0.1$, Model 3)**:
  Denominator penalty in $\frac{N_{c,w} + \alpha}{N_c + \alpha |V|}$ shrinks tenfold ($2\,000$ vs $20\,000$). Word probabilities become sharper and more discriminative, allowing specific neutral expressions (*"average"*, *"standard"*, *"okay"*) to trigger neutral classification.

---

## 6. Verified Benchmark Results (Table to Memorize)

Evaluated on $4\,099$ unseen test reviews from Dataset A:

| Metric | Model 1 (Baseline) | Model 2 (Uniform Prior) | Model 3 (Champion / Production) |
|---|:---:|:---:|:---:|
| **Smoothing ($\alpha$)** | $1.0$ (Laplace) | $1.0$ (Laplace) | **$0.1$ (Lidstone)** |
| **Prior (`fit_prior`)** | `True` (Empirical) | `False` (Uniform) | **`False` (Uniform)** |
| **Overall Accuracy** | $81.97\%$ | **$85.17\%$** | $80.12\%$ |
| **Macro $F_1$-Score** | $0.5249$ | $0.6153$ | **$0.6762$ (Peak)** |
| **Negative $F_1$** | $0.6802$ | **$0.7730$** | $0.7454$ |
| **Neutral Recall** | $0.00\%$ | $8.47\%$ | **$48.74\%$** |
| **Neutral $F_1$-Score** | **$0.0000$ (Collapsed)** | $0.1477$ | **$0.3883$ (Recovered)** |
| **Positive $F_1$** | $0.8945$ | **$0.9253$** | $0.8949$ |
| **Neutral Predictions** | **$0$** | $64$ | **$660$** |

### Why Model 3 is Superior Despite Slightly Lower Accuracy:
Accuracy on an imbalanced dataset is deceptive: predicting only the majority class guarantees high accuracy while producing an operational failure. **Macro $F_1$ is the only statistically honest metric** because it assigns equal weight ($1/3$) to all classes. Model 3 improves Macro $F_1$ by **$+0.1513$** over the baseline.

---

## 7. External Evaluation & Domain Shift Insights

### What Happened on Dataset B (Eco-hotel, 399 Reviews)?
- **Model Inferences**: Positive: $83.96\%$ ($335$), Neutral: $8.27\%$ ($33$), Negative: $7.77\%$ ($31$).
- **The Hidden Phenomenon**: Average confidence dropped to **$57.48\%$**. More than half of the reviews (**$57.64\%$ / $230$ reviews**) were predicted with less than $60\%$ confidence!
- **Why? Domain Shift**:
  - TripAdvisor vocabulary: *room, hotel, staff, clean, breakfast, location, price, wifi*.
  - Eco-resort vocabulary: *nature, sustainable, architecture, bio-pool, duck feathers, herbal tea, fire pits*.
- **Scientific Integrity Stance**:
  *"We do not report accuracy or F1 on Dataset B because there are no ratings or ground-truth labels. Doing so would violate basic scientific methodology."*

---

## 8. Full-Stack Architecture & Live Demo Scenario

### Architecture Highlights
- **Backend**: Flask 3.1 REST API with 5 blueprints (`health`, `predict`, `datasets`, `reviews`, `evaluation`) and 3 singleton services (`ModelService`, `DatasetService`, `EvaluationService`).
- **Frontend**: Responsive single-page dashboard adhering to the **Dark Intelligence** palette (`#0B0B0F` 75%, `#1E6BFF` 15%, `#E0122F` 10%).
- **Strict Decoupling**: Zero client-side heuristic fallbacks; all probabilities and salient n-grams are served live by the Python backend.

### Live Demo Scenario (Walkthrough for Defense)
1. **Open Dashboard**: Show active green status dot: `API Online (Model 3 Active)`.
2. **Review Input**: Type an ambiguous review:
   > *"The staff was exceptionally friendly and the bed was comfortable, but the room was quite small and breakfast was mediocre."*
3. **Execution**: Click **Classify Sentiment**.
4. **Explain Results**:
   - Class predicted: `NEUTRAL`.
   - Confidence: $\approx 46\%$.
   - Probabilities: Negative $\approx 25\%$, Neutral $\approx 46\%$, Positive $\approx 29\%$.
   - Salient Features: Show highlighted n-grams (*"exceptionally friendly"*, *"mediocre"*, *"quite small"*).
   - Point out that under Model 1, this review would have been forced into Positive due to empirical prior bias.

---

## 9. Key Text Mining Concepts to Explain to the Jury

1. **TF-IDF Weighting**: Explaining why raw word counts fail and how inverse document frequency penalizes ubiquitous non-informative words.
2. **N-Grams vs. Unigrams**: Why bigrams (*"not good"*, *"never again"*) are essential to capture localized sentiment reversal.
3. **Generative vs. Discriminative Models**: Naive Bayes models the joint probability $P(x, c) = P(c)P(x|c)$, making it fast and interpretable, but vulnerable to skewed class priors.
4. **Data Leakage**: Why fitting the TF-IDF vectorizer on the full dataset before splitting invalidates experimental conclusions.
5. **Macro vs. Micro/Weighted Metrics**: Why Macro $F_1$ is mandatory when assessing performance on imbalanced minority classes.
6. **Inter-Annotator Agreement (Cohen's $\kappa$)**: How chance-corrected agreement provides mathematical proof of ground-truth quality.

---

## 10. Anticipated Questions from the Professor & Model Answers

### Question 1: *"Why did you use Multinomial Naive Bayes instead of a Transformer like BERT or RoBERTa?"*
> **Answer**:  
> *"In a Text Mining coursework context, our primary objective was to thoroughly study the statistical mechanics of generative NLP models and class imbalance dynamics. Multinomial Naive Bayes provides complete transparency: we can trace exactly how prior probabilities $\ln P(c)$ and term likelihoods $\ln P(w_i|c)$ interact to cause minority class collapse. Furthermore, MNB trains in 13 seconds on CPU and provides microsecond inference latency with a 2 MB footprint, making it ideal for high-throughput, resource-constrained production environments. Deep learning transformers like RoBERTa or DeBERTa are our natural planned extension for future work to handle complex discourse structure."*

---

### Question 2: *"Why did your accuracy decrease from 85.17% (Model 2) to 80.12% (Model 3) if Model 3 is your best model?"*
> **Answer**:  
> *"This illustrates the classic 'Accuracy Paradox' in imbalanced classification. In our dataset, 74% of reviews are positive. A model can achieve high accuracy simply by predicting positive almost every time. Model 2 achieves 85% accuracy because it is overly conservative on neutral reviews (only 8.5% recall). Model 3, by using Lidstone smoothing $\alpha=0.1$, actively predicts the neutral class, lifting neutral recall from 8.5% to 48.7%. While this causes some inevitable false positives that slightly reduce accuracy, it increases the Macro F1-score from 0.6153 to 0.6762. In NLP, Macro F1 is the authoritative criterion because all three customer sentiment states matter equally to business decision-makers."*

---

### Question 3: *"Why did you retain stop words (`stop_words=None`) in your TF-IDF vectorizer? Isn't it standard to remove them?"*
> **Answer**:  
> *"In sentiment analysis, removing standard English stop words is dangerous because it eliminates crucial negation words such as 'not', 'no', 'never', and degree adverbs like 'very' or 'too'. Because our vectorizer uses unigrams and bigrams (`ngram_range=(1, 2)`), keeping stop words allows the model to learn critical valence-shifting bigrams like 'not clean', 'never again', or 'not recommend'. If we had removed stop words, 'not clean' would have collapsed into 'clean', directly inverting the review's polarity."*

---

### Question 4: *"Can you report accuracy or F1 on Dataset B (Eco-hotel) right now?"*
> **Answer**:  
> *"No, absolutely not. Reporting supervised metrics on Dataset B at this stage would be scientifically fraudulent. Dataset B contains only raw review texts without star ratings or verified human labels. What our model generates on Dataset B are model inferences, not ground-truth comparisons. To establish legitimate benchmark metrics, we designed a double-blind annotation protocol with two independent annotators and Cohen's Kappa consensus. Only when `annotation_consensus.csv` is completed will supervised metrics be calculated."*

---

### Question 5: *"What does the 57% low-confidence rate on Dataset B tell us from a data science perspective?"*
> **Answer**:  
> *"It provides quantifiable evidence of domain shift. The vocabulary of urban TripAdvisor hotels focuses on standard amenities (front desk, metro, wifi, elevator). Dataset B comes from a luxury boutique eco-resort (Areias do Seixo in Portugal), where reviews employ poetic vocabulary about nature, bio-pools, and sustainable architecture. When a bag-of-words model encounters out-of-vocabulary terms, its posterior probabilities flatten toward uniform distribution, resulting in low confidence scores. This demonstrates the critical importance of out-of-domain validation before deploying NLP pipelines in production."*

---

### Question 6: *"Why did you use ratings 1–2 for negative, 3 for neutral, and 4–5 for positive? Isn't that arbitrary?"*
> **Answer**:  
> *"This 3-class discretization is standard in hospitality literature (e.g., Wang et al., 2011; Hu & Liu, 2004). 1 and 2 stars represent explicit dissatisfaction, 4 and 5 stars represent satisfaction and recommendation, while 3 stars represent average, mixed, or lukewarm experiences. However, we also explicitly documented this as an academic limitation in our audit: customer star ratings carry subjective bias, as some guests write critical reviews while giving 4 stars, or conversely give 3 stars despite enjoying their stay. That is precisely why human textual annotation is superior to star ratings."*

---

## 11. Academic Limitations to Acknowledge Honestly

1. **Bag-of-Words Independence Assumption**: MNB cannot capture syntax, long-range dependencies, or sarcasm (*"The room was so clean that cockroaches were wearing slippers"*).
2. **Subjectivity of Rating-Derived Labels**: Star ratings are imperfect proxies for pure textual sentiment.
3. **Out-of-Vocabulary Vulnerability**: Unseen terms in external datasets cannot contribute to Bayesian evidence.
4. **Pending Annotation Consensus**: External quantitative benchmark remains pending completion of dual-annotator adjudication.

---

## 12. Checklist for the Oral Defense

- [x] Flask backend running locally (`python backend/app.py`).
- [x] Dashboard accessible at `http://127.0.0.1:5000/web/index.html`.
- [x] Unit test execution demonstrated (`python -m unittest discover tests` $\to$ 19 OK).
- [x] Know the 3 model Macro F1 numbers by heart ($0.5249 \to 0.6153 \to 0.6762$).
- [x] Know the neutral class recovery numbers ($0.0000 \to 0.3883$ F1, $0 \to 660$ predictions).
- [x] Know the Dataset A ($20\,491$) and Dataset B ($399$ unique) dimensions.
- [x] Emphasize the scientific integrity principle: zero fabricated external metrics.
- [x] Point to the public GitHub repository: `https://github.com/RYAD-BENYAKOUB/ReviewMind`.

