# Observed Prototype Predictions & Evaluation — ReviewLens AI

These outputs were observed during local empirical testing using the pre-trained ReviewLens AI model artifact (`model/reviewlens_model.pkl`). All numerical scores and class probability distributions are authentic outputs from the model.

---

## 1. Positive Prediction (High Confidence & Wide Margin)

- **Input Sentence**: `the laptop is very good`
- **Aspect Term**: `laptop`
- **Aspect Located in Text**: Yes (`the [ASPECT] laptop [/ASPECT] is very good`)
- **Predicted Sentiment**: **POSITIVE**
- **Top Probability**: **84.00%**
- **Second-Best Class**: Negative (8.46%)
- **Decision Margin**: **75.54%**
- **Heuristic Reliability Level**: **High**
- **Full Class Distribution**:
  - Positive: `84.00%`
  - Negative: `8.46%`
  - Neutral: `4.45%`
  - Conflict: `3.09%`
- **Observation**: Clear positive terminology with dominant class probability and wide decision separation.

---

## 2. Misclassified Negative Example — Keyboard (Observed Model Failure)

- **Input Sentence**: `The keyboard feels cheap and uncomfortable.`
- **Aspect Term**: `keyboard`
- **Aspect Located in Text**: Yes (`The [ASPECT] keyboard [/ASPECT] feels cheap and uncomfortable.`)
- **Predicted Sentiment**: **POSITIVE** *(Incorrect — Model Failure)*
- **Expected Human Sentiment**: **Negative**
- **Top Probability**: **42.33%**
- **Second-Best Class**: Negative (32.63%)
- **Decision Margin**: **9.70%**
- **Heuristic Reliability Level**: **Low / Ambiguous**
- **Full Class Distribution**:
  - Positive: `42.33%`
  - Negative: `32.63%`
  - Neutral: `18.18%`
  - Conflict: `6.86%`
- **Observation & Root Cause**: The linear TF-IDF model misclassifies this negative review as positive. However, the decision margin is only `9.70%` and top probability is below `45%`. The Task 3 intelligent reliability heuristic flags this as **Low / Ambiguous**, warning downstream consumers that the classification is uncertain.

---

## 3. Misclassified Negative Example — Battery Life (Observed Model Failure)

- **Input Sentence**: `The battery life is terrible and disappointing.`
- **Aspect Term**: `battery life`
- **Aspect Located in Text**: Yes (`The [ASPECT] battery life [/ASPECT] is terrible and disappointing.`)
- **Predicted Sentiment**: **POSITIVE** *(Incorrect — Model Failure)*
- **Expected Human Sentiment**: **Negative**
- **Top Probability**: **50.89%**
- **Second-Best Class**: Negative (29.18%)
- **Decision Margin**: **21.71%**
- **Heuristic Reliability Level**: **Moderate**
- **Full Class Distribution**:
  - Positive: `50.89%`
  - Negative: `29.18%`
  - Neutral: `11.95%`
  - Conflict: `7.98%`
- **Observation & Root Cause**: Despite explicitly negative words ("terrible", "disappointing"), positive class prevalence in the training set and sparse bigram interactions cause an erroneous positive prediction. This underscores the necessity of not treating raw softmax scores as calibrated certainty.

---

## 4. Neutral Review Example

- **Input Sentence**: `The laptop came preloaded with Windows 7.`
- **Aspect Term**: `Windows 7`
- **Aspect Located in Text**: Yes
- **Predicted Sentiment**: **NEUTRAL**
- **Top Probability**: **33.91%**
- **Second-Best Class**: Negative (33.30%)
- **Decision Margin**: **0.61%**
- **Heuristic Reliability Level**: **Low / Ambiguous**
- **Full Class Distribution**:
  - Neutral: `33.91%`
  - Negative: `33.30%`
  - Positive: `29.66%`
  - Conflict: `3.13%`
- **Observation**: The model correctly predicts Neutral as the leading category for a factual statement, but the separation over Negative is minimal (`0.61%`), triggering the ambiguity safeguard.

---

## 5. Contrastive / Ambiguous Review Example

- **Input Sentence**: `The display is vibrant and sharp, but the trackpad feels sluggish.`
- **Aspect Term**: `trackpad`
- **Aspect Located in Text**: Yes
- **Predicted Sentiment**: **POSITIVE** *(Model Limitation)*
- **Expected Sentiment for Trackpad**: **Negative**
- **Top Probability**: **46.40%**
- **Second-Best Class**: Negative (24.18%)
- **Decision Margin**: **22.22%**
- **Full Class Distribution**:
  - Positive: `46.40%`
  - Negative: `24.18%`
  - Conflict: `15.03%`
  - Neutral: `14.39%`
- **Observation**: Bag-of-words representations do not model clause structures or syntactic dependency parsing. The strongly positive adjectives in the first clause ("vibrant", "sharp") overwhelm the negative qualifier ("sluggish") associated with the trackpad.

---

## 6. Missing-Aspect Safeguard Example

- **Input Sentence**: `Great lightweight design and very durable.`
- **Aspect Term**: `battery`
- **Aspect Located in Text**: **No** (`aspect_found=False`)
- **Fallback Representation Applied**: `aspect: battery sentence: Great lightweight design and very durable.`
- **Warning Raised**: `Aspect 'battery' was not located in the sentence text. Model evaluated using fallback prefix formatting.`
- **Predicted Sentiment**: **POSITIVE** (Top: `79.04%`, Margin: `66.83%`)
- **Heuristic Reliability Level**: **Low / Ambiguous** *(Automatic downgrade due to missing aspect)*
- **Observation**: When an aspect is absent, the model evaluates global sentence sentiment rather than aspect-specific sentiment. The system explicitly alerts the user and downgrades reliability regardless of raw score magnitude.

---

## Summary of Heuristic Guidance Principles

1. **Uncalibrated Softmax**: Logistic regression probability estimates are monotonic transformations of decision boundary distances, not true empirical likelihoods.
2. **Decision Margin ($\Delta$)**: The gap between the first and second most probable classes ($P(\text{top}_1) - P(\text{top}_2)$) provides a vital indicator of classification ambiguity.
3. **Aspect Safeguards**: Absence of an aspect term in the review sentence triggers automatic warnings and reliability downgrades.
