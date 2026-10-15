# ReviewLens AI — Aspect-Based Sentiment Analysis

ReviewLens AI is an aspect-based sentiment analysis (ABSA) system designed to predict sentiment polarity toward specific aspects in laptop review sentences. Developed as part of the SWYNEX AI Internship (Tasks 2 & 3), it combines a scikit-learn machine-learning pipeline with intelligent validation, heuristic reliability guidance, automated testing, and an interactive Streamlit application.

---

## Task 3 Intelligent Feature Enhancements

Task 3 upgrades the baseline prototype into a robust, guided sentiment analysis system:

1. **Robust Input Validation & Error Handling**:
   - Validates that sentences and aspect terms are non-empty, non-whitespace strings (`ReviewLensValidationError`).
   - Safely verifies model artifact existence and compatibility before inference (`ReviewLensModelError`).
   - Wraps runtime and inference failures cleanly without exposing raw system tracebacks (`ReviewLensInferenceError`).
2. **Missing-Aspect Safeguard**:
   - Performs case-insensitive matching to verify if the aspect appears in the review sentence.
   - If the aspect is missing, the system activates the fallback prefix representation (`aspect: {aspect} sentence: {sentence}`), issues an explicit warning, and flags the output with `aspect_found=False`.
3. **Decision Margin & Heuristic Reliability Guidance**:
   - Calculates the decision margin between the top two predicted classes:
     $$\Delta = P(\text{top}_1) - P(\text{top}_2)$$
   - Assigns a heuristic **Reliability Level**:
     - **High**: Top probability $\ge 65\%$, margin $\ge 25\%$, and aspect is located in text.
     - **Moderate**: Top probability $\ge 45\%$, margin $\ge 12\%$, and aspect is located in text.
     - **Low / Ambiguous**: Margin $< 12\%$, top probability $< 45\%$, or aspect is missing.
   - Displays clear documentation that model scores are uncalibrated softmax estimates and heuristic reliability levels are not statistically validated confidence estimates.
4. **Interactive Streamlit Web Interface (`app.py`)**:
   - Clean, professional UI for entering custom reviews and aspect terms.
   - One-click preset scenarios demonstrating positive, neutral, ambiguous, missing-aspect, and known failure cases.
   - Metric cards displaying predicted polarity, top score, decision margin, and reliability badge.
   - Class probability distribution bar chart covering all four polarities.
5. **Automated Unit & Integration Test Suite (`tests/test_reviewlens.py`)**:
   - 23 automated tests covering input validation, aspect detection, error handling, probability constraints, and backward compatibility.

---

## Architecture & Preprocessing

- **Aspect-Aware Positioning**: Marks the target aspect in the text with position markers:
  `sentence[:start] + "[ASPECT] " + aspect + " [/ASPECT]" + sentence[end:]`
- **Fallback Representation**: When an aspect is absent:
  `aspect: {aspect} sentence: {sentence}`
- **Feature Pipeline**:
  - `TfidfVectorizer`: Unigrams and bigrams (`ngram_range=(1, 2)`), sublinear TF scaling, 10,000 feature vocabulary limit.
  - `LogisticRegression`: Multi-class classifier with `class_weight="balanced"` and `max_iter=2000`.
- **Target Polarities**: `positive`, `negative`, `neutral`, and `conflict`.

---

## Evaluation Results & Failure Analysis

The model was evaluated on a stratified held-out test split of 472 SemEval-2014 Task 4 laptop review samples:

| Metric | Result |
|---|---:|
| Test Samples | 472 |
| Overall Accuracy | 71.82% |
| Macro F1-Score | 55.87% |

### Class Support Breakdown
- **Conflict**: 9 samples (severely underrepresented)
- **Negative**: 173 samples
- **Neutral**: 92 samples
- **Positive**: 198 samples

### Observed Predictions & Failure Cases
Detailed empirical evaluation cases documented in [`examples/predictions.md`](examples/predictions.md):
- **Positive Example**: *"the laptop is very good"* (`laptop`) $\rightarrow$ **POSITIVE (84.00%)**, margin: `75.54%` (High Reliability).
- **Known Failure 1 (Keyboard)**: *"The keyboard feels cheap and uncomfortable."* (`keyboard`) $\rightarrow$ **POSITIVE (42.33%)** [Incorrect; expected Negative]. Margin is only `9.70%`, correctly triggering a **Low / Ambiguous** reliability alert.
- **Known Failure 2 (Battery Life)**: *"The battery life is terrible and disappointing."* (`battery life`) $\rightarrow$ **POSITIVE (50.89%)** [Incorrect; expected Negative]. Caused by training set class imbalance and bag-of-words limitations.
- **Neutral Example**: *"The laptop came preloaded with Windows 7."* (`Windows 7`) $\rightarrow$ **NEUTRAL (33.91%)**, margin: `0.61%` (Low / Ambiguous).
- **Missing-Aspect Safeguard**: *"Great lightweight design and very durable."* (`battery`) $\rightarrow$ **POSITIVE (79.04%)**, flagged with missing aspect warning and downgraded reliability.

---

## Setup & Execution

### 1. Environment Setup

Requires Python 3.9 or newer.

```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/Jyo-08/SWYNEX-Model-or-API-Integration.git
cd SWYNEX-Model-or-API-Integration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web App

Launch the Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### 3. Run the Command-Line Interface (CLI)

For terminal-based inference with full reliability guidance:

```bash
python predict.py
```

### 4. Run the Automated Test Suite

Execute the built-in test suite:

```bash
python -m unittest discover tests
```

To run with verbose test reporting:

```bash
python -m unittest -v tests/test_reviewlens.py
```

---

## Retraining & Dataset Provenance

To retrain the model:
1. Place the dataset at `data/Laptop_Train_v2.csv` after obtaining it from an authorized source.
2. Run the training script:

```bash
python train.py
```

> **Dataset Provenance Caveat**: The raw dataset CSV is excluded from version control via `.gitignore` because redistribution permissions have not been verified. Retraining will overwrite `model/reviewlens_model.pkl`.

---

## Project Directory Tree

```
swynex/
├── .gitignore               # Git exclusions (.venv, data/*.csv, caches, etc.)
├── README.md                # Comprehensive documentation, setup, and evaluation
├── requirements.txt         # Core dependencies (pandas, scikit-learn, joblib, streamlit)
├── app.py                   # Streamlit web interface with preset failure & success cases
├── predict.py               # Core inference engine with validation and reliability guidance
├── train.py                 # Pipeline training and evaluation script (manual execution)
├── model/
│   └── reviewlens_model.pkl # Pre-trained ReviewLens AI scikit-learn model artifact
├── examples/
│   └── predictions.md       # Empirical evaluation examples and failure analysis
├── tests/
│   └── test_reviewlens.py   # 23 automated unit and integration tests
└── data/                    # Dataset directory (raw CSV excluded from Git)
```

---

## Limitations

- **Heuristic Reliability**: Reliability levels and decision margins are rule-based heuristics designed to detect ambiguity; they are not calibrated Bayesian probabilities.
- **Bag-of-Words Representation**: TF-IDF n-grams cannot model syntactic dependencies or parse contrastive clauses (*"X is great, but Y is terrible"*).
- **Domain Specificity**: The model vocabulary is tailored to laptop customer reviews and may not generalize to other consumer electronics or domains.
- **Offline & Self-Contained**: Operates fully offline without external API keys or cloud credentials.

---

## References

- SemEval-2014 Task 4: Aspect Based Sentiment Analysis: https://alt.qcri.org/semeval2014/task4/
- Pontiki et al., *SemEval-2014 Task 4: Aspect Based Sentiment Analysis*: https://aclanthology.org/S14-2004/
