"""
ReviewLens AI — Streamlit Interactive Interface (Task 3)
Aspect-Based Sentiment Analysis with Intelligent Error Handling and Reliability Guidance.
"""

import streamlit as st
import pandas as pd
from predict import (
    predict_aspect_sentiment,
    ReviewLensValidationError,
    ReviewLensModelError,
    ReviewLensInferenceError,
    CALIBRATION_DISCLAIMER,
)

st.set_page_config(
    page_title="ReviewLens AI — Aspect Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
)

# Custom header styling
st.markdown(
    """
    <div style="padding: 1.2rem 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2.2rem; color: #1e293b;">
            🔍 ReviewLens AI
        </h1>
        <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 1.05rem;">
            Aspect-Based Sentiment Analysis with Heuristic Reliability Guidance & Error Handling
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Predefined evaluation scenarios
PRESET_SCENARIOS = {
    "— Select an example scenario —": {
        "sentence": "",
        "aspect": "",
        "note": "",
    },
    "Positive Case — 'laptop'": {
        "sentence": "the laptop is very good",
        "aspect": "laptop",
        "note": "Standard positive sentiment with dominant class probability.",
    },
    "Known Failure Case 1 — 'keyboard' (Cheap/Uncomfortable)": {
        "sentence": "The keyboard feels cheap and uncomfortable.",
        "aspect": "keyboard",
        "note": "OBSERVED FAILURE CASE: Intended sentiment is Negative, but linear TF-IDF erroneously predicts POSITIVE with a narrow margin.",
    },
    "Known Failure Case 2 — 'battery life' (Terrible/Disappointing)": {
        "sentence": "The battery life is terrible and disappointing.",
        "aspect": "battery life",
        "note": "OBSERVED FAILURE CASE: Intended sentiment is Negative, but linear TF-IDF predicts POSITIVE due to dataset imbalance and uncalibrated linear weights.",
    },
    "Neutral Case — 'Windows 7'": {
        "sentence": "The laptop came preloaded with Windows 7.",
        "aspect": "Windows 7",
        "note": "Factual statement regarding preloaded OS without explicit affective sentiment.",
    },
    "Ambiguous / Contrastive Case — 'trackpad'": {
        "sentence": "The display is vibrant and sharp, but the trackpad feels sluggish.",
        "aspect": "trackpad",
        "note": "Contrastive review containing opposing sentiments for different aspects in one sentence.",
    },
    "Missing Aspect Case — 'battery'": {
        "sentence": "Great lightweight design and very durable.",
        "aspect": "battery",
        "note": "Aspect term is missing from the review text; demonstrates validation warnings and prefix fallback.",
    },
}

# Sidebar configuration
with st.sidebar:
    st.header("📋 Preset Evaluation Examples")
    st.markdown(
        "Select a preset scenario below to evaluate correct predictions, "
        "missing-aspect handling, and genuine model failure modes."
    )

    selected_preset = st.selectbox(
        "Choose a test scenario:",
        list(PRESET_SCENARIOS.keys()),
        index=0,
    )

    if selected_preset != "— Select an example scenario —":
        preset_data = PRESET_SCENARIOS[selected_preset]
        st.session_state["input_sentence"] = preset_data["sentence"]
        st.session_state["input_aspect"] = preset_data["aspect"]
        if preset_data["note"]:
            if "OBSERVED FAILURE" in preset_data["note"]:
                st.warning(f"⚠️ {preset_data['note']}")
            else:
                st.info(f"ℹ️ {preset_data['note']}")

    st.markdown("---")
    st.subheader("💡 About ReviewLens AI")
    st.markdown(
        """
        - **Pipeline**: Scikit-Learn TF-IDF + Logistic Regression
        - **Task**: Aspect-Based Sentiment Analysis (SemEval-2014 Task 4)
        - **Classes**: `positive`, `negative`, `neutral`, `conflict`
        - **Artifact**: `model/reviewlens_model.pkl` (704 KB)
        - **Project**: SWYNEX AI Internship (Task 3)
        """
    )

# Main input layout
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Input Review & Aspect")
    default_sentence = st.session_state.get("input_sentence", "the laptop is very good")
    default_aspect = st.session_state.get("input_aspect", "laptop")

    sentence_input = st.text_area(
        "Review Sentence:",
        value=default_sentence,
        height=100,
        placeholder="e.g., The display is crisp and bright, but the battery runs out quickly.",
        help="Enter the customer review sentence to analyze.",
    )

    aspect_input = st.text_input(
        "Aspect Term:",
        value=default_aspect,
        placeholder="e.g., display or battery",
        help="Specify the target component, feature, or attribute within the review.",
    )

    analyze_clicked = st.button("🚀 Analyze Aspect Sentiment", type="primary", use_container_width=True)

with col_right:
    st.subheader("📖 Guidance & Principles")
    st.markdown(
        """
        1. **Aspect-Aware Positioning**: Identifies where the aspect term appears and wraps it with position tokens `[ASPECT] ... [/ASPECT]`.
        2. **Missing-Aspect Safeguard**: If the aspect does not exist in the review, an alert is raised and fallback prefix encoding is applied.
        3. **Decision Margin**: Measures $\Delta = P(\\text{top}_1) - P(\\text{top}_2)$ to differentiate high-certainty classifications from ambiguous splits.
        4. **Uncalibrated Heuristics**: Softmax scores reflect uncalibrated linear distances, not empirical probability certitude.
        """
    )

st.markdown("---")

# Execution and results presentation
if analyze_clicked or (sentence_input and aspect_input):
    try:
        result = predict_aspect_sentiment(sentence_input, aspect_input)

        st.subheader("📊 Sentiment Analysis Results")

        # Warning for missing aspect
        if not result["aspect_found"]:
            st.warning(f"⚠️ **Missing Aspect Warning**: {result['warning']}")

        # Highlight known failure cases
        lower_sent = sentence_input.lower()
        if "cheap and uncomfortable" in lower_sent and "keyboard" in aspect_input.lower():
            st.error(
                "🚨 **Known Model Failure Case**: The model predicted **POSITIVE** for this sentence, "
                "even though the human sentiment is clearly **NEGATIVE**. Notice the narrow decision margin "
                f"({result['margin'] * 100:.2f}%) and low top probability ({result['confidence'] * 100:.2f}%)."
            )
        elif "terrible and disappointing" in lower_sent and "battery" in aspect_input.lower():
            st.error(
                "🚨 **Known Model Failure Case**: The model predicted **POSITIVE** despite strong negative phrasing "
                "('terrible and disappointing'). This illustrates the linear model's blind spot due to TF-IDF bag-of-words "
                "and positive class imbalance in training data."
            )

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)

        sentiment_color = {
            "positive": "#16a34a",
            "negative": "#dc2626",
            "neutral": "#475569",
            "conflict": "#9333ea",
        }.get(result["prediction"], "#2563eb")

        with m1:
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">PREDICTED SENTIMENT</span>
                    <h2 style="margin: 4px 0 0 0; color: {sentiment_color}; font-size: 1.8rem; text-transform: uppercase;">
                        {result['prediction']}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m2:
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">TOP CLASS SCORE</span>
                    <h2 style="margin: 4px 0 0 0; color: #1e293b; font-size: 1.8rem;">
                        {result['confidence'] * 100:.2f}%
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">DECISION MARGIN</span>
                    <h2 style="margin: 4px 0 0 0; color: #1e293b; font-size: 1.8rem;">
                        {result['margin'] * 100:.2f}%
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m4:
            reliability_badge_color = {
                "High": "#16a34a",
                "Moderate": "#d97706",
                "Low / Ambiguous": "#dc2626",
            }.get(result["reliability_level"], "#64748b")

            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center;">
                    <span style="font-size: 0.85rem; color: #64748b; font-weight: 600;">RELIABILITY LEVEL</span>
                    <h2 style="margin: 4px 0 0 0; color: {reliability_badge_color}; font-size: 1.35rem;">
                        {result['reliability_level']}
                    </h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probabilities distribution chart
        st.subheader("📈 Class Probability Distribution")
        probs_df = pd.DataFrame(
            {
                "Sentiment Class": [c.capitalize() for c in result["probabilities"].keys()],
                "Probability (%)": [p * 100 for p in result["probabilities"].values()],
            }
        ).sort_values(by="Probability (%)", ascending=False)

        st.bar_chart(
            data=probs_df.set_index("Sentiment Class"),
            y="Probability (%)",
            height=250,
            use_container_width=True,
        )

        # Technical details and model input inspection
        with st.expander("🛠️ Inspect Preprocessed Representation & Guidance Details", expanded=False):
            st.write("**Preprocessed String Fed to Model:**")
            st.code(result["marked_text"], language="text")

            st.write("**Heuristic Reliability Assessment:**")
            st.write(f"- **Level**: `{result['reliability_level']}`")
            st.write(f"- **Rationale**: {result['guidance']}")
            st.write(f"- **Leading Class vs. Runner-Up**: `{result['top_classes'][0]}` vs. `{result['top_classes'][1]}` (margin: {result['margin'] * 100:.2f}%)")

            st.info(f"ℹ️ **Calibration Notice**: {result['disclaimer']}")

    except ReviewLensValidationError as val_err:
        st.error(f"❌ **Validation Error**: {val_err}")
    except ReviewLensModelError as model_err:
        st.error(f"❌ **Model Error**: {model_err}")
    except ReviewLensInferenceError as inf_err:
        st.error(f"❌ **Inference Error**: {inf_err}")
    except Exception as exc:
        st.error(f"❌ **Unexpected Error**: {exc}")
