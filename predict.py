import os
import re
import joblib

MODEL_PATH = "model/reviewlens_model.pkl"

CALIBRATION_DISCLAIMER = (
    "Model scores are uncalibrated softmax estimates from a linear TF-IDF classifier. "
    "Heuristic reliability levels are rule-based guidelines, not statistically calibrated probabilities."
)


class ReviewLensError(Exception):
    """Base exception class for ReviewLens AI errors."""
    pass


class ReviewLensModelError(ReviewLensError):
    """Raised when the model artifact is missing, unreadable, or invalid."""
    pass


class ReviewLensValidationError(ReviewLensError):
    """Raised when input text or aspect term fails validation."""
    pass


class ReviewLensInferenceError(ReviewLensError):
    """Raised when an unexpected error occurs during preprocessing or inference."""
    pass


def load_model(model_path=MODEL_PATH):
    """
    Safely load the trained ReviewLens model artifact.
    """
    if not os.path.exists(model_path):
        raise ReviewLensModelError(
            f"Model artifact not found at '{model_path}'. "
            "Please ensure 'model/reviewlens_model.pkl' exists."
        )

    try:
        loaded_model = joblib.load(model_path)
    except Exception as exc:
        raise ReviewLensModelError(
            f"Failed to load model artifact from '{model_path}': {exc}"
        ) from exc

    if not hasattr(loaded_model, "predict") or not hasattr(loaded_model, "predict_proba"):
        raise ReviewLensModelError(
            f"Loaded object from '{model_path}' is not a valid scikit-learn estimator."
        )

    return loaded_model


# Cached model instance for performance and backward compatibility
_cached_model = None


def get_model(model_path=MODEL_PATH):
    """Retrieve or lazily load the default model instance."""
    global _cached_model
    if _cached_model is None or model_path != MODEL_PATH:
        loaded = load_model(model_path)
        if model_path == MODEL_PATH:
            _cached_model = loaded
        return loaded
    return _cached_model


# Initialize model module attribute safely for backward compatibility
try:
    if os.path.exists(MODEL_PATH):
        model = get_model(MODEL_PATH)
    else:
        model = None
except Exception:
    model = None


def validate_input(sentence, aspect):
    """
    Validate the review sentence and aspect term.
    Ensures both inputs are non-empty strings.
    """
    if not isinstance(sentence, str):
        raise ReviewLensValidationError(
            f"Review sentence must be a string, received {type(sentence).__name__}."
        )
    if not isinstance(aspect, str):
        raise ReviewLensValidationError(
            f"Aspect term must be a string, received {type(aspect).__name__}."
        )

    clean_sentence = sentence.strip()
    clean_aspect = aspect.strip()

    if not clean_sentence:
        raise ReviewLensValidationError(
            "Review sentence cannot be empty or contain only whitespace."
        )
    if not clean_aspect:
        raise ReviewLensValidationError(
            "Aspect term cannot be empty or contain only whitespace."
        )

    return clean_sentence, clean_aspect


def check_aspect_present(sentence, aspect):
    """Case-insensitive check for whether the aspect appears in the sentence."""
    pattern = re.compile(re.escape(aspect), re.IGNORECASE)
    return bool(pattern.search(sentence))


def mark_aspect(sentence, aspect):
    """
    Mark the first occurrence of the aspect in the sentence.
    Matches the preprocessing used during model training.
    Preserves Task 2 compatibility.
    """
    pattern = re.compile(re.escape(aspect), re.IGNORECASE)
    match = pattern.search(sentence)

    if match:
        start, end = match.span()
        return (
            sentence[:start]
            + "[ASPECT] "
            + sentence[start:end]
            + " [/ASPECT]"
            + sentence[end:]
        )

    # Fallback if aspect is not found in the sentence
    return f"aspect: {aspect} sentence: {sentence}"


def assess_reliability(confidence, margin, aspect_found):
    """
    Determine heuristic prediction reliability and generate guidance.
    """
    if not aspect_found:
        return (
            "Low / Ambiguous",
            "Aspect term was not found in the review text. The fallback prefix representation was used.",
        )

    if confidence >= 0.65 and margin >= 0.25:
        return (
            "High",
            "Prediction has high class separation and dominant probability.",
        )
    elif confidence >= 0.45 and margin >= 0.12:
        return (
            "Moderate",
            "Prediction leads the second-best class with moderate separation.",
        )
    else:
        return (
            "Low / Ambiguous",
            "Narrow separation between candidate classes or low top probability. Treat with caution.",
        )


def predict_aspect_sentiment(sentence, aspect, model_path=MODEL_PATH, model_obj=None):
    """
    Predict sentiment with robust validation, aspect presence detection,
    prediction margin, and heuristic reliability guidance.
    """
    clean_sentence, clean_aspect = validate_input(sentence, aspect)

    active_model = model_obj if model_obj is not None else get_model(model_path)
    if active_model is None:
        raise ReviewLensModelError("Model is not initialized or could not be loaded.")

    aspect_found = check_aspect_present(clean_sentence, clean_aspect)
    model_text = mark_aspect(clean_sentence, clean_aspect)

    try:
        prediction = active_model.predict([model_text])[0]
        raw_probs = active_model.predict_proba([model_text])[0]
        classes = active_model.classes_
    except Exception as exc:
        raise ReviewLensInferenceError(f"Inference computation failed: {exc}") from exc

    prob_dict = {
        str(cls_name): float(prob)
        for cls_name, prob in zip(classes, raw_probs)
    }

    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_class, top_prob = sorted_probs[0]
    second_class, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else (top_class, 0.0)

    margin = float(top_prob - second_prob)
    reliability_level, guidance_reason = assess_reliability(top_prob, margin, aspect_found)

    warning = None
    if not aspect_found:
        warning = (
            f"Aspect '{clean_aspect}' was not located in the sentence text. "
            "Model evaluated using fallback prefix formatting."
        )

    return {
        "sentence": clean_sentence,
        "aspect": clean_aspect,
        "marked_text": model_text,
        "aspect_found": aspect_found,
        "prediction": str(prediction),
        "confidence": float(top_prob),
        "probabilities": prob_dict,
        "margin": margin,
        "top_classes": (top_class, second_class),
        "reliability_level": reliability_level,
        "guidance": guidance_reason,
        "warning": warning,
        "disclaimer": CALIBRATION_DISCLAIMER,
    }


def predict_sentiment(sentence, aspect):
    """
    Predict sentiment and return class probabilities.
    Maintains full backward compatibility with Task 2.
    """
    result = predict_aspect_sentiment(sentence, aspect)
    return result["prediction"], result["confidence"], result["probabilities"]


def main():
    print("=" * 60)
    print("                 ReviewLens AI (Task 3)")
    print("   Intelligent Aspect-Based Sentiment Analysis & Guidance")
    print("=" * 60)

    try:
        sentence = input("\nEnter review sentence: ")
        aspect = input("Enter aspect term: ")

        result = predict_aspect_sentiment(sentence, aspect)

        print("\n===== Prediction & Reliability =====")
        print(f"Sentence          : {result['sentence']}")
        print(f"Aspect            : {result['aspect']}")
        print(f"Aspect In Text    : {'Yes' if result['aspect_found'] else 'No (Fallback Used)'}")
        print(f"Sentiment         : {result['prediction'].upper()}")
        print(f"Confidence (Top)  : {result['confidence'] * 100:.2f}%")
        print(f"Decision Margin   : {result['margin'] * 100:.2f}% (over {result['top_classes'][1]})")
        print(f"Reliability Level : {result['reliability_level']}")
        print(f"Guidance          : {result['guidance']}")

        if result["warning"]:
            print(f"\n[Warning] {result['warning']}")

        print("\n===== Class Probabilities =====")
        for sentiment in ["positive", "negative", "neutral", "conflict"]:
            prob = result["probabilities"].get(sentiment, 0.0)
            print(f"{sentiment.capitalize():<10}: {prob * 100:.2f}%")

        print(f"\nNotice: {result['disclaimer']}")
        print("=" * 60)

    except ReviewLensValidationError as val_err:
        print(f"\n[Input Error]: {val_err}")
    except ReviewLensModelError as model_err:
        print(f"\n[Model Error]: {model_err}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as exc:
        print(f"\n[Unexpected Error]: {exc}")


if __name__ == "__main__":
    main()
