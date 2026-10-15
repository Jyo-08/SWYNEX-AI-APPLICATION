"""
Automated unit and integration tests for ReviewLens AI (Task 3).
Tests input validation, aspect detection, error handling,
margin calculations, and backward compatibility.
"""

import os
import tempfile
import unittest
import joblib

from predict import (
    MODEL_PATH,
    ReviewLensModelError,
    ReviewLensValidationError,
    ReviewLensInferenceError,
    validate_input,
    check_aspect_present,
    mark_aspect,
    assess_reliability,
    load_model,
    predict_aspect_sentiment,
    predict_sentiment,
)


class TestInputValidation(unittest.TestCase):
    """Test input validation for sentences and aspect terms."""

    def test_empty_sentence_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input("", "laptop")

    def test_whitespace_sentence_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input("   \t\n  ", "keyboard")

    def test_empty_aspect_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input("The screen is bright", "")

    def test_whitespace_aspect_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input("The screen is bright", "   ")

    def test_non_string_sentence_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input(12345, "battery")

    def test_non_string_aspect_raises_error(self):
        with self.assertRaises(ReviewLensValidationError):
            validate_input("The laptop is fast", None)

    def test_valid_input_strips_whitespace(self):
        clean_s, clean_a = validate_input("  The battery life is decent.  ", "  battery life  ")
        self.assertEqual(clean_s, "The battery life is decent.")
        self.assertEqual(clean_a, "battery life")


class TestAspectDetectionAndPreprocessing(unittest.TestCase):
    """Test aspect presence detection and marking."""

    def test_aspect_present_case_insensitive(self):
        self.assertTrue(check_aspect_present("The Battery Life is great", "battery life"))
        self.assertTrue(check_aspect_present("the KEYBOARD feels nice", "keyboard"))

    def test_aspect_absent(self):
        self.assertFalse(check_aspect_present("The screen is vibrant", "touchpad"))

    def test_mark_aspect_when_present(self):
        marked = mark_aspect("The screen is clear", "screen")
        self.assertEqual(marked, "The [ASPECT] screen [/ASPECT] is clear")

    def test_mark_aspect_when_absent(self):
        marked = mark_aspect("Fast delivery", "screen")
        self.assertEqual(marked, "aspect: screen sentence: Fast delivery")


class TestModelLoadingAndErrorHandling(unittest.TestCase):
    """Test robust error handling for missing and corrupted model files."""

    def test_missing_model_file_raises_model_error(self):
        with self.assertRaises(ReviewLensModelError) as ctx:
            load_model("model/non_existent_file_12345.pkl")
        self.assertIn("Model artifact not found", str(ctx.exception))

    def test_corrupted_model_file_raises_model_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"NOT_A_VALID_PICKLE_CONTENT")
            tmp_path = tmp.name

        try:
            with self.assertRaises(ReviewLensModelError) as ctx:
                load_model(tmp_path)
            self.assertIn("Failed to load model artifact", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_invalid_estimator_object_raises_model_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump({"not": "an_estimator"}, tmp.name)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ReviewLensModelError) as ctx:
                load_model(tmp_path)
            self.assertIn("not a valid scikit-learn estimator", str(ctx.exception))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestInferenceAndPredictionStructure(unittest.TestCase):
    """Test inference outputs, probability bounds, and return structures."""

    @classmethod
    def setUpClass(cls):
        cls.has_model = os.path.exists(MODEL_PATH)

    def test_actual_model_prediction_structure(self):
        if not self.has_model:
            self.skipTest("Trained model artifact not found.")

        result = predict_aspect_sentiment("the laptop is very good", "laptop")

        self.assertIn("sentence", result)
        self.assertIn("aspect", result)
        self.assertIn("marked_text", result)
        self.assertIn("aspect_found", result)
        self.assertIn("prediction", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("margin", result)
        self.assertIn("reliability_level", result)
        self.assertIn("guidance", result)
        self.assertIn("disclaimer", result)

        self.assertTrue(result["aspect_found"])
        self.assertIn(result["prediction"], ["positive", "negative", "neutral", "conflict"])

    def test_probabilities_sum_to_one(self):
        if not self.has_model:
            self.skipTest("Trained model artifact not found.")

        result = predict_aspect_sentiment("The keyboard feels cheap and uncomfortable.", "keyboard")
        probs = result["probabilities"]
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_missing_aspect_warning_and_fallback(self):
        if not self.has_model:
            self.skipTest("Trained model artifact not found.")

        result = predict_aspect_sentiment("Great lightweight design and very durable.", "battery")
        self.assertFalse(result["aspect_found"])
        self.assertIsNotNone(result["warning"])
        self.assertEqual(result["reliability_level"], "Low / Ambiguous")

    def test_backward_compatibility_predict_sentiment(self):
        if not self.has_model:
            self.skipTest("Trained model artifact not found.")

        prediction, confidence, probs = predict_sentiment("the laptop is very good", "laptop")
        self.assertIsInstance(prediction, str)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(probs, dict)
        self.assertEqual(len(probs), 4)
        self.assertEqual(prediction, "positive")


class TestReliabilityAndGuidanceHeuristics(unittest.TestCase):
    """Test decision margin and heuristic reliability assessment rules."""

    def test_high_reliability_criteria(self):
        level, reason = assess_reliability(confidence=0.80, margin=0.30, aspect_found=True)
        self.assertEqual(level, "High")

    def test_moderate_reliability_criteria(self):
        level, reason = assess_reliability(confidence=0.52, margin=0.15, aspect_found=True)
        self.assertEqual(level, "Moderate")

    def test_low_reliability_due_to_narrow_margin(self):
        level, reason = assess_reliability(confidence=0.42, margin=0.08, aspect_found=True)
        self.assertEqual(level, "Low / Ambiguous")

    def test_low_reliability_when_aspect_missing(self):
        level, reason = assess_reliability(confidence=0.85, margin=0.50, aspect_found=False)
        self.assertEqual(level, "Low / Ambiguous")


class TestInferenceFailureHandling(unittest.TestCase):
    """Test handling of unexpected inference exceptions."""

    class BrokenEstimator:
        classes_ = ["positive", "negative", "neutral", "conflict"]

        def predict(self, X):
            raise RuntimeError("Internal numerical solver failed.")

        def predict_proba(self, X):
            raise RuntimeError("Internal numerical solver failed.")

    def test_broken_estimator_raises_inference_error(self):
        broken = self.BrokenEstimator()
        with self.assertRaises(ReviewLensInferenceError) as ctx:
            predict_aspect_sentiment("Valid review", "aspect", model_obj=broken)
        self.assertIn("Inference computation failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
