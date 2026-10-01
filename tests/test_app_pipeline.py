"""Tests of the analysis pipeline that need the model weights (checkpoints/ must be present)."""

import json
import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from app.analysis import analyze_fundus, assess_fundus_validity, build_governance_html, create_sample_fundus
from core.agents import GovernanceAgent
from core.config import AppConfig
from core.models import full_model
from core.preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = sorted((ROOT / "app" / "samples").glob("stage*.jpg"))
GOV_STATUS, HERO, PROBS, RESEARCH = 0, 1, 2, 20  # positions in analyze_fundus' output tuple


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class TestUploadsAlwaysUseTheModel(unittest.TestCase):
    """Uploaded images, even the exact sample files, must never get fixed preset values."""

    def test_sample_files_are_real_predictions(self):
        self.assertEqual(len(SAMPLES), 5)
        for path in SAMPLES:
            out = analyze_fundus(load_rgb(path), 0.70, [])
            self.assertEqual(out[RESEARCH]["result_source"], "model_prediction", path.name)
            self.assertNotIn("DEMO PRESET", out[HERO], path.name)
            expected = full_model(preprocess_image(load_rgb(path))[None], training=False).numpy()[0]
            got = np.array([out[PROBS][name] for name in AppConfig.CLASS_NAMES])
            np.testing.assert_allclose(got, expected, atol=1e-5, err_msg=path.name)

    def test_cached_sample_arrays_are_real_predictions(self):
        # create_sample_fundus returns exactly what a preset button loads into the upload box.
        for stage in range(5):
            out = analyze_fundus(create_sample_fundus(stage), 0.70, [])
            self.assertEqual(out[RESEARCH]["result_source"], "model_prediction")


class TestPresetButtons(unittest.TestCase):
    def test_preset_results_are_labelled(self):
        out = analyze_fundus(create_sample_fundus(2), 0.70, [], preset_stage=2)
        self.assertEqual(out[RESEARCH]["result_source"], "demo_preset_fixed_values")
        self.assertIn("DEMO PRESET", out[HERO])
        self.assertIn("FIXED DEMO VALUE", out[HERO])

    def test_preset_buttons_say_preset_demo(self):
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        self.assertIn("PRESET DEMOS (fixed illustrative values, not model output)", source)
        for label in ("Normal", "Moderate", "Proliferative"):
            self.assertIn(f"{label} — preset demo", source)


class TestGovernanceReasons(unittest.TestCase):
    """The banner must name the real reason: confidence, image quality, or both."""

    def _evaluate(self, confidence, qc_passed):
        diagnosis = {"stage": 2, "stage_name": "Moderate", "confidence": confidence,
                     "probabilities": {n: 0.2 for n in AppConfig.CLASS_NAMES}}
        explanation = {"lesion_pct": 3.0, "affected_quadrants_count": 3,
                       "overlap_analysis": {"dice": 0.5, "lesion_in_cam_pct": 60.0}}
        advisory = {"urgency": "u", "plan": "p", "followup": "f", "disclaimer": "d"}
        qc = {"passed": qc_passed, "issues": [] if qc_passed else ["Severe blur detected"]}
        return GovernanceAgent(0.70).evaluate(diagnosis, explanation, advisory, qc=qc)

    def test_low_confidence_only(self):
        result = self._evaluate(0.55, True)
        self.assertEqual(result["flag_reasons"], ["confidence"])
        html = build_governance_html(result, 0.70)
        self.assertIn("below your threshold", html)
        self.assertIn("55.0%", html)
        self.assertNotIn("quality check failed", html)

    def test_quality_failure_only(self):
        result = self._evaluate(0.965, False)
        self.assertEqual(result["flag_reasons"], ["quality"])
        self.assertTrue(result["flagged"])
        html = build_governance_html(result, 0.70)
        self.assertIn("Image quality check failed", html)
        self.assertNotIn("below your threshold", html)
        self.assertNotIn("96.5%", html)  # the confidence is not presented as the reason
        self.assertIn("image-quality check failed", result["message"])

    def test_both_reasons(self):
        result = self._evaluate(0.40, False)
        self.assertEqual(result["flag_reasons"], ["confidence", "quality"])
        html = build_governance_html(result, 0.70)
        self.assertIn("below your threshold", html)
        self.assertIn("Image quality check failed", html)

    def test_no_hallucination_wording(self):
        for confidence, qc_passed in ((0.55, True), (0.965, False)):
            result = self._evaluate(confidence, qc_passed)
            text = result["message"] + build_governance_html(result, 0.70)
            self.assertNotIn("hallucination", text.lower())
            self.assertIn("to avoid acting on an uncertain result", text)

    def test_high_confidence_card_hidden_when_flagged(self):
        blurred = cv2.GaussianBlur(load_rgb(SAMPLES[2]), (0, 0), 12)
        out = analyze_fundus(blurred, 0.50, [])
        self.assertIn("ISSUES DETECTED", out[21])
        self.assertNotIn("HIGH CONFIDENCE", out[21])
        self.assertIn("Image quality check failed", out[GOV_STATUS])


class TestValidityGate(unittest.TestCase):
    def test_non_fundus_images_are_rejected(self):
        rng = np.random.default_rng(0)
        for name, img in {
            "black": np.zeros((512, 512, 3), np.uint8),
            "white": np.full((512, 512, 3), 255, np.uint8),
            "noise": rng.integers(0, 256, (512, 512, 3), dtype=np.uint8),
        }.items():
            self.assertFalse(assess_fundus_validity(img)["valid"], name)
            out = analyze_fundus(img, 0.70, [])
            self.assertIn("Not a usable fundus photograph", out[GOV_STATUS], name)
            self.assertEqual(out[HERO], "", name)
            self.assertEqual(out[PROBS], {}, name)

    def test_sample_images_are_accepted(self):
        for path in SAMPLES:
            self.assertTrue(assess_fundus_validity(load_rgb(path))["valid"], path.name)


class TestWeightsNoneMatchesImagenetBuild(unittest.TestCase):
    """Building the backbone with weights=None must give the same probabilities as before."""

    def test_sample_probabilities_unchanged(self):
        fixture = json.loads((ROOT / "tests" / "fixtures" / "sample_probabilities.json").read_text())
        for path in SAMPLES:
            got = full_model(preprocess_image(str(path))[None], training=False).numpy()[0]
            np.testing.assert_allclose(got, fixture["probabilities"][path.name], atol=1e-5, err_msg=path.name)

    def test_backbone_built_without_imagenet_download(self):
        source = (ROOT / "core" / "models.py").read_text(encoding="utf-8")
        self.assertTrue(re.search(r"EfficientNetB3\(\s*include_top=False,\s*weights=None", source))


if __name__ == "__main__":
    unittest.main()
