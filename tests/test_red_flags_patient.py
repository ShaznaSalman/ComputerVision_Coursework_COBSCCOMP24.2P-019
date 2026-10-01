"""Red-flag symptoms, patient ID/eye and the Patient view (needs checkpoints/ for the end-to-end tests)."""

import os
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

from app import chatbot
from app.analysis import analyze_fundus, build_governance_html, build_patient_card_html
from app.patient import clean_patient_id, view_visibility
from app.reports import generate_document_pdf, generate_full_report_pdf
from app.triage import update_triage_routing
from core.agents import (AdvisoryAgent, GovernanceAgent, RED_FLAG_SYMPTOMS, RED_FLAG_WARNING,
                         URGENT_OUTCOME)
from core.config import AppConfig

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = sorted((ROOT / "app" / "samples").glob("stage*.jpg"))
GOV_STATUS, HERO, PROBS, EHR, CONTEXT, HISTORY_STATE, RESEARCH, CARD = 0, 1, 2, 14, 18, 19, 20, 30


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def evaluate(confidence=0.95, qc_passed=True, red_flags=None, stage=2):
    diagnosis = {"stage": stage, "stage_name": AppConfig.CLASS_NAMES[stage], "confidence": confidence,
                 "probabilities": {n: 0.2 for n in AppConfig.CLASS_NAMES}}
    explanation = {"lesion_pct": 3.0, "affected_quadrants_count": 3,
                   "overlap_analysis": {"dice": 0.5, "lesion_in_cam_pct": 60.0}}
    advisory = AdvisoryAgent().process(stage)
    qc = {"passed": qc_passed, "issues": [] if qc_passed else ["Severe blur detected"]}
    return GovernanceAgent(0.70).evaluate(diagnosis, explanation, advisory, qc=qc, red_flags=red_flags)


class TestRedFlagRule(unittest.TestCase):
    def test_each_red_flag_forces_urgent_at_high_confidence(self):
        for key, label in RED_FLAG_SYMPTOMS.items():
            for given in (key, label):
                result = evaluate(confidence=0.97, red_flags=[given])
                self.assertTrue(result["urgent"], given)
                self.assertEqual(result["outcome"], URGENT_OUTCOME)
                self.assertEqual(result["flag_reasons"], ["red_flag"])
                self.assertTrue(result["flagged"])  # automated plan withheld
                self.assertEqual(result["advisory"]["urgency"], URGENT_OUTCOME)
                self.assertNotIn("Moderate NPDR (dot/blot", result["advisory"]["plan"])
                self.assertIn(label, result["message"])
                html = build_governance_html(result, 0.70)
                self.assertIn(URGENT_OUTCOME, html)
                self.assertIn(label, html)
                self.assertIn(RED_FLAG_WARNING, html)

    def test_no_flags_leaves_behaviour_unchanged(self):
        for flags in (None, [], ["not a real symptom"]):
            result = evaluate(confidence=0.95, red_flags=flags)
            self.assertFalse(result["urgent"])
            self.assertEqual(result["flag_reasons"], [])
            self.assertEqual(result["outcome"], "AUTOMATION APPROVED")
            self.assertEqual(result["advisory"], AdvisoryAgent().process(2))
        low = evaluate(confidence=0.5, red_flags=[])
        self.assertEqual(low["flag_reasons"], ["confidence"])
        self.assertEqual(low["outcome"], "FLAGGED FOR HUMAN REVIEW")

    def test_combined_reasons_are_all_listed(self):
        flag = RED_FLAG_SYMPTOMS["curtain"]
        result = evaluate(confidence=0.40, qc_passed=False, red_flags=[flag])
        self.assertEqual(result["flag_reasons"], ["red_flag", "confidence", "quality"])
        self.assertEqual(result["outcome"], URGENT_OUTCOME)
        html = build_governance_html(result, 0.70)
        for text in (flag, "below your threshold", "Image quality check failed", RED_FLAG_WARNING):
            self.assertIn(text, html)
        for text in (flag, "below the threshold", "image-quality check failed"):
            self.assertIn(text, result["message"])

    def test_flags_never_change_the_stage(self):
        flags = list(RED_FLAG_SYMPTOMS.values())
        for path in SAMPLES:
            plain = analyze_fundus(load_rgb(path), 0.70, [])
            flagged = analyze_fundus(load_rgb(path), 0.70, [], flags)
            self.assertEqual(plain[CONTEXT]["stage"], flagged[CONTEXT]["stage"], path.name)
            self.assertEqual(plain[PROBS], flagged[PROBS], path.name)
            self.assertTrue(flagged[CONTEXT]["urgent"])
            self.assertIn(URGENT_OUTCOME, flagged[GOV_STATUS])

    def test_invalid_image_shows_only_the_validity_message(self):
        out = analyze_fundus(np.zeros((512, 512, 3), np.uint8), 0.70, [], list(RED_FLAG_SYMPTOMS.values()))
        self.assertIn("Not a usable fundus photograph", out[GOV_STATUS])
        self.assertNotIn("URGENT", out[GOV_STATUS])
        self.assertEqual(out[HERO], "")

    def test_red_flag_reaches_ehr_history_and_chatbot(self):
        flag = RED_FLAG_SYMPTOMS["vision_loss"]
        out = analyze_fundus(load_rgb(SAMPLES[0]), 0.70, [], [flag], "DEMO-0001", "Right eye (OD)")
        self.assertIn(URGENT_OUTCOME, out[EHR])
        self.assertIn(flag, out[EHR])
        self.assertEqual(out[HISTORY_STATE][-1]["urgency"], URGENT_OUTCOME)
        reply = chatbot.run_chat_pipeline("why was this urgent?", pred_context=out[CONTEXT])
        self.assertIn(flag, reply)
        self.assertIn("same-day", reply)


class TestPatientId(unittest.TestCase):
    def test_blank_and_spaces(self):
        for raw in (None, "", "   "):
            self.assertEqual(clean_patient_id(raw), "Not provided")
        self.assertEqual(clean_patient_id("  DEMO 0001 "), "DEMO0001")

    def test_long_id_is_cut_to_40(self):
        self.assertEqual(clean_patient_id("A" * 60), "A" * 40)

    def test_odd_characters_are_removed(self):
        self.assertEqual(clean_patient_id("<script>../😀"), "script")
        self.assertEqual(clean_patient_id("MRN_12-ab"), "MRN_12-ab")
        self.assertEqual(clean_patient_id("../../😀"), "Not provided")


class TestPatientView(unittest.TestCase):
    def test_card_for_each_stage(self):
        agent = AdvisoryAgent()
        for stage in range(5):
            summary = agent.patient_summary(stage, [])
            self.assertFalse(summary["flagged"])
            self.assertEqual(summary["lines"][0], agent.PATIENT_TEXT["stage"][stage])
            self.assertIn(agent.GUIDANCE[stage][2].lower(), summary["next_step"])
            html = build_patient_card_html(summary)
            self.assertIn("What this means for you", html)
            self.assertIn("research prototype", html)

    def test_card_for_gate_and_red_flag_cases(self):
        agent = AdvisoryAgent()
        for reasons, phrase in ((["confidence"], "wasn't sure enough"), (["quality"], "wasn't clear enough"),
                                (["red_flag"], "emergency"), (["red_flag", "confidence"], "emergency")):
            summary = agent.patient_summary(4, reasons)
            self.assertTrue(summary["flagged"])
            self.assertIn(phrase, summary["lines"][0])
            self.assertEqual(summary["next_step"], "")  # no stage advice
            self.assertNotIn(agent.PATIENT_TEXT["stage"][4], " ".join(summary["lines"]))

    def test_end_to_end_card_and_view_switch(self):
        out = analyze_fundus(load_rgb(SAMPLES[2]), 0.95, [])  # 95% threshold: gate fires
        self.assertIn("wasn't sure enough", out[CARD])
        self.assertEqual(view_visibility("Patient"),
                         {"patient_card": True, "probabilities": False, "explainability": False, "retrieval": False})
        self.assertEqual(view_visibility("Clinician"),
                         {"patient_card": False, "probabilities": True, "explainability": True, "retrieval": True})


class TestDocumentHeaders(unittest.TestCase):
    """Patient ID and eye must appear in the text of all three PDFs."""

    def setUp(self):
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            self.skipTest("pymupdf is needed to read PDF text")

    def _pdf_text(self, path):
        import pymupdf
        with pymupdf.open(path) as doc:
            return " ".join(page.get_text() for page in doc)

    def test_id_eye_and_red_flag_in_all_pdfs(self):
        flag = RED_FLAG_SYMPTOMS["floaters"]
        out = analyze_fundus(load_rgb(SAMPLES[1]), 0.70, [], [flag], " DEMO-0001 ", "Right eye (OD)")
        context, probs = out[CONTEXT], out[PROBS]
        _, referral_text = update_triage_routing(probs, 7.5, 10, 55, 135, "Type 2", context)
        pdfs = {
            "EHR note": generate_document_pdf("RetinaTrace Clinical Session Note", out[EHR]),
            "referral": generate_document_pdf("RetinaTrace Example Referral Summary (Illustrative)", referral_text),
            "full report": generate_full_report_pdf(out[24], out[25], probs, out[26], out[27], out[28], out[EHR],
                                                    out[RESEARCH], load_rgb(SAMPLES[1]), out[3], out[4], context),
        }
        for name, path in pdfs.items():
            text = " ".join(self._pdf_text(path).split())
            self.assertIn("DEMO-0001", text, name)
            self.assertIn("Right eye (OD)", text, name)
            self.assertIn("model prediction", text, name)
            self.assertIn("URGENT", text, name)
            os.remove(path)

    def test_blank_id_prints_not_provided(self):
        out = analyze_fundus(load_rgb(SAMPLES[0]), 0.70, [], [], "", "Not specified")
        self.assertIn("Patient ID / MRN        : Not provided", out[EHR])
        self.assertIn("Eye                     : Not specified", out[EHR])


if __name__ == "__main__":
    unittest.main()
