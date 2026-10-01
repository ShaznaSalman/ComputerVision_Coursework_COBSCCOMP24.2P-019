"""Unit tests for the app logic that does not need model weights (chatbot, triage, reports, metrics loader)."""

import json
import os
import tempfile
import unittest

from app import chatbot, reports, triage


class TestReportedMetricsLoader(unittest.TestCase):
    """The app must only show test metrics that come from a saved notebook run."""

    def _write(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_missing_file_gives_no_numbers(self):
        self.assertIsNone(reports.load_reported_test_metrics("does_not_exist.csv"))

    def test_valid_file_is_read(self):
        path = self._write("argmax_qwk,argmax_accuracy\n0.8123,0.6789\n")
        self.assertEqual(reports.load_reported_test_metrics(path), {"qwk": 0.8123, "accuracy": 0.6789})

    def test_malformed_file_is_ignored(self):
        self.assertIsNone(reports.load_reported_test_metrics(self._write("x,y\n1,2\n")))

    def test_description_uses_the_saved_numbers(self):
        text = reports.describe_reported_metrics({"qwk": 0.8123, "accuracy": 0.6789})
        self.assertIn("0.812", text)
        self.assertIn("67.9%", text)


class TestChatbot(unittest.TestCase):
    """Routing, safety refusals and knowledge-base content."""

    def test_greeting(self):
        self.assertIn("Clinical Knowledge Assistant", chatbot.run_chat_pipeline("hello"))

    def test_treatment_questions_are_refused(self):
        reply = chatbot.run_chat_pipeline("should I start treatment")
        self.assertIn("cannot recommend", reply)

    def test_medicine_dose_questions_are_refused(self):
        for question in ("What dose of insulin should I take?", "how much metformin should I take"):
            self.assertIn("cannot recommend", chatbot.run_chat_pipeline(question), question)

    def test_long_question_is_not_fuzzy_matched_to_a_definition(self):
        reply = chatbot.run_chat_pipeline("What is the 4-2-1 rule in diabetic retinopathy?")
        self.assertIn("4-2-1 rule", reply)

    def test_definition_still_matches_with_a_typo(self):
        reply = chatbot.run_chat_pipeline("what is diabetic retinopaty")
        self.assertIn("damage to the retinal blood vessels", reply)

    def test_dataset_answer_describes_aptos(self):
        reply = chatbot.run_chat_pipeline("tell me about the dataset in detail")
        self.assertIn("APTOS 2019", reply)
        self.assertIn("3,662", reply)
        self.assertNotIn("38,034", reply)

    def test_no_hard_coded_performance_claims(self):
        reply = chatbot.run_chat_pipeline("what is qwk in detail")
        self.assertNotIn("0.842", reply)

    def test_knowledge_answers_carry_the_clinical_disclaimer(self):
        reply = chatbot.run_chat_pipeline("what is the 4-2-1 rule in detail")
        self.assertIn("4-2-1", reply)
        self.assertIn("qualified ophthalmologist", reply)

    def test_demo_preset_results_are_labelled(self):
        context = {"stage_name": "Moderate", "confidence": 0.885, "peak_quadrant": "-", "lesion_pct": 0.0,
                   "demo_preset": True}
        reply = chatbot.run_chat_pipeline("explain why this image was classified", context)
        self.assertTrue(reply.startswith("⚠️ *The current result is a demo preset"))

    def test_real_predictions_are_not_labelled_as_demo(self):
        context = {"stage_name": "Moderate", "confidence": 0.7, "peak_quadrant": "-", "lesion_pct": 0.0,
                   "demo_preset": False}
        reply = chatbot.run_chat_pipeline("explain why this image was classified", context)
        self.assertFalse(reply.startswith("⚠️"))

    def test_empty_message_changes_nothing(self):
        self.assertEqual(chatbot.respond_to_clinical_query("   ", []), ([], ""))


class TestTriageSimulator(unittest.TestCase):
    """The simulator must stay labelled as illustrative and keep its risk within bounds."""

    def test_disclaimer_and_neutral_wording(self):
        html, ticket = triage.calculate_multimodal_risk(3, 9.0, 15, 60, 150, "Type 1")
        self.assertIn("Illustrative only — not clinically validated. Do not use for patient decisions.", html)
        for banned in ("RetinaRisk", "Dispatch", "Mandatory"):
            self.assertNotIn(banned, html + ticket)
        self.assertIn("EXAMPLE REFERRAL SUMMARY (ILLUSTRATIVE)", ticket)
        self.assertIn("Illustrative only, not a validated clinical model", ticket)

    def test_risk_is_clamped(self):
        for args in ((0, 5.0, 0, 18, 90, "Type 2"), (4, 14.0, 40, 90, 220, "Type 1")):
            html, _ = triage.calculate_multimodal_risk(*args)
            percent = float(html.split('font-weight="900"')[1].split(">")[1].split("%")[0])
            self.assertGreaterEqual(percent, 1.5)
            self.assertLessEqual(percent, 98.5)

    def test_routing_uses_top_probability_stage(self):
        probabilities = {"No DR": 0.1, "Mild": 0.1, "Moderate": 0.1, "Severe": 0.1, "Proliferative DR": 0.6}
        _, ticket = triage.update_triage_routing(probabilities, 7.5, 10, 55, 135, "Type 2")
        self.assertIn("Stage 4", ticket)


class TestReports(unittest.TestCase):
    """EHR note and JSON report content."""

    def _note(self, demo_preset: bool) -> str:
        diag = {"stage": 2, "stage_name": "Moderate", "confidence": 0.8}
        expl = {"quadrant_desc": "**Superior-Temporal**", "vessel_density": 1.0, "affected_quadrants_count": 1}
        adv = {"urgency": "Routine", "followup": "6 months", "plan": "Review"}
        return reports.build_ehr_note(diag, expl, adv, {"status": "CONSISTENT"}, {"iou": 0.1}, False, 0.7,
                                      demo_preset=demo_preset)

    def test_ehr_note_marks_demo_presets_only(self):
        self.assertIn("DEMO PRESET", self._note(True))
        self.assertNotIn("DEMO PRESET", self._note(False))

    def test_json_report(self):
        path = reports.generate_report_json("Moderate", 0.8, {"Moderate": 0.8, "Mild": 0.2},
                                            "Routine", "6 months", "Review", "note text")
        self.addCleanup(os.remove, path)
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)["retinatrace_report"]
        self.assertEqual(report["diagnosis"]["confidence_pct"], 80.0)
        self.assertIn("disclaimer", report)

    def test_json_report_needs_a_diagnosis(self):
        self.assertIsNone(reports.generate_report_json("", 0.0, {}, "", "", "", ""))


if __name__ == "__main__":
    unittest.main()
