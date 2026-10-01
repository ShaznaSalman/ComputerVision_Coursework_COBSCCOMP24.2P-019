"""Clinical decision pipeline agents."""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf

from core.config import AppConfig
from core.explainability import (
    compute_gradcam,
    extract_classical_cv_biomarkers,
    extract_retinal_vessels,
    find_similar_cases,
    generate_quadrant_description,
    localize_optic_disc,
    overlay_heatmap,
    segment_retinal_lesions_with_mask,
)
from core.advanced_cv import (
    analyze_attention_lesion_agreement,
    compute_retinal_severity_map,
    evaluate_prediction_evidence_consistency,
    explain_similar_cases,
)
from core.models import full_model, unet_model


class DiagnosisAgent:
    def process(self, preproc_img: np.ndarray, preset_stage: Optional[int] = None) -> Dict[str, Any]:
        if preset_stage is not None and 0 <= preset_stage < 5:
            # Fixed illustrative values for the preset demo buttons only. They are NOT model
            # output and are never used for uploaded images (see app/analysis.py).
            demo_preset_distributions = {
                0: [0.938, 0.042, 0.012, 0.005, 0.003],
                1: [0.081, 0.865, 0.041, 0.008, 0.005],
                2: [0.015, 0.062, 0.885, 0.026, 0.012],
                3: [0.005, 0.018, 0.082, 0.871, 0.024],
                4: [0.002, 0.008, 0.016, 0.053, 0.921],
            }
            probs = demo_preset_distributions[preset_stage]
            stage = preset_stage
            return {
                "stage": stage,
                "stage_name": AppConfig.CLASS_NAMES[stage],
                "confidence": float(probs[stage]),
                "probabilities": {AppConfig.CLASS_NAMES[i]: float(probs[i]) for i in range(5)},
            }

        probabilities = full_model(preproc_img[np.newaxis, ...], training=False).numpy()[0]
        stage = int(np.argmax(probabilities))
        return {"stage": stage, "stage_name": AppConfig.CLASS_NAMES[stage],
                "confidence": float(probabilities[stage]),
                "probabilities": {AppConfig.CLASS_NAMES[i]: float(probabilities[i]) for i in range(5)}}


class ExplainabilityAgent:
    def process(self, preproc_img: np.ndarray, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        stage = diagnosis["stage"]
        try:
            heatmap = compute_gradcam(preproc_img, stage)
        except (tf.errors.OpError, ValueError, RuntimeError):
            heatmap = np.zeros((7, 7), dtype=np.float32)

        quadrant_desc, quadrant_scores, peak_quadrant, peak_value = generate_quadrant_description(heatmap, stage)

        try:
            lesion_segmentation, lesion_pct, raw_mask = segment_retinal_lesions_with_mask(preproc_img, heatmap, stage)
        except (tf.errors.OpError, cv2.error, ValueError, RuntimeError):
            base = (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)
            lesion_segmentation, lesion_pct, raw_mask = base, 0.0, None

        try:
            vessel_analysis = extract_retinal_vessels(preproc_img)
        except (cv2.error, ValueError, TypeError, RuntimeError):
            vessel_analysis = {
                "vessel_mask": np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE), dtype=np.uint8),
                "vessel_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8),
                "vessel_density": 0.0,
                "vessel_status": "Unavailable",
                "method_description": "Vessel analysis unavailable; classifier output unaffected.",
            }
        try:
            optic_disc = localize_optic_disc(preproc_img)
        except (cv2.error, ValueError, TypeError, RuntimeError):
            optic_disc = {
                "optic_disc_found": False,
                "optic_disc_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8),
                "optic_disc_center": None,
                "optic_disc_score": 0.0,
                "method_description": "Optic-disc analysis unavailable; classifier output unaffected.",
            }

        h, w = AppConfig.IMG_SIZE, AppConfig.IMG_SIZE
        cam_resized = cv2.resize(heatmap, (w, h)) if (heatmap is not None and heatmap.size > 0) else np.zeros((h, w), dtype=np.float32)
        lesion_binary = (raw_mask > 0.35) & (cam_resized > 0.30) if (raw_mask is not None and stage > 0) else np.zeros((h, w), dtype=bool)

        # 1. Feature 1: Lesion-Grad-CAM Overlap Analysis (Agreement)
        try:
            overlap_analysis = analyze_attention_lesion_agreement(preproc_img, heatmap, raw_mask, stage)
        except (ValueError, cv2.error, ZeroDivisionError, RuntimeError):
            overlap_analysis = {"iou": 0.0, "dice": 0.0, "lesion_in_cam_pct": 0.0,
                                "interpretation": "Overlap analysis unavailable.", "agreement_level": "Insufficient evidence",
                                "combined_vis": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)}

        # 2. Feature 2: Retinal Severity Map (4 Anatomical Quadrants)
        try:
            severity_map = compute_retinal_severity_map(preproc_img, lesion_binary, vessel_analysis.get("vessel_mask"), heatmap)
        except (ValueError, KeyError, cv2.error, RuntimeError):
            severity_map = {"quadrants": {}, "affected_quadrants_count": 0, "total_lesion_count": 0,
                            "quadrant_overlay": (np.clip(preproc_img, 0.0, 1.0) * 255).astype(np.uint8)}

        similar_cases = find_similar_cases(preproc_img)
        cbr_explanation = explain_similar_cases(similar_cases, stage)

        return {
            "overlay_cam": overlay_heatmap(preproc_img, heatmap),
            "lesion_seg": lesion_segmentation,
            "lesion_pct": lesion_pct,
            "raw_mask": raw_mask,
            "lesion_binary": lesion_binary,
            "quadrant_desc": quadrant_desc,
            "quadrant_scores": quadrant_scores,
            "peak_quadrant": peak_quadrant,
            "peak_val": peak_value,
            "similar_cases": similar_cases,
            "cbr_explanation": cbr_explanation,
            "classical_cv": extract_classical_cv_biomarkers(preproc_img),
            "vessel_method_description": vessel_analysis.get("method_description", "Exploratory classical-CV vessel analysis."),
            "optic_disc_method_description": optic_disc.get("method_description", "Exploratory optic-disc heuristic."),
            "overlap_analysis": overlap_analysis,
            "severity_map": severity_map,
            "affected_quadrants_count": severity_map.get("affected_quadrants_count", 0),
            **vessel_analysis,
            **optic_disc,
        }


# Patient-reported red-flag symptoms (second Governance rule). Keys are stable identifiers; the
# labels are what the app shows. Illustrative, not clinically validated.
RED_FLAG_SYMPTOMS = {
    "vision_loss": "Sudden loss or major drop in vision",
    "curtain": "A curtain or shadow over part of the vision",
    "floaters": "A sudden shower of new floaters or flashes of light",
    "pain_redness": "Eye pain with redness",
}
RED_FLAG_WARNING = ("These symptoms can signal retinal detachment, vitreous haemorrhage or another emergency "
                    "that a photograph cannot rule out.")
URGENT_OUTCOME = "URGENT — seek same-day eye care"


def normalise_red_flags(red_flags) -> List[str]:
    """Return the ticked red-flag labels in a fixed order; accepts keys or labels, ignores anything else."""
    if not red_flags:
        return []
    if isinstance(red_flags, str):
        red_flags = [red_flags]
    ticked = {str(flag).strip() for flag in red_flags}
    return [label for key, label in RED_FLAG_SYMPTOMS.items() if key in ticked or label in ticked]


class AdvisoryAgent:
    GUIDANCE = {
        0: ("Routine Screening", "No diabetic microvascular abnormalities observed. Recommend annual dilated retinal examination and continued glycemic management (HbA1c < 7.0%).", "12 Months"),
        1: ("Non-Urgent Clinical Monitoring", "Mild NPDR (isolated microaneurysms). Primary care management: optimize blood pressure, cholesterol, and glycemic control.", "6–9 Months"),
        2: ("Comprehensive Specialist Referral", "Moderate NPDR (dot/blot hemorrhages, hard exudates). Significant risk of macular edema; schedule dilated examination and optical coherence tomography (OCT).", "3–6 Months"),
        3: ("Urgent Specialist Evaluation", "Severe NPDR (fulfills '4-2-1 rule'). High progression risk to proliferative retinopathy. Immediate ophthalmologist evaluation required.", "2–4 Weeks"),
        4: ("EMERGENCY Vitreoretinal Intervention", "Proliferative DR (active neovascularization, vitreous hemorrhage). Immediate retina specialist referral for panretinal photocoagulation (PRP) or intravitreal anti-VEGF therapy.", "24–48 Hours"),
    }
    DISCLAIMER = "CLINICAL DISCLAIMER: RetinaTrace AI is an investigational decision-support tool. It does not replace independent clinical judgment or formal diagnostic verification by a licensed ophthalmologist."

    # Plain-language texts for the Patient view (about a 12-year-old reading level).
    # Illustrative, not clinically validated.
    PATIENT_TEXT = {
        "stage": {
            0: ("No signs of diabetic eye damage were found in your photo. That is good news. "
                "Keeping your blood sugar and blood pressure under control helps keep your eyes healthy."),
            1: ("Your photo shows very small, early changes caused by diabetes. They do not usually hurt your sight yet. "
                "Good control of your blood sugar and blood pressure can stop them getting worse."),
            2: ("Your photo shows some damage to the tiny blood vessels at the back of your eye. "
                "Your sight may still be fine, but an eye specialist needs to keep a close watch on it."),
            3: ("Your photo shows a lot of damage to the blood vessels at the back of your eye. "
                "This is serious. An eye specialist should see you soon to help protect your sight."),
            4: ("Your photo shows signs of the most serious stage, where new, weak blood vessels can grow and bleed. "
                "This can harm your sight quickly, so you need to see an eye specialist very soon."),
        },
        "next_step": {
            "routine": "Next step: have another eye check in {followup}.",
            "soon": "Next step: see an eye specialist within {followup}.",
        },
        "flag": {
            "red_flag": ("You told us about symptoms that can be an emergency. Please get eye care today. "
                         "Do not wait because of this result."),
            "confidence": "The computer wasn't sure enough, so an eye specialist needs to look at your photo.",
            "quality": ("The photo wasn't clear enough for the computer to read, so an eye specialist needs to "
                        "look at it. You may need a new photo."),
        },
        "demo": "These are fixed demo values, not a result from your photo.",
        "reminder": "This is a research prototype, not a medical device. Always follow your eye doctor's advice.",
    }

    def process(self, stage: int) -> Dict[str, str]:
        urgency, plan, followup = self.GUIDANCE.get(stage, ("Unknown", "Manual ophthalmological review mandatory.", "Immediate"))
        return {"urgency": urgency, "plan": plan, "followup": followup, "disclaimer": self.DISCLAIMER}

    def patient_summary(self, stage: int, reasons: Optional[List[str]] = None,
                        demo_preset: bool = False) -> Dict[str, Any]:
        """Plain-language result for the Patient view.

        When a red flag, the confidence gate or the quality check fires, the summary says so first
        and gives no stage advice.
        """
        texts = self.PATIENT_TEXT
        fired = [r for r in ("red_flag", "confidence", "quality") if r in (reasons or [])]
        if fired:
            return {"flagged": True, "lines": [texts["flag"][r] for r in fired],
                    "next_step": "", "reminder": texts["reminder"]}
        followup = self.GUIDANCE[stage][2].lower()
        kind = "soon" if stage >= 3 else "routine"
        lines = [texts["stage"][stage]] + ([texts["demo"]] if demo_preset else [])
        return {"flagged": False, "lines": lines,
                "next_step": texts["next_step"][kind].format(followup=followup),
                "reminder": texts["reminder"]}


REASON_TEXT = {
    "quality": "the image-quality check failed",
    "evidence": "the exploratory visual-evidence check did not support the predicted stage",
}


def governance_message(reasons, confidence: float, threshold: float, red_flags: Optional[List[str]] = None) -> str:
    """One sentence naming the real reason(s) a case was flagged, or confirming it passed."""
    if not reasons:
        return (f"Safety Verified: Model confidence ({confidence*100:.1f}%) satisfies the clinical safety "
                f"threshold ({threshold*100:.0f}%) and the image-quality check passed.")
    parts = []
    for reason in reasons:
        if reason == "red_flag":
            parts.append("the patient reported red-flag symptoms (" + "; ".join(red_flags or []) + ")")
        elif reason == "confidence":
            parts.append(f"model confidence ({confidence*100:.1f}%) is below the threshold ({threshold*100:.0f}%)")
        else:
            parts.append(REASON_TEXT[reason])
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    if "red_flag" in reasons:
        return (f"{URGENT_OUTCOME}: flagged because {joined}. {RED_FLAG_WARNING} Automated treatment guidance "
                "has been withheld to avoid acting on an uncertain result.")
    if "confidence" in reasons or "quality" in reasons:
        return (f"Flagged for human review because {joined}. Automated treatment guidance has been withheld "
                "to avoid acting on an uncertain result.")
    return f"Clinician review recommended because {joined}."


class GovernanceAgent:
    def __init__(self, threshold: float = AppConfig.DEFAULT_CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate(
        self,
        diagnosis: Dict[str, Any],
        explanation: Dict[str, Any],
        advisory: Dict[str, str],
        qc: Optional[Dict[str, Any]] = None,
        red_flags=None,
    ) -> Dict[str, Any]:
        confidence = diagnosis["confidence"]
        qc = qc or {"passed": True}
        flags = normalise_red_flags(red_flags)

        # Feature 4: Prediction-Evidence Consistency Analysis
        consistency_analysis = evaluate_prediction_evidence_consistency(
            diagnosis=diagnosis,
            lesion_pct=explanation.get("lesion_pct", 0.0),
            affected_quadrants=explanation.get("affected_quadrants_count", 0),
            overlap_data=explanation.get("overlap_analysis", {}),
            qc=qc,
        )

        # The actual reason(s) for flagging, so every message states what really happened.
        # Red flags only escalate: they never change the predicted stage in `diagnosis`.
        reasons = []
        if flags:
            reasons.append("red_flag")
        if confidence < self.threshold:
            reasons.append("confidence")
        if not qc.get("passed", True):
            reasons.append("quality")
        if consistency_analysis.get("status") == "POTENTIALLY INCONSISTENT":
            reasons.append("evidence")
        urgent = bool(flags)
        # Advice is withheld for red flags, low confidence or a failed quality check.
        flagged = urgent or "confidence" in reasons or "quality" in reasons
        message = governance_message(reasons, confidence, self.threshold, flags)

        if urgent:
            controlled_advisory = {"urgency": URGENT_OUTCOME, "plan": message,
                                   "followup": "Same day — do not wait for a routine appointment",
                                   "disclaimer": advisory["disclaimer"]}
        elif flagged:
            controlled_advisory = {"urgency": "HUMAN SPECIALIST TRIAGE MANDATORY", "plan": message,
                                   "followup": "Withheld — Manual Slit-Lamp Examination Required Immediately",
                                   "disclaimer": advisory["disclaimer"]}
        else:
            controlled_advisory = advisory

        if urgent:
            outcome = URGENT_OUTCOME
        elif flagged:
            outcome = "FLAGGED FOR HUMAN REVIEW"
        elif reasons:
            outcome = "CLINICIAN REVIEW RECOMMENDED"
        else:
            outcome = "AUTOMATION APPROVED"

        return {
            "flagged": flagged,
            "flagged_for_review": bool(reasons),
            "flag_reasons": reasons,
            "urgent": urgent,
            "red_flags": flags,
            "outcome": outcome,
            "confidence": confidence,
            "message": message,
            "diagnosis": diagnosis,
            "explanation": explanation,
            "advisory": controlled_advisory,
            "consistency_analysis": consistency_analysis,
        }


def run_pipeline(
    preproc_img: np.ndarray,
    threshold: float = AppConfig.DEFAULT_CONFIDENCE_THRESHOLD,
    qc: Optional[Dict[str, Any]] = None,
    preset_stage: Optional[int] = None,
    red_flags=None,
) -> Dict[str, Any]:
    diagnosis = DiagnosisAgent().process(preproc_img, preset_stage=preset_stage)
    explanation = ExplainabilityAgent().process(preproc_img, diagnosis)
    advisory = AdvisoryAgent().process(diagnosis["stage"])
    return GovernanceAgent(threshold).evaluate(diagnosis, explanation, advisory, qc=qc, red_flags=red_flags)
