"""Patient identification, document headers and the Clinician/Patient view switch.

Illustrative, not clinically validated: the prototype stores nothing; IDs exist only in the
current session and in documents the user downloads. Do not enter real patient data.
"""

import datetime as _dt
import re
from typing import Any, Dict, List, Optional

PATIENT_ID_PLACEHOLDER = "e.g. DEMO-0001 (do not enter real patient data)"
PATIENT_ID_MAX_LENGTH = 40
NOT_PROVIDED = "Not provided"
EYE_CHOICES = ["Right eye (OD)", "Left eye (OS)", "Not specified"]
EYE_DEFAULT = "Not specified"
VIEW_CHOICES = ["Clinician", "Patient"]


def clean_patient_id(raw: Optional[str]) -> str:
    """Keep only letters, digits, '-' and '_' (no spaces), at most 40 characters; blank -> 'Not provided'."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", str(raw or ""))[:PATIENT_ID_MAX_LENGTH]
    return cleaned or NOT_PROVIDED


def clean_eye(raw: Optional[str]) -> str:
    """One of the eye choices; anything else becomes 'Not specified'."""
    return raw if raw in EYE_CHOICES else EYE_DEFAULT


def patient_record(patient_id: Optional[str], eye: Optional[str], demo_preset: bool,
                   when: Optional[_dt.datetime] = None) -> Dict[str, str]:
    """Identification fields printed on every document for one analysis."""
    return {
        "patient_id": clean_patient_id(patient_id),
        "eye": clean_eye(eye),
        "analysed_at": (when or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M"),
        "result_source": "demo preset (fixed illustrative values)" if demo_preset else "model prediction",
    }


def patient_header_lines(context: Optional[Dict[str, Any]]) -> List[str]:
    """Header block (ID, eye, date/time, result source and any red-flag outcome) for the documents."""
    context = context or {}
    lines = [
        "PATIENT AND SESSION",
        f"Patient ID / MRN        : {context.get('patient_id') or NOT_PROVIDED}",
        f"Eye                     : {context.get('eye') or EYE_DEFAULT}",
        f"Date / time             : {context.get('analysed_at') or 'Not recorded'}",
        f"Result source           : {context.get('result_source') or 'Not recorded'}",
    ]
    if context.get("urgent"):
        lines += [
            f"Outcome                 : {context.get('outcome', 'URGENT')}",
            "Red-flag symptoms       : " + "; ".join(context.get("red_flags", [])),
        ]
    return lines


def patient_header_text(context: Optional[Dict[str, Any]]) -> str:
    return "\n".join(patient_header_lines(context))


def view_visibility(view: Optional[str]) -> Dict[str, bool]:
    """Which result panels are shown in each view (Patient view hides the technical panels)."""
    patient = view == "Patient"
    return {
        "patient_card": patient,
        "probabilities": not patient,
        "explainability": not patient,
        "retrieval": not patient,
    }
