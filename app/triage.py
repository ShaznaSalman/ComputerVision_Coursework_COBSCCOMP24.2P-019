"""Multimodal triage and 10-year risk simulator.

ILLUSTRATIVE, NOT VALIDATED: the risk numbers come from hand-set multipliers loosely
inspired by UKPDS 33 / WESDR and are for demonstration only.
"""

from typing import Tuple

from core.config import AppConfig
from app.reports import build_referral_ticket


def calculate_multimodal_risk(stage: int = 2, hba1c: float = 7.5, duration_years: float = 10.0, age: float = 55.0, systolic_bp: float = 135.0, diabetes_type: str = "Type 2") -> Tuple[str, str]:
    """Illustrative 10-year progression risk and triage routing (UKPDS/WESDR-inspired, NOT validated).

    The stage baselines and multipliers are hand-set for demonstration; they were not fitted
    to or validated against any cohort, so the output must not be used for clinical decisions.
    """
    try:
        stage = int(stage) if stage is not None else 2
    except (ValueError, TypeError):
        stage = 2

    # Base stage risk (10-year baseline from UKPDS 33 / WESDR epidemiological cohorts)
    base_risks = {0: 3.5, 1: 12.0, 2: 29.5, 3: 58.0, 4: 84.0}
    base_risk = base_risks.get(stage, 25.0)

    # Systemic risk multipliers
    hba1c_factor = max(0.5, 1.0 + (hba1c - 7.0) * 0.18)
    duration_factor = max(0.6, 1.0 + (duration_years - 5.0) * 0.025)
    bp_factor = max(0.7, 1.0 + (systolic_bp - 120.0) * 0.008)
    age_factor = max(0.8, 1.0 + (age - 50.0) * 0.005)
    type_factor = 1.15 if diabetes_type == "Type 1" else 1.0

    risk_pct = min(98.5, max(1.5, base_risk * hba1c_factor * duration_factor * bp_factor * age_factor * type_factor))

    triage_info = {
        0: ("P4 — ROUTINE SURVEILLANCE", "Primary Community Optometry Clinic", "12 Months", "#10b981", "Routine annual digital fundus screening. Maintain glycemic control (HbA1c < 7.0%) and BP < 130/80 mmHg."),
        1: ("P3 — PRIMARY CARE GLYCEMIC ROUTE", "General Practice / Diabetes Care Team", "6–9 Months", "#0284c7", "Optimize systemic risk factors. Intensify medical therapy and blood pressure management."),
        2: ("P2 — SECONDARY HOSPITAL OPHTHALMOLOGY", "Hospital Outpatient Ophthalmology & OCT Clinic", "3–6 Months", "#d97706", "Comprehensive dilated examination + Macular OCT to evaluate subclinical Diabetic Macular Edema (DME)."),
        3: ("P2+ — URGENT VITREORETINAL EVALUATION", "Vitreoretinal Specialist Service", "2–4 Weeks", "#ea580c", "Pre-proliferative severity. Assess readiness for panretinal photocoagulation (PRP) laser therapy."),
        4: ("P1 — EMERGENCY VITREORETINAL SURGICAL ROUTE", "Tertiary Vitreoretinal Emergency Unit", "≤ 24–48 Hours", "#e11d48", "Active neovascularization / vitreous hemorrhage risk. Immediate anti-VEGF or emergency PRP laser intervention.")
    }

    triage_code, facility, wait_time, color, protocol = triage_info.get(stage, triage_info[2])

    if risk_pct < 15.0:
        risk_label, risk_color = "Low 10-Yr Progression Risk", "#10b981"
    elif risk_pct < 40.0:
        risk_label, risk_color = "Moderate 10-Yr Progression Risk", "#0284c7"
    elif risk_pct < 70.0:
        risk_label, risk_color = "High 10-Yr Progression Risk", "#ea580c"
    else:
        risk_label, risk_color = "CRITICAL VISION-THREATENING RISK", "#e11d48"

    # SVG speedometer gauge for the illustrative risk estimate
    clamped_risk = min(max(risk_pct, 1.0), 99.0)
    arc_length = 267.0
    dash_offset = arc_length * (1.0 - (clamped_risk / 100.0))

    # Calculate screening ladder steps (highlight the recommended tier)
    tiers = [
        ("12 Mo", "P4 Surveillance", "#10b981", stage == 0),
        ("6–9 Mo", "P3 Primary Care", "#0284c7", stage == 1),
        ("3–6 Mo", "P2 Hospital OCT", "#d97706", stage == 2),
        ("2–4 Wk", "P2+ Vitreoretinal", "#ea580c", stage == 3),
        ("≤ 48 Hr", "P1 Emergency", "#e11d48", stage == 4),
    ]

    ladder_html = ""
    for interval, tier_name, tcolor, is_active in tiers:
        if is_active:
            ladder_html += f"""
            <div style="flex:1; background:{tcolor}; color:#fff; border-radius:8px; padding:8px 4px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.15); border:2px solid #fff;">
                <div style="font-size:13px; font-weight:800;">{interval}</div>
                <div style="font-size:9.5px; font-weight:700; text-transform:uppercase; margin-top:2px; opacity:0.95;">{tier_name}</div>
                <div style="font-size:9px; background:rgba(255,255,255,0.25); border-radius:4px; padding:1px 3px; margin-top:3px; font-weight:800;">RECOMMENDED</div>
            </div>
            """
        else:
            ladder_html += f"""
            <div style="flex:1; background:#f1f5f9; color:#64748b; border-radius:8px; padding:8px 4px; text-align:center; border:1px solid #e2e8f0; opacity:0.75;">
                <div style="font-size:12px; font-weight:700;">{interval}</div>
                <div style="font-size:9px; margin-top:2px;">{tier_name}</div>
            </div>
            """

    risk_card_html = f"""
    <div class="risk-card" style="padding:16px; background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:12px;">
        <div class="risk-card-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
            <div style="font-size:13px; font-weight:800; text-transform:uppercase; color:#0369a1; display:flex; align-items:center; gap:8px;">
                <span>⏱️</span> Illustrative Screening Interval & 10-Year Risk Simulator
            </div>
            <span style="background:{color}; color:#fff; font-size:11.5px; font-weight:800; padding:4px 10px; border-radius:6px; letter-spacing:0.5px;">
                {triage_code.split('—')[0].strip()}
            </span>
        </div>
        <div style="margin-bottom:12px; padding:8px 12px; background:#fef2f2; border:1px solid #fca5a5; border-radius:8px; color:#991b1b; font-size:12.5px; font-weight:700;">
            ⚠️ Illustrative only — not clinically validated. Do not use for patient decisions.
        </div>
        <div class="responsive-grid responsive-grid-two" style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:12px;">
            <!-- Speedometer Gauge Card -->
            <div style="background:#fff; border-radius:10px; padding:16px; border:1px solid #e2e8f0; border-top:4px solid {risk_color}; text-align:center;">
                <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Illustrative 10-Year Progression Estimate</div>
                <div style="margin:8px auto; max-width:240px;">
                    <svg viewBox="0 0 240 135" style="width:100%; height:auto; overflow:visible;">
                        <defs>
                            <linearGradient id="retinaRiskGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#10b981" />
                                <stop offset="30%" stop-color="#0284c7" />
                                <stop offset="65%" stop-color="#f59e0b" />
                                <stop offset="100%" stop-color="#ef4444" />
                            </linearGradient>
                        </defs>
                        <!-- Background track -->
                        <path d="M 30 115 A 85 85 0 0 1 210 115" fill="none" stroke="#f1f5f9" stroke-width="14" stroke-linecap="round" />
                        <!-- Active Progress Arc -->
                        <path d="M 30 115 A 85 85 0 0 1 210 115" fill="none" stroke="url(#retinaRiskGrad)" stroke-width="14" stroke-linecap="round"
                              stroke-dasharray="267" stroke-dashoffset="{dash_offset:.1f}" />
                        <!-- Gauge Center Text -->
                        <text x="120" y="86" text-anchor="middle" font-size="30" font-weight="900" fill="{risk_color}">{risk_pct:.1f}%</text>
                        <text x="120" y="104" text-anchor="middle" font-size="10" font-weight="800" fill="#64748b" letter-spacing="0.5">IN 10 YEARS</text>
                        <text x="30" y="130" font-size="9.5" font-weight="700" fill="#10b981">0% (Low)</text>
                        <text x="120" y="130" text-anchor="middle" font-size="9.5" font-weight="700" fill="#f59e0b">Moderate</text>
                        <text x="210" y="130" text-anchor="end" font-size="9.5" font-weight="700" fill="#ef4444">100% (High)</text>
                    </svg>
                </div>
                <div style="font-size:13px; font-weight:800; color:{risk_color}; margin-top:2px;">{risk_label}</div>
                <div style="font-size:11px; color:#64748b; margin-top:4px;">
                    Illustrative, not validated &bull; UKPDS 33 / WESDR-inspired multipliers
                </div>
            </div>

            <!-- Suggested follow-up card (illustrative) -->
            <div style="background:#fff; border-radius:10px; padding:16px; border:1px solid #e2e8f0; border-top:4px solid {color}; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Suggested Follow-Up Interval (Illustrative)</div>
                    <div style="font-size:16px; font-weight:800; color:#0f172a; margin-top:4px;">{facility}</div>
                    <div style="display:inline-flex; align-items:center; gap:6px; background:#eff6ff; color:{color}; padding:4px 10px; border-radius:6px; font-size:12px; font-weight:800; margin:6px 0;">
                        <span>⏱️</span> Suggested timeframe: {wait_time}
                    </div>
                    <div style="font-size:12px; color:#334155; line-height:1.5; margin-top:6px;">
                        <strong>Example action (illustrative):</strong> {protocol}
                    </div>
                </div>
                <div style="font-size:11px; color:#64748b; background:#f8fafc; border-radius:6px; padding:8px 10px; margin-top:8px;">
                    <strong>Patient Profile:</strong> Age {age:.0f}y &bull; {diabetes_type} &bull; HbA1c {hba1c:.1f}% &bull; Duration {duration_years:.0f}y &bull; BP {systolic_bp:.0f} mmHg
                </div>
            </div>
        </div>

        <!-- Illustrative screening-interval ladder -->
        <div style="background:#fff; border-radius:10px; padding:14px; border:1px solid #e2e8f0;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:11.5px; font-weight:700; text-transform:uppercase; color:#0369a1;">
                    Illustrative Follow-Up Interval vs. Standard Annual Screening
                </div>
                <div style="font-size:11px; color:#64748b;">
                    Standard Recall: <strong style="color:#0f172a;">12 Months</strong> &bull; Personalized: <strong style="color:{color};">{wait_time}</strong>
                </div>
            </div>
            <div class="risk-ladder" style="display:flex; gap:8px;">
                {ladder_html}
            </div>
        </div>
    </div>
    """

    referral_ticket = build_referral_ticket(
        triage_code, facility, wait_time, stage, risk_pct, risk_label,
        age, diabetes_type, hba1c, duration_years, systolic_bp, protocol,
    )

    return risk_card_html, referral_ticket


def update_triage_routing(prob_dict, hba1c, duration, age, bp, d_type):
    """Recompute the triage card from the current top-probability stage (defaults to stage 2)."""
    st = 2
    if isinstance(prob_dict, dict) and prob_dict:
        stage_map = {name: i for i, name in enumerate(AppConfig.CLASS_NAMES)}
        try:
            top_name = max(prob_dict, key=prob_dict.get)
            st = stage_map.get(top_name, 2)
        except (KeyError, ValueError, TypeError):
            st = 2
    return calculate_multimodal_risk(st, hba1c, duration, age, bp, d_type)
