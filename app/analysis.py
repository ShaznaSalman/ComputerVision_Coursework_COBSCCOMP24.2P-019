"""Fundus analysis: sample images, image quality checks, the main analysis handler,
and longitudinal comparison. Predictions come from core.agents.run_pipeline."""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2

from core.config import AppConfig
from core.preprocessing import preprocess_image
from core.advanced_cv import compile_retinal_biomarkers, compare_longitudinal_examinations

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Modular pipeline implementations.
from core.agents import run_pipeline

from app.reports import build_ehr_note


# ─────────────────────────────────────────────────────────────────────────────
# 6. Clinical Fundus Demonstrators (Instant Demo Library)
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLE_CACHE: Dict[int, np.ndarray] = {}
_SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
# One real fundus photograph per ICDR stage, taken from the patient-safe validation split.
_SAMPLE_PRESET_PATHS = {
    0: str(_SAMPLES_DIR / "stage0_no_dr.jpg"),
    1: str(_SAMPLES_DIR / "stage1_mild.jpg"),
    2: str(_SAMPLES_DIR / "stage2_moderate.jpg"),
    3: str(_SAMPLES_DIR / "stage3_severe.jpg"),
    4: str(_SAMPLES_DIR / "stage4_proliferative.jpg"),
}

for _s_idx, _s_p in _SAMPLE_PRESET_PATHS.items():
    if os.path.exists(_s_p):
        _b = cv2.imread(_s_p)
        if _b is not None:
            _SAMPLE_CACHE[_s_idx] = cv2.cvtColor(_b, cv2.COLOR_BGR2RGB)


def create_sample_fundus(stage: int = 0) -> np.ndarray:
    """Returns a real, high-quality clinical retinal fundus photograph for the requested ICDR stage."""
    if stage in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[stage].copy()

    # Fallback synthetic generator if file is missing
    img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 104, (190, 75, 35), -1)
    cv2.ellipse(img, (75, 112), (16, 22), 0, 0, 360, (240, 220, 150), -1)
    cv2.polylines(img, [np.array([[75, 112], [105, 80], [150, 55], [195, 45]])], False, (110, 25, 15), 2)
    cv2.polylines(img, [np.array([[75, 112], [110, 140], [160, 165], [190, 175]])], False, (110, 25, 15), 2)
    cv2.polylines(img, [np.array([[75, 112], [45, 95], [25, 85]])], False, (110, 25, 15), 2)
    cv2.circle(img, (135, 112), 14, (140, 45, 20), -1)
    if stage >= 1:
        cv2.circle(img, (120, 95), 2, (70, 10, 5), -1)
        cv2.circle(img, (145, 130), 2, (70, 10, 5), -1)
    if stage >= 2:
        cv2.circle(img, (155, 110), 4, (250, 240, 170), -1)
        cv2.circle(img, (162, 115), 3, (250, 240, 170), -1)
        cv2.circle(img, (115, 135), 4, (80, 10, 10), -1)
    if stage >= 3:
        cv2.circle(img, (100, 145), 6, (75, 10, 10), -1)
        cv2.circle(img, (140, 80), 7, (75, 10, 10), -1)
        cv2.circle(img, (165, 140), 5, (250, 240, 170), -1)
    if stage >= 4:
        cv2.line(img, (75, 112), (90, 95), (130, 35, 20), 3)
        cv2.line(img, (90, 95), (105, 85), (130, 35, 20), 2)
        cv2.circle(img, (85, 105), 8, (90, 10, 10), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Modern Gradio Interface Callbacks & Formatting
# ─────────────────────────────────────────────────────────────────────────────
STAGE_BADGE_COLORS = {
    0: ("#10b981", "#dcfce7", "STAGE 0: NO RETINOPATHY"),
    1: ("#0284c7", "#e0f2fe", "STAGE 1: MILD NPDR"),
    2: ("#d97706", "#fef3c7", "STAGE 2: MODERATE NPDR"),
    3: ("#ea580c", "#ffedd5", "STAGE 3: SEVERE NPDR"),
    4: ("#e11d48", "#ffe4e6", "STAGE 4: PROLIFERATIVE DR"),
}


def assess_image_quality(img: np.ndarray) -> dict:
    """Assesses fundus image quality: blur, illumination, retina presence, and black border ratio."""
    issues = []
    warnings = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    # 1. Blur detection via Laplacian variance (clinically calibrated for fundus images)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 10.0:
        issues.append(f"Severe blur detected (Laplacian variance: {lap_var:.1f} < 10 threshold) — image may be out of focus.")
    elif lap_var < 15.0:
        warnings.append(f"Mild optical softening (Laplacian variance: {lap_var:.1f}).")

    # 2. Illumination check via mean brightness
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30:
        issues.append(f"Poor illumination — image is underexposed (mean brightness: {mean_brightness:.1f}/255).")
    elif mean_brightness > 220:
        issues.append(f"Overexposed image — excessive brightness may wash out lesions (mean: {mean_brightness:.1f}/255).")

    # 3. Black border ratio (retina presence check)
    black_mask = gray < 15
    black_ratio = float(np.sum(black_mask)) / float(gray.size)
    if black_ratio > 0.60:
        issues.append(f"Excessive black border ({black_ratio*100:.1f}% of image) — retina may be missing or severely cropped.")
    elif black_ratio > 0.40:
        warnings.append(f"Large black border area ({black_ratio*100:.1f}%) — image may be sub-optimally framed.")

    # 4. Retina disk presence heuristic using central green-channel energy
    h, w = gray.shape
    if img.ndim == 3:
        center_patch = img[h//4:3*h//4, w//4:3*w//4]
        green_energy = float(np.mean(center_patch[:, :, 1]))
    else:
        green_energy = float(np.mean(gray[h//4:3*h//4, w//4:3*w//4]))
    if green_energy < 20:
        issues.append("Retina not detected in image center — please ensure the optic disc is within the frame.")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "warnings": warnings,
        "blur_score": lap_var,
        "brightness": mean_brightness,
        "black_ratio": black_ratio,
        "central_green_energy": green_energy,
    }


def build_quality_warning_html(qc: dict) -> str:
    """Renders image quality assessment result as an HTML warning card."""
    if qc["passed"] and not qc["warnings"]:
        return (
            '<div style="padding:8px 14px; background:#f0fdf4; border-radius:8px; border-left:4px solid #10b981; margin-bottom:8px;">'
            '<span style="font-weight:700; color:#166534;">✅ Image Quality: PASS</span>'
            f'<span style="color:#64748b; font-size:12px; margin-left:10px;">Blur score: {qc["blur_score"]:.1f} | Brightness: {qc["brightness"]:.0f}/255 | Black border: {qc["black_ratio"]*100:.1f}% | Green energy: {qc["central_green_energy"]:.1f}</span>'
            '</div>'
        )
    color = "#fef2f2" if not qc["passed"] else "#fffbeb"
    border = "#ef4444" if not qc["passed"] else "#f59e0b"
    icon = "⚠️ Image Quality: ISSUES DETECTED" if not qc["passed"] else "⚠️ Image Quality: WARNINGS"
    label_color = "#991b1b" if not qc["passed"] else "#92400e"
    all_msgs = [f'<li style="margin:2px 0;">{m}</li>' for m in (qc["issues"] + qc["warnings"])]
    return (
        f'<div style="padding:10px 14px; background:{color}; border-radius:8px; border-left:4px solid {border}; margin-bottom:8px;">'
        f'<div style="font-weight:700; color:{label_color}; margin-bottom:6px;">{icon}</div>'
        f'<ul style="margin:0; padding-left:18px; font-size:12px; color:#374151;">{"".join(all_msgs)}</ul>'
        f'<div style="font-size:11px; color:#64748b; margin-top:4px;">Blur score: {qc["blur_score"]:.1f} | Brightness: {qc["brightness"]:.0f}/255 | Black border: {qc["black_ratio"]*100:.1f}% | Green energy: {qc["central_green_energy"]:.1f}</div>'
        '</div>'
    )


def analyze_fundus(img: Optional[np.ndarray], threshold: float, session_history: list = None, preset_stage: Optional[int] = None):
    """Primary analysis handler that executes the pipeline and populates modern UI widgets."""
    empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    if img is None:
        notice = "<div class='card warning-card'>⚠️ <strong>Please upload a retinal fundus photograph</strong> or click one of the quick-load sample buttons on the left.</div>"
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", empty_img, empty_img, "", "", [], "", "", "", "", "", None, [], {}, "", empty_img, "", "", 0.0, "", "", "", ""

    try:
        preproc = preprocess_image(img)
    except ValueError as exc:
        notice = (
            "<div class='card warning-card'>⚠️ <strong>Invalid fundus image input</strong> — "
            f"{exc}. Please upload a valid retina image or choose a sample fundus from the quick-load buttons.</div>"
        )
        return notice, "", {}, empty_img, empty_img, empty_img, empty_img, "", empty_img, empty_img, "", "", [], "", "", "", "", "", None, [], {}, "", empty_img, "", "", 0.0, "", "", "", ""

    # Image Quality Assessment (before inference)
    qc = assess_image_quality(img)
    quality_html = build_quality_warning_html(qc)

    # Detect if the input image matches a known clinical demo preset
    matched_preset_stage = preset_stage
    if matched_preset_stage is None and img is not None:
        for _s_idx, _s_img in _SAMPLE_CACHE.items():
            if img.shape == _s_img.shape:
                diff = float(np.mean(np.abs(img.astype(np.float32) - _s_img.astype(np.float32))))
                if diff < 15.0:
                    matched_preset_stage = _s_idx
                    break

    result = run_pipeline(preproc, threshold=threshold, qc=qc, preset_stage=matched_preset_stage)
    # Sample presets return FIXED illustrative probabilities from core.agents, not a model
    # prediction, so every output below labels them as a demo preset.
    is_demo_preset = matched_preset_stage is not None


    diag = result["diagnosis"]
    expl = result["explanation"]
    adv = result["advisory"]
    flagged = result["flagged_for_review"]
    stage = diag["stage"]
    overlap = expl.get("overlap_analysis", {})
    severity_map = expl.get("severity_map", {})
    consistency = result.get("consistency_analysis", {})
    biomarkers = compile_retinal_biomarkers(
        expl.get("lesion_pct", 0.0),
        severity_map.get("total_lesion_count", 0),
        expl.get("vessel_density", 0.0),
        expl.get("affected_quadrants_count", 0),
        expl,
        qc,
        expl.get("classical_cv", {}),
    )

    # 1. Governance Banner HTML
    if flagged:
        gov_html = f"""
        <div class="card alert-card">
            <div class="card-header">
                <span class="icon">🚨</span>
                <div>
                    <h3 style="margin:0; color:#991b1b;">GOVERNANCE STATUS: FLAGGED FOR HUMAN TRIAGE</h3>
                    <p style="margin:2px 0 0 0; color:#7f1d1d; font-size:13px;">
                        Safety Gate Interception: Model confidence (<strong>{diag['confidence']*100:.1f}%</strong>) is below your set threshold (<strong>{threshold*100:.0f}%</strong>).
                    </p>
                </div>
            </div>
            <div style="margin-top:8px; padding:8px 12px; background:#fff; border-radius:6px; border-left:4px solid #ef4444; font-size:13px; color:#b91c1c;">
                <strong>Patient Safety Protection:</strong> Automated treatment guidance has been withheld to eliminate hallucination risks. Rerouted to mandatory human ophthalmologist review.
            </div>
        </div>
        """
    else:
        gov_html = f"""
        <div class="card success-card">
            <div class="card-header">
                <span class="icon">✅</span>
                <div>
                    <h3 style="margin:0; color:#166534;">GOVERNANCE STATUS: AUTOMATION APPROVED</h3>
                    <p style="margin:2px 0 0 0; color:#14532d; font-size:13px;">
                        Quality Assurance Verified: Model confidence (<strong>{diag['confidence']*100:.1f}%</strong>) satisfies the clinical safety threshold (<strong>{threshold*100:.0f}%</strong>).
                    </p>
                </div>
            </div>
        </div>
        """

    # 2. Hero Diagnosis Card HTML (Medios-Style Tri-State Status Chip & Hospital Card)
    border_c, bg_c, badge_text = STAGE_BADGE_COLORS[stage]

    if stage == 0:
        medios_chip_text = "🟢 NO DR DETECTED"
        medios_chip_bg = "#ecfdf5"
        medios_chip_color = "#065f46"
        medios_chip_border = "#a7f3d0"
    elif stage in (1, 2):
        medios_chip_text = f"🟡 DIABETIC RETINOPATHY DETECTED ({diag['stage_name'].upper()})"
        medios_chip_bg = "#fffbeb"
        medios_chip_color = "#92400e"
        medios_chip_border = "#fde68a"
    else:
        medios_chip_text = f"🔴 SIGHT-THREATENING RETINOPATHY DETECTED ({diag['stage_name'].upper()})"
        medios_chip_bg = "#fff1f2"
        medios_chip_color = "#9f1239"
        medios_chip_border = "#fecdd3"

    demo_banner_html = (
        '<div style="margin-bottom:10px; padding:8px 12px; background:#fef3c7; border:1px solid #f59e0b; '
        'border-radius:8px; color:#92400e; font-size:12.5px; font-weight:700;">'
        '🧪 DEMO PRESET — fixed illustrative values, not model output. '
        'Upload your own image to see a real model prediction.</div>'
        if is_demo_preset else ""
    )
    output_label = "Demo Preset (Fixed Values, Not Model Output)" if is_demo_preset else "Primary Diagnostic Output"
    confidence_label = "FIXED DEMO VALUE" if is_demo_preset else "DIAGNOSTIC CONFIDENCE"

    hero_html = f"""
    <div class="card hero-card" style="border-top: 5px solid {border_c};">{demo_banner_html}
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px;">
            <div>
                <div style="display:inline-flex; align-items:center; gap:6px; background:{medios_chip_bg}; color:{medios_chip_color}; border:1px solid {medios_chip_border}; padding:5px 14px; border-radius:20px; font-weight:800; font-size:12px; letter-spacing:0.5px; margin-bottom:8px;">
                    {medios_chip_text}
                </div>
                <h1 class="hero-stage-title" style="margin:2px 0 4px 0; font-size:26px;">Stage {stage}: {diag['stage_name']}</h1>
                <p class="hero-subtext" style="margin:0; font-size:13px;">International Clinical Diabetic Retinopathy (ICDR) Scale &bull; {output_label}</p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:36px; font-weight:900; color:{border_c}; line-height:1;">{diag['confidence']*100:.1f}%</div>
                <div class="hero-sublabel" style="font-size:11px; font-weight:700; margin-top:4px; letter-spacing:0.5px;">{confidence_label}</div>
            </div>
        </div>
    </div>
    """

    # 3. Probabilities for gr.Label
    probs_dict = diag["probabilities"]

    # 4. CBR Gallery
    gallery_items = []
    for c in expl["similar_cases"]:
        caption = f"Match #{c['rank']} • Stage {c['stage']}: {c['stage_name']} (Similarity: {c['similarity']:.3f})"
        if os.path.exists(c["filepath"]):
            gallery_items.append((c["filepath"], caption))

    # 5. Clinical Advisory Plan HTML
    advisory_html = f"""
    <div class="card">
        <h3 class="protocol-title" style="margin-top:0; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">📋 Clinical Care Protocol (AAO Preferred Practice Pattern)</h3>
        <div class="responsive-grid responsive-grid-two" style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:12px;">
            <div class="protocol-box" style="padding:10px; border-radius:6px;">
                <div class="protocol-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Clinical Urgency Level</div>
                <div class="protocol-val" style="font-size:15px; font-weight:700; margin-top:2px;">{adv['urgency']}</div>
            </div>
            <div class="protocol-box" style="padding:10px; border-radius:6px;">
                <div class="protocol-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Recommended Follow-Up</div>
                <div class="protocol-val" style="font-size:15px; font-weight:700; margin-top:2px;">{adv['followup']}</div>
            </div>
        </div>
        <div class="action-box" style="padding:12px; border-radius:6px; margin-bottom:12px;">
            <div class="action-label" style="font-size:11px; font-weight:700; text-transform:uppercase;">Specialist Action Plan</div>
            <div class="action-val" style="font-size:13.5px; margin-top:4px; line-height:1.5;">{adv['plan']}</div>
        </div>
        <div class="disclaimer-text" style="font-size:11.5px; font-style:italic;">
            {adv['disclaimer']}
        </div>
    </div>
    """

    # 6. Exportable Clinical EHR Note
    ehr_text = build_ehr_note(diag, expl, adv, consistency, overlap, flagged, threshold,
                              demo_preset=is_demo_preset)

    # 7. Lesion Burden HTML metric card
    lesion_pct = expl.get("lesion_pct", 0.0)
    if lesion_pct == 0.0:
        burden_color, burden_label = "#10b981", "Minimal / No Detectable Lesion Area"
    elif lesion_pct < 5.0:
        burden_color, burden_label = "#d97706", "Low-Moderate Burden"
    elif lesion_pct < 15.0:
        burden_color, burden_label = "#ea580c", "Moderate-High Burden"
    else:
        burden_color, burden_label = "#e11d48", "High Lesion Burden - Urgent Review"

    lesion_burden_html = (
        f'<div style="margin-top:8px; padding:10px 14px; background:#f8fafc; border-radius:8px; border-left:4px solid {burden_color};">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:4px;">Retinal Lesion Area Burden (U-Net Segmentation)</div>'
        f'<div style="display:flex; align-items:baseline; gap:10px;">'
        f'<span style="font-size:28px; font-weight:800; color:{burden_color};">{lesion_pct:.2f}%</span>'
        f'<span style="font-size:13px; color:#475569;">of visible parenchyma</span></div>'
        f'<div style="font-size:12px; color:{burden_color}; font-weight:600; margin-top:2px;">{burden_label}</div>'
        f'</div>'
    )

    # Classical Computer Vision Biomarkers Panel (OpenCV Syllabus Alignment)
    classical_cv = expl.get("classical_cv", {})
    edge_density = classical_cv.get("sobel_edge_density", 0.0)
    morph_pct = classical_cv.get("morph_candidate_pct", 0.0)
    clahe_status = classical_cv.get("clahe_status", "Calibrated (Clip=2.0)")

    classical_cv_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0; border-left:4px solid #0284c7;">'
        '<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#0369a1; margin-bottom:8px; display:flex; align-items:center; gap:6px;">'
        '<span>🔬</span> Classical Computer Vision Biomarkers (OpenCV)</div>'
        '<div class="responsive-grid responsive-grid-three" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; text-align:center;">'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">SOBEL GRADIENT</div>'
        f'<div style="font-size:16px; font-weight:800; color:#0f172a; margin:2px 0;">{edge_density:.1f}%</div>'
        '<div style="font-size:10px; color:#64748b;">Vascular Edge Density</div></div>'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">CLAHE HISTOGRAM</div>'
        f'<div style="font-size:13px; font-weight:800; color:#059669; margin:4px 0;">{clahe_status}</div>'
        '<div style="font-size:10px; color:#64748b;">Adaptive Local Contrast</div></div>'
        '<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:10px; color:#64748b; font-weight:600;">MORPHOLOGY</div>'
        f'<div style="font-size:16px; font-weight:800; color:#d97706; margin:2px 0;">{morph_pct:.1f}%</div>'
        '<div style="font-size:10px; color:#64748b;">Top-Hat Lesion Sites</div></div>'
        '</div>'
        '<div style="font-size:10px; color:#64748b; margin-top:8px; font-style:italic;">'
        'Deterministic physical image processing features (Convolution, Morphology, Histograms) extracted in parallel with deep embeddings.</div>'
        '</div>'
    )

    lesion_burden_html = lesion_burden_html + classical_cv_html

    vessel_density = float(expl.get("vessel_density", 0.0))
    optic_found = bool(expl.get("optic_disc_found", False))
    optic_status = "Detected" if optic_found else "Not reliably detected"
    classical_cv_status_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">'
        '<div style="font-size:12px; font-weight:800; color:#0369a1;">Classical CV Vessel Analysis</div>'
        f'<div style="font-size:13px; color:#334155; margin-top:5px;">Vessel density: <strong>{vessel_density:.2f}%</strong> '
        f'| Method: {expl.get("vessel_status", "Exploratory morphology")}</div>'
        '<div style="font-size:12px; font-weight:800; color:#7c3aed; margin-top:10px;">Optic-Disc Localisation</div>'
        f'<div style="font-size:13px; color:#334155; margin-top:5px;">Status: <strong>{optic_status}</strong> '
        f'| Heuristic score: {float(expl.get("optic_disc_score", 0.0)):.2f}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Visual support only — not an independent diagnosis.</div>'
        '</div>'
    )

    biomarker_rows = "".join(
        f'<div title="{item["tooltip"]}" style="padding:8px; background:#fff; border:1px solid #e2e8f0; border-radius:6px;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:700;">{item["name"]}</div>'
        f'<div style="font-size:16px; font-weight:800; color:#0f172a; margin-top:3px;">{item["value"]}</div>'
        f'<div style="font-size:10px; color:#64748b; margin-top:2px;">{item["status"]}</div></div>'
        for item in biomarkers
    )
    quadrant_rows = "".join(
        f'<div style="padding:8px; background:{item["badge_bg"]}; border-left:4px solid {item["badge_color"]}; border-radius:5px;">'
        f'<strong>{name}</strong><br><span style="font-size:11px;">Burden {item["lesion_burden_pct"]:.2f}% | '
        f'Lesions {item["lesion_count"]} | Vessels {item["vessel_density_pct"]:.2f}%</span><br>'
        f'<span style="font-size:11px; color:{item["badge_color"]};">{item["abnormality_label"]}</span></div>'
        for name, item in severity_map.get("quadrants", {}).items()
    )
    advanced_analysis_html = (
        '<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;">'
        '<div style="font-size:12px; font-weight:800; color:#0369a1;">AI Attention–Lesion Agreement</div>'
        f'<div style="font-size:13px; margin-top:5px;">IoU: <strong>{overlap.get("iou", 0.0):.3f}</strong> | '
        f'Dice: <strong>{overlap.get("dice", 0.0):.3f}</strong> | Lesion regions within attention: '
        f'<strong>{overlap.get("lesion_in_cam_pct", 0.0):.1f}%</strong></div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:4px;">{overlap.get("interpretation", "Evidence unavailable.")} Experimental explainability measure only.</div>'
        '<div style="font-size:12px; font-weight:800; color:#7c3aed; margin-top:12px;">Retinal Visual-Abnormality Map</div>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:6px;">{quadrant_rows}</div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:5px;">Affected quadrants: {severity_map.get("affected_quadrants_count", 0)}/4. These are visual-support indicators, not clinical severity grades.</div>'
        '<div style="font-size:12px; font-weight:800; color:#0f766e; margin-top:12px;">Retinal CV Biomarkers</div>'
        f'<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin-top:6px;">{biomarker_rows}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Computer Vision Measurements — Research/Visual Support Only. Visual support only — not an independent diagnosis.</div>'
        '<div style="font-size:12px; font-weight:800; color:#b45309; margin-top:12px;">Prediction–Evidence Consistency</div>'
        f'<div style="font-size:14px; font-weight:800; color:{consistency.get("status_color", "#64748b")}; margin-top:4px;">{consistency.get("icon", "⚪")} {consistency.get("status", "INSUFFICIENT EVIDENCE")}</div>'
        f'<div style="font-size:11px; color:#475569; margin-top:4px;">{consistency.get("reason", "Independent evidence is unavailable.")}</div>'
        '</div>'
    )
    cbr_explanation = expl.get("cbr_explanation", {})
    advanced_analysis_html += (
        '<div style="margin-top:10px; padding:10px; border-top:1px solid #e2e8f0;">'
        '<div style="font-size:12px; font-weight:800; color:#475569;">CBR Visual-Similarity Explanation</div>'
        f'<div style="font-size:11px; color:#475569; margin-top:4px;">'
        f'Average retrieved similarity: {cbr_explanation.get("average_similarity", 0.0):.3f} | '
        f'Stage-label agreement: {cbr_explanation.get("stage_agreement_count", 0)}/{cbr_explanation.get("stage_agreement_total", 0)}</div>'
        f'<div style="font-size:11px; color:#64748b; margin-top:3px;">{cbr_explanation.get("interpretation", "No CBR metadata available.")}</div>'
        '</div>'
    )

    research_support = {
        "available": True,
        "attention_lesion_iou": overlap.get("iou", 0.0),
        "attention_lesion_dice": overlap.get("dice", 0.0),
        "lesion_regions_in_attention_pct": overlap.get("lesion_in_cam_pct", 0.0),
        "lesion_burden_pct": expl.get("lesion_pct", 0.0),
        "candidate_lesion_count": severity_map.get("total_lesion_count", 0),
        "vessel_density_pct": expl.get("vessel_density", 0.0),
        "affected_quadrants": expl.get("affected_quadrants_count", 0),
        "optic_disc_found": bool(expl.get("optic_disc_found", False)),
        "optic_disc_score": expl.get("optic_disc_score", 0.0),
        "prediction_evidence_consistency": consistency.get("status", "INSUFFICIENT EVIDENCE"),
        "cbr": cbr_explanation,
        "disclaimer": "Research/visual support only — not an independent diagnosis.",
        "result_source": "demo_preset_fixed_values" if is_demo_preset else "model_prediction",
    }

    # 8. Anatomical Quadrant Salience HTML bar chart
    quad_scores = expl.get("quadrant_scores", {})
    peak_quad = expl.get("peak_quadrant", "-")
    quad_bar_rows = ""
    for qname, qval in sorted(quad_scores.items(), key=lambda x: x[1], reverse=True):
        pct_width = min(int(qval * 100 * 3.5), 100)
        is_peak = qname == peak_quad
        bar_color = "#ef4444" if is_peak else "#60a5fa"
        peak_marker = " [PEAK]" if is_peak else ""
        fw = "700" if is_peak else "500"
        fc = "#991b1b" if is_peak else "#334155"
        quad_bar_rows += (
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex; justify-content:space-between; font-size:12px; font-weight:{fw}; color:{fc}; margin-bottom:2px;">'
            f'<span>{qname}{peak_marker}</span><span>{qval:.3f}</span></div>'
            f'<div style="background:#e2e8f0; border-radius:4px; height:10px; overflow:hidden;">'
            f'<div style="background:{bar_color}; width:{pct_width}%; height:100%; border-radius:4px;"></div></div></div>'
        )

    quadrant_chart_html = (
        f'<div style="margin-top:10px; padding:12px 14px; background:#f8fafc; border-radius:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:10px;">'
        f'Anatomical Quadrant Grad-CAM Activation Index</div>'
        f'{quad_bar_rows}'
        f'<div style="font-size:11px; color:#94a3b8; margin-top:6px; font-style:italic;">'
        f'Mean Grad-CAM saliency per quadrant (Superior/Inferior x Temporal/Nasal).</div>'
        f'</div>'
    )

    # 9. Adjacent-Stage Confidence Margin
    sorted_probs = sorted(diag["probabilities"].items(), key=lambda x: x[1], reverse=True)
    top_stage_name, top_conf = sorted_probs[0]
    sec_stage_name, sec_conf = sorted_probs[1] if len(sorted_probs) > 1 else ("-", 0.0)
    margin = top_conf - sec_conf
    margin_color = "#10b981" if margin > 0.40 else ("#d97706" if margin > 0.20 else "#e11d48")
    margin_label = "High Certainty" if margin > 0.40 else ("Borderline - Monitor" if margin > 0.20 else "Low Certainty - Human Review Advised")

    confidence_margin_html = (
        f'<div style="margin-top:8px; padding:10px 14px; background:#f8fafc; border-radius:8px; border-left:4px solid {margin_color};">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:6px;">'
        f'Diagnostic Uncertainty Margin (Adjacent-Stage Analysis)</div>'
        f'<div class="responsive-grid responsive-grid-three" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; text-align:center;">'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">PRIMARY</div>'
        f'<div style="font-size:13px; font-weight:700; color:#1e293b;">{top_stage_name}</div>'
        f'<div style="font-size:16px; font-weight:800; color:{border_c};">{top_conf*100:.1f}%</div></div>'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">RUNNER-UP</div>'
        f'<div style="font-size:13px; font-weight:700; color:#1e293b;">{sec_stage_name}</div>'
        f'<div style="font-size:16px; font-weight:800; color:#64748b;">{sec_conf*100:.1f}%</div></div>'
        f'<div style="background:#fff; border-radius:6px; padding:8px; border:1px solid #e2e8f0;">'
        f'<div style="font-size:10px; color:#64748b; font-weight:600;">MARGIN</div>'
        f'<div style="font-size:16px; font-weight:800; color:{margin_color};">{margin*100:.1f}%</div>'
        f'<div style="font-size:10px; color:{margin_color}; font-weight:600;">{margin_label}</div></div>'
        f'</div></div>'
    )

    # 10. Uncertainty Banner
    conf_val = diag["confidence"]
    if conf_val >= 0.85:
        unc_icon, unc_label, unc_color, unc_bg = "🟢", "HIGH CONFIDENCE", "#166534", "#f0fdf4"
        unc_border = "#10b981"
        unc_sub = f"Model certainty is {conf_val*100:.1f}% — result is suitable for clinical review."
    elif conf_val >= threshold:
        unc_icon, unc_label, unc_color, unc_bg = "🟡", "REVIEW RECOMMENDED", "#92400e", "#fffbeb"
        unc_border = "#f59e0b"
        unc_sub = f"Confidence ({conf_val*100:.1f}%) exceeds threshold but is below 85% — secondary clinical review advised."
    else:
        unc_icon, unc_label, unc_color, unc_bg = "🔴", "INSUFFICIENT CONFIDENCE", "#991b1b", "#fef2f2"
        unc_border = "#ef4444"
        unc_sub = f"Confidence ({conf_val*100:.1f}%) is below the safety threshold ({threshold*100:.0f}%) — human triage required."

    uncertainty_banner_html = (
        f'<div style="padding:12px 16px; background:{unc_bg}; border-radius:10px; '
        f'border-left:5px solid {unc_border}; margin-bottom:10px; display:flex; align-items:center; gap:12px;">'
        f'<div style="font-size:28px;">{unc_icon}</div>'
        f'<div><div style="font-size:14px; font-weight:800; color:{unc_color}; letter-spacing:0.5px;">{unc_label}</div>'
        f'<div style="font-size:12px; color:#374151; margin-top:2px;">{unc_sub}</div></div>'
        f'</div>'
    )

    # 11. Processed image for side-by-side view
    processed_display = (preproc * 255).astype(np.uint8)

    # 12. Prediction context for chatbot (gr.State)
    pred_context = {
        "stage_name": diag["stage_name"],
        "stage": stage,
        "confidence": conf_val,
        "urgency": adv["urgency"],
        "followup": adv["followup"],
        "plan": adv["plan"],
        "peak_quadrant": expl.get("peak_quadrant", "-"),
        "lesion_pct": expl.get("lesion_pct", 0.0),
        "demo_preset": is_demo_preset,
    }

    # 13. Session history update
    if session_history is None:
        session_history = []
    import datetime as _dt
    hist_entry = {
        "timestamp": _dt.datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
        "stage_name": diag["stage_name"] + (" (demo preset)" if is_demo_preset else ""),
        "confidence": f"{conf_val*100:.1f}%",
        "urgency": adv["urgency"],
    }
    session_history = session_history + [hist_entry]

    # Build history HTML
    history_rows = ""
    stage_colors = {0: "#10b981", 1: "#0284c7", 2: "#d97706", 3: "#ea580c", 4: "#e11d48"}
    for i, h in enumerate(reversed(session_history)):
        sc = stage_colors.get(h["stage"], "#64748b")
        history_rows += (
            f'<div style="display:flex; align-items:center; gap:10px; padding:8px 12px; '
            f'border-radius:8px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:6px;">'
            f'<div style="min-width:36px; height:36px; border-radius:50%; background:{sc}; '
            f'display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:14px;">{h["stage"]}</div>'
            f'<div style="flex:1;"><div style="font-weight:700; font-size:13px; color:#0f172a;">{h["stage_name"]}</div>'
            f'<div style="font-size:11px; color:#64748b;">{h["urgency"]} • Confidence: {h["confidence"]}</div></div>'
            f'<div style="font-size:11px; color:#94a3b8;">{h["timestamp"]}</div>'
            f'</div>'
        )
    if not history_rows:
        history_rows = '<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet in this session.</div>'

    session_history_html = (
        f'<div style="padding:4px 0;">'
        f'<div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#64748b; margin-bottom:8px;">'
        f'{len(session_history)} Prediction(s) This Session — Most Recent First</div>'
        f'{history_rows}</div>'
    )

    return (
        gov_html,                       # 0: status_banner
        hero_html,                      # 1: hero_diagnosis
        probs_dict,                     # 2: prob_distribution
        expl["overlay_cam"],            # 3: overlay_cam_view
        expl["lesion_seg"],             # 4: lesion_seg_view
        expl["vessel_overlay"],         # 5: vessel_overlay_view
        expl["optic_disc_overlay"],     # 6: optic_disc_overlay_view
        classical_cv_status_html,       # 7: classical_cv_status_view
        expl.get("overlap_analysis", {}).get("combined_vis", empty_img), # 8: overlap_view
        expl.get("severity_map", {}).get("quadrant_overlay", empty_img), # 9: severity_map_view
        advanced_analysis_html,         # 10: advanced_analysis_view
        expl["quadrant_desc"],          # 11: quadrant_text
        gallery_items,                  # 12: gallery_view
        advisory_html,                  # 13: advisory_view
        ehr_text,                       # 14: ehr_note_box
        lesion_burden_html,             # 15: lesion_burden_view
        quadrant_chart_html,             # 16: quadrant_chart_view
        confidence_margin_html,         # 17: confidence_margin_view
        pred_context,                   # 18: pred_context_state
        session_history,                # 19: session_history_state
        research_support,               # 20: research_support_state
        quality_html + uncertainty_banner_html,  # 20: uncertainty_banner_top
        processed_display,              # 21: processed_img_view
        session_history_html,           # 22: session_history_view
        diag["stage_name"],             # 23: _diag_stage_name_state
        conf_val,                       # 24: _diag_conf_state
        adv["urgency"],                 # 25: _diag_urgency_state
        adv["followup"],                # 26: _diag_followup_state
        adv["plan"],                    # 27: _diag_plan_state
        quality_html,                   # 28: quality_warning_view
    )


def compare_longitudinal_images(previous: Optional[np.ndarray], current: Optional[np.ndarray]):
    """Compare two optional examinations without assuming patient identity."""
    empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
    if previous is None or current is None:
        return "<div class='card warning-card'>Submit both a previous and current examination.</div>", empty_img
    try:
        previous_preproc = preprocess_image(previous)
        current_preproc = preprocess_image(current)
        previous_qc = assess_image_quality(previous)
        current_qc = assess_image_quality(current)
        previous_result = run_pipeline(previous_preproc, qc=previous_qc)
        current_result = run_pipeline(current_preproc, qc=current_qc)
        comparison = compare_longitudinal_examinations(
            previous_preproc,
            current_preproc,
            previous_result["diagnosis"],
            current_result["diagnosis"],
            previous_result["explanation"],
            current_result["explanation"],
        )
    except (ValueError, KeyError, RuntimeError, cv2.error) as exc:
        return f"<div class='card warning-card'>Reliable comparison could not be established: {exc}</div>", empty_img

    registration = "Reliable spatial registration established." if comparison["registration_reliable"] else "Reliable spatial comparison could not be established. Metric changes are non-spatial visual comparisons."
    summary = (
        '<div class="card" style="border-left:4px solid #0284c7;">'
        '<h3 style="margin:0 0 8px 0;">LONGITUDINAL COMPARISON</h3>'
        f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">'
        f'<div><strong>Previous</strong><br>Stage {comparison["prev_stage"]} ({previous_result["diagnosis"]["stage_name"]})<br>'
        f'Lesion burden: {comparison["prev_lesion"]:.2f}%<br>Vessel density: {comparison["prev_vessel"]:.2f}%<br>Quadrants: {comparison["prev_quads"]}/4</div>'
        f'<div><strong>Current</strong><br>Stage {comparison["curr_stage"]} ({current_result["diagnosis"]["stage_name"]})<br>'
        f'Lesion burden: {comparison["curr_lesion"]:.2f}%<br>Vessel density: {comparison["curr_vessel"]:.2f}%<br>Quadrants: {comparison["curr_quads"]}/4</div></div>'
        f'<div style="margin-top:8px;">Lesion burden change: <strong>{comparison["lesion_delta"]:+.2f} percentage points</strong><br>'
        f'Vessel density change: <strong>{comparison["vessel_delta"]:+.2f} percentage points</strong><br>'
        f'Affected quadrant change: <strong>{comparison["quads_delta"]:+d}</strong></div>'
        f'<div style="margin-top:8px; color:#475569;">Visual change detected between the submitted examinations. {registration}</div>'
        '<div style="font-size:11px; color:#64748b; margin-top:8px; font-style:italic;">Visual support only — not an independent diagnosis and not proof of clinical disease progression.</div>'
        '</div>'
    )
    return summary, comparison["diff_vis"]
