"""Gradio interface: builds the tabs, wires buttons to handlers, and launches the app."""

import os

import numpy as np
import gradio as gr

from core.config import AppConfig
from app.analysis import analyze_fundus, compare_longitudinal_images, create_sample_fundus
from app.patient import EYE_CHOICES, EYE_DEFAULT, PATIENT_ID_PLACEHOLDER, VIEW_CHOICES, view_visibility
from core.agents import RED_FLAG_SYMPTOMS
from app.chatbot import respond_to_clinical_query
from app.reports import (
    describe_reported_metrics,
    generate_document_pdf,
    generate_full_report_pdf,
    load_reported_test_metrics,
)
from app.triage import calculate_multimodal_risk, update_triage_routing
from app.ui_style import CUSTOM_CSS, HEAD_SCRIPT, HEADER_HTML, LAYOUT_CSS, SIDEBAR_HTML, theme

# System-specification text is driven by the notebook's saved test metrics, if present.
_REPORTED_METRICS = load_reported_test_metrics()
_REPORTED_METRICS_TEXT = describe_reported_metrics(_REPORTED_METRICS)
_BACKBONE_TEXT = (
    "EfficientNetB3 (ImageNet transfer learning, two-phase fine-tuning)."
    if _REPORTED_METRICS is not None
    else "EfficientNetB3 with historical saved weights. The corrected input-scale pipeline "
    "and duplicate-safe split have not yet been retrained and evaluated."
)


with gr.Blocks(title="RetinaTrace — DR Research Prototype") as demo:
    # Inject Custom Clinical Styling & Theme Detection
    gr.HTML(CUSTOM_CSS)

    # ── Sidebar (injected as fixed HTML, JS drives tab navigation) ───
    gr.HTML(SIDEBAR_HTML)

    # Session state
    pred_context_state = gr.State({})
    session_history_state = gr.State([])
    research_support_state = gr.State({})
    _diag_stage_name_state = gr.State("")
    _diag_conf_state = gr.State(0.0)
    _diag_urgency_state = gr.State("")
    _diag_followup_state = gr.State("")
    _diag_plan_state = gr.State("")

    # ── Top Title Strip ──────
    gr.HTML(HEADER_HTML)

    gr.Markdown(
        "> **Research prototype — not for clinical decisions.** This tool is still being tested, "
        "and its results have not been fully verified. Do not use its predictions to make medical "
        "decisions; always ask a qualified healthcare professional."
    )

    # 2. Main Workspace (custom compact horizontal split)
    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(label="📥 Upload Fundus Photo", type="numpy", height=180)

            with gr.Row(elem_classes=["action-row"]):
                submit_btn = gr.Button("🚀 Run Diagnostic Analysis", variant="primary", size="lg", scale=3, elem_classes=["action-btn"])
                btn_clear = gr.Button("🔄 Reset", variant="secondary", size="lg", scale=1)

            gr.Markdown("<div style='font-size:11px; font-weight:700; color:#64748b; margin:3px 0 1px 0;'>⚡ PRESET DEMOS (fixed illustrative values, not model output)</div>")
            with gr.Row(elem_classes=["quick-samples-row"]):
                btn_normal = gr.Button("🟢 Normal — preset demo", size="sm")
                btn_moderate = gr.Button("🟡 Moderate — preset demo", size="sm")
                btn_prolif = gr.Button("🔴 Proliferative — preset demo", size="sm")

            with gr.Accordion("🛡️ Safety gate & patient context", open=False):
                threshold_slider = gr.Slider(
                    minimum=0.50, maximum=0.95, value=0.70, step=0.05,
                    label="Governance Confidence Threshold",
                    info="Predictions below this confidence trigger an active safety override and withhold automated guidance.",
                )
                btn_override_test = gr.Button("🧪 Simulate Safety Override (Set to 95%)", variant="secondary", size="sm")
                red_flag_group = gr.CheckboxGroup(
                    choices=list(RED_FLAG_SYMPTOMS.values()), value=[],
                    label="Red-flag symptoms (patient-reported)",
                    info="Any ticked symptom makes the outcome URGENT — seek same-day eye care. It never changes the "
                         "predicted stage. Illustrative, not clinically validated.",
                )
                with gr.Row():
                    patient_id_box = gr.Textbox(label="Patient ID / MRN", value="", placeholder=PATIENT_ID_PLACEHOLDER,
                                                max_lines=1, scale=3)
                    eye_dropdown = gr.Dropdown(choices=EYE_CHOICES, value=EYE_DEFAULT, label="Eye", scale=2)
                gr.Markdown("**Optional patient context**")
                patient_age = gr.Slider(minimum=18, maximum=90, value=55, step=1, label="Patient Age (years)")
                diabetes_type = gr.Dropdown(
                    choices=["Type 1", "Type 2", "Gestational", "Not Specified"],
                    value="Type 2", label="Diabetes Type"
                )
                hba1c_level = gr.Slider(minimum=5.0, maximum=14.0, value=7.5, step=0.1, label="HbA1c (%)")
                diabetes_duration = gr.Slider(minimum=0, maximum=40, value=10, step=1, label="Duration of Diabetes (years)")
                systolic_bp = gr.Slider(minimum=90, maximum=220, value=135, step=1, label="Systolic Blood Pressure (mmHg)")

        with gr.Column(scale=7):
            # Structured Tabs
            with gr.Tabs():
                # Tab 1: Primary Diagnosis & 3-Layer Explainability
                with gr.TabItem("🏥 Diagnostic Assessment & Explainability", elem_id="rg-tab-diag"):
                    view_toggle = gr.Radio(VIEW_CHOICES, value="Clinician", label="View", elem_id="rt-view-toggle",
                                           info="Patient view shows a plain-language summary (illustrative, not clinically validated).")
                    patient_card_view = gr.HTML("", visible=False)
                    hero_diagnosis = gr.HTML()
                    uncertainty_banner_top = gr.HTML()
                    prob_distribution = gr.Label(label="5-Stage Disease Probability Distribution (Softmax)", num_top_classes=5)

                    with gr.Accordion("🔬 Explainability & visual evidence", open=False) as explain_accordion:
                        gr.Markdown("Grad-CAM, U-Net, classical CV, and experimental research measurements")
                        with gr.Row():
                            with gr.Column(scale=5):
                                gr.Markdown("**Layer 2: Regional Attention (Grad-CAM)**")
                                overlay_cam_view = gr.Image(label="Grad-CAM Saliency Overlay", type="numpy", height=200)
                            with gr.Column(scale=5):
                                gr.Markdown("**Layer 3: U-Net pseudo-mask demo (not segmentation)**")
                                lesion_seg_view = gr.Image(label="U-Net pseudo-mask output (demo)", type="numpy", height=200)
                        with gr.Row():
                            with gr.Column(scale=5):
                                gr.Markdown("**Classical CV Vessel Analysis**")
                                vessel_overlay_view = gr.Image(label="Retinal Vessel Overlay", type="numpy", height=180)
                            with gr.Column(scale=5):
                                gr.Markdown("**Optic-Disc Localisation**")
                                optic_disc_overlay_view = gr.Image(label="Optic-Disc Localisation Overlay", type="numpy", height=180)
                        classical_cv_status_view = gr.HTML(
                            "<div style='color:#64748b; font-size:12px;'>Visual support only — not an independent diagnosis.</div>"
                        )
                        with gr.Row():
                            overlap_view = gr.Image(label="Attention–Lesion Agreement", type="numpy", height=180)
                            severity_map_view = gr.Image(label="Retinal Visual-Abnormality Map", type="numpy", height=180)
                        advanced_analysis_view = gr.HTML()
                        lesion_burden_view = gr.HTML()
                        quadrant_text = gr.Markdown()
                        quadrant_chart_view = gr.HTML()
                        confidence_margin_view = gr.HTML()

                # Tab 2: Case-Based Reasoning (CBR) Evidence
                with gr.TabItem("📚 Case-Based Reasoning (CBR) Evidence", elem_id="rg-tab-cbr"):
                    gr.Markdown("### 🔎 Similar Dataset-Labeled Fundus Images")
                    gr.Markdown(
                        "The query image was projected into the 256-D penultimate feature bottleneck. "
                        "When image paths are available, the **Top-3 closest reference images** are retrieved via cosine similarity. "
                        "Displayed stages are dataset labels, not independent clinical confirmation."
                    )
                    gallery_view = gr.Gallery(columns=3, rows=1, height=260, object_fit="contain")

                # Tab 3: Clinical Care Protocol & EHR Note
                with gr.TabItem("📋 Clinical Management & EHR Note", elem_id="rg-tab-care"):
                    advisory_view = gr.HTML()
                    gr.Markdown("### 📄 Exportable Electronic Health Record (EHR) Summary Note")
                    ehr_note_box = gr.Textbox(label="Clinical Session Note", lines=12, interactive=False, elem_classes=["clinical-document-box"])
                    ehr_pdf_btn = gr.Button("📄 Download Session Note (PDF)", variant="secondary", size="sm")
                    ehr_pdf_file = gr.File(label="Session Note PDF", interactive=False)

                # Tab 4: AI Clinical Chatbot & SaMD Guidelines
                with gr.TabItem("💬 AI Clinical Chatbot", elem_id="rg-tab-chat"):
                    gr.HTML('<button class="rg-chat-close" onclick="window.retinaCloseChat()" title="Close AI Clinical Assistant">&times;</button>')
                    gr.Markdown("""
                    ### 🤖 RetinaTrace Clinical Knowledge Assistant
                    Ask about diabetic retinopathy, the RetinaTrace model, or clinical guidelines.
                    """)
                    gr.Markdown("**Quick Prompts:**")
                    with gr.Row():
                        chip_classify = gr.Button("Explain why this was classified", size="sm", variant="secondary")
                        chip_rule421 = gr.Button("What is the 4-2-1 rule?", size="sm", variant="secondary")
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot_widget = gr.Chatbot(
                                label="Clinical Knowledge Assistant",
                                height=180,
                                value=[],
                            )
                            with gr.Row(elem_id="rg-chat-input-row", elem_classes=["rg-chat-input-row"]):
                                chat_input = gr.Textbox(
                                    placeholder="Ask a question about DR staging, the model, or clinical guidelines…",
                                    label="",
                                    scale=5,
                                    container=False,
                                )
                                chat_send_btn = gr.Button("Send", variant="primary", scale=1)
                            chat_clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                        with gr.Column(scale=2, elem_classes=["rg-chat-specs"]):
                            gr.Markdown(rf"""
                            ### ⚙️ System Specifications:
                            * **Deep Learning Backbone:** {_BACKBONE_TEXT}
                            * **Ordinal Metric (QWK):** {_REPORTED_METRICS_TEXT}
                            * **Threshold Workflow Flag:** Low-confidence predictions are flagged for specialist review; the prototype cannot enforce a clinical referral.
                            * **Intended Use:** Coursework prototype for research and demonstration only; not validated for clinical use.
                            """)

                # Tab 5: Multimodal Clinical Triage & Risk Simulator
                with gr.TabItem("🚦 Multimodal Triage & 10-Yr Risk Simulator", elem_id="rg-tab-triage"):
                    gr.Markdown("""
                    ### 🏥 Illustrative Triage & Progression Risk Simulator (Educational)
                    Combines the **image-derived DR severity stage** with systemic indicators 
                    (**HbA1c, Diabetes Duration, Blood Pressure, Patient Age**) using hand-set multipliers 
                    loosely inspired by the **UKPDS 33** and **WESDR** cohorts.

                    > ⚠️ **Illustrative, not validated.** These risk figures and routing tiers are a 
                    > demonstration only and must not be used for clinical decisions.
                    """)
                    initial_risk_html, initial_ticket_text = calculate_multimodal_risk(2, 7.5, 10.0, 55.0, 135.0, "Type 2")
                    triage_risk_card = gr.HTML(initial_risk_html)
                    with gr.Row():
                        btn_recalc_triage = gr.Button("⚡ Recalculate Illustrative Risk & Follow-Up", variant="primary", scale=2)
                    gr.Markdown("### 📄 Example Referral Summary (Illustrative)")
                    referral_ticket_view = gr.Textbox(
                        label="Example referral summary — illustrative, not for clinical use", 
                        value=initial_ticket_text,
                        lines=11, 
                        interactive=False,
                        elem_classes=["clinical-document-box"],
                    )
                    referral_pdf_btn = gr.Button("📄 Download Example Referral Summary (PDF)", variant="secondary", size="sm")
                    referral_pdf_file = gr.File(label="Example Referral Summary PDF", interactive=False)

                # Tab 6: Prediction History
                with gr.TabItem("📜 Session Prediction History", elem_id="rg-tab-history"):
                    gr.HTML('<button class="rg-history-close" onclick="window.retinaCloseHistory()" title="Close Prediction History">&times;</button>')
                    gr.Markdown("### 🕐 Prediction History — Current Session")
                    gr.Markdown("Each analysis run is logged here for comparison during the same session. History resets on page refresh.")
                    session_history_view = gr.HTML('<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet.</div>')

                # Tab 7: Image Comparison & Download
                with gr.TabItem("🖼️ Image Comparison & Report", elem_id="rg-tab-compare"):
                    gr.Markdown("### 📸 Original vs. Preprocessed Fundus Image")
                    gr.Markdown("Left: raw upload. Right: after Ben Graham enhancement, circular crop, and 224×224 resize.")
                    with gr.Row():
                        original_img_view = gr.Image(label="Original Upload", type="numpy", height=260, interactive=False)
                        processed_img_view = gr.Image(label="Preprocessed (Ben Graham Enhanced)", type="numpy", height=260, interactive=False)
                    quality_warning_view = gr.HTML('<div style="color:#94a3b8; font-size:12px;">Upload an image to see quality assessment.</div>')
                    gr.Markdown("---")
                    gr.Markdown("### 📥 Download Full Diagnosis Report")
                    gr.Markdown("Downloads a formatted PDF containing the diagnosis, confidence, probabilities, advisory, and clinical session note.")
                    download_btn = gr.Button("⬇️ Generate & Download Report (PDF)", variant="primary")
                    download_file = gr.File(label="Download PDF", visible=False)

                with gr.TabItem("📊 Longitudinal Analysis", elem_id="rg-tab-longitudinal"):
                    gr.Markdown("### Optional Previous vs Current Examination Comparison")
                    gr.Markdown("Do not assume the images belong to the same patient. Results are visual comparisons only, not confirmed disease progression.")
                    with gr.Row():
                        longitudinal_previous = gr.Image(label="Previous Examination", type="numpy", height=240)
                        longitudinal_current = gr.Image(label="Current Examination", type="numpy", height=240)
                    longitudinal_compare_btn = gr.Button("Compare Examinations", variant="primary")
                    longitudinal_summary_view = gr.HTML("<div style='color:#64748b;'>Submit two images to begin.</div>")
                    longitudinal_diff_view = gr.Image(label="Registration/Difference Visualisation", type="numpy", height=240)

            status_banner = gr.HTML("")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Event Connections
    # ─────────────────────────────────────────────────────────────────────────
    # Reusable analysis outputs tuple for all analyze_fundus call sites
    analysis_outputs = [
        status_banner,
        hero_diagnosis,
        prob_distribution,
        overlay_cam_view,
        lesion_seg_view,
        vessel_overlay_view,
        optic_disc_overlay_view,
        classical_cv_status_view,
        overlap_view,
        severity_map_view,
        advanced_analysis_view,
        quadrant_text,
        gallery_view,
        advisory_view,
        ehr_note_box,
        lesion_burden_view,
        quadrant_chart_view,
        confidence_margin_view,
        pred_context_state,
        session_history_state,
        research_support_state,
        uncertainty_banner_top,
        processed_img_view,
        session_history_view,
        _diag_stage_name_state,
        _diag_conf_state,
        _diag_urgency_state,
        _diag_followup_state,
        _diag_plan_state,
        quality_warning_view,
        patient_card_view,
    ]
    analysis_inputs = [input_image, threshold_slider, session_history_state, red_flag_group, patient_id_box, eye_dropdown]

    # Main Analysis Event
    submit_btn.click(
        fn=analyze_fundus,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    download_btn.click(
        fn=generate_full_report_pdf,
        inputs=[_diag_stage_name_state, _diag_conf_state, prob_distribution,
            _diag_urgency_state, _diag_followup_state, _diag_plan_state, ehr_note_box,
            research_support_state, input_image, overlay_cam_view, lesion_seg_view, pred_context_state],
        outputs=[download_file],
    ).then(
        fn=lambda: gr.File(visible=True),
        outputs=[download_file],
    )

    ehr_pdf_btn.click(
        fn=lambda body: generate_document_pdf("RetinaTrace Clinical Session Note", body),
        inputs=[ehr_note_box],
        outputs=[ehr_pdf_file],
    )
    referral_pdf_btn.click(
        fn=lambda body: generate_document_pdf("RetinaTrace Example Referral Summary (Illustrative)", body),
        inputs=[referral_ticket_view],
        outputs=[referral_pdf_file],
    )

    longitudinal_compare_btn.click(
        fn=compare_longitudinal_images,
        inputs=[longitudinal_previous, longitudinal_current, patient_id_box, eye_dropdown],
        outputs=[longitudinal_summary_view, longitudinal_diff_view],
    )

    # Interactive Multimodal Risk Recalculation Handlers
    btn_recalc_triage.click(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    hba1c_level.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    diabetes_duration.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    systolic_bp.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    patient_age.release(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )
    diabetes_type.change(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    )

    def reset_workspace():
        empty_img = np.zeros((AppConfig.IMG_SIZE, AppConfig.IMG_SIZE, 3), dtype=np.uint8)
        initial_banner = "<div class='card'><em>Upload a retinal fundus photograph, or click a preset demo button to see fixed example values.</em></div>"
        default_risk_html, default_ticket_text = calculate_multimodal_risk(0, 7.0, 5.0, 50.0, 120.0, "Type 2")
        return (
            None,
            0.70,
            initial_banner,
            "",
            {},
            empty_img,
            empty_img,
            empty_img,
            empty_img,
            "<div style='color:#64748b; font-size:12px;'>Visual support only — not an independent diagnosis.</div>",
            empty_img,
            empty_img,
            "",
            "",
            [],
            "",
            "",
            "",
            "",
            "",
            default_risk_html,
            default_ticket_text,
            {},
            [],
            {},
            "",
            empty_img,
            '<div style="color:#94a3b8; font-size:13px; padding:12px;">No predictions yet.</div>',
            "",
            0.0,
            "",
            "",
            "",
            '<div style="color:#94a3b8; font-size:12px;">Upload an image to see quality assessment.</div>',
            "",
            [],
            "",
            EYE_DEFAULT,
        )

    btn_clear.click(
        fn=reset_workspace,
        outputs=[
            input_image,
            threshold_slider,
            status_banner,
            hero_diagnosis,
            prob_distribution,
            overlay_cam_view,
            lesion_seg_view,
            vessel_overlay_view,
            optic_disc_overlay_view,
            classical_cv_status_view,
            overlap_view,
            severity_map_view,
            advanced_analysis_view,
            quadrant_text,
            gallery_view,
            advisory_view,
            ehr_note_box,
            lesion_burden_view,
            quadrant_chart_view,
            confidence_margin_view,
            triage_risk_card,
            referral_ticket_view,
            pred_context_state,
            session_history_state,
            research_support_state,
            uncertainty_banner_top,
            processed_img_view,
            session_history_view,
            _diag_stage_name_state,
            _diag_conf_state,
            _diag_urgency_state,
            _diag_followup_state,
            _diag_plan_state,
            quality_warning_view,
            patient_card_view,
            red_flag_group,
            patient_id_box,
            eye_dropdown,
        ]
    )

    # Clinician / Patient view: Patient view hides the technical panels and shows the plain-language card.
    def apply_view(view):
        shown = view_visibility(view)
        return (gr.update(visible=shown["patient_card"]), gr.update(visible=shown["probabilities"]),
                gr.update(visible=shown["explainability"]), gr.update(visible=shown["retrieval"]))

    view_toggle.change(
        fn=apply_view,
        inputs=[view_toggle],
        outputs=[patient_card_view, prob_distribution, explain_accordion, gallery_view],
    )

    # Live threshold adjustment re-evaluates active prediction upon release
    threshold_slider.release(
        fn=analyze_fundus,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    # Preset Sample Button Handlers
    btn_normal.click(
        fn=lambda: (create_sample_fundus(0), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=lambda img, thr, hist, flags, pid, eye: analyze_fundus(img, thr, hist, flags, pid, eye, preset_stage=0),
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    btn_moderate.click(
        fn=lambda: (create_sample_fundus(2), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=lambda img, thr, hist, flags, pid, eye: analyze_fundus(img, thr, hist, flags, pid, eye, preset_stage=2),
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    btn_prolif.click(
        fn=lambda: (create_sample_fundus(4), 0.70),
        outputs=[input_image, threshold_slider],
    ).then(
        fn=lambda img, thr, hist, flags, pid, eye: analyze_fundus(img, thr, hist, flags, pid, eye, preset_stage=4),
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )

    # Safety Override Simulation Button Handler (Sets threshold to 95% and executes)
    btn_override_test.click(
        fn=lambda: 0.95,
        outputs=[threshold_slider],
    ).then(
        fn=analyze_fundus,
        inputs=analysis_inputs,
        outputs=analysis_outputs,
    ).then(
        fn=update_triage_routing,
        inputs=[prob_distribution, hba1c_level, diabetes_duration, patient_age, systolic_bp, diabetes_type, pred_context_state],
        outputs=[triage_risk_card, referral_ticket_view],
    ).then(
        fn=lambda img: img,
        inputs=[input_image],
        outputs=[original_img_view],
    )



    # ── Chatbot Event Handlers ──────────────────────────────────────────────
    # -- Quick-Prompt Chip Handlers
    chip_classify.click(fn=lambda: 'Explain why this image was classified with the current DR stage.', outputs=[chat_input])
    chip_rule421.click(fn=lambda: 'What is the 4-2-1 rule in diabetic retinopathy?', outputs=[chat_input])

    chat_send_btn.click(
        fn=respond_to_clinical_query,
        inputs=[chat_input, chatbot_widget, pred_context_state],
        outputs=[chatbot_widget, chat_input],
    )
    chat_input.submit(
        fn=respond_to_clinical_query,
        inputs=[chat_input, chatbot_widget, pred_context_state],
        outputs=[chatbot_widget, chat_input],
    )
    chat_clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot_widget, chat_input])


def launch() -> None:
    """Launch the RetinaTrace interface.

    The port comes from PORT (set by Render and similar hosts), then GRADIO_SERVER_PORT
    (Gradio), then 7860. The app listens on 0.0.0.0 unless
    GRADIO_SERVER_NAME says otherwise. No public share link is created.
    """
    port = os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or "7860"
    demo.launch(
        head=HEAD_SCRIPT,
        theme=theme,
        css=LAYOUT_CSS,
        share=False,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(port),
    )


if __name__ == "__main__":
    launch()
