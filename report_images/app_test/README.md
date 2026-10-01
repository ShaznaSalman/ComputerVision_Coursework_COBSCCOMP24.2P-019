# RetinaTrace app — full test (1 October 2026)

App run locally (`python app.py`, http://127.0.0.1:7861) with `checkpoints/final_model.weights.h5`. Driven with Playwright and the installed Chromium-based Microsoft Edge (no separate Chromium is installed on this machine), viewport 1600×1000, full-page screenshots. PDF items show page 1 of the generated PDF rendered to PNG.

## Five stages from real uploads (files in `app/samples/`, validation-split images)

| File | True grade (validation split) | App prediction | Confidence | Correct | Source of result |
|---|---|---|---|---|---|
| stage0_no_dr.jpg | 0: No DR | 0: No DR | 74.7% | yes | model prediction |
| stage1_mild.jpg | 1: Mild | 1: Mild | 58.0% | yes | model prediction |
| stage2_moderate.jpg | 2: Moderate | 2: Moderate | 70.1% | yes | model prediction |
| stage3_severe.jpg | 3: Severe | 4: Proliferative DR | 53.4% | **no** (model error, kept as evidence) | model prediction |
| stage4_proliferative.jpg | 4: Proliferative DR | 4: Proliferative DR | 53.6% | yes | model prediction |

The Severe sample is predicted Proliferative DR (53.4%): an honest model error, consistent with the weak Severe/Proliferative recall in the report. All five uploads are real model predictions; before the fix, uploading these files returned fixed preset values.

## Screenshot index

| File | What was done | What the app showed | Result |
|---|---|---|---|
| 01_home_dark.png | Opened the app (default theme) | Home screen; dark mode = True | PASS |
| 02_home_light.png | Clicked the theme toggle | Home screen; light mode = True | PASS |
| 03_upload_stage0_no_dr.png | Uploaded app/samples/stage0_no_dr.jpg (true grade 0: No DR) and ran the analysis | Stage 0: No DR, 74.7% (model prediction; correct); 5.9 s | PASS |
| 04_upload_stage1_mild.png | Uploaded app/samples/stage1_mild.jpg (true grade 1: Mild) and ran the analysis | Stage 1: Mild, 58.0% (model prediction; correct); 5.3 s | PASS |
| 05_upload_stage2_moderate.png | Uploaded app/samples/stage2_moderate.jpg (true grade 2: Moderate) and ran the analysis | Stage 2: Moderate, 70.1% (model prediction; correct); 5.6 s | PASS |
| 06_upload_stage3_severe.png | Uploaded app/samples/stage3_severe.jpg (true grade 3: Severe) and ran the analysis | Stage 4: Proliferative DR, 53.4% (model prediction; WRONG (honest model error)); 5.6 s | PASS |
| 07_upload_stage4_proliferative.png | Uploaded app/samples/stage4_proliferative.jpg (true grade 4: Proliferative DR) and ran the analysis | Stage 4: Proliferative DR, 53.6% (model prediction; correct); 5.3 s | PASS |
| 08_explainability_open.png | Opened 'Explainability & visual evidence' after the last upload | Grad-CAM overlay, U-Net demo panel, vessel and optic-disc overlays | PASS |
| 09_preset_normal.png | Clicked 'Normal — preset demo' | Stage 0: No DR, 93.8%, labelled as preset = True; heading shows 'PRESET DEMOS' = True | PASS |
| 10_preset_moderate.png | Clicked 'Moderate — preset demo' | Stage 2: Moderate, 88.5%, labelled as preset = True; heading shows 'PRESET DEMOS' = True | PASS |
| 11_preset_proliferative.png | Clicked 'Proliferative — preset demo' | Stage 4: Proliferative DR, 92.1%, labelled as preset = True; heading shows 'PRESET DEMOS' = True | PASS |
| 12_tab_cbr_similar_cases.png | Opened the CBR tab after a real upload (Moderate sample) | 3 gallery images shown with dataset grades and similarity | PASS |
| 13_tab_care_ehr_note.png | Opened the Clinical Management & EHR Note tab | EHR note shown (1093 characters) | PASS |
| 14_tab_triage_recalculated.png | Set HbA1c 9.5%, age 68, systolic BP 165, then recalculated triage | Triage uses the new values = True; illustrative 10-year risk 71.3% | PASS |
| 15_tab_history.png | Opened Prediction History after 9 analyses (5 uploads, 3 presets, 1 upload) | 9 predictions listed (demo presets marked '(demo preset)') | PASS |
| 16_tab_image_comparison.png | Opened Image Comparison & Report | Original and preprocessed images side by side | PASS |
| 17_pdf_full_report_page1.png | Generated the full diagnostic report PDF | PDF with 3 page(s), 938,171 bytes; page 1 rendered | PASS |
| 18_pdf_ehr_note_page1.png | Generated the EHR session note PDF | PDF with 1 page(s), 2,746 bytes; page 1 rendered | PASS |
| 19_pdf_referral_summary_page1.png | Generated the referral summary PDF | PDF with 1 page(s), 2,609 bytes; page 1 rendered | PASS |
| 20_tab_longitudinal.png | Compared stage1_mild.jpg (previous) with stage4_proliferative.jpg (current) | Longitudinal comparison: shown | PASS |
| 21_gate_confidence_95.png | Uploaded the No DR sample and set the threshold to 95% (Simulate Safety Override) | Banner: 🚨 GOVERNANCE STATUS: FLAGGED FOR HUMAN REVIEW Reason: Model confidence (74.7%) is below your threshold (95%). Patient safety: automated treatment guid | PASS |
| 22_gate_quality_fail.png | Threshold 50%; uploaded a heavily blurred copy of the Moderate sample | Prediction 4: Proliferative DR 99.9%; banner: 🚨 GOVERNANCE STATUS: FLAGGED FOR HUMAN REVIEW Reason: Image quality check failed (see the image-quality card). Patient safety: automated treatment gui; confidence also named = False; 'HIGH CONFIDENCE' card hidden = True | PASS |
| 23_invalid_black.png | Uploaded a non-fundus image (black.png) and ran the analysis | Banner: 🚫 Not a usable fundus photograph — please upload a colour retinal photograph No circular retinal region was found. Overa; stage shown = False | PASS |
| 24_invalid_noise.png | Uploaded a non-fundus image (noise.png) and ran the analysis | Banner: 🚫 Not a usable fundus photograph — please upload a colour retinal photograph The image has no dark background around a c; stage shown = False | PASS |
| 25_invalid_mindmap.png | Uploaded a non-fundus image (mindmap.png) and ran the analysis | Banner: 🚫 Not a usable fundus photograph — please upload a colour retinal photograph The image has no dark background around a c; stage shown = False | PASS |
| 26_chat_421_rule.png | Chatbot: 'What is the 4-2-1 rule?' | nal photocoagulation (PRP) prophylactically in high-risk patients. This is AI-assisted information. Confirm clinical decisions with a qualified ophthalmologist. | PASS |
| 27_chat_explain_classification.png | After uploading stage4_proliferative.jpg: 'Explain why this was classified' | .0% of the retinal area. Because this result is AI-assisted, it should be confirmed by an ophthalmologist. This is AI-assisted information. Confirm clinical decisions with a qualified ophthalmologist. | PASS |
| 28_chat_out_of_scope_refusal.png | Chatbot: 'What dose of insulin should I take?' | ould I take? I cannot recommend starting, stopping, or changing treatment from an AI chat response. Please discuss medication or procedures with a qualified ophthalmologist or the patient's clinician. | PASS |
| 29_theme_light_with_flagged_result.png | Switched to light mode with a flagged (quality-fail) result showing | Light mode = True; banners readable in light mode | PASS |
| 30_theme_dark_with_flagged_result.png | Switched back to dark mode | Same banners in dark mode | PASS |
| 31_reset.png | Clicked Reset | 5-Stage Disease Probability Distribution (Softmax) 🔬 Explainability & visual evidence ▼ | PASS |
| 32_new_upload_after_reset.png | Uploaded stage3_severe.jpg after a previous result and Reset | Stage 4: Proliferative DR, 53.4% | PASS |

## Input-validity gate (run on the image files, `app/analysis.py: assess_fundus_validity`)

| Input | Expected | Result |
|---|---|---|
| app/cbr_reference/ (250 fundus images) | pass | 250/250 pass |
| app/samples/ (5 images) | pass | 5/5 pass |
| heavily blurred Moderate sample | pass (then fails the quality check) | pass |
| black, white, grey images | fail | fail |
| random noise, dark random noise | fail | fail |
| mindmap.png, architecture diagram, two charts | fail | fail |
| app screenshot | fail | fail |

## Timing, errors and logs

- Time per analysis (click to finished output, CPU, AMD Ryzen 5 7535HS): stage0_no_dr.jpg 5.9 s, stage1_mild.jpg 5.3 s, stage2_moderate.jpg 5.6 s, stage3_severe.jpg 5.6 s, stage4_proliferative.jpg 5.3 s, tabs: stage2_moderate.jpg 5.2 s.
- Browser console errors: 0.
- Python tracebacks in the server log: 0 in the final run. (A first run found one real environment problem: `reportlab`, listed in requirements.txt, was not installed on this machine, so the three PDF buttons failed; it was installed and the run repeated.)

## Observations

- The heavily blurred image is predicted Proliferative DR with 99.9% confidence. The model is over-confident on images unlike its training data; the image-quality gate is what flags it.
- The out-of-scope question 'What dose of insulin should I take?' was not refused before this round (it got a generic fallback answer); dosing and medicine questions are now refused.
- Screenshot 31 (Reset) shows the cleared result area; screenshot 32 shows that a new upload after Reset runs normally.
- The sidebar appears twice in some full-page screenshots (for example 14): this is how the fixed-position sidebar renders in a full-page capture, not a layout bug in the browser window.
