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

## Red flags, patient ID/eye, Patient view and video-demo uploads (second run, 1 October 2026)

Same set-up (local app, Playwright with Chromium-based Edge, 1600×1000). Red flags, patient ID/eye and Patient view are illustrative, not clinically validated.

| File | What was done | What the app showed | Result |
|---|---|---|---|
| - | Switched back to Clinician view | probabilities and explainability panels restored = True | PASS |
| - | app_06: uploaded 2_81.8_891392c9683c.png, asked 'What is the 4-2-1 rule?' | Full answer shown = True; reply ends: s is AI-assisted information. Confirm clinical decisions with a qualified ophthalmologist. | PASS |
| 33_red_flag_high_confidence.png | Uploaded 0_100.0_165634a6167e.png (No DR, 100.0%) and ticked 'Sudden loss or major drop in vision' | Stage still 0: No DR 100.0%; banner: 🚨 URGENT — seek same-day eye care Reason: Patient-reported red-flag symptoms: Sudden loss or major drop in vision These symptoms can signal retinal detachment, vitreous h | PASS |
| 34_red_flag_plus_confidence_gate.png | Uploaded 2_81.8_891392c9683c.png with the red flag ticked and the threshold at 95% | Stage 2: Moderate 81.8%; both reasons listed = True; banner: 🚨 URGENT — seek same-day eye care Reasons: Patient-reported red-flag symptoms: Sudden loss or major drop in vision Model confidence (81.8%) is below your threshold (95%). | PASS |
| 35_patient_id_eye_filled.png | Filled Patient ID 'DEMO-0001' and chose 'Right eye (OD)' in the safety-gate accordion | ID box = DEMO-0001; eye = Right eye (OD); red-flag checkboxes visible | PASS |
| 36_pdf_full_report_id_eye_red_flag_page1.png | Generated the full-report PDF for case 34 | Page 1 header has ID DEMO-0001, Right eye (OD), date/time, 'model prediction', URGENT outcome and the red-flag symptom = True | PASS |
| 37_patient_view_not_flagged.png | Uploaded 2_81.8_891392c9683c.png (threshold 70%, no red flags) and switched to Patient view | Card: 💬 What this means for you Your photo shows some damage to the tiny blood vessels at the back of your eye. Your sight may still be fine, but an eye spe; probabilities/Grad-CAM hidden = True | PASS |
| 38_patient_view_flagged.png | Same image with the red flag ticked, still in Patient view | Card: 💬 What this means for you You told us about symptoms that can be an emergency. Please get eye care today. Do not wait because of this result. This is a research prototype | PASS |
| 39_video_demo_0_no_dr.png | Uploaded video_demo_images/0_no_dr/0_100.0_165634a6167e.png (true grade 0: No DR) | App: Stage 0: No DR, 100.0% (file name says 0, 100.0%) — match = True | PASS |
| 40_video_demo_1_mild.png | Uploaded video_demo_images/1_mild/1_90.7_384631079d1e.png (true grade 1: Mild) | App: Stage 1: Mild, 90.7% (file name says 1, 90.7%) — match = True | PASS |
| 41_video_demo_2_moderate.png | Uploaded video_demo_images/2_moderate/2_81.8_891392c9683c.png (true grade 2: Moderate) | App: Stage 2: Moderate, 81.8% (file name says 2, 81.8%) — match = True | PASS |
| 42_video_demo_3_severe.png | Uploaded video_demo_images/3_severe/3_96.3_f64214bed40e.png (true grade 3: Severe) | App: Stage 3: Severe, 96.3% (file name says 3, 96.3%) — match = True | PASS |
| 43_video_demo_4_proliferative.png | Uploaded video_demo_images/4_proliferative/4_95.1_b90bc89ce8d8.png (true grade 4: Proliferative DR) | App: Stage 4: Proliferative DR, 95.1% (file name says 4, 95.1%) — match = True | PASS |
| 44_video_demo_misclassified.png | Uploaded video_demo_images/misclassified/true-1_pred-2_57.0_6298468d7d75.png (true grade 1: Mild) | App: Stage 2: Moderate, 57.0% (file name says 2, 57.0%) — match = True | PASS |

### Video-demo uploads: app result vs file name

| File | File name says | App showed | Match |
|---|---|---|---|
| video_demo_images/0_no_dr/0_100.0_165634a6167e.png | 0 100.0% | 0 100.0% | yes |
| video_demo_images/1_mild/1_90.7_384631079d1e.png | 1 90.7% | 1 90.7% | yes |
| video_demo_images/2_moderate/2_81.8_891392c9683c.png | 2 81.8% | 2 81.8% | yes |
| video_demo_images/3_severe/3_96.3_f64214bed40e.png | 3 96.3% | 3 96.3% | yes |
| video_demo_images/4_proliferative/4_95.1_b90bc89ce8d8.png | 4 95.1% | 4 95.1% | yes |
| video_demo_images/misclassified/true-1_pred-2_57.0_6298468d7d75.png | 2 57.0% | 2 57.0% | yes |

Browser console errors in this run: 0. Python tracebacks in the server log: 0.

## App vs notebook on all 550 test images (`app_vs_notebook_check.csv`)

Each test image was run through the app's real path (`preprocess_image` → `DiagnosisAgent.process`, no preset) and compared with `report_images/test_predictions.npz`.

| Metric | App | Notebook |
|---|---|---|
| images | 550 | 550 |
| app prediction equals notebook y_pred_argmax | 549/550 |  |
| largest absolute probability difference vs notebook softmax | 0.006609 |  |
| mean absolute probability difference (max per image) | 0.000873 |  |
| accuracy vs y_true | 0.762 | 0.760 |
| stage 0 No DR: correct / total (recall) | 268/271 (0.989) | 268/271 (0.989) |
| stage 1 Mild: correct / total (recall) | 39/56 (0.696) | 39/56 (0.696) |
| stage 2 Moderate: correct / total (recall) | 83/150 (0.553) | 82/150 (0.547) |
| stage 3 Severe: correct / total (recall) | 17/29 (0.586) | 17/29 (0.586) |
| stage 4 Proliferative DR: correct / total (recall) | 12/44 (0.273) | 12/44 (0.273) |
| images passing the validity gate | 550/550 |  |
| images passing the image-quality check | 402/550 |  |
| disagreement 91e8af9ceee9 | app 2 (0.425) | notebook 1 (0.425) |

The single disagreement is a near-tie (top stage at 42.5% in both runs); the app picks the correct stage there, so its accuracy is 0.762 against the notebook's 0.760. Probability differences come from float32 CPU inference in the app against mixed-precision GPU inference on PNG-cached inputs in the notebook.
148 of the 550 test images fail the app's image-quality check (123 for the blur threshold, 26 for poor illumination; one fails both), so the quality gate would flag about a quarter of real APTOS test photographs.
