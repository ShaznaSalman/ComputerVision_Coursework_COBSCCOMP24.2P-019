---
title: RetinaTrace AI - Diabetic Retinopathy Research Prototype
emoji: 👁️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.27.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: other
short_description: Coursework prototype for diabetic-retinopathy image analysis
---

# Diabetic Retinopathy Stage Detection: Coursework Research Prototype

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Academic%20Coursework-lightgrey.svg)]()

> **BSc (Hons) in Computing (Batch 2024.2) — Computer Vision Coursework**  
> **Coventry University (UK) & National Institute of Business Management (NIBM)**  
> **Student Registered Name:** M.S.F. Shazna  
> **Coventry Index:** 16115859 | **NIBM Index:** COBSCCOMP242P-019  
> **Assessment Weighting:** 100 Marks (Individual Project Report with Video Demonstration)

---

## 📋 Executive Summary & Project Purpose

Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults worldwide. Early detection and precise severity staging are critical: while early stages (Mild/Moderate NPDR) require monitoring and primary care glycemic optimization, advanced stages (Severe NPDR and Proliferative DR) demand urgent specialist laser photocoagulation or anti-VEGF pharmacotherapy to avert permanent visual loss.

This coursework project implements a research prototype for **five-class diabetic-retinopathy image analysis** (ICDR Stages 0 to 4), using an **EfficientNetB3** classifier and experimental explainability and retrieval components. It has not been clinically validated and is not intended for diagnosis or treatment decisions.
1. **Embedding-Based Similar-Case Retrieval (Case-Based Reasoning):** Prototype retrieval using image embeddings.
2. **Multi-Agent Decision Workflow:** Separate components for classification, visual explanations, advisory text, and a threshold-based low-confidence flag; this workflow does not enforce clinical review.

---

## 📊 Dataset Specification

* **Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) (Kaggle competition). Only the **3,662 labelled images** in `train_images/` are used (`train.csv`: `id_code`, `diagnosis`); the competition's `test_images/` have no public labels and are not used.
* **Disease staging (International Clinical Diabetic Retinopathy scale), published class counts** — the notebook recomputes them in Section 1.9:
  * `Stage 0`: No DR (1,805)
  * `Stage 1`: Mild Non-Proliferative DR (NPDR) (370)
  * `Stage 2`: Moderate NPDR (999)
  * `Stage 3`: Severe NPDR (193)
  * `Stage 4`: Proliferative DR (PDR) (295)
* **Class imbalance:** about **9.4×** between the largest (No DR) and smallest (Severe) class. Class weights are computed from the training split only: `balanced` (default), `sqrt` (softer) or `none`. On-the-fly oversampling is compared as a separate pilot trial and is never combined with class weights.
* **Duplicate-aware split:** every image gets a perceptual hash; near-duplicates (Hamming distance ≤ 4) are grouped with union-find, and whole groups are assigned to a **70 / 15 / 15** train / validation / test split. Groups larger than 50 images always go to training; every other group is placed whole, largest first, into the split that most needs its classes, and the best of 200 seeded orderings (closest to 70/15/15 overall and per class, using labels and group IDs only) is kept. The notebook asserts that no image or duplicate group appears in two splits; groups with conflicting grades are kept together and listed in `duplicate_groups.csv`. The split files are saved to `splits/`.
* **Limitations:** single source (Aravind Eye Hospital, India), so results may not transfer to other cameras or populations and there is no external test set; **no patient IDs**, so two photographs of the same patient that are not near-duplicates can land in different splits; a small test set (550 images, only 29 Severe), so per-class test metrics have wide uncertainty; and No DR images are mostly much lower-resolution than diseased ones (median 1.1 vs about 5 megapixels), which points to a different camera and is a possible shortcut.

---

## 🏗️ Repository Architecture & File Structure

```
ComputerVision_Coursework_COBSCCOMP24.2P-019/
├── notebooks/
│   └── diabetic_retinopathy_full_retrain_v2.ipynb   # End-to-end research notebook (Sections 1-13)
├── app.py                                # Tiny entry point: imports app.main and launches it
├── app/                                  # Gradio application package
│   ├── main.py                           # Builds the tabs, wires buttons to handlers, launches
│   ├── analysis.py                       # analyze_fundus(), image quality checks, longitudinal comparison
│   ├── chatbot.py                        # Clinical knowledge base + router/knowledge/governance chat agents
│   ├── reports.py                        # JSON report, PDF generation, EHR note, referral summary
│   ├── triage.py                         # Risk simulator (illustrative, not validated)
│   ├── ui_style.py                       # CSS, head JavaScript, sidebar/header HTML
│   ├── samples/                          # 5 real fundus images (one per stage) for the sample buttons
│   └── cbr_reference/                    # Reference images (50 per stage) for similar-case retrieval
├── core/                                 # Reusable backend modules (the notebook clones this repo to import them)
│   ├── config.py                         # App settings and checkpoint paths
│   ├── preprocessing.py                  # Border crop, resize, Ben Graham, optional CLAHE/denoise/edges
│   ├── augmentation.py                   # Augmentation helpers
│   ├── models.py                         # EfficientNetB3 classifier, U-Net, Grad-CAM model, embeddings
│   ├── training.py                       # Training helpers and QWK
│   ├── explainability.py                 # Grad-CAM, retrieval, segmentation, classical CV
│   ├── advanced_cv.py                    # Overlap, biomarkers, consistency, longitudinal analysis
│   ├── research_evidence.py              # Error analysis and reproducibility artifacts
│   └── agents.py                         # Diagnosis, explainability, advisory and governance agents
├── scripts/
│   └── build_reference_set.py            # After training: rebuild app samples, CBR images and embeddings.npz
├── tests/                                # Unit tests (preprocessing and app logic)
├── splits/                               # train/validation/test split CSVs (written by the notebook)
├── checkpoints/                          # Trained weights used by the app (add after a training run)
├── report_images/                        # Figures and metrics copied from a run (add after a training run)
├── requirements.txt
└── README.md
```

Each notebook run writes its outputs to a run folder (`retinatrace_runs/run_<timestamp>/` locally, `/kaggle/working/retinatrace_runs/...` on Kaggle): `report_images/` (split files, duplicate audit, per-trial pilot evidence under `pilots/<trial_id>/`, learning curves, evaluation metrics, Grad-CAM figures) and `checkpoints/`. The notebook's *Notebook Artifact Guide* lists every file. Copy the ones you keep into the repository's `checkpoints/` and `report_images/` folders.

```

### 🏛️ Software Engineering & Architectural Design Rationale

To align with clean-architecture principles and deployment requirements:
1. **Decoupled Backend Package (`core/`):** All domain business logic is modularized into testable units:
   - `core/preprocessing.py`: single source of truth for border cropping and Ben Graham enhancement, shared by the research notebook (which clones this repository on Kaggle to import it) and the Gradio application.
   - `core/models.py` & `core/training.py`: model architecture definitions, the input-rescaling adapter, and training helpers.
   - `core/explainability.py` & `core/advanced_cv.py`: Mathematical Grad-CAM formulation, U-Net inference, and Case-Based Reasoning (CBR) embedding similarity.
   - `core/agents.py`: Decoupled 4-agent clinical governance architecture with typed error recovery.
2. **Presentation Layer (`app/` package + root `app.py`):**
   - **Deployment Architecture:** Hugging Face Spaces looks for a top-level `app_file: app.py`, so the root `app.py` is kept as a few-line entry point that imports `app.main` and launches it; `python app.py` still works.
   - **Module split:** `app/main.py` only builds the layout and wires events; analysis, chatbot, reports, triage and styling each live in their own module.
   - **Separation of Concerns:** the `app/` package contains *no* raw neural network layer definitions or image-processing mathematics; it delegates domain processing to `core/`.
3. **Automated Unit Testing (`tests/`):**
   - `tests/test_preprocessing.py` verifies tensor dimensionality, range $[0.0, 1.0]$, border cropping, ablation flags, and corrupted-input handling; `tests/test_app_logic.py` covers the chatbot, triage labels, reports and the metrics loader. Run them with `python -m pytest tests` (or `python -m unittest discover tests`).

> **Checkpoint status:** `core/config.py` matches the APTOS 2019 final model from Kaggle run
> `run_20260930_155003` (300×300, dropout 0.5). The repository includes everything the app needs:
> `checkpoints/final_model.weights.h5` (a 46 MB weights-only copy of the run's `best_phase2.weights.h5`,
> saved with `model.save_weights` and giving identical predictions), `checkpoints/unet_pseudomask.weights.h5`,
> `embeddings.npz` and the reference images in `app/cbr_reference/`. The run's own `best_phase1` and
> 187 MB `best_phase2` files are not in git. To rebuild the reference set, run
> `scripts/build_reference_set.py` (it needs the run's `train_split.csv` and `validation_split.csv` in
> `splits/` and the APTOS images). The app shows test metrics only when the
> notebook's `test_loss_and_metrics.csv` has been copied into `report_images/`.

---

## 🔬 Technical Innovation Features (Grounded in Module Lecture Materials)

All core innovations in this project are directly grounded in and adapted from prior coursework and laboratory materials:
* **Innovation 1 (Multi-Agent Safety Pipeline):** Derived from `6. Multi-Agent_AI_Blueprint.pdf` and `7/8. Defense_Multi_Agent_LLM.ipynb` (military ISR/cyber/governance decision pattern adapted to clinical safety).
* **Innovation 2 (Embedding-Based Case Retrieval):** Derived from `11. siamese_network_tutorial.ipynb` (AT&T Faces similarity learning adapted to retinal pathology Case-Based Reasoning).
* **Innovation 3 (Lesion-Level Segmentation):** Derived from `9. U-Net_Brain_Tumor_Segmentation.pdf` and `8. Brain_MRI_Segmentation_Kaggle_UNet_Dice_Report.pdf` (Brain tumor U-Net with Soft Dice loss adapted to microvascular fundus lesions).
* **Metric Formulation (Quadratic Weighted Kappa):** Derived from official APTOS/Kaggle competition evaluation standards for ordinal clinical disease grading.

### Innovation Feature A: Embedding-Based Similar-Case Retrieval (CBR Engine)

* **Theoretical Inspiration:** Inspired by **Siamese networks** and deep metric learning (traditionally deployed in facial verification and one-shot matching), this feature repurposes deep latent representations for clinical case comparison.
* **Implementation:** We tap the penultimate dense representation layer ($D = 256$) immediately prior to the 5-class softmax output layer. When a verified reference image library is available, 256-dimensional feature vectors are extracted and $L_2$-normalized such that Euclidean distance is strictly monotonic with cosine distance:
  $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{for } \|u\|_2 = \|v\|_2 = 1)$$
* **Clinical Rationale:** Medical practitioners rarely rely on an isolated probabilistic number. In ophthalmic practice, clinicians reason by **analogy to definitive historical cases** (Case-Based Reasoning). When the reference library is verified, returning the top-3 nearest cases with their labels offers **inter-case comparative explainability**, complementing the **intra-image spatial explainability** provided by Grad-CAM.

### Innovation Feature B: Multi-Agent Clinical Decision Pipeline & Safety Governance

* **Theoretical Rationale:** Single-model "monolithic" architectures and unstructured LLM chatbot wrappers are dangerous in clinical healthcare because they conflate perception with risk governance. A single high-probability hallucination can be presented as medical guidance without safety checks.
* **Architecture:** We architected a 4-agent decoupled clinical decision support pipeline:
  1. **`DiagnosisAgent`**: Dedicated to perceptual classification through the fine-tuned CNN, outputting discrete ICDR stages and softmax confidence scores.
  2. **`ExplainabilityAgent`**: Formulates evidence dossiers: computes spatial Grad-CAM saliency heatmaps, derives anatomical quadrant lesion descriptions (e.g., Superior-Temporal microaneurysm concentration), and executes similar-case retrieval.
  3. **`AdvisoryAgent`**: Aligns predicted stages with international clinical protocols (American Academy of Ophthalmology Preferred Practice Patterns & NHS Diabetic Eye Screening protocols), formulating concrete referral timeframes and management steps with an unambiguous legal disclaimer.
  4. **`GovernanceAgent` (Active Safety Gate)**: An autonomous safety officer. If `DiagnosisAgent` confidence falls below a configurable threshold (default: $70\%$), the `GovernanceAgent` **actively intercepts and overrides** the pipeline output. Rather than merely logging a warning, it withholds automated treatment advice, sets `flagged_for_review: True`, and issues an urgent clinical triage alert requiring manual ophthalmologist review.

---

### Innovation Feature C (Stretch Goal): Lesion-Level U-Net Segmentation (3-Layer Explainability Hierarchy)

* **Theoretical Translation:** Adapts the classical **U-Net** architecture (Ronneberger et al., 2015), originally designed for biomedical microscopy and brain tumor segmentation, to the domain of retinal microvascular lesions.
* **The 3-Layer Explainability Hierarchy:**
  1. **Layer 1 (Global Classification):** EfficientNetB3 outputs the 5-stage ICDR disease grade ($0-4$) and confidence score.
  2. **Layer 2 (Regional Attention):** Grad-CAM visualizes class-discriminative heatmap activations and maps peak pathology quadrants (*Superior-Temporal, Inferior-Nasal*, etc.).
  3. **Layer 3 (Pixel-level candidate regions):** An auxiliary U-Net with skip connections outputs a pixel-level map of candidate lesion regions. It is trained on synthetic pseudo-masks (below), so its output is a visual aid, not a validated lesion segmentation.
* **Hybrid Soft Dice Loss:** Because retinal lesions occupy $< 1-3\%$ of total pixels, standard binary cross-entropy collapses to predicting background. The network optimizes a **hybrid Soft Dice + BCE loss**:
  $$\mathcal{L} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\left(1 - \frac{2\sum y_i\hat{y}_i + \epsilon}{\sum y_i + \sum \hat{y}_i + \epsilon}\right)$$
  ensuring stable gradient propagation while penalizing boundary overlap errors on tiny microvascular lesions.
* **Mask Synthesis Methodology (pseudo-masks):** APTOS 2019 has no pixel-level lesion annotations. Pseudo-masks are therefore synthesized: green-channel top-hat and black-hat morphology (fixed threshold) marks bright and dark candidate structures, and only those inside the classifier's Grad-CAM attention (saliency > 0.35) are kept. These masks train the auxiliary U-Net for qualitative visual explanation only; any Dice score measures agreement with these synthetic targets, not with expert masks.

## 🌟 Bonus Features Implemented

* **Bonus C: Interactive Gradio UI**: Complete clinical web dashboard featuring image drag-and-drop, adjustable governance threshold sliders, live Grad-CAM heatmaps, and case retrieval galleries; runs locally at `http://localhost:7860` (set `GRADIO_SERVER_PORT` to change the port).
* **Bonus D: Cloud Hosting & Hugging Face Spaces Readiness**: Standalone `app.py` and `requirements.txt` structured specifically for zero-configuration deployment to Hugging Face Spaces using the Gradio SDK.
* **Bonus E: Multi-Stage Classification Verification**: The network strictly performs 5-class ordinal disease staging across all ICDR grades (`No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative DR`), rejecting binary (DR present/absent) simplification.

---

## 🚀 Step-by-Step Reproduction Guide

### Option 1: Training on Kaggle (recommended)

1. On Kaggle, open the [APTOS 2019 competition](https://www.kaggle.com/competitions/aptos2019-blindness-detection), **join it and accept the rules**.
2. Create a notebook, import `notebooks/diabetic_retinopathy_full_retrain_v2.ipynb`, and use **Add Input → Competitions → APTOS 2019 Blindness Detection**.
3. Turn on a **GPU** and **Internet** (for ImageNet weights and cloning this repository's `core/`).
4. Use **Save Version → Save & Run All**. The pipeline runs end to end:
   - Data loading, corrupt-image and resolution audits
   - Border cropping and Ben Graham enhancement
   - Duplicate-aware stratified 70/15/15 split with leakage assertions
   - **17 pilot trials** on the full train/validation splits (22 defined; `Config.PILOT_SKIP_TRIALS` skips five): 9 selectable (fine-tuning depth, plateau vs warm-up + cosine schedule, weight decay, 224/300/380 px, balanced vs sqrt vs no class weights, oversampling) and 8 for comparison (preprocessing and augmentation ablations, EfficientNetB0 / ResNet-50 / MobileNetV2 / DenseNet-121, and a CNN trained from scratch). The winner is the selectable trial with the highest validation QWK. Set `RUN_COMPARISON_TRIALS=False` to run only the selectable trials. `Config.PILOT_TIME_BUDGET_HOURS` (5 h) and `Config.RESUME_RUN_ID` protect long runs; the 17 pilots took about 2.5 h of training time on 2× T4 (sum of `training_time_seconds`).
   - Two-phase EfficientNetB3 training of the selected configuration (frozen backbone, then top-120 fine-tuning), with early stopping and checkpointing on validation QWK (patience 6); curves show augmented-train, clean-train-subset and validation lines plus validation QWK
   - Evaluation on the held-out test split, once per run (accuracy, QWK, macro-F1 for both the argmax and validation-QWK-threshold rules, per-class reports, confusion matrix, ROC, screening metrics)
   - Grad-CAM, similar-case retrieval, multi-agent pipeline, U-Net demo, error analysis
   - Building the optional Gradio UI (launch is commented out in the notebook)

On **Google Colab** or locally, the notebook instead downloads the competition data in Section 1.5 with `kaggle competitions download -c aptos2019-blindness-detection` (upload `kaggle.json` when asked; you must have accepted the competition rules). You can also set `APTOS_DATA_DIR` to an existing copy containing `train.csv` and `train_images/`.

### Option 2: Local Execution

```bash
# 1. Clone the repository
git clone https://github.com/ShaznaSalman/ComputerVision_Coursework_COBSCCOMP24.2P-019.git
cd ComputerVision_Coursework_COBSCCOMP24.2P-019

# 2. Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the interactive Gradio clinical application (http://localhost:7860)
python app.py
```

### Option 3: Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) selecting the **Gradio SDK**.
2. Push `app.py`, the `app/` and `core/` folders, `requirements.txt`, `README.md`, your trained checkpoints in `checkpoints/`, and `embeddings.npz` (optional; enables similar-case retrieval) to the Space repository. An optional `assets/retinatrace_icon.png` replaces the built-in logo.
3. Hugging Face Spaces will automatically build the environment and host your clinical AI app at a permanent public URL.

---

## 📈 Clinical Evaluation Methodology: Why QWK?

Standard classification accuracy treats a Mild vs. Severe error identically to a Mild vs. Moderate error. In ophthalmology, adjacent-stage errors (Mild vs Moderate) represent minor monitoring adjustments, whereas distant errors (classifying Severe NPDR or Proliferative DR as No DR) risk catastrophic vision loss due to omitted treatment.

The model is evaluated using **Quadratic Weighted Kappa (QWK)** via `cohen_kappa_score(weights='quadratic')`:
$$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(N - 1)^2}$$
Quadratic penalties ($|i - j|^2$) heavily penalize distant staging mistakes, reflecting true clinical safety requirements.

## 📊 Results (Kaggle run `run_20260930_155003`)

The selected pilot trial was `weight_decay_1e3` (EfficientNetB3, 300×300, dropout 0.5, top-120 fine-tuning, AdamW weight decay 1e-3, balanced class weights; validation QWK 0.856). Test split: 550 images (29 Severe NPDR).

| Decision rule | Accuracy | QWK | Macro-F1 |
| :--- | ---: | ---: | ---: |
| Argmax (used by the app) | 0.760 | 0.828 | 0.597 |
| Validation-selected QWK thresholds | 0.793 | 0.892 | 0.556 |
| Majority class (always No DR) | 0.493 | 0.000 | 0.132 |

* **Per-stage recall (argmax):** No DR 0.99, Mild 0.70, Moderate 0.55, Severe 0.59, Proliferative 0.27. Macro one-vs-rest ROC-AUC 0.929.
* **Screening:** any DR vs No DR sensitivity 0.975 / specificity 0.989 (argmax); referable DR (stage ≥ 2) sensitivity 0.987 / specificity 0.881 with the thresholds.
* **Pilots:** ImageNet pretraining mattered most (validation QWK 0.773 for the EfficientNetB3 baseline vs 0.088 for a CNN from scratch); CLAHE instead of Ben Graham (0.852) and no augmentation (0.845) scored above the baseline in these single-seed pilots.
* **Caveats:** an earlier final fit (run A) stopped on validation loss and restored weak epochs, because validation loss and QWK disagreed under balanced class weights (Phase 1 restored epoch 1; Phase 2 restored epoch 3 while validation QWK and accuracy were still rising). The stopping rule was therefore changed to validation QWK **on validation evidence only**, and run B was fixed as the reported model **before its test results were produced**; run A's test metrics were not used in any decision. The test split is small (550 images, 29 Severe), comes from a single source with no external test set, and No DR images are mostly lower-resolution than diseased ones, which may make "any DR" detection look easier than it is.

---

## ⚖️ Ethical, Regulatory, & Safety Statement

This system is an investigational computer science coursework project and clinical decision-support research prototype. It is **not** certified as Software as a Medical Device (SaMD) by the FDA, EMA, or MHRA. It does not provide medical diagnoses or prescribe treatment regimens. In clinical deployment, all automated outputs must be verified by a board-certified ophthalmologist or licensed optometrist.
