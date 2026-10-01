"""
app.py — RetinaTrace AI entry point: `python app.py` starts the Gradio app (also the Docker CMD).
Research prototype for five-stage diabetic retinopathy grading; not for clinical use.

BSc (Hons) Computer Science — Computer Vision (BSCCOMP24.2P)
Repository: https://github.com/ShaznaSalman/ComputerVision_Coursework_COBSCCOMP24.2P-019

Features:
---------
1. Three explainability layers:
   - Layer 1: EfficientNetB3 five-stage classification with the probability for each stage.
   - Layer 2: Grad-CAM heatmap with a quadrant description.
   - Layer 3: U-Net pseudo-mask demonstration (not lesion segmentation; trained on heuristic masks, shown in green).
2. Additions:
   - Similar-case retrieval: cosine similarity between 256-d embeddings and a bundled reference library.
   - Four-agent pipeline (diagnosis, explainability, advisory, governance) with a 70% confidence
     safety gate, an image-quality check and patient-reported red-flag symptoms (URGENT outcome).
3. Interface:
   - Fundus-validity check that rejects non-fundus uploads before inference.
   - Preset demo buttons (fixed illustrative values, not model output) and an adjustable safety threshold.
   - Optional patient ID / eye, Clinician / Patient view, three PDF reports and a rule-based chatbot.

The interface itself lives in the app/ package; see app/__init__.py for the module map.
"""

from app.main import demo, launch

__all__ = ["demo", "launch"]  # demo is exposed for Gradio tooling / Spaces

if __name__ == "__main__":
    launch()
