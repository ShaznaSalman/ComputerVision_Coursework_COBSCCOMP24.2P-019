"""
app.py — RetinaTrace AI entry point (Hugging Face Spaces looks for this file).
Modern, Unique & User-Friendly Gradio Web Application.

BSc (Hons) Computer Science — Computer Vision (BSCCOMP24.2P)
Repository: https://github.com/ShaznaSalman/ComputerVision_Coursework_COBSCCOMP24.2P-019

Architecture Features:
----------------------
1. 3-Layer Clinical Explainability Dossier:
   - Layer 1: EfficientNetB3 5-Stage Disease Classification & Probability Bar Breakdown.
   - Layer 2: Regional Grad-CAM Attention Heatmap & Automated Anatomical Quadrant Narrative.
   - Layer 3: Auxiliary U-Net Pixel-Level Retinal Lesion Segmentation (Microaneurysms/Exudates in Green).
2. Technical Innovations:
   - Innovation A: Case-Based Reasoning (CBR) Metric Embedding Retrieval (Cosine Dot-Product).
   - Innovation B: Decoupled 4-Agent Decision Pipeline with Active Governance Safety Gate (70% Threshold).
3. Modern UX / UI:
   - Custom Biomedical Clinical CSS Theme with Responsive Cards & Status Badges.
   - One-Click Preset Fundus Sample Loaders (Normal, Moderate NPDR, Proliferative DR).
   - One-Click Interactive Safety Gate Override Simulation.
   - Exportable Clinical EHR Summary Note.

The interface itself lives in the app/ package; see app/__init__.py for the module map.
"""

from app.main import demo, launch

__all__ = ["demo", "launch"]  # demo is exposed for Gradio tooling / Spaces

if __name__ == "__main__":
    launch()
