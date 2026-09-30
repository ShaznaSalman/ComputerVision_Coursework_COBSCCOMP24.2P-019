"""Application configuration for RetinaTrace AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig:
    """Centralized configuration parameters for RetinaTrace AI.

    IMG_SIZE and DROPOUT_RATE must match the notebook's final configuration
    (Section 6.6 prints it) and the checkpoint in checkpoints/. The current
    values match the earlier checkpoint; after the APTOS retraining set
    IMG_SIZE to the selected image size (300 by default) and DROPOUT_RATE to 0.5.
    """

    IMG_SIZE: int = 224
    NUM_CLASSES: int = 5
    CLASS_NAMES: list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    BEN_GRAHAM_SIGMA: int = 10
    BEN_GRAHAM_ALPHA: float = 4.0
    BEN_GRAHAM_BETA: float = -4.0
    BEN_GRAHAM_GAMMA: float = 128.0
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.70
    ENABLE_OPTIC_DISC_REMOVAL: bool = False

    # Advanced Preprocessing Configuration
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: tuple = (8, 8)
    EDGE_SHARPEN_STRENGTH: float = 1.2
    EDGE_SHARPEN_SIGMA: float = 3.0

    # ── Training defaults (mirror the notebook's Config baseline) ─────────
    # Used by the optional helpers in core/training.py; the notebook has its own
    # Config and the pilot trials may select different values (Section 6.6).
    # Phase 1: Transfer Learning Feature Extraction (Frozen Backbone)
    PHASE1_EPOCHS: int = 8
    PHASE1_LR: float = 1e-3
    PHASE1_BATCH_SIZE: int = 16
    PHASE1_OPTIMIZER: str = "Adam(learning_rate=1e-3)"
    PHASE1_FROZEN_LAYERS: str = "all backbone layers (385 on Kaggle's Keras 3.13, 384 on Keras 3.15)"

    # Phase 2: Fine-Tuning (top layers unfrozen, BatchNorm frozen)
    PHASE2_EPOCHS: int = 25
    PHASE2_LR: float = 1e-5
    PHASE2_MIN_LR: float = 1e-7
    PHASE2_BATCH_SIZE: int = 16
    PHASE2_OPTIMIZER: str = "AdamW(learning_rate=1e-5, weight_decay=1e-4)"
    PHASE2_UNFROZEN_LAYERS: int = 120  # notebook Config.UNFREEZE_TOP_N

    # Regularization & Optimization Guardrails
    # DROPOUT_RATE must match the checkpoint the app loads (0.30 for the earlier
    # checkpoint; set 0.5 when the APTOS-trained weights are copied in).
    DROPOUT_RATE: float = 0.30
    L2_WEIGHT_DECAY: float = 1e-4  # AdamW decoupled weight decay in Phase 2
    LABEL_SMOOTHING: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 4
    REDUCE_LR_PATIENCE: int = 2
    REDUCE_LR_FACTOR: float = 0.50

    WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "best_phase2.weights.h5")
    UNET_WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "unet_lesion_best.weights.h5")  # earlier 2-level U-Net
    # Notebook Section 12.4 output; used (with its own 3-level architecture) whenever it exists.
    NOTEBOOK_UNET_WEIGHTS_PATH: str = str(PROJECT_ROOT / "checkpoints" / "unet_pseudomask.weights.h5")
    EMBEDDINGS_PATH: str = str(PROJECT_ROOT / "embeddings.npz")
