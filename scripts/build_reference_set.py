"""
Rebuild the app's sample images, similar-case reference images and embeddings.npz
from a finished notebook run (APTOS 2019).

Run this AFTER copying the new checkpoints into checkpoints/ and setting
core/config.py (IMG_SIZE, DROPOUT_RATE) to the notebook's final configuration:

    python scripts/build_reference_set.py --images-dir <folder with APTOS train_images/*.png>

What it does:
  * Reads splits/train_split.csv and splits/validation_split.csv (written by notebook Section 4.1).
  * app/cbr_reference/<stage>/ : 50 images per stage from the TRAINING split (seed 42), saved
    border-cropped and resized as JPEG so the repository stays small.
  * app/samples/stage<k>_*.jpg  : one image per stage from the VALIDATION split (never test),
    plus SOURCES.csv recording where each came from.
  * embeddings.npz              : 256-D L2-normalised head_dense embeddings of the reference
    images, computed with the app's own model and preprocessing, with relative file paths.
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig  # noqa: E402  (after sys.path setup)

STAGE_NAMES = {0: "no_dr", 1: "mild", 2: "moderate", 3: "severe", 4: "proliferative"}
REFERENCE_SIDE = 512  # pixels; display/storage size of saved reference and sample images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", required=True,
                        help="Folder containing the APTOS images (<id_code>.png), e.g. train_images/.")
    parser.add_argument("--splits-dir", default=str(PROJECT_ROOT / "splits"),
                        help="Folder with train_split.csv and validation_split.csv (default: splits/).")
    parser.add_argument("--per-stage", type=int, default=50, help="Reference images per stage (default 50).")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42).")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT),
                        help="Where app/ and embeddings.npz are written (default: project root).")
    parser.add_argument("--checkpoint", default=None,
                        help="Classifier weights to use instead of AppConfig.WEIGHTS_PATH.")
    return parser.parse_args()


def resolve_image(manifest_path: str, images_dir: Path) -> Path:
    """Map a manifest file path (e.g. a Kaggle path) to the same file name inside images_dir."""
    candidate = images_dir / Path(str(manifest_path).replace("\\", "/")).name
    if not candidate.is_file():
        raise FileNotFoundError(f"{candidate} not found; check --images-dir.")
    return candidate


def save_display_copy(source: Path, target: Path) -> None:
    """Save a border-cropped, resized JPEG copy of a fundus photograph."""
    from core.preprocessing import crop_image_from_gray

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read {source}")
    cropped = crop_image_from_gray(image)
    resized = cv2.resize(cropped, (REFERENCE_SIDE, REFERENCE_SIDE), interpolation=cv2.INTER_AREA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not write {target}")


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    splits_dir = Path(args.splits_dir)
    output_root = Path(args.output_root)

    train_df = pd.read_csv(splits_dir / "train_split.csv")
    val_df = pd.read_csv(splits_dir / "validation_split.csv")
    if set(train_df["split"]) != {"train"} or set(val_df["split"]) != {"validation"}:
        raise ValueError("Split files must contain only their own split; the test split is never used here.")

    # ---- reference images: training split only --------------------------------
    reference_dir = output_root / "app" / "cbr_reference"
    if reference_dir.exists():
        shutil.rmtree(reference_dir)
    reference_rows = []
    for stage in range(AppConfig.NUM_CLASSES):
        stage_rows = train_df[train_df["label"] == stage]
        chosen = stage_rows.sample(min(args.per_stage, len(stage_rows)), random_state=args.seed + stage)
        for manifest_path in chosen["filepath"]:
            source = resolve_image(manifest_path, images_dir)
            relative = Path("app") / "cbr_reference" / str(stage) / f"{source.stem}.jpg"
            save_display_copy(source, output_root / relative)
            reference_rows.append((relative.as_posix(), stage))

    # ---- sample buttons: validation split only --------------------------------
    samples_dir = output_root / "app" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_sources = []
    for stage in range(AppConfig.NUM_CLASSES):
        chosen = val_df[val_df["label"] == stage].sample(1, random_state=args.seed + stage).iloc[0]
        source = resolve_image(chosen["filepath"], images_dir)
        save_display_copy(source, samples_dir / f"stage{stage}_{STAGE_NAMES[stage]}.jpg")
        sample_sources.append({"sample": f"stage{stage}_{STAGE_NAMES[stage]}.jpg",
                               "source_id_code": source.stem, "split": "validation"})
    pd.DataFrame(sample_sources).to_csv(samples_dir / "SOURCES.csv", index=False)

    # ---- embeddings with the app's own model and preprocessing ----------------
    if args.checkpoint:
        AppConfig.WEIGHTS_PATH = args.checkpoint
    AppConfig.EMBEDDINGS_PATH = str(output_root / "embeddings.npz.building")  # do not load the old library
    from core.models import embedding_extractor  # noqa: E402  (loads the classifier)
    from core.preprocessing import preprocess_image  # noqa: E402

    batch = np.stack([
        preprocess_image(str(output_root / relative), img_size=AppConfig.IMG_SIZE)
        for relative, _ in reference_rows
    ]).astype(np.float32)
    embeddings = embedding_extractor.predict(batch, batch_size=16, verbose=0).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    np.savez_compressed(
        output_root / "embeddings.npz",
        embeddings=embeddings,
        labels=np.array([stage for _, stage in reference_rows], dtype=np.int64),
        filepaths=np.array([relative for relative, _ in reference_rows]),
    )

    counts = pd.Series([stage for _, stage in reference_rows]).value_counts().sort_index().to_dict()
    print(f"[Reference set] {len(reference_rows)} training-split images; per stage: {counts}")
    print(f"[Reference set] images -> {reference_dir}")
    print(f"[Samples] 5 validation-split images -> {samples_dir}")
    print(f"[Embeddings] {embeddings.shape} -> {output_root / 'embeddings.npz'} "
          f"(model input {AppConfig.IMG_SIZE}px, checkpoint {AppConfig.WEIGHTS_PATH})")


if __name__ == "__main__":
    main()
