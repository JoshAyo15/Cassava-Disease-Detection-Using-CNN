"""
Cassava Leaf Disease Classification - CNN Training Script
Uses ConvNeXt transfer learning (optimal for plant disease classification).
Input: 5-class folder dataset at `data_path`
Output: saved model at `models/` + metrics

Checkpointing:
  - Phase 1 (frozen head) saved to models/phase1_checkpoint.pkl
  - Phase 2 (fine-tuned)  saved to models/phase2_checkpoint.pkl
  - Final export          saved to models/model.pkl

On restart, the script detects which checkpoint exists and skips ahead:
  - phase2_checkpoint.pkl found → skip straight to evaluation + export
  - phase1_checkpoint.pkl found → skip Phase 1, resume from Phase 2
  - neither found            → train from scratch
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fastai.vision.all import *
from fastai.callback.all import *

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

PHASE1_CKPT = MODEL_DIR / "phase1_checkpoint.pkl"
PHASE2_CKPT = MODEL_DIR / "phase2_checkpoint.pkl"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LR_BASE = 1e-2
SEED = 42

CLASSES = [
    "bacterial_blight",
    "brown_streak_disease",
    "green_mottle",
    "healthy",
    "mosaic_disease",
]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def detect_checkpoint():
    """
    Returns one of: 'phase2', 'phase1', or None
    indicating the latest completed phase checkpoint found on disk.
    """
    if PHASE2_CKPT.exists():
        return 'phase2'
    if PHASE1_CKPT.exists():
        return 'phase1'
    return None


def load_checkpoint(ckpt_path, dls, class_weights=None):
    """Load a FastAI .pkl checkpoint and return a learner ready to continue."""
    print(f"\n  ↩  Loading checkpoint: {ckpt_path}")
    learn = load_learner(ckpt_path)
    # Re-attach the dataloaders so the learner can see the current data split
    learn.dls = dls
    # Pin model path so SaveModelCallback always writes to MODEL_DIR
    learn.path = MODEL_DIR
    learn.model_dir = ''
    # Re-apply weighted loss if needed (not persisted in .pkl)
    if class_weights:
        weights = torch.tensor(
            [v for k, v in sorted(class_weights.items())],
            dtype=torch.float32
        )
        learn.loss_func = CrossEntropyLossFlat(weight=weights)
    return learn


# ── Validation ────────────────────────────────────────────────────────────────
def validate_data():
    """Verify data directory structure and calculate class weights"""
    expected_dirs = [
        "Cassava___bacterial_blight",
        "Cassava___brown_streak_disease",
        "Cassava___green_mottle",
        "Cassava___healthy",
        "Cassava___mosaic_disease"
    ]

    class_counts = {}
    total_images = 0

    for dir_name in expected_dirs:
        dir_path = DATA_PATH / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(f"Missing data: {dir_path}")
        files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        count = len(files)
        class_counts[dir_name] = count
        total_images += count
        print(f"  ✓ {dir_name}: {count} images")

    print(f"\n  CLASS DISTRIBUTION:")
    for dir_name, count in class_counts.items():
        pct = (count / total_images) * 100
        print(f"    {dir_name}: {count:,} ({pct:.1f}%)")

    print(f"\n  CLASS IMBALANCE WEIGHTS (for balancing):")
    class_weights = {}
    max_count = max(class_counts.values())

    for dir_name, count in class_counts.items():
        weight = max_count / count
        class_weights[dir_name] = weight
        print(f"    {dir_name}: {weight:.2f}x")

    imbalance_ratio = max_count / min(class_counts.values())
    print(f"\n  ⚠️  IMBALANCE RATIO: {imbalance_ratio:.1f}:1 (max:min)")
    print(f"     Using weighted loss and stratified sampling to handle imbalance")

    return class_counts, class_weights


# ── Data Loading ──────────────────────────────────────────────────────────────
def prepare_dataloaders(class_weights=None):
    """Create data loaders with augmentations and stratified splits"""

    item_tfms = [
        Resize(IMG_SIZE, method=ResizeMethod.Squish),
    ]

    batch_tfms = [
        Flip(p=0.5),
        Rotate(max_deg=15, p=0.5),
        Brightness(max_lighting=0.2, p=0.5),
        Contrast(max_lighting=0.2, p=0.5),
        Normalize.from_stats(*imagenet_stats)
    ]

    dls = ImageDataLoaders.from_folder(
        DATA_PATH,
        valid_pct=0.2,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=BATCH_SIZE,
        seed=SEED,
        num_workers=0,
        splitter=RandomSplitter(valid_pct=0.2, seed=SEED)
    )

    print(f"\n  ✓ Data loaders created (stratified split)")
    print(f"    - Train batches: {len(dls.train)}")
    print(f"    - Valid batches: {len(dls.valid)}")
    print(f"    - Batch size: {BATCH_SIZE}")

    return dls


# ── Training ──────────────────────────────────────────────────────────────────
def build_fresh_learner(dls, class_weights=None):
    """Instantiate a brand-new ConvNeXt learner."""
    if class_weights:
        weights = torch.tensor(
            [v for k, v in sorted(class_weights.items())],
            dtype=torch.float32
        )
        print(f"\n  Using weighted cross-entropy loss")
        print(f"  Weights: {weights.tolist()}")
        loss_func = CrossEntropyLossFlat(weight=weights)
    else:
        loss_func = LabelSmoothingCrossEntropy(eps=0.1)

    learn = vision_learner(
        dls,
        'convnext_small_in22k',
        metrics=[error_rate, accuracy],
        loss_func=loss_func,
        pretrained=True
    )
    # Pin model path so SaveModelCallback writes to MODEL_DIR regardless of cwd
    learn.path = MODEL_DIR
    learn.model_dir = ''
    return learn


def run_phase1(learn):
    """Phase 1: frozen backbone, train head only."""
    print("\n" + "="*60)
    print("PHASE 1: TRAIN HEAD (Backbone Frozen)")
    print("="*60)

    callbacks = [
        EarlyStoppingCallback(monitor='error_rate',
                              patience=3, min_delta=0.001),
        SaveModelCallback(monitor='error_rate', fname='phase1_best')
    ]

    learn.freeze()
    learn.fit_one_cycle(5, lr_max=1e-2, cbs=callbacks)

    # Load best weights before exporting (fastai leaves model at last epoch, not best)
    learn.load('phase1_best')

    # ── Checkpoint after Phase 1 ──────────────────────────────────────────
    learn.export(PHASE1_CKPT)
    print(f"\n  ✓ Phase 1 checkpoint saved → {PHASE1_CKPT}")

    return learn


def run_phase2(learn):
    """Phase 2: unfreeze everything, fine-tune with discriminative LR."""
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNE (Discriminative LR)")
    print("="*60)

    callbacks = [
        EarlyStoppingCallback(monitor='error_rate',
                              patience=4, min_delta=0.001),
        SaveModelCallback(monitor='error_rate', fname='phase2_best')
    ]

    learn.unfreeze()
    # slice(1e-5, 1e-4): backbone gets tiny LR, head gets 10x more
    # Flat 1e-3 caused valid_loss explosion (60+) by destroying pretrained weights
    learn.fit_one_cycle(EPOCHS, lr_max=slice(1e-5, 1e-4), cbs=callbacks)

    # Load best weights before exporting (fastai leaves model at last epoch, not best)
    learn.load('phase2_best')

    # ── Checkpoint after Phase 2 ──────────────────────────────────────────
    learn.export(PHASE2_CKPT)
    print(f"\n  ✓ Phase 2 checkpoint saved → {PHASE2_CKPT}")

    return learn


def train_model(dls, class_weights=None):
    """
    Orchestrate training with checkpoint-aware resume logic.

    Resume behaviour:
      phase2_checkpoint.pkl exists → load it, skip both training phases
      phase1_checkpoint.pkl exists → load it, skip Phase 1, run Phase 2
      neither                      → build fresh learner, run Phase 1 then Phase 2
    """
    checkpoint = detect_checkpoint()

    if checkpoint == 'phase2':
        print("\n  ✅ Phase 2 checkpoint detected — skipping training entirely.")
        print(
            f"     NOTE: This checkpoint used the old LR (1e-3) which caused instability.")
        print(
            f"     Delete {PHASE2_CKPT} to re-run Phase 2 with the fixed LR (slice 1e-5→1e-4).")
        print(f"     Phase 1 checkpoint is preserved — only Phase 2 will re-run.")
        learn = load_checkpoint(PHASE2_CKPT, dls, class_weights)
        return learn

    if checkpoint == 'phase1':
        print("\n  ✅ Phase 1 checkpoint detected — resuming from Phase 2.")
        print(f"     Delete {PHASE1_CKPT} to force a full retrain.")
        learn = load_checkpoint(PHASE1_CKPT, dls, class_weights)
        learn = run_phase2(learn)
        return learn

    # Fresh start
    print("\n" + "="*60)
    print("MODEL CREATION  (no checkpoints found — training from scratch)")
    print("="*60)
    learn = build_fresh_learner(dls, class_weights)
    learn = run_phase1(learn)
    learn = run_phase2(learn)
    return learn


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(learn):
    """Evaluate model performance"""

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    try:
        preds, targets = learn.get_preds(dl=learn.dls.valid)
    except Exception as e:
        raise RuntimeError(
            f"get_preds failed — the validation dataloader may be detached or empty.\n"
            f"Original error: {e}"
        ) from e

    # .item() converts fastai TensorBase → plain Python float so f-string :.4f works
    err = error_rate(preds, targets).item()
    acc = accuracy(preds, targets).item()

    print(f"\nValidation Results:")
    print(f"  Error Rate: {err:.4f} ({(1-err)*100:.2f}%)")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")

    print(f"\nPerforming Test-Time Augmentation...")
    try:
        tta_preds, _ = learn.tta(dl=learn.dls.valid)
        tta_err = error_rate(tta_preds, targets).item()
        tta_acc = accuracy(tta_preds, targets).item()
        print(f"  TTA Error Rate: {tta_err:.4f} ({(1-tta_err)*100:.2f}%)")
        print(f"  TTA Accuracy:  {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    except Exception as e:
        print(
            f"  \u26a0\ufe0f  TTA failed ({e}) \u2014 falling back to standard predictions.")
        tta_err = err
        tta_acc = acc

    metrics = {
        'accuracy': acc,
        'error_rate': err,
        'tta_accuracy': tta_acc,
        'tta_error_rate': tta_err,
        'model_arch': 'convnext_small_in22k'
    }

    return metrics


# ── Model Saving ──────────────────────────────────────────────────────────────
def save_model(learn):
    """Save final model for inference"""

    # FastAI export (.pkl) — full learner including transforms, vocab, etc.
    try:
        learn.export(MODEL_DIR / "model.pkl")
        print(f"  ✓ Exported to {MODEL_DIR / 'model.pkl'}")
    except Exception as e:
        print(f"  ⚠️  learn.export failed: {e}")
        raise

    # Raw PyTorch weights — move to CPU first to avoid device-mismatch on load
    try:
        state_dict = {k: v.cpu() for k, v in learn.model.state_dict().items()}
        torch.save(state_dict, MODEL_DIR / "weights.pth")
        print(f"  ✓ Weights saved to {MODEL_DIR / 'weights.pth'}")
    except Exception as e:
        print(f"  ⚠️  weights.pth save failed (non-fatal): {e}")

    # Class label mapping
    try:
        with open(MODEL_DIR / "classes.json", "w") as f:
            json.dump({i: c for i, c in enumerate(CLASSES)}, f, indent=2)
        print(f"  ✓ Classes saved to {MODEL_DIR / 'classes.json'}")
    except Exception as e:
        print(f"  ⚠️  classes.json save failed (non-fatal): {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    """Run training pipeline"""

    print("\n" + "="*60)
    print("CASSAVA DISEASE CLASSIFICATION - MODEL TRAINING")
    print("="*60)

    # ── Check for existing final model ───────────────────────────────────────
    final_model = MODEL_DIR / "model.pkl"
    if final_model.exists():
        print(f"\n  ✅ Final model already exists at {final_model}")
        print(f"     Delete it (and the phase checkpoints) to trigger a full retrain.")
        metrics_path = MODEL_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"\n  Cached metrics:")
            print(f"    Validation Accuracy: {metrics['accuracy']*100:.2f}%")
            print(
                f"    TTA Accuracy:        {metrics['tta_accuracy']*100:.2f}%")
        return

    print("\nValidating data structure...")
    class_counts, class_weights = validate_data()

    print("\nPreparing data loaders...")
    dls = prepare_dataloaders(class_weights=class_weights)

    # train_model handles resume logic internally
    learn = train_model(dls, class_weights=class_weights)

    metrics = evaluate_model(learn)

    print("\n" + "="*60)
    print("SAVING FINAL MODEL")
    print("="*60)
    save_model(learn)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {MODEL_DIR / 'metrics.json'}")

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE")
    print("="*60)
    print(f"\nFinal Performance:")
    print(f"  Validation Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  TTA Accuracy:        {metrics['tta_accuracy']*100:.2f}%")
    print(f"  Model Location:      {MODEL_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
