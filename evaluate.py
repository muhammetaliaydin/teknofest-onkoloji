"""
============================================================================
EVALUATE — Test Set Evaluation & Visualization
============================================================================
Features:
  - Test set evaluation with torch.no_grad()
  - Confusion matrix generation
  - Mask overlay visualization (predicted mask on original MRI)
  - Training curves from CSV log
  - Sample predictions grid
============================================================================
"""

import sys
import csv
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.amp import autocast
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
)

import config
from dataset import create_dataloaders
from model import BrainTumorSegModel, CombinedLoss, compute_metrics
from train import load_checkpoint

sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════
# 1. PLOT TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(log_file: Path, output_dir: Path):
    """
    Reads the CSV training log and plots loss, Dice, IoU curves.
    """
    epochs, train_loss, val_loss = [], [], []
    train_dice, val_dice = [], []
    train_iou, val_iou = [], []
    lrs = []
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_dice.append(float(row["train_dice"]))
            val_dice.append(float(row["val_dice"]))
            train_iou.append(float(row["train_iou"]))
            val_iou.append(float(row["val_iou"]))
            lrs.append(float(row["lr"]))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
    
    # Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice
    axes[0, 1].plot(epochs, train_dice, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_dice, 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title("Dice Coefficient")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(epochs, train_iou, 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, val_iou, 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_title("IoU (Intersection over Union)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs, lrs, 'g-', linewidth=2)
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Training curves: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX (per-slice classification)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_confusion_matrix(model, test_loader, device, output_dir: Path):
    """
    Generates confusion matrix for slice-level tumor detection.
    Each slice is classified as tumor/no-tumor based on the segmentation output.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, masks in test_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            seg_logits, cls_logits = model(images, return_classification=True)
        
        # Slice-level label: 1 if any tumor pixel exists
        labels = (masks.sum(dim=(1, 2, 3)) > 0).cpu().numpy()
        
        # Slice-level prediction: 1 if predicted mask has tumor pixels
        probs = torch.sigmoid(cls_logits).squeeze(-1).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)
        
        del images, masks
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Tumor", "Tumor"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Slice-Level Classification)", fontsize=14)
    save_path = output_dir / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Confusion matrix: {save_path}")
    
    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Slice-Level)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    save_path = output_dir / "roc_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] ROC curve: {save_path}")
    
    # ── Classification Report ──
    report = classification_report(all_labels, all_preds,
                                    target_names=["No Tumor", "Tumor"])
    print(f"\nClassification Report:\n{report}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    return roc_auc


# ═══════════════════════════════════════════════════════════════════════════
# 3. SAMPLE PREDICTIONS WITH MASK OVERLAY
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_sample_predictions(model, test_loader, device, output_dir: Path, n_samples=10):
    """
    Generates overlay visualizations: original MRI + ground truth + prediction.
    """
    model.eval()
    
    samples_collected = 0
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    fig.suptitle("Sample Predictions", fontsize=16, fontweight="bold", y=1.01)
    
    column_titles = ["T1c Input", "Ground Truth Mask", "Predicted Mask", "Overlay"]
    
    for images, masks in test_loader:
        if samples_collected >= n_samples:
            break
        
        images = images.to(device, non_blocking=True)
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            seg_logits = model(images)
        
        seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
        images_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        
        for i in range(images.shape[0]):
            if samples_collected >= n_samples:
                break
            
            idx = samples_collected
            
            # T1c channel (first modality) for display
            t1c = images_np[i, 0]  # (H, W)
            gt_mask = masks_np[i, 0]  # (H, W)
            pred_mask = seg_probs[i, 0]  # (H, W)
            pred_binary = (pred_mask > 0.5).astype(float)
            
            # Normalize T1c for display
            t1c_display = (t1c - t1c.min()) / (t1c.max() - t1c.min() + 1e-8)
            
            # Column 1: T1c slice
            axes[idx, 0].imshow(t1c_display, cmap='gray')
            axes[idx, 0].set_title(column_titles[0] if idx == 0 else "")
            axes[idx, 0].axis('off')
            
            # Column 2: Ground truth mask
            axes[idx, 1].imshow(t1c_display, cmap='gray')
            axes[idx, 1].imshow(gt_mask, cmap='Reds', alpha=0.5)
            axes[idx, 1].set_title(column_titles[1] if idx == 0 else "")
            axes[idx, 1].axis('off')
            
            # Column 3: Predicted mask
            axes[idx, 2].imshow(t1c_display, cmap='gray')
            axes[idx, 2].imshow(pred_binary, cmap='Blues', alpha=0.5)
            axes[idx, 2].set_title(column_titles[2] if idx == 0 else "")
            axes[idx, 2].axis('off')
            
            # Column 4: Overlay (green=TP, red=FN, blue=FP)
            overlay = np.zeros((*t1c_display.shape, 3))
            overlay[..., :] = np.stack([t1c_display]*3, axis=-1)
            overlay[gt_mask > 0, 1] = 0.7  # Ground truth in green
            overlay[pred_binary > 0, 2] = 0.7  # Prediction in blue
            overlay[(gt_mask > 0) & (pred_binary > 0), :] = [0, 1, 0]  # TP in bright green
            overlay[(gt_mask > 0) & (pred_binary == 0), :] = [1, 0, 0]  # FN in red
            overlay[(gt_mask == 0) & (pred_binary > 0), :] = [0, 0, 1]  # FP in blue
            
            axes[idx, 3].imshow(np.clip(overlay, 0, 1))
            axes[idx, 3].set_title(column_titles[3] if idx == 0 else "")
            axes[idx, 3].axis('off')
            
            # Dice for this sample
            dice = compute_metrics(
                torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0),
                torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0),
            )["dice"]
            axes[idx, 0].set_ylabel(f"Sample {idx+1}\nDice: {dice:.3f}", fontsize=10)
            
            samples_collected += 1
        
        del images, masks
    
    plt.tight_layout()
    save_path = output_dir / "sample_predictions.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Sample predictions: {save_path}")
    
    # Save individual overlay images
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    print(f"[INFO] Individual overlays saved to: {overlay_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(args):
    """Run full evaluation pipeline."""
    print("=" * 70)
    print("BRAIN TUMOR SEGMENTATION - EVALUATION")
    print("=" * 70)
    
    output_dir = config.OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    
    # ── Plot training curves ──
    if config.LOG_FILE.exists():
        print("\n[1/3] Plotting training curves...")
        plot_training_curves(config.LOG_FILE, output_dir)
    else:
        print(f"[SKIP] No training log found at {config.LOG_FILE}")
    
    # ── Load model ──
    print("\n[2/3] Loading best model...")
    model = BrainTumorSegModel().to(config.DEVICE)
    
    ckpt_path = Path(args.checkpoint) if args.checkpoint else config.CHECKPOINT_DIR / "best_model.pt"
    if ckpt_path.exists():
        epoch, best_dice = load_checkpoint(ckpt_path, model)
        print(f"  Loaded from: {ckpt_path}")
        print(f"  Epoch: {epoch+1}, Best Dice: {best_dice:.4f}")
    else:
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return
    
    # ── Create test DataLoader ──
    _, _, test_loader = create_dataloaders(
        subset_ratio=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # ── Full test evaluation ──
    print("\n[3/3] Running evaluation...")
    criterion = CombinedLoss()
    
    from train import validate
    test_metrics = validate(model, test_loader, criterion, config.DEVICE)
    
    print(f"\n{'=' * 50}")
    print(f"TEST RESULTS")
    print(f"{'=' * 50}")
    for k, v in test_metrics.items():
        print(f"  {k:>12s}: {v:.4f}")
    
    # ── Confusion matrix & ROC ──
    print("\nGenerating confusion matrix and ROC curve...")
    roc_auc = generate_confusion_matrix(model, test_loader, config.DEVICE, output_dir)
    
    # ── Sample predictions ──
    print("\nGenerating sample predictions...")
    generate_sample_predictions(model, test_loader, config.DEVICE, output_dir,
                                n_samples=args.n_samples)
    
    print(f"\n{'=' * 70}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Dataset subset ratio for evaluation")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help="DataLoader workers")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of sample predictions to visualize")
    
    args = parser.parse_args()
    evaluate(args)
