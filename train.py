"""
============================================================================
TRAIN — Full Training Loop for Brain Tumor Segmentation
============================================================================
Features:
  - Mixed precision training (torch.cuda.amp) — ~40% memory savings
  - Gradient accumulation for effective larger batch sizes
  - Cosine annealing LR scheduler with linear warmup
  - Checkpoint saving every epoch + best model tracking
  - Resume-from-checkpoint capability
  - Early stopping with patience
  - CSV metric logging every epoch
  - Memory management (empty_cache after validation)
  - Auto batch-size halving on OOM error
============================================================================
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np

import config
from dataset import create_dataloaders
from model import BrainTumorSegModel, CombinedLoss, compute_metrics

sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER: Cosine Annealing with Linear Warmup
# ═══════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for warmup_epochs, then cosine decay to min_lr.
    """
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


# ═══════════════════════════════════════════════════════════════════════════
# CSV LOGGER
# ═══════════════════════════════════════════════════════════════════════════

class CSVLogger:
    """Logs metrics to a CSV file every epoch."""
    
    FIELDS = [
        "epoch", "lr", "train_loss", "train_dice", "train_iou",
        "val_loss", "val_dice", "val_iou",
        "val_precision", "val_recall", "val_accuracy",
        "epoch_time_sec",
    ]
    
    def __init__(self, filepath):
        self.filepath = filepath
        if not Path(filepath).exists():
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()
    
    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v 
                           for k, v in row.items()})


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    grad_accum_steps=1, epoch=0, total_epochs=0,
):
    """
    Train for one epoch with mixed precision and gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Classification target: 1 if any tumor pixel in the sample
        cls_targets = (masks.sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        
        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            seg_logits, cls_logits = model(images, return_classification=True)
            total_loss, seg_loss, cls_loss = criterion(
                seg_logits, masks, cls_logits, cls_targets
            )
            # Scale loss for gradient accumulation
            total_loss = total_loss / grad_accum_steps
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        
        # Step optimizer every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        running_loss += total_loss.item() * grad_accum_steps
        metrics = compute_metrics(seg_logits.detach(), masks)
        running_dice += metrics["dice"]
        running_iou += metrics["iou"]
        num_batches += 1
        
        # Print progress
        if (batch_idx + 1) % config.PRINT_EVERY_N_BATCHES == 0:
            avg_loss = running_loss / num_batches
            avg_dice = running_dice / num_batches
            print(f"  Epoch [{epoch+1}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")
        
        # Release batch from GPU
        del images, masks, seg_logits, cls_logits
        
        # GPU memory report after first batch
        if batch_idx == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            print(f"  [GPU] After 1st batch: {alloc:.0f} MB allocated / "
                  f"{reserved:.0f} MB reserved / {total:.0f} MB total")
    
    return {
        "loss": running_loss / max(num_batches, 1),
        "dice": running_dice / max(num_batches, 1),
        "iou": running_iou / max(num_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate model on val/test set in inference mode.
    """
    model.eval()
    running_loss = 0.0
    running_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0, "accuracy": 0}
    num_batches = 0
    
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        cls_targets = (masks.sum(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        
        with autocast(device_type='cuda', enabled=config.USE_AMP):
            seg_logits, cls_logits = model(images, return_classification=True)
            total_loss, _, _ = criterion(seg_logits, masks, cls_logits, cls_targets)
        
        running_loss += total_loss.item()
        metrics = compute_metrics(seg_logits, masks)
        for k in running_metrics:
            running_metrics[k] += metrics[k]
        num_batches += 1
        
        del images, masks, seg_logits, cls_logits
    
    n = max(num_batches, 1)
    return {
        "loss": running_loss / n,
        "dice": running_metrics["dice"] / n,
        "iou": running_metrics["iou"] / n,
        "precision": running_metrics["precision"] / n,
        "recall": running_metrics["recall"] / n,
        "accuracy": running_metrics["accuracy"] / n,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dice, filepath):
    """Save training state for resumption."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_dice": best_dice,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load training state from checkpoint."""
    checkpoint = torch.load(filepath, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint["epoch"], checkpoint.get("best_dice", 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    """Main training function."""
    
    print("=" * 70)
    print("BRAIN TUMOR SEGMENTATION — TRAINING")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ── Create DataLoaders ─────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        subset_ratio=args.subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # ── Create Model ───────────────────────────────────────────────────
    model = BrainTumorSegModel().to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: U-Net + {config.ENCODER_NAME}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # ── Loss, Optimizer, Scheduler ─────────────────────────────────────
    criterion = CombinedLoss(seg_weight=0.8, cls_weight=0.2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=args.epochs,
    )
    scaler = GradScaler('cuda', enabled=config.USE_AMP)
    
    # ── Resume from checkpoint if available ────────────────────────────
    start_epoch = 0
    best_dice = 0.0
    
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"\nResuming from checkpoint: {ckpt_path}")
            start_epoch, best_dice = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
            start_epoch += 1  # Start from next epoch
            print(f"  Resuming from epoch {start_epoch}, best dice: {best_dice:.4f}")
    
    # ── CSV Logger ─────────────────────────────────────────────────────
    logger = CSVLogger(config.LOG_FILE)
    
    # ── GPU memory check after first batch ─────────────────────────────
    if torch.cuda.is_available():
        print("\n[GPU] Memory after model creation:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"  Cached:    {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
    
    # ── Training loop ──────────────────────────────────────────────────
    patience_counter = 0
    
    print(f"\n{'=' * 70}")
    print(f"Starting training: {args.epochs - start_epoch} epochs")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accumulation: {config.GRAD_ACCUMULATION_STEPS}")
    print(f"  Effective batch: {args.batch_size * config.GRAD_ACCUMULATION_STEPS}")
    print(f"  Mixed precision: {config.USE_AMP}")
    print(f"  Gradient checkpoint: {getattr(config, 'GRADIENT_CHECKPOINTING', False)}")
    print(f"  Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"{'=' * 70}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # ── Train ──────────────────────────────────────────────────
        try:
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                config.DEVICE, config.GRAD_ACCUMULATION_STEPS,
                epoch, args.epochs,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] GPU out of memory! Halving batch size...")
                torch.cuda.empty_cache()
                args.batch_size = max(1, args.batch_size // 2)
                print(f"  New batch size: {args.batch_size}")
                print(f"  Recreating DataLoaders...")
                train_loader, val_loader, test_loader = create_dataloaders(
                    subset_ratio=args.subset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                continue  # Retry this epoch
            raise
        
        # ── Validate ───────────────────────────────────────────────
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        
        # Free GPU memory after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ── Step scheduler ─────────────────────────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        epoch_time = time.time() - epoch_start
        
        # ── Log metrics ────────────────────────────────────────────
        logger.log({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_sec": epoch_time,
        })
        
        # ── Print epoch summary ────────────────────────────────────
        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch+1}/{args.epochs}  ({epoch_time:.1f}s)  LR: {current_lr:.6f}")
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f} | IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
        
        # ── Save checkpoint ────────────────────────────────────────
        ckpt_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1:03d}.pt"
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dice, ckpt_path)
        
        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_path = config.CHECKPOINT_DIR / "best_model.pt"
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dice, best_path)
            print(f"  >> New best model! Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  >> No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        # ── Early stopping ─────────────────────────────────────────
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[EARLY STOPPING] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs.")
            break
    
    # ── Final evaluation on test set ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'=' * 70}")
    
    # Load best model
    best_path = config.CHECKPOINT_DIR / "best_model.pt"
    if best_path.exists():
        load_checkpoint(best_path, model)
        print(f"Loaded best model (Dice: {best_dice:.4f})")
    
    test_metrics = validate(model, test_loader, criterion, config.DEVICE)
    print(f"\nTest Results:")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Dice:      {test_metrics['dice']:.4f}")
    print(f"  IoU:       {test_metrics['iou']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    
    # ── Export as TorchScript ──────────────────────────────────────────
    print(f"\nExporting model to TorchScript...")
    model.eval()
    model_cpu = model.to("cpu")
    example_input = torch.randn(1, 4, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    try:
        traced = torch.jit.trace(model_cpu, example_input)
        export_path = config.PROJECT_ROOT / "brain_tumor_segmentation.pt"
        traced.save(str(export_path))
        print(f"  Model exported to: {export_path}")
    except Exception as e:
        print(f"  [WARN] TorchScript tracing failed: {e}")
        print(f"  Saving state_dict instead...")
        torch.save(model_cpu.state_dict(), config.PROJECT_ROOT / "brain_tumor_segmentation_weights.pt")
    
    print(f"\nTraining complete! Best validation Dice: {best_dice:.4f}")
    print(f"Logs saved to: {config.LOG_FILE}")
    print(f"Checkpoints in: {config.CHECKPOINT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain tumor segmentation model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--subset", type=float, default=config.SUBSET_RATIO,
                        help="Fraction of patients to use (0.0-1.0)")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS,
                        help="DataLoader num_workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    train(args)
