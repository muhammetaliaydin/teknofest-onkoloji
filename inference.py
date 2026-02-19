"""
============================================================================
INFERENCE — Single-Image Prediction Pipeline
============================================================================
Usage:
    python inference.py --input "Data/PatientID_0003/Timepoint_1" --output "results/"

Input: Path to a timepoint folder containing 4 MRI modality NIfTI files
Output: 
  - Per-slice tumor masks saved as PNGs
  - Overlay visualizations
  - Classification result (cancerous / non-cancerous) with confidence
============================================================================
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.amp import autocast

import config

sys.stdout.reconfigure(encoding='utf-8')


def load_model(checkpoint_path: str = None):
    """
    Load the trained model from checkpoint.
    Tries TorchScript first, falls back to state_dict.
    """
    # Try TorchScript model first
    ts_path = config.PROJECT_ROOT / "brain_tumor_segmentation.pt"
    if ts_path.exists() and checkpoint_path is None:
        print(f"[MODEL] Loading TorchScript model: {ts_path}")
        model = torch.jit.load(str(ts_path), map_location=config.DEVICE)
        model.eval()
        return model, "torchscript"
    
    # Fall back to checkpoint
    from model import BrainTumorSegModel
    model = BrainTumorSegModel().to(config.DEVICE)
    
    ckpt_path = Path(checkpoint_path) if checkpoint_path else config.CHECKPOINT_DIR / "best_model.pt"
    if ckpt_path.exists():
        print(f"[MODEL] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[ERROR] No model found at {ckpt_path}")
        sys.exit(1)
    
    model.eval()
    return model, "checkpoint"


def load_timepoint_data(timepoint_dir: Path):
    """
    Load all 4 MRI modalities from a timepoint directory.
    Returns: (volume_4d, affine) where volume_4d is (H, W, D, 4)
    """
    modality_volumes = []
    
    for mod in config.MODALITY_SUFFIXES:
        # Find the file matching this modality
        matching = list(timepoint_dir.glob(f"*{mod}.nii.gz"))
        if not matching:
            raise FileNotFoundError(f"Missing modality {mod} in {timepoint_dir}")
        
        nii = nib.load(str(matching[0]))
        vol = nii.get_fdata(dtype=np.float32)
        modality_volumes.append(vol)
        affine = nii.affine
    
    # Stack as (H, W, D, 4)
    volume_4d = np.stack(modality_volumes, axis=-1)
    return volume_4d, affine


def normalize_slice(slice_4ch):
    """Z-score normalize each channel of a 2D slice."""
    normalized = np.copy(slice_4ch)
    for c in range(slice_4ch.shape[-1]):
        ch = normalized[:, :, c]
        nonzero = ch[ch > 0]
        if len(nonzero) > 0:
            mean = nonzero.mean()
            std = nonzero.std() + 1e-8
            ch[ch > 0] = (ch[ch > 0] - mean) / std
        normalized[:, :, c] = ch
    return normalized


@torch.no_grad()
def predict_volume(model, volume_4d, model_type="checkpoint"):
    """
    Run inference on an entire 3D volume, slice by slice.
    
    Args:
        model: trained model
        volume_4d: (H, W, D, 4) numpy array
        model_type: "torchscript" or "checkpoint"
    
    Returns:
        predictions: (H_out, W_out, D) probability map
        classifications: (D,) per-slice cancer probability
    """
    H, W, D, C = volume_4d.shape
    predictions = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, D), dtype=np.float32)
    classifications = np.zeros(D, dtype=np.float32)
    
    for z in range(D):
        slice_4ch = volume_4d[:, :, z, :]  # (H, W, 4)
        
        # Skip mostly-empty slices
        brain_fraction = np.count_nonzero(slice_4ch[:, :, 0]) / (H * W)
        if brain_fraction < config.MIN_BRAIN_FRACTION:
            continue
        
        # Normalize
        slice_norm = normalize_slice(slice_4ch)
        
        # Resize to model input size
        from skimage.transform import resize
        slice_resized = resize(slice_norm, (config.IMAGE_SIZE, config.IMAGE_SIZE, C),
                               preserve_range=True, anti_aliasing=True)
        
        # To tensor: (1, 4, H, W)
        tensor = torch.from_numpy(slice_resized.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(config.DEVICE)
        
        with autocast(device_type='cuda', enabled=config.USE_AMP and config.DEVICE.type == 'cuda'):
            if model_type == "torchscript":
                seg_logits = model(tensor)
                cls_prob = 0.0  # TorchScript may not support dual output
            else:
                seg_logits, cls_logits = model(tensor, return_classification=True)
                cls_prob = torch.sigmoid(cls_logits).item()
        
        seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        predictions[:, :, z] = seg_prob
        classifications[z] = cls_prob
    
    return predictions, classifications


def generate_visualizations(volume_4d, predictions, classifications, output_dir: Path):
    """
    Generate overlay visualizations for each clinically relevant slice.
    """
    H, W, D, C = volume_4d.shape
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find slices with significant predictions
    tumor_slices = []
    for z in range(D):
        pred_slice = predictions[:, :, z]
        if pred_slice.max() > 0.3:  # At least some tumor probability
            tumor_slices.append((z, pred_slice.max()))
    
    tumor_slices.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n[VIS] Found {len(tumor_slices)} slices with tumor predictions")
    
    # Generate overlay for top slices
    n_show = min(20, len(tumor_slices))
    
    if n_show > 0:
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i, (z, max_prob) in enumerate(tumor_slices[:n_show]):
            from skimage.transform import resize
            t1c = volume_4d[:, :, z, 0]
            t1c_resized = resize(t1c, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                                 preserve_range=True, anti_aliasing=True)
            t1c_display = (t1c_resized - t1c_resized.min()) / (t1c_resized.max() - t1c_resized.min() + 1e-8)
            
            pred = predictions[:, :, z]
            pred_binary = (pred > 0.5).astype(float)
            
            # Original
            axes[i, 0].imshow(t1c_display, cmap='gray')
            axes[i, 0].set_title(f"Slice {z} - T1c" if i == 0 else f"Slice {z}")
            axes[i, 0].axis('off')
            
            # Prediction heatmap
            axes[i, 1].imshow(t1c_display, cmap='gray')
            axes[i, 1].imshow(pred, cmap='hot', alpha=0.6, vmin=0, vmax=1)
            axes[i, 1].set_title("Tumor Probability" if i == 0 else "")
            axes[i, 1].axis('off')
            
            # Binary overlay
            overlay = np.stack([t1c_display]*3, axis=-1)
            overlay[pred_binary > 0] = [1, 0.2, 0.2]  # Red for tumor
            axes[i, 2].imshow(np.clip(overlay, 0, 1))
            axes[i, 2].set_title("Tumor Region" if i == 0 else "")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / "tumor_overlay_grid.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] Tumor overlay grid: {save_path}")
    
    # Save individual overlays for the top 5 slices
    for i, (z, max_prob) in enumerate(tumor_slices[:5]):
        from skimage.transform import resize
        t1c = volume_4d[:, :, z, 0]
        t1c_resized = resize(t1c, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                             preserve_range=True, anti_aliasing=True)
        t1c_display = (t1c_resized - t1c_resized.min()) / (t1c_resized.max() - t1c_resized.min() + 1e-8)
        pred = predictions[:, :, z]
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(t1c_display, cmap='gray')
        ax.imshow(pred, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        ax.set_title(f"Slice {z} | Tumor Prob: {max_prob:.2f}")
        ax.axis('off')
        
        save_path = output_dir / f"overlay_slice_{z:03d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    
    print(f"[SAVED] Individual overlays in: {output_dir}")


def main(args):
    """Main inference pipeline."""
    print("=" * 70)
    print("BRAIN TUMOR SEGMENTATION - INFERENCE")
    print("=" * 70)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)
    
    # ── Load model ──
    model, model_type = load_model(args.checkpoint)
    print(f"  Model type: {model_type}")
    
    # ── Load MRI data ──
    print(f"\n[DATA] Loading MRI from: {input_dir}")
    volume_4d, affine = load_timepoint_data(input_dir)
    print(f"  Volume shape: {volume_4d.shape}")
    print(f"  Modalities: {config.MODALITY_SUFFIXES}")
    
    # ── Run prediction ──
    print(f"\n[PREDICT] Running inference on {volume_4d.shape[2]} slices...")
    predictions, classifications = predict_volume(model, volume_4d, model_type)
    
    # ── Overall classification ──
    # Average classification confidence across all valid slices
    valid_cls = classifications[classifications > 0]
    if len(valid_cls) > 0:
        avg_confidence = valid_cls.mean()
        max_confidence = valid_cls.max()
    else:
        avg_confidence = 0.0
        max_confidence = 0.0
    
    is_cancerous = max_confidence > 0.5
    
    # Count tumor voxels in prediction
    tumor_volume = (predictions > 0.5).sum()
    total_volume = predictions.size
    tumor_fraction = tumor_volume / total_volume
    
    print(f"\n{'=' * 50}")
    print(f"DIAGNOSIS RESULT")
    print(f"{'=' * 50}")
    print(f"  Classification:     {'CANCEROUS' if is_cancerous else 'NON-CANCEROUS'}")
    print(f"  Max Confidence:     {max_confidence:.4f}")
    print(f"  Avg Confidence:     {avg_confidence:.4f}")
    print(f"  Tumor volume:       {tumor_volume} voxels ({tumor_fraction*100:.2f}% of brain)")
    print(f"  Affected slices:    {(predictions.max(axis=(0,1)) > 0.5).sum()}/{volume_4d.shape[2]}")
    
    # ── Generate visualizations ──
    print(f"\n[VIS] Generating visualizations...")
    generate_visualizations(volume_4d, predictions, classifications, output_dir)
    
    # ── Save prediction as NIfTI ──
    pred_nii_path = output_dir / "predicted_mask.nii.gz"
    from skimage.transform import resize
    # Resize back to original dimensions
    pred_full = resize(predictions, volume_4d.shape[:3], preserve_range=True, anti_aliasing=True)
    pred_nii = nib.Nifti1Image((pred_full > 0.5).astype(np.uint8), affine)
    nib.save(pred_nii, str(pred_nii_path))
    print(f"[SAVED] Predicted mask NIfTI: {pred_nii_path}")
    
    print(f"\n{'=' * 70}")
    print(f"Inference complete! Results saved to: {output_dir}")
    print(f"{'=' * 70}")
    
    # ── Clinical disclaimer ──
    print(f"""
 ** IMPORTANT DISCLAIMER **
 This model is for research and competition purposes only.
 It has NOT been clinically validated and should NEVER be used
 as the sole basis for medical diagnosis or treatment decisions.
 Always consult qualified medical professionals.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a brain MRI scan")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to timepoint folder with MRI NIfTI files")
    parser.add_argument("--output", type=str, default="inference_output",
                        help="Output directory for results")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    
    args = parser.parse_args()
    main(args)
