"""
============================================================================
MODEL — 2D U-Net with EfficientNet Encoder for Brain Tumor Segmentation
============================================================================
Architecture:
  - Encoder: EfficientNet-B0 (pretrained on ImageNet, optimized for 4GB VRAM)
  - Decoder: U-Net decoder with skip connections
  - Input: (B, 4, H, W) — 4 MRI modalities as channels
  - Output: (B, 1, H, W) — binary tumor mask (sigmoid activated)
  
Uses segmentation_models_pytorch (smp) for clean implementation.
The first conv layer is adapted from 3-channel (ImageNet) to 4-channel input.
============================================================================
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import segmentation_models_pytorch as smp

import config


class BrainTumorSegModel(nn.Module):
    """
    U-Net segmentation model for brain tumor detection.
    
    Also includes a classification head that outputs a binary
    cancer/no-cancer prediction from the encoder features.
    """
    
    def __init__(
        self,
        encoder_name: str = None,
        encoder_weights: str = None,
        in_channels: int = None,
        num_classes: int = None,
        gradient_checkpointing: bool = None,
    ):
        super().__init__()
        
        encoder_name = encoder_name or config.ENCODER_NAME
        encoder_weights = encoder_weights or config.ENCODER_WEIGHTS
        in_channels = in_channels or config.NUM_MODALITIES
        num_classes = num_classes or config.NUM_CLASSES
        self.use_gradient_checkpointing = (
            gradient_checkpointing if gradient_checkpointing is not None
            else getattr(config, 'GRADIENT_CHECKPOINTING', False)
        )
        
        # ── Segmentation backbone (U-Net with EfficientNet encoder) ────
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # We apply sigmoid in forward/loss
        )
        
        # Enable gradient checkpointing on encoder to save VRAM
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # ── Classification head (from encoder bottleneck features) ─────
        # Get the encoder output channels from smp
        encoder_channels = self.segmentation_model.encoder.out_channels
        bottleneck_channels = encoder_channels[-1]  # Deepest feature map
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(bottleneck_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),               # Binary: cancer / no cancer
        )
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on encoder blocks to save ~30% VRAM."""
        encoder = self.segmentation_model.encoder
        for name, module in encoder.named_children():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
            # Wrap major encoder blocks with checkpointing
            for child_name, child in module.named_children():
                if isinstance(child, nn.Sequential) and len(list(child.children())) > 0:
                    original_forward = child.forward
                    def make_ckpt_forward(mod):
                        orig = mod.forward
                        def ckpt_forward(*args, **kwargs):
                            if self.training:
                                return cp.checkpoint(orig, *args, use_reentrant=False, **kwargs)
                            return orig(*args, **kwargs)
                        return ckpt_forward
                    child.forward = make_ckpt_forward(child)
    
    def forward(self, x, return_classification=False):
        """
        Forward pass.
        
        Args:
            x: (B, 4, H, W) input tensor
            return_classification: if True, also return classification logits
            
        Returns:
            seg_logits: (B, 1, H, W) segmentation logits (pre-sigmoid)
            cls_logits: (B, 1) classification logits (only if return_classification)
        """
        # Use the full segmentation model's forward for segmentation output
        seg_logits = self.segmentation_model(x)
        
        if return_classification:
            # Separately get encoder features for classification head
            features = self.segmentation_model.encoder(x)
            cls_logits = self.classification_head(features[-1])
            return seg_logits, cls_logits
        
        return seg_logits
    
    def predict(self, x):
        """
        Inference-mode prediction with sigmoid activation.
        Returns:
            seg_probs: (B, 1, H, W) probabilities [0, 1]
            cls_probs: (B, 1) classification probabilities [0, 1]
        """
        self.eval()
        with torch.no_grad():
            seg_logits, cls_logits = self.forward(x, return_classification=True)
            seg_probs = torch.sigmoid(seg_logits)
            cls_probs = torch.sigmoid(cls_logits)
        return seg_probs, cls_probs


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.
    
    Dice Loss: directly optimizes the Dice coefficient (overlap metric).
    BCE Loss: provides stable gradient signal, especially for small tumors.
    
    Combined loss = alpha * DiceLoss + (1 - alpha) * BCELoss
    """
    
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, logits, targets):
        """Computes Dice loss from logits."""
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice
    
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce(logits, targets)
        return self.alpha * dice + (1.0 - self.alpha) * bce


class CombinedLoss(nn.Module):
    """
    Full loss: segmentation loss + classification loss.
    
    Total = seg_weight * DiceBCE(seg_logits, seg_targets)
          + cls_weight * BCE(cls_logits, cls_targets)
    """
    
    def __init__(self, seg_weight: float = 0.8, cls_weight: float = 0.2):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss_fn = DiceBCELoss()
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, seg_logits, seg_targets, cls_logits=None, cls_targets=None):
        seg_loss = self.seg_loss_fn(seg_logits, seg_targets)
        
        if cls_logits is not None and cls_targets is not None:
            cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
            total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
            return total, seg_loss, cls_loss
        
        return seg_loss, seg_loss, torch.tensor(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(seg_logits, seg_targets, threshold=0.5):
    """
    Computes segmentation metrics from logits and targets.
    
    Returns dict with: dice, iou, precision, recall, accuracy
    """
    with torch.no_grad():
        probs = torch.sigmoid(seg_logits)
        preds = (probs > threshold).float()
        
        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = seg_targets.view(-1)
        
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1 - targets_flat)).sum()
        fn = ((1 - preds_flat) * targets_flat).sum()
        tn = ((1 - preds_flat) * (1 - targets_flat)).sum()
        
        smooth = 1e-7
        
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
        
        return {
            "dice": dice.item(),
            "iou": iou.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "accuracy": accuracy.item(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing model architecture...")
    
    model = BrainTumorSegModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 4, 256, 256)
    
    # Segmentation only
    seg_out = model(x)
    print(f"\nSegmentation output:    {seg_out.shape}")
    
    # Segmentation + Classification
    seg_out, cls_out = model(x, return_classification=True)
    print(f"Segmentation output:    {seg_out.shape}")
    print(f"Classification output:  {cls_out.shape}")
    
    # Test loss
    target_seg = torch.randint(0, 2, (2, 1, 256, 256)).float()
    target_cls = torch.randint(0, 2, (2, 1)).float()
    
    loss_fn = CombinedLoss()
    total_loss, seg_loss, cls_loss = loss_fn(seg_out, target_seg, cls_out, target_cls)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"Seg loss:   {seg_loss.item():.4f}")
    print(f"Cls loss:   {cls_loss.item():.4f}")
    
    # Test metrics
    metrics = compute_metrics(seg_out, target_seg)
    print(f"\nMetrics: {metrics}")
    
    print("\nModel test PASSED!")
