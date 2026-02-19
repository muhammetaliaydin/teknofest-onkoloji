"""
============================================================================
AUGMENTATIONS — MRI-Appropriate On-The-Fly Data Augmentation
============================================================================
Uses albumentations for fast, GPU-friendly augmentations.
All transforms are applied during DataLoader iteration — no extra disk usage.
Medical imaging constraints:
  - No heavy warping that distorts anatomy
  - Conservative rotation (±15°)
  - Mild brightness/contrast jitter
  - Elastic deformation (helps segmentation generalization)
============================================================================
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


def get_train_transforms():
    """
    Training augmentations — applied on-the-fly to each 2D slice.
    Both image (4-channel) and mask (1-channel) are transformed together
    to maintain spatial alignment.
    """
    return A.Compose([
        # -- Spatial transforms (applied to both image and mask) --
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(
            limit=config.AUG_ROTATION_LIMIT,
            border_mode=0,  # zero-padding at borders
            p=0.5
        ),
        # Elastic deformation — good for segmentation, mild settings
        A.ElasticTransform(
            alpha=config.AUG_ELASTIC_ALPHA,
            sigma=config.AUG_ELASTIC_SIGMA,
            p=0.3
        ),
        # -- Intensity transforms (applied to image only, not mask) --
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=0.4
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        # Convert to tensor
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Validation/test transforms — deterministic, no randomness.
    Only resize + tensor conversion.
    """
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        ToTensorV2(),
    ])
