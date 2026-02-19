"""
============================================================================
CONFIG — Central Configuration for Brain MRI Tumor Segmentation
============================================================================
All hyperparameters, paths, and device settings in one place.
============================================================================
"""

import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "Data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = PROJECT_ROOT / "training_log.csv"

# Create directories
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2       # 2 for laptop (fewer CPU cores, shared thermal)
PIN_MEMORY = torch.cuda.is_available()
GPU_VRAM_GB = 4       # RTX 3050 Laptop GPU

# ── Data ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224               # Optimized for 4GB VRAM (224 vs 256 saves ~25% memory)
NUM_MODALITIES = 4             # T1c, T1n, T2-FLAIR, T2w
MODALITY_SUFFIXES = [          # Order matters — stacked as input channels
    "brain_t1c",
    "brain_t1n",
    "brain_t2f",
    "brain_t2w",
]
MASK_SUFFIX = "tumorMask"

# Subset ratio for prototyping (0.15 = 15% of patients)
SUBSET_RATIO = 0.15

# Minimum brain tissue fraction in a slice to include it
# (filters out mostly-empty slices that add noise)
MIN_BRAIN_FRACTION = 0.02

# Train/Val/Test split ratios (patient-level, not slice-level)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE = 4                 # Small batch for 4GB VRAM
GRAD_ACCUMULATION_STEPS = 8    # Effective batch = 4 * 8 = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
WARMUP_EPOCHS = 5
GRADIENT_CHECKPOINTING = True  # Saves ~30% VRAM by recomputing activations
EARLY_STOPPING_PATIENCE = 10
USE_AMP = True                 # Mixed precision (FP16)

# ── Model ──────────────────────────────────────────────────────────────────
ENCODER_NAME = "efficientnet-b0"  # B0 instead of B3 — fits 4GB VRAM comfortably
ENCODER_WEIGHTS = "imagenet"      # Pre-trained weights
NUM_CLASSES = 1                   # Binary segmentation (tumor / no tumor)

# ── Augmentation ───────────────────────────────────────────────────────────
AUG_ROTATION_LIMIT = 15        # degrees
AUG_BRIGHTNESS_LIMIT = 0.1
AUG_CONTRAST_LIMIT = 0.1
AUG_ELASTIC_ALPHA = 50
AUG_ELASTIC_SIGMA = 10

# ── Logging ────────────────────────────────────────────────────────────────
PRINT_EVERY_N_BATCHES = 50     # Print progress every N batches
