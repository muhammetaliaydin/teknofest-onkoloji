"""
============================================================================
DATASET — Memory-Efficient Brain MRI Dataset with 2D Slice Extraction
============================================================================
Key design decisions:
  1. LAZY LOADING: NIfTI volumes are NOT loaded into RAM at init time.
     Only metadata (paths, slice indices) are cached.
  2. ON-THE-FLY SLICING: Each __getitem__ loads only ONE 2D slice from 
     the 3D volume using nibabel's proxy object (memory-mapped).
  3. PATIENT-LEVEL SPLIT: Train/val/test are split by patient ID, not
     by slice, to prevent data leakage between sets.
  4. EMPTY SLICE FILTERING: Slices with <2% brain tissue are excluded.
============================================================================
"""

import os
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config
from augmentations import get_train_transforms, get_val_transforms


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Scan dataset and build a slice index WITHOUT loading image data
# ═══════════════════════════════════════════════════════════════════════════

def build_slice_index(
    data_root: Path,
    subset_ratio: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> List[Dict]:
    """
    Scans the dataset folder structure and builds an index of all valid
    2D slices across all patients/timepoints.
    
    Returns a list of dicts, each containing:
      - patient_id: str
      - timepoint: str
      - modality_paths: dict mapping modality suffix -> file path
      - mask_path: path to tumor mask (or None)
      - slice_idx: int (axial slice index)
      - has_tumor: bool (whether this slice has any tumor voxels)
    
    IMPORTANT: This function loads each volume BRIEFLY to check dimensions
    and identify valid slices, then immediately releases the data.
    Mask files are small (~30KB) and loaded to determine per-slice tumor presence.
    """
    
    # -- Step 1: Collect all patient directories --
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    # -- Step 2: Apply subset sampling (patient-level) --
    if subset_ratio < 1.0:
        random.seed(seed)
        n_patients = max(1, int(len(patient_dirs) * subset_ratio))
        patient_dirs = sorted(random.sample(patient_dirs, n_patients))
        if verbose:
            print(f"[SUBSET] Using {n_patients}/{len(list(data_root.iterdir()))} "
                  f"patients ({subset_ratio*100:.0f}%)")
    
    slice_index = []
    total_slices = 0
    tumor_slices = 0
    skipped_empty = 0
    
    for i, patient_dir in enumerate(patient_dirs):
        patient_id = patient_dir.name
        
        for tp_dir in sorted(patient_dir.iterdir()):
            if not tp_dir.is_dir():
                continue
            timepoint = tp_dir.name
            
            # -- Collect modality file paths --
            modality_paths = {}
            mask_path = None
            
            for f in tp_dir.iterdir():
                if not f.is_file() or not f.name.endswith('.nii.gz'):
                    continue
                
                fname = f.name.replace('.nii.gz', '')
                for mod in config.MODALITY_SUFFIXES:
                    if fname.endswith(mod):
                        modality_paths[mod] = f
                        break
                
                if config.MASK_SUFFIX in fname:
                    mask_path = f
            
            # Skip if not all 4 modalities are present
            if len(modality_paths) != config.NUM_MODALITIES:
                if verbose:
                    print(f"[WARN] Missing modalities in {patient_id}/{timepoint}, skipping")
                continue
            
            # -- Load ONE modality header to get volume dimensions --
            first_mod_path = list(modality_paths.values())[0]
            try:
                nii = nib.load(str(first_mod_path))
                vol_shape = nii.shape  # e.g., (240, 240, 155)
                num_slices = vol_shape[2]  # axial slices
            except Exception as e:
                if verbose:
                    print(f"[WARN] Failed to read {first_mod_path}: {e}")
                continue
            
            # -- Load mask to determine per-slice tumor presence --
            # Masks are very small (~30KB compressed) so this is fast
            mask_data = None
            if mask_path is not None:
                try:
                    mask_nii = nib.load(str(mask_path))
                    mask_data = mask_nii.get_fdata(dtype=np.float32)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] Failed to read mask {mask_path}: {e}")
            
            # -- Also load one modality to check brain content per slice --
            # (We only check sum > threshold, very fast)
            try:
                ref_data = nii.get_fdata(dtype=np.float32)
            except Exception:
                continue
            
            # -- Build slice entries --
            for slice_idx in range(num_slices):
                # Check if slice has enough brain tissue
                ref_slice = ref_data[:, :, slice_idx]
                brain_fraction = np.count_nonzero(ref_slice) / ref_slice.size
                
                if brain_fraction < config.MIN_BRAIN_FRACTION:
                    skipped_empty += 1
                    continue
                
                # Check tumor presence in this slice
                has_tumor = False
                if mask_data is not None:
                    mask_slice = mask_data[:, :, slice_idx]
                    has_tumor = np.any(mask_slice > 0)
                
                slice_entry = {
                    "patient_id": patient_id,
                    "timepoint": timepoint,
                    "modality_paths": {k: str(v) for k, v in modality_paths.items()},
                    "mask_path": str(mask_path) if mask_path else None,
                    "slice_idx": slice_idx,
                    "has_tumor": has_tumor,
                    "vol_shape": vol_shape,
                }
                slice_index.append(slice_entry)
                total_slices += 1
                if has_tumor:
                    tumor_slices += 1
            
            # Release memory immediately
            del ref_data, mask_data
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Scanned {i+1}/{len(patient_dirs)} patients...")
    
    if verbose:
        non_tumor = total_slices - tumor_slices
        print(f"\n[INDEX] Slice index built:")
        print(f"  Total valid slices: {total_slices}")
        print(f"  Tumor slices:       {tumor_slices} ({tumor_slices/max(1,total_slices)*100:.1f}%)")
        print(f"  Non-tumor slices:   {non_tumor} ({non_tumor/max(1,total_slices)*100:.1f}%)")
        print(f"  Skipped empty:      {skipped_empty}")
    
    return slice_index


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Patient-level train/val/test split
# ═══════════════════════════════════════════════════════════════════════════

def patient_split(
    slice_index: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits slice_index into train/val/test by PATIENT ID.
    This prevents data leakage — no patient appears in multiple splits.
    """
    # Group slices by patient
    patients = defaultdict(list)
    for entry in slice_index:
        patients[entry["patient_id"]].append(entry)
    
    patient_ids = sorted(patients.keys())
    random.seed(seed)
    random.shuffle(patient_ids)
    
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_pids = set(patient_ids[:n_train])
    val_pids = set(patient_ids[n_train:n_train + n_val])
    test_pids = set(patient_ids[n_train + n_val:])
    
    train_slices = [e for e in slice_index if e["patient_id"] in train_pids]
    val_slices = [e for e in slice_index if e["patient_id"] in val_pids]
    test_slices = [e for e in slice_index if e["patient_id"] in test_pids]
    
    print(f"\n[SPLIT] Patient-level split (no leakage):")
    print(f"  Train: {len(train_pids)} patients, {len(train_slices)} slices")
    print(f"  Val:   {len(val_pids)} patients, {len(val_slices)} slices")
    print(f"  Test:  {len(test_pids)} patients, {len(test_slices)} slices")
    
    return train_slices, val_slices, test_slices


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════

class BrainMRIDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for brain MRI tumor segmentation.
    
    Each sample is a 2D axial slice with:
      - Input:  (4, H, W) tensor — 4 MRI modalities stacked as channels
      - Target: (1, H, W) tensor — binary tumor mask
    
    Key features:
      - Lazy loading: only the requested slice is loaded per __getitem__
      - On-the-fly augmentation via albumentations
      - Per-modality intensity normalization (z-score)
    """
    
    def __init__(
        self,
        slice_index: List[Dict],
        transform=None,
        normalize: bool = True,
    ):
        self.slice_index = slice_index
        self.transform = transform
        self.normalize = normalize
        
        # Cache for loaded volumes (LRU-style, limited to save RAM)
        # Key: modality_path, Value: nibabel proxy image
        # We don't cache pixel data — only the nib object for fast slicing
        self._nii_cache = {}
        self._cache_max_size = 50  # Keep at most 50 nii objects cached
    
    def __len__(self):
        return len(self.slice_index)
    
    def _load_nii(self, path: str):
        """Load a NIfTI file, using cache to avoid repeated disk reads."""
        if path not in self._nii_cache:
            # Evict oldest if cache full
            if len(self._nii_cache) >= self._cache_max_size:
                oldest_key = next(iter(self._nii_cache))
                del self._nii_cache[oldest_key]
            self._nii_cache[path] = nib.load(path)
        return self._nii_cache[path]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.slice_index[idx]
        slice_idx = entry["slice_idx"]
        
        # ── Load 4 modality slices ─────────────────────────────────────
        channels = []
        for mod in config.MODALITY_SUFFIXES:
            nii = self._load_nii(entry["modality_paths"][mod])
            # get_fdata loads data; we take only the slice we need
            vol = nii.dataobj[..., slice_idx].astype(np.float32)
            channels.append(vol)
        
        # Stack as (H, W, 4) for albumentations (expects HWC)
        image = np.stack(channels, axis=-1)  # (H, W, 4)
        
        # ── Load mask slice ────────────────────────────────────────────
        if entry["mask_path"] is not None:
            mask_nii = self._load_nii(entry["mask_path"])
            mask = mask_nii.dataobj[..., slice_idx].astype(np.float32)
            # Binarize mask (any tumor label > 0 becomes 1)
            mask = (mask > 0).astype(np.float32)
        else:
            # No mask = no tumor
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # ── Normalize per-modality (z-score) ───────────────────────────
        if self.normalize:
            for c in range(image.shape[-1]):
                ch = image[:, :, c]
                # Only normalize on non-zero voxels (brain region)
                nonzero = ch[ch > 0]
                if len(nonzero) > 0:
                    mean = nonzero.mean()
                    std = nonzero.std() + 1e-8
                    ch[ch > 0] = (ch[ch > 0] - mean) / std
                image[:, :, c] = ch
        
        # ── Apply augmentations ────────────────────────────────────────
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]  # (4, H, W) tensor
            mask = transformed["mask"]    # (H, W) tensor
        else:
            # Manual conversion to tensor if no transform
            image = torch.from_numpy(image.transpose(2, 0, 1))  # (4, H, W)
            mask = torch.from_numpy(mask)
        
        # Ensure mask has channel dimension: (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return image.float(), mask.float()
    
    def get_tumor_weights(self) -> List[float]:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Tumor slices get higher weight to balance class distribution.
        """
        tumor_count = sum(1 for e in self.slice_index if e["has_tumor"])
        non_tumor_count = len(self.slice_index) - tumor_count
        
        if tumor_count == 0 or non_tumor_count == 0:
            return [1.0] * len(self.slice_index)
        
        # Weight inversely proportional to class frequency
        w_tumor = len(self.slice_index) / (2.0 * tumor_count)
        w_non_tumor = len(self.slice_index) / (2.0 * non_tumor_count)
        
        weights = []
        for entry in self.slice_index:
            weights.append(w_tumor if entry["has_tumor"] else w_non_tumor)
        
        return weights


# ═══════════════════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_dataloaders(
    subset_ratio: float = None,
    batch_size: int = None,
    num_workers: int = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train/val/test DataLoaders with proper configuration.
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if subset_ratio is None:
        subset_ratio = config.SUBSET_RATIO
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    print("=" * 60)
    print("BUILDING DATA PIPELINE")
    print("=" * 60)
    
    # Step 1: Build slice index
    print("\n[1/3] Scanning dataset and building slice index...")
    slice_index = build_slice_index(
        config.DATA_ROOT,
        subset_ratio=subset_ratio,
        seed=seed,
    )
    
    # Step 2: Patient-level split
    print("\n[2/3] Splitting by patient ID...")
    train_idx, val_idx, test_idx = patient_split(
        slice_index,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=seed,
    )
    
    # Step 3: Create datasets
    print("\n[3/3] Creating DataLoaders...")
    train_dataset = BrainMRIDataset(train_idx, transform=get_train_transforms())
    val_dataset = BrainMRIDataset(val_idx, transform=get_val_transforms())
    test_dataset = BrainMRIDataset(test_idx, transform=get_val_transforms())
    
    # Weighted sampler for training (handles class imbalance)
    train_weights = train_dataset.get_tumor_weights()
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # WeightedRandomSampler replaces shuffle
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    
    print(f"\n[READY] DataLoaders created:")
    print(f"  Train: {len(train_dataset)} slices, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} slices, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} slices, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Pin memory: {config.PIN_MEMORY}")
    
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST (run this file directly to verify pipeline)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Testing data pipeline with 5% subset...\n")
    train_loader, val_loader, test_loader = create_dataloaders(
        subset_ratio=0.05,
        batch_size=4,
        num_workers=0,  # 0 workers for testing
    )
    
    # Grab one batch
    print("\nLoading first batch...")
    images, masks = next(iter(train_loader))
    print(f"\n[BATCH]")
    print(f"  Images shape: {images.shape}")    # Expected: (4, 4, 256, 256)
    print(f"  Masks shape:  {masks.shape}")      # Expected: (4, 1, 256, 256)
    print(f"  Images dtype: {images.dtype}")
    print(f"  Masks dtype:  {masks.dtype}")
    print(f"  Images range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Masks unique: {masks.unique().tolist()}")
    print(f"  Tumor pixels in batch: {masks.sum().item():.0f}")
    print("\nData pipeline test PASSED!")
