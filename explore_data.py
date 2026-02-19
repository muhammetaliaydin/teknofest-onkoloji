"""
============================================================================
STEP 1: MEMORY-SAFE DATA EXPLORATION
============================================================================
Scans the entire dataset folder structure using pathlib ONLY.
Does NOT load any image data into RAM.
Reports: patient count, timepoint distribution, file types, mask availability,
         class distribution, and estimated dataset size.
============================================================================
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# -- Configuration ----------------------------------------------------------
DATA_ROOT = Path(r"c:\Users\mali\Documents\Projects\TeknofestOnkoloji\Data")

# -- 1. Scan all patient directories ----------------------------------------
print("=" * 70)
print("STEP 1: MEMORY-SAFE DATA EXPLORATION")
print("=" * 70)

patient_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
print(f"\n[DIR] Total patient directories: {len(patient_dirs)}")

# Extract patient IDs
patient_ids = [d.name for d in patient_dirs]
id_numbers = [int(pid.split("_")[-1]) for pid in patient_ids]
print(f"   Patient ID range: {min(id_numbers)} - {max(id_numbers)}")
missing_ids = set(range(min(id_numbers), max(id_numbers)+1)) - set(id_numbers)
print(f"   Missing IDs in range: {len(missing_ids)} IDs missing")

# -- 2. Scan timepoints per patient -----------------------------------------
print(f"\n{'-' * 70}")
print("TIMEPOINT ANALYSIS")
print(f"{'-' * 70}")

timepoints_per_patient = {}
total_timepoints = 0

for patient_dir in patient_dirs:
    tp_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
    tp_names = [d.name for d in tp_dirs]
    timepoints_per_patient[patient_dir.name] = tp_names
    total_timepoints += len(tp_names)

tp_counts = Counter(len(v) for v in timepoints_per_patient.values())
print(f"\n[STATS] Total timepoints across all patients: {total_timepoints}")
print(f"   Timepoints per patient distribution:")
for count, freq in sorted(tp_counts.items()):
    print(f"     {count} timepoint(s): {freq} patients")

# -- 3. Scan files within each timepoint -------------------------------------
print(f"\n{'-' * 70}")
print("FILE TYPE ANALYSIS")
print(f"{'-' * 70}")

all_extensions = Counter()
all_file_suffixes = Counter()
total_files = 0
total_size_bytes = 0
has_mask = 0
no_mask = 0
file_sizes_by_type = defaultdict(list)

samples_with_mask = []
samples_without_mask = []

for patient_dir in patient_dirs:
    for tp_dir in sorted(patient_dir.iterdir()):
        if not tp_dir.is_dir():
            continue
        
        files_in_tp = list(tp_dir.iterdir())
        mask_found = False
        
        for f in files_in_tp:
            if f.is_file():
                total_files += 1
                size = f.stat().st_size
                total_size_bytes += size
                
                fname = f.name
                if fname.endswith('.nii.gz'):
                    ext = '.nii.gz'
                else:
                    ext = f.suffix
                all_extensions[ext] += 1
                
                file_sizes_by_type[ext].append(size)
                
                if 'tumorMask' in fname or 'mask' in fname.lower():
                    mask_found = True
                
                parts = fname.replace('.nii.gz', '').replace('.nii', '')
                modality_parts = parts.split('_')
                if len(modality_parts) >= 5:
                    modality = '_'.join(modality_parts[4:])
                    all_file_suffixes[modality] += 1
        
        if mask_found:
            has_mask += 1
            samples_with_mask.append(f"{patient_dir.name}/{tp_dir.name}")
        else:
            no_mask += 1
            samples_without_mask.append(f"{patient_dir.name}/{tp_dir.name}")

print(f"\n[FILES] Total files: {total_files}")
print(f"   Total dataset size: {total_size_bytes / (1024**3):.2f} GB")
print(f"\n   File extensions found:")
for ext, count in all_extensions.most_common():
    avg_size = sum(file_sizes_by_type[ext]) / len(file_sizes_by_type[ext])
    print(f"     {ext}: {count} files (avg size: {avg_size / (1024**2):.1f} MB)")

print(f"\n   Modality/file type breakdown:")
for suffix, count in all_file_suffixes.most_common():
    print(f"     {suffix}: {count} files")

# -- 4. Segmentation mask availability (CLASS DISTRIBUTION) -----------------
print(f"\n{'-' * 70}")
print("SEGMENTATION MASK ANALYSIS (CLASS DISTRIBUTION)")
print(f"{'-' * 70}")

total_tp = has_mask + no_mask
print(f"\n[MASKS] Timepoints WITH tumor mask:    {has_mask} ({has_mask/total_tp*100:.1f}%)")
print(f"   Timepoints WITHOUT tumor mask: {no_mask} ({no_mask/total_tp*100:.1f}%)")
print(f"   Total timepoint samples:       {total_tp}")

# -- 5. Patient-level analysis ----------------------------------------------
print(f"\n{'-' * 70}")
print("PATIENT-LEVEL ANALYSIS")
print(f"{'-' * 70}")

patients_all_masks = 0
patients_some_masks = 0
patients_no_masks = 0

for patient_dir in patient_dirs:
    tp_count = 0
    mask_count = 0
    for tp_dir in sorted(patient_dir.iterdir()):
        if not tp_dir.is_dir():
            continue
        tp_count += 1
        for f in tp_dir.iterdir():
            if f.is_file() and 'tumorMask' in f.name:
                mask_count += 1
                break
    
    if mask_count == tp_count:
        patients_all_masks += 1
    elif mask_count > 0:
        patients_some_masks += 1
    else:
        patients_no_masks += 1

print(f"\n[PATIENTS] ALL timepoints have masks: {patients_all_masks}")
print(f"   SOME timepoints have masks: {patients_some_masks}")
print(f"   NO timepoints have masks:   {patients_no_masks}")

# -- 6. Show some samples without masks -------------------------------------
print(f"\n{'-' * 70}")
print("SAMPLE TIMEPOINTS WITHOUT MASKS (first 15)")
print(f"{'-' * 70}")
for s in samples_without_mask[:15]:
    print(f"   [NO MASK] {s}")

# -- 7. Priority decision ---------------------------------------------------
print(f"\n{'=' * 70}")
print("PRIORITY DECISION")
print(f"{'=' * 70}")
print(f"""
>> PRIORITY 1 -- CANCER REGION SEGMENTATION confirmed!

Evidence:
  - Dataset contains .nii.gz (NIfTI) 3D volumetric brain MRI scans
  - 4 MRI modalities per scan: T1c, T1n, T2-FLAIR, T2w
  - Pixel-level tumor segmentation masks (tumorMask.nii.gz) are present
  - {has_mask} out of {total_tp} timepoints have tumor masks
  - This is a standard brain tumor segmentation dataset (BraTS-style)

Recommended approach:
  -> U-Net-based 3D/2D segmentation model
  -> Multi-modal input (stack 4 MRI sequences as channels)
  -> Binary or multi-class tumor segmentation
  -> Post-segmentation classification (tumor present vs absent)
""")

# -- 8. GPU memory estimation -----------------------------------------------
print(f"{'-' * 70}")
print("GPU MEMORY ESTIMATION")
print(f"{'-' * 70}")
print(f"""
Assumptions (to be verified after loading 1 sample):
  - NIfTI volumes likely ~240x240x155 voxels (standard BraTS)
  - 4 modalities x 240x240x155 x 4 bytes (float32) = ~57 MB per sample
  - With batch_size=2 + model + gradients: ~4-6 GB GPU memory
  - Mixed precision (FP16) would reduce to ~3-4 GB
  
Recommendation:
  - Start with batch_size=2 for 3D volumes
  - Use 2D slice-based approach if GPU memory < 8 GB
  - 2D slices: 4x240x240 x 4 bytes = ~0.9 MB per slice (very manageable)
""")

# -- 9. Save scan results to JSON -------------------------------------------
scan_results = {
    "total_patients": len(patient_dirs),
    "total_timepoints": total_timepoints,
    "total_files": total_files,
    "total_size_gb": round(total_size_bytes / (1024**3), 2),
    "timepoints_with_mask": has_mask,
    "timepoints_without_mask": no_mask,
    "file_extensions": dict(all_extensions),
    "modalities": dict(all_file_suffixes),
    "priority": "PRIORITY_1_SEGMENTATION",
    "samples_without_mask": samples_without_mask,
}

output_path = DATA_ROOT.parent / "data_scan_results.json"
with open(output_path, 'w') as f:
    json.dump(scan_results, f, indent=2)
print(f"\n[SAVED] Scan results saved to: {output_path}")
print("=" * 70)
