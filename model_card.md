# Brain MRI Tumor Segmentation — Model Card

## Model Overview

| Property | Value |
|---|---|
| **Task** | Brain tumor segmentation from MRI scans |
| **Architecture** | 2D U-Net with EfficientNet-B3 encoder |
| **Input** | 4-channel 2D slice (T1c, T1n, T2-FLAIR, T2w), 256×256 |
| **Output** | Binary tumor mask (256×256) + cancer/no-cancer classification |
| **Framework** | PyTorch + segmentation_models_pytorch |
| **Training** | Mixed precision (FP16), AdamW optimizer, cosine warmup LR |
| **Competition** | Teknofest — Oncology AI Track |

## Dataset

- **Source**: BraTS-style brain MRI dataset (NIfTI format)
- **Size**: 11.07 GB, 203 patients, 596 timepoints
- **Modalities**: T1 contrast (T1c), T1 native (T1n), T2-FLAIR (T2f), T2 weighted (T2w)
- **Annotations**: Pixel-level tumor segmentation masks (594/596 with masks)
- **Split**: 80/10/10 patient-level (no data leakage)

## Training Details

- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **LR Schedule**: Linear warmup (5 epochs) + cosine annealing
- **Batch Size**: 8 (effective 32 with gradient accumulation)
- **Loss**: 0.5×Dice + 0.5×BCE (segmentation) + 0.2×BCE (classification)
- **Augmentation**: Random flip, rotation ±15°, elastic deformation, brightness/contrast jitter
- **Early Stopping**: Patience = 10 epochs
- **Precision**: Mixed precision (FP16) throughout

## Metrics

*To be filled after training completion.*

| Metric | Validation | Test |
|---|---|---|
| Dice Coefficient | — | — |
| IoU | — | — |
| Precision | — | — |
| Recall | — | — |
| Accuracy | — | — |
| AUC-ROC | — | — |

## Limitations & Honest Assessment

1. **Not clinically validated** — This model is for competition/research purposes only. Clinical deployment would require extensive prospective validation, regulatory approval, and clinical trials.

2. **2D slice-based approach** — While memory-efficient, 2D processing loses inter-slice spatial context. A 3D model could provide better volumetric consistency.

3. **Binary segmentation** — The current model treats all tumor subtypes equally. BraTS datasets often have multi-class labels (necrotic core, edema, enhancing tumor) which we binarize.

4. **Dataset size** — 203 patients is a reasonable dataset but smaller than clinical-grade training sets. Model generalization to different MRI scanners, protocols, and demographics may be limited.

5. **Transfer learning limitation** — EfficientNet-B3 was pre-trained on natural images (ImageNet), which differ significantly from medical images. MRI-specific pre-training could improve performance.

## Usage

```bash
# Training (15% subset for prototyping)
python train.py --epochs 50 --subset 0.15 --batch_size 8

# Training (full dataset)
python train.py --epochs 100 --subset 1.0 --batch_size 8

# Evaluation
python evaluate.py --subset 1.0

# Inference on a single scan
python inference.py --input "Data/PatientID_0003/Timepoint_1" --output "results/"
```

## License

Competition use only. Not for clinical deployment.
