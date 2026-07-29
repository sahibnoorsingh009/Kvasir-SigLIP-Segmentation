# Kvasir-SigLIP Segmentation

A reproducible project for comparing the Kvasir-SEG paper's ResUNet baseline with a
SigLIP2 encoder-based segmentation model on the author-released split.

## Study design

- Dataset files: Kaggle `debeshjha1/kvasirseg`
- Canonical split: authors' GitHub repository `DebeshJha/Kvasir-SEG`
- Official `train.txt`: model-development pool
- Official `val.txt`: untouched final benchmark set
- Internal split: 90%/10% split of official training pool, generated once and reused
- Primary metrics: per-image Dice and IoU
- Statistical comparison: paired bootstrap confidence interval and paired permutation test
- Seeds: 42, 43, 44

The original paper used 320×320 images, Dice loss, batch size 8, Nadam with learning rate
1e-4, at most 150 epochs, and threshold 0.5. This project reproduces that recipe for ResUNet.
For SigLIP2, the decoder uses learning rate 1e-4 while the pretrained encoder uses 1e-5.

## Included models

1. `resunet`: five-stage residual U-Net reproduction
2. `siglip2_unet`: SigLIP2 NaFlex vision encoder + four hidden-state projections + decoder

The default SigLIP2 checkpoint is:

`google/siglip2-base-patch16-naflex`

NaFlex is used because the paper protocol uses 320×320 inputs and the model accepts flexible
spatial shapes. The training code reads the processor-generated spatial shape instead of assuming
a fixed token grid.

## Quick start on RunPod

```bash
unzip Kvasir-SigLIP-Segmentation.zip
cd Kvasir-SigLIP-Segmentation

bash scripts/00_setup.sh
bash scripts/01_download_data.sh
bash scripts/02_prepare_splits.sh
bash scripts/03_audit.sh
```

Review:

- `results/audit/dataset_summary.json`
- `results/audit/overlays/`

Then run a smoke test:

```bash
python -m src.smoke_test --config configs/siglip2_full.yaml
```

Train ResUNet:

```bash
bash scripts/10_train_resunet_all_seeds.sh
```

Train SigLIP2:

```bash
bash scripts/11_train_siglip2_all_seeds.sh
```

Evaluate each checkpoint:

```bash
bash scripts/20_evaluate_all.sh
```

Run paired statistical analysis:

```bash
python -m src.statistical_analysis \
  --baseline results/predictions/resunet_seed42.csv \
  --candidate results/predictions/siglip2_full_seed42.csv \
  --name-a ResUNet \
  --name-b SigLIP2 \
  --out-dir results/statistics/resunet_vs_siglip2_seed42 \
  --bootstrap 10000 \
  --permutations 10000 \
  --seed 2026
```

## Kaggle authentication

`kagglehub` usually downloads public datasets without a legacy API token. If authentication is
requested, configure Kaggle credentials on RunPod before rerunning the download script.

## Expected data discovery

The downloader searches recursively for image and mask folders. If automatic discovery fails,
set these paths in the YAML files:

```yaml
data:
  image_dir: /workspace/data/kvasir-seg/Kvasir-SEG/images
  mask_dir: /workspace/data/kvasir-seg/Kvasir-SEG/masks
```

## Fairness rules

- Do not tune on `splits/official_test.txt`.
- Use identical internal splits for all methods.
- Use the same geometric augmentations for image and mask.
- Preserve separate learning rates for pretrained encoder and random decoder.
- Report all three seeds, not only the best seed.
- Compare per-image predictions on the same official test images.
- Treat the paper's published score as historical context; the main comparison is reproduced
  ResUNet versus SigLIP2 on the same author-released evaluation images.

## Output structure

```text
checkpoints/<experiment>/seed_<seed>/best.pt
results/history/<experiment>_seed<seed>.csv
results/predictions/<experiment>_seed<seed>.csv
results/qualitative/<experiment>_seed<seed>/
results/statistics/
```

## Important metric note

The original paper's reported Dice and mean IoU are unusually close. This project calculates
transparent foreground-only per-image metrics:

- Dice = (2TP + eps)/(2TP + FP + FN + eps)
- IoU = (TP + eps)/(TP + FP + FN + eps)

It also writes global/micro metrics so aggregation differences can be investigated.

## Main Results

Experiments were conducted on the official 120-image Kvasir-SEG test split using three random seeds.

| Method | Mean Dice | Seed SD | Mean IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| ResUNet | 0.8182 | 0.0072 | 0.7326 | 0.8739 | 0.8354 |
| SigLIP2 Frozen | 0.8745 | 0.0010 | 0.8024 | 0.9015 | 0.8903 |
| SigLIP2 Partial | 0.8813 | 0.0039 | 0.8116 | 0.8943 | 0.9072 |
| SigLIP2 Full | **0.9013** | **0.0026** | **0.8447** | **0.9170** | **0.9211** |

Full SigLIP2 improved mean Dice over ResUNet by 0.0832.

- Paired bootstrap 95% CI: [0.0546, 0.1147]
- Paired permutation test: p < 0.001
- Practically meaningful improvement threshold of 0.02 was exceeded
- Full SigLIP2 was clearly better on 59/120 images
- It was approximately tied on 52/120 images
- It was clearly worse on 9/120 images

### Lesion-size subgroup results

| Lesion size | ResUNet Dice | SigLIP2 Full Dice | Difference |
|---|---:|---:|---:|
| Small | 0.8285 | 0.8976 | +0.0691 |
| Medium | 0.8240 | 0.9157 | +0.0918 |
| Large | 0.7985 | 0.8918 | +0.0934 |

Detailed tables are available under `results/publication/`.
