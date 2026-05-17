# SegRAD
## Segmentation-Guided Dual-Signal Reconstruction for Annotation-Free Anomaly Detection in Panoramic Dental Radiographs

---

## Folder structure

```
SegRAD/
│
├── unet50.pth              ← your trained segmentation model (place here)
│
├── segmentation.py         loads unet50.pth, produces binary teeth mask
├── autoencoder.py          DentalAutoencoder architecture (1-ch, 224×224)
├── dataset_ae.py           NormalROIDataset — grayscale ROIs, normal only
├── train_ae.py             training loop (MSE + SSIM combined loss)
├── utils.py                dual-signal fusion, NMS, bboxes, visualisation
├── inference.py            end-to-end pipeline + threshold calibration CLI
├── requirements.txt
│
├── data/
│   ├── normal/
│   │   ├── images/         ← ONLY healthy/normal OPG scans  (no anomalies)
│   │   └── masks/          ← binary masks (auto-generated in Step 1)
│   └── test/
│       ├── images/         ← mix of normal + anomalous OPGs
│       └── masks/          ← binary masks (auto-generated in Step 1)
│
├── checkpoints/            ← auto-created by train_ae.py
│   └── autoencoder.pth
│
└── results/                ← auto-created by inference.py
    ├── result_*.png        5-panel output figures
    └── segrad_summary.csv  per-image anomaly scores
```

---

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:
- `torch`, `torchvision`
- `segmentation-models-pytorch` (your UNet)
- `opencv-python`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `Pillow`

---

## Step-by-step run guide

### Step 1 — Pre-save masks (run once)

Runs `unet50.pth` on every image and saves binary mask PNGs to disk.
This means training and inference load masks instantly without re-running the UNet each time.

```python
# Run from inside the SegRAD/ folder
from segmentation import presave_masks_for_dir

# Normal images
presave_masks_for_dir(
    image_dir  = "data/normal/images",
    output_dir = "data/normal/masks",
    checkpoint = "unet50.pth",
    device     = "cpu",        # change to "cuda" if you have a GPU
)

# Test images
presave_masks_for_dir(
    image_dir  = "data/test/images",
    output_dir = "data/test/masks",
    checkpoint = "unet50.pth",
    device     = "cpu",
)
```

Or save as `presave.py` and run:
```bash
python presave.py
```

---

### Step 2 — Train the autoencoder

Edit the `CONFIG` block at the top of `train_ae.py` — only the paths need changing:

```python
CONFIG = {
    "normal_image_dir" : "data/normal/images",
    "mask_dir"         : "data/normal/masks",
    "seg_checkpoint"   : "unet50.pth",
    "checkpoint"       : "checkpoints/autoencoder.pth",
    ...
}
```

Then run:
```bash
python train_ae.py
```

**What to watch during training:**
- Both `train loss` and `val loss` should decrease together
- If `val loss` rises while `train loss` falls → reduce `latent_channels` to 64 or increase `val_split`
- Training on CPU: ~10–20 min for 100 images over 60 epochs
- Training on GPU: ~2–5 min

---

### Step 3 — Calibrate the anomaly threshold

This step finds the right threshold for YOUR data automatically.
Run it on your normal images — the 95th percentile error becomes the threshold.

```bash
python inference.py \
    --calibrate \
    --image_dir data/normal/images \
    --seg_ckpt  unet50.pth \
    --ae_ckpt   checkpoints/autoencoder.pth
```

Expected output:
```
[Calibrate] Normal error stats:
            mean=0.0318  std=0.0091
            p90=0.0412   p95=0.0461   p99=0.0538
[Calibrate] Suggested threshold (95th pct): 0.0461

→ Re-run with: --threshold 0.0461
```

Use that value in Step 4.

---

### Step 4 — Run inference

**Single image:**
```bash
python inference.py \
    --image     data/test/images/patient042.jpg \
    --seg_ckpt  unet50.pth \
    --ae_ckpt   checkpoints/autoencoder.pth \
    --threshold 0.0461
```

**Whole test folder (batch):**
```bash
python inference.py \
    --image_dir  data/test/images \
    --seg_ckpt   unet50.pth \
    --ae_ckpt    checkpoints/autoencoder.pth \
    --threshold  0.0461 \
    --output_dir results/
```

Each image produces a 5-panel PNG saved to `results/` showing:
1. Original OPG
2. Autoencoder reconstruction ("normal" version)
3. MSE error map (intensity anomalies)
4. SSIM structural error map (shape/geometry anomalies)
5. Fused map + final bounding boxes

A summary CSV is written to `results/segrad_summary.csv`.

---

## Tuning guide

| Parameter | Where | Effect | Start value |
|-----------|-------|--------|-------------|
| `--threshold` | inference CLI | Anomaly sensitivity — lower = more detections, more false positives | from Step 3 calibration |
| `--alpha` | inference CLI | MSE weight in fusion (0=SSIM only, 1=MSE only, 0.5=equal) | `0.5` |
| `--min_area` | inference CLI | Min pixel area to count as a detection | `80` |
| `latent_channels` | train_ae.py CONFIG | Bottleneck size — reduce to `64` if dataset < 50 images | `128` |
| `ssim_weight` | train_ae.py CONFIG | SSIM term in training loss | `0.5` |
| `patience` / `early_stop` | train_ae.py CONFIG | Training stop conditions | `12` / `20` |

**Most common issue:** Too many small scattered boxes → increase `--min_area` to 150–200.
**Second most common:** Missing large anomalies → lower `--threshold` by 0.005–0.01.

---

## What makes this novel (for report / viva)

Three contributions distinguish SegRAD from prior autoencoder anomaly detection work:

**1. Segmentation-guided ROI extraction** — the UNet isolates the dental arch before the autoencoder sees it, so the model learns only tooth morphology and is not distracted by jaw bone, sinuses, or background variation.

**2. Dual-signal error fusion** — rather than using only pixel-wise MSE as the anomaly signal (standard in the literature), SegRAD computes two complementary signals: MSE detects intensity/density anomalies (cavities, carious lesions), and a spatial SSIM map detects structural/geometric anomalies (missing teeth, irregular root forms, bone level changes). The weighted fusion `fused = α × MSE + (1-α) × SSIM_error` is ablatable — running `α=0, 0.25, 0.5, 0.75, 1.0` and comparing precision/recall gives a proper ablation study for the report.

**3. Annotation-free** — the system requires only normal images for training. No cavity labels, no lesion annotations, no bounding box supervision.

---

## Report description (paste-ready)

> "SegRAD is a segmentation-guided dual-signal autoencoder framework for
> annotation-free anomaly localisation in panoramic dental radiographs.
> A pre-trained UNet isolates the dental ROI from the OPG background;
> the resulting greyscale region is passed to a convolutional autoencoder
> trained exclusively on normal teeth. At inference, the model reconstructs
> the input as its healthy approximation. Two complementary reconstruction
> error signals are computed: a pixel-wise MSE map capturing intensity
> deviations and a spatial SSIM error map capturing structural deviations.
> These are fused via a weighted sum parameterised by α, thresholded at the
> 95th-percentile error of normal images, and morphologically cleaned to
> produce a binary anomaly mask from which bounding boxes are derived by
> contour analysis and non-maximum suppression."
