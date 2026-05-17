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

A summary CSV is written to `results/segrad_summary.csv`.

---
