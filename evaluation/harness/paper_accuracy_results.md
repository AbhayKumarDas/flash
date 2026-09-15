# Synthetic-Anomaly Accuracy Results — Combined Paper Tables

Five detection models × 8 MVTec AD 2 categories × 4 anomaly sources. Real is the oracle
reference, while Perlin, AnoStyler, and FLASH (Ours) calibrate thresholds with synthetic
anomalies and transfer them to the real test set. **All metrics are percentages (0–100).**

## Setting → calibration mapping

| setting | calibration data | evaluation data |
|---|---|---|
| Real (oracle) | Real test anomalies | Real test set |
| Perlin | Held-out normals + Perlin anomalies | Real test set |
| AnoStyler | Held-out normals + AnoStyler anomalies | Real test set |
| FLASH (Ours) | Held-out normals + FLASH anomalies | Real test set |

Real and Perlin are measured in both cross-model comparisons (three seeds each) and
averaged for lower variance. Dinomaly is evaluated in its dedicated comparison. AnoStyler
and FLASH are evaluated wherever their calibration data are available.

## image F1 — mean ± std over categories (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 80.37±6.52 | 69.41±14.18 | 62.37±22.95 | 79.11±6.59 |
| PatchCore | 82.46±7.23 | 67.50±16.07 | 43.02±25.03 | 74.64±12.31 |
| AnomalyDINO | 81.47±6.97 | 62.19±19.47 | 49.85±19.68 | 77.05±10.73 |
| Dinomaly | 81.81±5.99 | 60.42±24.92 | 55.37±18.86 | 77.51±11.17 |
| SuperADD (ViT-H+) | 83.64±8.47 | 79.12±10.38 | 64.36±22.75 | 78.13±8.37 |

## pixel F1 (SegF1) — mean ± std over categories (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 7.63±5.40 | 3.05±3.56 | 5.37±5.06 | 3.89±5.07 |
| PatchCore | 26.09±14.99 | 14.90±18.07 | 16.08±17.54 | 18.14±15.04 |
| AnomalyDINO | 33.55±20.77 | 17.47±17.21 | 13.99±15.71 | 26.11±21.14 |
| Dinomaly | 31.90±19.15 | 15.45±13.71 | 3.35±3.92 | 22.53±14.75 |
| SuperADD (ViT-H+) | 51.53±22.76 | 19.44±19.51 | 36.99±21.92 | 38.38±25.18 |

---

# Full per-metric tables

Each table: rows = 5 models, columns = 4 methods, mean ± std over 8 categories (percentages).

## image AUROC (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 61.79±11.34 | 61.81±11.33 | 61.80±12.76 | 61.52±11.20 |
| PatchCore | 72.50±17.12 | 72.49±17.12 | 72.48±17.59 | 70.73±16.46 |
| AnomalyDINO | 69.93±16.66 | 70.05±16.39 | 70.06±16.27 | 69.92±15.87 |
| Dinomaly | 68.49±13.07 | 68.29±12.93 | 68.30±12.93 | 68.87±12.22 |
| SuperADD (ViT-H+) | 74.50±16.46 | 74.13±16.73 | 74.19±16.70 | 74.01±16.70 |

## image F1 (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 80.37±6.52 | 69.41±14.18 | 62.37±22.95 | 79.11±6.59 |
| PatchCore | 82.46±7.23 | 67.50±16.07 | 43.02±25.03 | 74.64±12.31 |
| AnomalyDINO | 81.47±6.97 | 62.19±19.47 | 49.85±19.68 | 77.05±10.73 |
| Dinomaly | 81.81±5.99 | 60.42±24.92 | 55.37±18.86 | 77.51±11.17 |
| SuperADD (ViT-H+) | 83.64±8.47 | 79.12±10.38 | 64.36±22.75 | 78.13±8.37 |

## image AUPR (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 75.38±12.10 | 75.31±11.59 | 73.88±13.17 | 76.29±10.46 |
| PatchCore | 82.57±13.91 | 81.81±15.08 | 81.81±15.38 | 83.92±12.51 |
| AnomalyDINO | 81.56±14.35 | 80.83±15.54 | 80.81±15.47 | 82.28±13.01 |
| Dinomaly | 79.92±12.74 | 77.59±14.87 | 77.64±14.85 | 80.59±12.29 |
| SuperADD (ViT-H+) | 85.30±12.30 | 83.84±14.64 | 83.61±14.45 | 83.77±14.61 |

## image Prec (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 69.74±10.79 | 70.59±12.24 | 68.09±12.74 | 66.35±8.48 |
| PatchCore | 73.90±12.01 | 75.71±17.25 | 82.77±18.56 | 69.38±14.03 |
| AnomalyDINO | 73.39±12.89 | 67.76±15.57 | 82.66±16.79 | 66.37±8.45 |
| Dinomaly | 73.18±11.38 | 69.72±11.94 | 79.74±15.20 | 68.03±7.99 |
| SuperADD (ViT-H+) | 77.12±14.20 | 72.57±17.27 | 84.76±16.49 | 66.31±8.53 |

## image Rec (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 96.65±2.07 | 79.36±27.14 | 67.60±27.74 | 98.75±3.31 |
| PatchCore | 95.25±4.49 | 71.90±29.80 | 33.88±24.18 | 87.48±22.10 |
| AnomalyDINO | 94.37±6.23 | 65.58±30.09 | 41.87±21.43 | 93.70±16.66 |
| Dinomaly | 95.01±6.27 | 65.34±33.48 | 51.09±26.62 | 92.27±17.86 |
| SuperADD (ViT-H+) | 94.08±6.42 | 91.61±13.79 | 58.43±26.17 | 95.97±10.66 |

## pixel AUROC (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 81.37±5.47 | 81.37±5.47 | 81.20±5.38 | 81.55±5.60 |
| PatchCore | 87.29±7.22 | 87.29±7.22 | 87.32±7.17 | 86.96±7.76 |
| AnomalyDINO | 87.12±12.14 | 87.27±11.89 | 87.29±11.73 | 87.13±12.13 |
| Dinomaly | 88.17±10.58 | 88.17±10.58 | 88.17±10.58 | 88.17±10.58 |
| SuperADD (ViT-H+) | 90.92±10.50 | 90.92±10.50 | 90.88±10.57 | 90.94±10.52 |

## pixel F1 (SegF1) (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 7.63±5.40 | 3.05±3.56 | 5.37±5.06 | 3.89±5.07 |
| PatchCore | 26.09±14.99 | 14.90±18.07 | 16.08±17.54 | 18.14±15.04 |
| AnomalyDINO | 33.55±20.77 | 17.47±17.21 | 13.99±15.71 | 26.11±21.14 |
| Dinomaly | 31.90±19.15 | 15.45±13.71 | 3.35±3.92 | 22.53±14.75 |
| SuperADD (ViT-H+) | 51.53±22.76 | 19.44±19.51 | 36.99±21.92 | 38.38±25.18 |

## pixel AUPR (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 3.41±2.71 | 3.41±2.69 | 3.43±2.67 | 3.34±2.76 |
| PatchCore | 20.57±15.62 | 20.57±15.62 | 20.61±15.65 | 20.52±15.60 |
| AnomalyDINO | 27.38±18.89 | 27.26±18.94 | 27.14±18.90 | 27.41±18.85 |
| Dinomaly | 23.60±15.89 | 23.75±15.82 | 23.56±15.91 | 23.71±15.98 |
| SuperADD (ViT-H+) | 46.74±23.28 | 46.87±23.30 | 46.67±23.22 | 46.80±23.25 |

## pixel AUPRO (0.3) (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 42.10±9.97 | 42.10±9.97 | 42.46±9.91 | 41.74±10.18 |
| PatchCore | 54.02±19.38 | 54.02±19.38 | 53.99±19.40 | 54.05±19.36 |
| AnomalyDINO | 51.67±22.09 | 51.42±22.01 | 50.87±21.79 | 51.62±22.00 |
| Dinomaly | 58.74±20.91 | 58.74±20.91 | 58.74±20.91 | 58.74±20.91 |
| SuperADD (ViT-H+) | 50.53±22.35 | 50.53±22.35 | 50.69±22.44 | 50.37±22.30 |

## pixel AUPRO (0.05) (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| PaDiM | 9.96±7.04 | 9.96±7.04 | 9.69±6.91 | 10.23±7.24 |
| PatchCore | 28.72±19.86 | 28.72±19.86 | 28.64±19.88 | 28.80±19.83 |
| AnomalyDINO | 30.38±21.35 | 30.29±21.04 | 30.21±20.96 | 30.37±21.12 |
| Dinomaly | 30.85±19.93 | 30.85±19.93 | 30.85±19.93 | 30.85±19.93 |
| SuperADD (ViT-H+) | 29.36±25.63 | 29.36±25.63 | 29.29±25.55 | 29.43±25.72 |

---

# Per-category breakdown (headline metrics)

Each cell is the mean over 3 seeds (real/Perlin additionally averaged over the two
cross-model comparisons). Values are percentages.

## pixel F1 (SegF1) — per model (%)

### PaDiM

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 0.16 | 0.05 | 0.05 | 0.05 |
| fabric | 3.36 | 1.44 | 0.93 | 1.01 |
| fruit_jelly | 12.41 | 4.46 | 5.13 | 2.96 |
| rice | 6.26 | 2.03 | 6.21 | 1.74 |
| sheet_metal | 11.95 | 3.70 | 11.27 | 9.08 |
| vial | 10.01 | 0.18 | 3.22 | 0.37 |
| wallplugs | 1.09 | 0.88 | 0.94 | 0.72 |
| walnuts | 15.83 | 11.63 | 15.23 | 15.21 |

### PatchCore

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 0.06 | 0.04 | 0.00 | 0.00 |
| fabric | 15.27 | 1.88 | 2.05 | 15.19 |
| fruit_jelly | 40.02 | 38.92 | 36.06 | 14.28 |
| rice | 22.81 | 2.64 | 0.00 | 3.26 |
| sheet_metal | 30.69 | 5.12 | 0.01 | 26.32 |
| vial | 32.23 | 17.66 | 27.54 | 28.14 |
| wallplugs | 16.22 | 2.76 | 15.94 | 8.53 |
| walnuts | 51.41 | 50.15 | 47.04 | 49.38 |

### AnomalyDINO

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 0.06 | 0.04 | 0.00 | 0.00 |
| fabric | 46.11 | 16.61 | 20.11 | 37.66 |
| fruit_jelly | 40.22 | 15.92 | 13.42 | 21.26 |
| rice | 58.47 | 31.29 | 1.84 | 56.34 |
| sheet_metal | 32.09 | 9.37 | 3.12 | 8.16 |
| vial | 32.45 | 9.00 | 25.64 | 27.30 |
| wallplugs | 2.39 | 1.57 | 0.00 | 1.83 |
| walnuts | 56.59 | 55.95 | 47.82 | 56.30 |

### Dinomaly

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 0.03 | 0.03 | 0.00 | 0.00 |
| fabric | 27.45 | 15.68 | 3.77 | 19.46 |
| fruit_jelly | 52.89 | 28.04 | 9.76 | 37.18 |
| rice | 46.46 | 22.35 | 0.00 | 21.21 |
| sheet_metal | 44.24 | 15.00 | 0.00 | 30.90 |
| vial | 35.46 | 0.58 | 9.36 | 23.97 |
| wallplugs | 2.36 | 1.27 | 0.00 | 2.18 |
| walnuts | 46.27 | 40.70 | 3.90 | 45.35 |

### SuperADD (ViT-H+)

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 0.02 | 0.01 | 0.00 | 0.00 |
| fabric | 78.36 | 18.57 | 44.30 | 69.06 |
| fruit_jelly | 56.07 | 40.02 | 55.93 | 55.57 |
| rice | 58.75 | 6.21 | 25.36 | 51.90 |
| sheet_metal | 35.87 | 2.53 | 7.79 | 7.39 |
| vial | 57.05 | 56.46 | 46.63 | 39.41 |
| wallplugs | 54.30 | 1.98 | 50.79 | 17.41 |
| walnuts | 71.86 | 29.78 | 65.12 | 66.28 |

## image F1 — per model (%)

### PaDiM

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 71.49 | 44.60 | 64.67 | 68.93 |
| fabric | 73.89 | 73.17 | 58.37 | 73.17 |
| fruit_jelly | 88.27 | 69.37 | 87.24 | 85.71 |
| rice | 81.31 | 81.08 | 6.27 | 81.08 |
| sheet_metal | 89.64 | 88.24 | 65.13 | 88.24 |
| vial | 86.10 | 48.79 | 78.46 | 85.71 |
| wallplugs | 75.24 | 75.00 | 63.80 | 75.00 |
| walnuts | 77.06 | 75.00 | 75.00 | 75.00 |

### PatchCore

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 70.92 | 37.86 | 42.56 | 45.33 |
| fabric | 79.47 | 73.17 | 16.33 | 73.17 |
| fruit_jelly | 92.16 | 46.64 | 54.16 | 73.56 |
| rice | 80.73 | 81.08 | 11.42 | 81.08 |
| sheet_metal | 89.00 | 88.24 | 26.75 | 88.24 |
| vial | 91.02 | 64.28 | 84.67 | 85.71 |
| wallplugs | 74.53 | 75.00 | 32.53 | 75.00 |
| walnuts | 81.88 | 73.76 | 75.71 | 75.00 |

### AnomalyDINO

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 71.34 | 46.75 | 44.28 | 52.45 |
| fabric | 74.44 | 73.17 | 24.40 | 73.17 |
| fruit_jelly | 86.83 | 24.76 | 48.14 | 85.71 |
| rice | 84.26 | 81.08 | 17.98 | 81.08 |
| sheet_metal | 88.69 | 88.24 | 58.08 | 88.24 |
| vial | 91.61 | 54.81 | 74.65 | 85.71 |
| wallplugs | 75.07 | 54.98 | 55.17 | 75.00 |
| walnuts | 79.49 | 73.70 | 76.11 | 75.00 |

### Dinomaly

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 71.43 | 43.21 | 51.48 | 50.81 |
| fabric | 77.42 | 73.17 | 29.18 | 73.17 |
| fruit_jelly | 87.03 | 6.32 | 45.02 | 85.28 |
| rice | 81.16 | 81.08 | 40.71 | 81.08 |
| sheet_metal | 88.66 | 88.24 | 50.27 | 88.24 |
| vial | 89.05 | 56.19 | 87.57 | 85.71 |
| wallplugs | 76.27 | 56.15 | 55.82 | 75.63 |
| walnuts | 83.43 | 79.02 | 82.94 | 80.15 |

### SuperADD (ViT-H+)

| category | Real | Perlin | AnoStyler | FLASH (Ours) |
|---|---|---|---|---|
| can | 71.19 | 61.14 | 61.46 | 61.10 |
| fabric | 75.60 | 73.17 | 41.03 | 73.17 |
| fruit_jelly | 85.86 | 80.74 | 81.54 | 85.71 |
| rice | 86.60 | 81.08 | 25.17 | 81.08 |
| sheet_metal | 89.75 | 88.24 | 70.47 | 88.24 |
| vial | 99.52 | 98.61 | 99.84 | 85.71 |
| wallplugs | 76.47 | 75.00 | 52.87 | 75.00 |
| walnuts | 84.09 | 75.00 | 82.51 | 75.00 |
