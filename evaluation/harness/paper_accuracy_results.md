# Synthetic-Anomaly Accuracy Results — F1 Tables

Five detectors × 8 MVTec AD 2 categories × 4 calibration sources. **Metrics are percentages.**

## Setting → calibration mapping

| setting | calibration data | evaluation data |
|---|---|---|
| Real (oracle) | Real test anomalies | Real test set |
| Perlin | Held-out normals + Perlin anomalies | Real test set |
| AnoStyler | Held-out normals + AnoStyler anomalies | Real test set |
| FLASH | Held-out normals + FLASH anomalies | Real test set |

Real and Perlin use three seeds in the cross-model comparisons. AnoStyler and FLASH are
included where their calibration data are available; Dinomaly has a dedicated comparison.

## image F1 — mean ± std over categories (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH |
|---|---|---|---|---|
| PaDiM | 80.37±6.52 | 69.41±14.18 | 62.37±22.95 | 79.11±6.59 |
| PatchCore | 82.46±7.23 | 67.50±16.07 | 43.02±25.03 | 74.64±12.31 |
| AnomalyDINO | 81.47±6.97 | 62.19±19.47 | 49.85±19.68 | 77.05±10.73 |
| Dinomaly | 81.81±5.99 | 60.42±24.92 | 55.37±18.86 | 77.51±11.17 |
| SuperADD (ViT-H+) | 83.64±8.47 | 79.12±10.38 | 64.36±22.75 | 78.13±8.37 |

## pixel F1 (SegF1) — mean ± std over categories (%)

| model | Real (oracle) | Perlin | AnoStyler | FLASH |
|---|---|---|---|---|
| PaDiM | 7.63±5.40 | 3.05±3.56 | 5.37±5.06 | 3.89±5.07 |
| PatchCore | 26.09±14.99 | 14.90±18.07 | 16.08±17.54 | 18.14±15.04 |
| AnomalyDINO | 33.55±20.77 | 17.47±17.21 | 13.99±15.71 | 26.11±21.14 |
| Dinomaly | 31.90±19.15 | 15.45±13.71 | 3.35±3.92 | 22.53±14.75 |
| SuperADD (ViT-H+) | 51.53±22.76 | 19.44±19.51 | 36.99±21.92 | 38.38±25.18 |

---

# Per-category F1 breakdown

Each cell is the mean over 3 seeds (real/Perlin additionally averaged over the two
cross-model comparisons). Values are percentages.

## pixel F1 (SegF1) — per model (%)

### PaDiM

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
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

| category | Real | Perlin | AnoStyler | FLASH |
|---|---|---|---|---|
| can | 71.19 | 61.14 | 61.46 | 61.10 |
| fabric | 75.60 | 73.17 | 41.03 | 73.17 |
| fruit_jelly | 85.86 | 80.74 | 81.54 | 85.71 |
| rice | 86.60 | 81.08 | 25.17 | 81.08 |
| sheet_metal | 89.75 | 88.24 | 70.47 | 88.24 |
| vial | 99.52 | 98.61 | 99.84 | 85.71 |
| wallplugs | 76.47 | 75.00 | 52.87 | 75.00 |
| walnuts | 84.09 | 75.00 | 82.51 | 75.00 |
