## 3. DiffMask

The next twelve cells define the mask recovery code. Run them in order; they define functions and
produce no output.

Recovery works in six steps: coarse scale and translation search on edge maps, ECC refinement with
a rotation-capable fallback, optional dense flow for residual drift, photometric matching with a
tolerance band, five difference cues fused into one score, and a local z-score followed by
hysteresis and component selection.

| Algorithm 1 step | section | function |
|---|---|---|
| 1. scale and translation search | 3.5 | `coarse_similarity` |
| 2. ECC refinement, feature fallback | 3.5 | `ecc_refine`, `feature_similarity`, `register` |
| 3. dense flow | 3.6 | `flow_refine` |
| 4. photometric match, tolerance band | 3.7 | `photometric_fit`, `band_residual` |
| 5. five residuals fused | 3.7 to 3.9 | `low_freq_*`, `sharpness_z`, `difference_score` |
| 6. local statistics | 3.8 | `local_zscore` |
| 7. hysteresis | 3.10 | `hysteresis` |
| 8. component ranking | 3.10 | `select_components` |
| 9. direction-matched completion | 3.10 | `complete_object` |
