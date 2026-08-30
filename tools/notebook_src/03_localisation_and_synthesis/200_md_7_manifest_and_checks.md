## 7. Manifest and checks

Writes `manifest.csv` and prints per-category statistics.

`d_in` is the mean difference between the composite and the host inside the ground truth, on a 0 to
255 scale. A sample below `DISSOLVED_T` carries no findable defect: the label marks a region the
blend left unchanged.

Those samples are counted, not removed. If one placement mode or one category dominates the count,
that is worth investigating before changing any threshold.
