## 9. Inspect the output

Shows a sample of the generated images with their masks outlined, read back from disk.

Three things to check.

The defect is on the product, not the background. If it is not, object detection failed for that
host.

There is no visible rectangular seam or halo around the defect. A visible seam is a shortcut
feature: a detector will learn it instead of the defect.

The defect is still present. Compare against the crop shown in section 2 if you are unsure.
