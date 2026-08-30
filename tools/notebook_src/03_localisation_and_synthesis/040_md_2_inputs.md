## 2. Inputs

Locates the two mounts and loads the bank.

Expected bank layout:

```
<bank>/manifest.csv
<bank>/<category>/<key>.png
<bank>/<category>/<key>_alpha.png
```

Both image files are required. The crop keeps its own background, which the harmoniser measures,
and the mask becomes the ground truth.

`manifest.csv` supplies `defect_frac`, the defect's size relative to its original frame. It cannot
be recovered from the PNG. If the manifest is missing, the notebook falls back to an assumed source
resolution and says so; check the `replay@1024` column before trusting the run.

Host images come from `<mvtec>/<category>/train/good`. Donor images are removed from the host pool,
and the number dropped is printed. If that count is zero, donors and hosts are not being matched
and the same frame may appear on both sides.
