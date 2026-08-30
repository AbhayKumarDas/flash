# Module 2: Defect extraction, validation and banking

Takes the normal and anomaly image pairs from Module 1, recovers the defect mask from each pair,
validates the result, and writes a defect bank for Module 3.

The mask is recovered from the pair rather than prescribed in advance, so the label always matches
the image.

## Requirements

Kaggle notebook, accelerator set to GPU T4 x2, internet off.

Attach two mounts:

1. `Qwen/Qwen2.5-VL-7B-Instruct`, as a Model.
2. Your Module 1 output, as a Dataset, laid out as `<category>/<id>_regular.png` and
   `<category>/<id>_anomaly.png`.

## Steps

1. Open section 1 and set the mount paths and the output directory.
2. Run all cells in order. Expect a few seconds per pair for mask recovery and a few seconds per
   crop for validation.
3. Check the overlays in section 5. Every candidate region is drawn on the frame it came from.
4. Check the verdict overlay in section 8, which colours each crop by whether it was banked.
5. Read the funnel in section 11. A category reaching zero cannot be used by Module 3.

## Output

```
defect_bank/
  <category>/<key>.png            the defect crop, with its surrounding substrate
  <category>/<key>_alpha.png      the defect mask for that crop
  manifest.csv                    geometry, gate results, model verdict, timing

Defect Masks and Anomalies/
  <category>/<id>_anomaly.png     the generated anomaly frame
  <category>/<id>_mask.png        the mask recovered from it
```

Both files in the bank are required by Module 3. The crop keeps its background because the
harmoniser measures it, and the mask becomes the ground truth.
