# Module 3: Localisation and synthesis

Takes the defect bank from Module 2 and composites those defects onto new defect-free images,
producing synthetic anomalies with pixel-accurate masks. No image generation model is used.

## Requirements

Kaggle notebook, accelerator set to GPU T4 x2. No language or vision model is needed.

Attach two mounts:

1. Your Module 2 defect bank, as a Dataset.
2. MVTec AD 2, as a Dataset, for the host images.

## Steps

1. Open section 1 and set the mount paths, `N_SYNTH` and `SEEDS`. Corpus size is
   `categories x SEEDS x N_SYNTH`.
2. Run all cells in order. Start with `N_SYNTH = 1` to check the setup before committing time.
3. Check the `replay@1024` column printed in section 2. It is the size each defect will appear at.
   If those numbers look wrong, the bank manifest is wrong.
4. Check the sanity figure in section 4, which shows the detected object region and the placement
   blob for one host per category.
5. Read the summary in section 7 and look at the images in section 9.

## Output

```
Final Synthetic Images/
  <category>/<seed>/<n>_<entry>.png          the synthetic anomaly
  <category>/<seed>/<n>_<entry>_mask.png     its ground-truth mask
  manifest.csv                               one row per image
  run_config.json                            the settings used
```
