# Module 2: Defect extraction, validation and banking

Paper Stages 2 and 3. Notebook
[`notebooks/02_extraction_and_banking.ipynb`](../../notebooks/02_extraction_and_banking.ipynb).
Library: [`src/flash/diffmask.py`](../../src/flash/diffmask.py),
[`src/flash/bank.py`](../../src/flash/bank.py).

## What this stage is for

Module 1 produced pairs: a normal image `I_n` and a generated anomaly image `I_a`. Nobody told the
generator where to put the defect, so nobody knows where it is. Stage 2 finds out by comparing the
two frames, and Stage 3 decides whether what it found is worth keeping.

This is the step that makes FLASH's labels trustworthy. The mask is measured from the image rather
than prescribed before it, so it cannot disagree with the image it labels.

## Stage 2: DiffMask

### The framing

Almost everything changes between `I_n` and `I_a`. The frames are not aligned, the exposure
drifts, and a re-render is noisy on texture and silent on flat surfaces. So the difference is
treated as a **signal recovery problem**: alignment error, illumination and reconstruction noise
are nuisance, the introduced defect is signal.

The two inputs do not have to be aligned, the same size, or the same exposure. Resizing them onto
a common canvas does not help and misreads the problem, because the subject sits at a different
scale and offset inside each frame, so matching the canvases still leaves the content tens of
pixels apart.

### The pipeline

This is Algorithm 1 in the paper. The notebook reproduces `diffmask.py` cell by cell, in order,
with the section numbers below.

| step | what happens | why | code |
|:--:|:--|:--|:--|
| 1 | Coarse scale and translation search on edge maps | Edges rather than pixels, because the frames differ in exposure and edges do not care | `coarse_similarity` |
| 2 | ECC refinement to an affine or homography warp, with an ORB and RANSAC fallback | ECC handles smooth drift; the feature route also handles rotation. The fallback must beat the incumbent by a margin, so a pair that was already registered correctly cannot be talked out of it | `ecc_refine`, `feature_similarity`, `register` |
| 3 | Optional smoothed dense optical flow | Absorbs slow non-rigid drift. Computed small and blurred hard on purpose, so it soaks up drift without deforming itself around the defect | `flow_refine` |
| 4 | Photometric matching, then a tolerance band | A residual only counts once it leaves a band built from the reference's own local range, so sub-pixel misregistration costs nothing | `photometric_fit`, `band_residual` |
| 5 | Five cues fused into one difference score | Luminance, chromaticity, gradient structure, low-frequency appearance and sharpness. Different defects announce themselves in different channels | `difference_score`, `low_freq_residual`, `sharpness_z` |
| 6 | Local z-score | The important one, see below | `local_zscore` |
| 7 | Hysteresis | Strong seeds grow through connected weak evidence, so faint extensions of a real defect survive while isolated weak noise does not | `hysteresis` |
| 8 | Components ranked by accumulated evidence, then area filtered | Evidence, not size, decides which component is the defect | `select_components` |
| 9 | Direction-matched completion | Recovers parts of a defect that changed the same *way* as the detected part but less strongly | `complete_object` |

### The local z-score

Step 6 is where DiffMask differs most from a plain image difference. Rather than thresholding the
fused score `S` globally, each pixel is judged against the noise in its own neighbourhood
`N(p)`. This is Eq. 1 in the paper:

```
Z(p) = (S(p) - mu_N(p)) / max(sigma_N(p), eps)
```

A fixed global cutoff is either deaf on texture or hallucinating on flat surfaces, because
re-render noise is not uniform across the frame. Judging a pixel against its own surroundings
removes that asymmetry. `eps` keeps the normalisation stable where the local variance is near
zero.

### The flags

The same four flags run on every category. They were fitted on 12 pairs and are not the library
defaults, which fail on this data. They live in `config.DIFFMASK_FLAGS`.

| flag | value | effect |
|:--|:--|:--|
| `--w-sharp` | `0` | Disables the sharpness cue. It fires on re-render differences rather than defects, and on one frame it placed 29.9 % of the image inside the mask. |
| `--shrink` | `0.4` | Post-detection erosion. At `1.2` an 1835 px seed shattered into 13 fragments totalling 166 px. |
| `--norm-frac` | `0.10` | Width of the local noise window. `0.04` is too local on flat substrate. |
| `--auto-low` | `2.5` | Enables the low-frequency cue. Without it, low-contrast foreign objects do not detect at all. |

`--sat-guard` stays at its default `0.99`. It drops pixels clipped in every channel in either
frame, which removes false positives where one frame blows out to white and the other does not.

There is deliberately **no per-category override table**. The paper claims a fixed,
category-agnostic configuration, and a rescue config for one category would contradict it. If you
change a flag, re-run every pair, not just the category you were fixing.

### Known limitation

The score is sized for compact regions. A defect thinner than the score blur, or one that differs
from its background in a single colour channel, may not be recovered at all. That pair produces an
empty mask, is reported as `EMPTY`, and contributes nothing to the bank. For thin defect recovery,
raise `--combine-p` and lower `--smooth`, then re-check every category.

## Stage 3: banking and validation

### What a bank entry is

`bank.regions_of` splits the recovered mask into connected components and drops anything under
`MIN_REGION_PX = 500`, which is extraction speckle. `bank.extract_entry` turns each surviving
region into an entry.

**The crop keeps its background.** This surprises people, because the obvious thing to store is a
cut-out of the defect alone. But the harmoniser in Module 3 matches the crop's own background
against the host's background, and a cut-out has no background to measure. So the crop side is
`CTX_EXPAND = 1.8` times the defect's bounding box long side, and typically 76 to 99 % of it is
ordinary product surface. The mask is stored **beside** the crop as a separate PNG, never as an
alpha channel, for exactly this reason.

**`defect_frac` cannot be thrown away.** It is the defect's long side as a fraction of the
original frame. A 322 px defect cut from a 2448 px photograph and a 322 px defect cut from a
1024 px photograph are completely different sizes relative to the product, and a crop PNG cannot
tell you which it was. Without this number every defect is replayed at the wrong scale.

Each entry carries:

| field | meaning |
|:--|:--|
| `key`, `donor`, `cat` | identity, `<category>-<id>` because ids collide across categories |
| `rgb`, `alpha`, `side` | the crop, its mask, its side length |
| `defect_long_px`, `defect_frac` | absolute and frame-relative defect size |
| `area_px`, `area_frac` | defect area |
| `contrast`, `cfrac` | contrast gate measurements, below |
| `kind` | coarse appearance prior from `DEFECT_KIND` |

### The contrast gate

A recovered mask can land on plain background. DiffMask fires on registration seams and texture
shifts as well as on real defects, and such an entry is worse than useless: replayed, it produces
an image whose ground truth marks a region identical to the host, so the detector is scored on
finding something that is not there and every metric for that category sinks.

The gate measures, in CIELAB, how far the masked pixels sit from the crop's **own** substrate:

* `contrast` is the median distance.
* `cfrac` is the fraction of masked pixels beyond `DEFECT_T = 6.0`.
* Entries below `MIN_ENTRY_CFRAC = 0.10` are rejected before banking.

### The VLM-2 gate

Each crop is then judged semantically. VLM-2 sees the crop together with its surrounding
substrate, and accepts it only when the region is a plausible category-specific defect. It is
shown three panels side by side rather than one, so it can compare the candidate against clean
substrate from the same image.

Two decisions in this gate are worth stating plainly.

**The visual-token budget was raised, not lowered.** `MAX_PIXELS` went from 1280 to
`2048 x 28 x 28`. At the old budget each of three panels got about 578 px and the defect inside it
about 180 px, which is *smaller* than it was under the older single-panel prompt. Context is
useless if it costs you the acuity to see what you are judging. `VLM_CTX = 2.2` sets how much
wider the judged crop is than the banked one, and defect size in the model's view is roughly
`panel_px / VLM_CTX`, so every extra unit of context shrinks the thing being judged.

**The gate uses the verdict, not the confidence.** `USE_VLM_CONFIDENCE = False`. A 7B VLM's
self-reported confidence is close to uncalibrated: it clusters at 0.8 to 0.9 regardless of
correctness, so thresholding it mostly removes correct answers. The value is still recorded, and
it is still used to break ties between two regions of the *same* donor, which is the one
comparison it is fit for.

**One defect per donor.** `ONE_DEFECT_PER_IMAGE = True`, because Stage 1's hard rule 3 introduced
exactly one. When several components come back from one donor, at most one is the defect and the
rest are extraction artifacts, most often a registration seam, which appears as a long thin strip.

### Generation is deterministic

`VLM_TEMPERATURE = 0.0`. Validation has to reproduce exactly on a re-run, otherwise the bank is
not the same bank twice and no downstream number is comparable.

## Parameters

| name | value | stage |
|:--|:--|:--|
| `WORK_SIZE` | 1024 | synthesis resolution everything is sized against |
| `DIFFMASK_FLAGS` | see above | Stage 2 |
| `MIN_REGION_PX` | 500 | Stage 3, speckle floor |
| `CTX_EXPAND` | 1.8 | Stage 3, crop side relative to defect bbox |
| `ALPHA_SOFTEN` | 1.0 | Stage 3, anti-alias sigma on the recovered mask edge |
| `DEFECT_T` | 6.0 | Stage 3, CIELAB distance for "defect, not substrate" |
| `MIN_ENTRY_CFRAC` | 0.10 | Stage 3, contrast gate floor |
| `VLM_CTX` | 2.2 | Stage 3, context multiplier for the judged crop |
| `USE_VLM_CONFIDENCE` | False | Stage 3, gate on verdict only |
| `ONE_DEFECT_PER_IMAGE` | True | Stage 3, one defect per donor |
| `VLM_MAX_PIXELS` | 2048 x 28 x 28 | Stage 3, visual-token cap for three panels |
| `VLM_TEMPERATURE` | 0.0 | Stage 3, greedy decoding |

## Input and output

**Input.** Module 1's pairs, laid out as `<category>/<id>_regular.png` and
`<category>/<id>_anomaly.png`, plus `Qwen/Qwen2.5-VL-7B-Instruct` as VLM-2.

**Output.**

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

## Running it

1. Set the mount paths and the output directory in section 1.
2. Run all cells in order. Expect a few seconds per pair for mask recovery and a few seconds per
   crop for validation.
3. Check the overlays in section 5, where every candidate region is drawn on the frame it came
   from.
4. Check the verdict overlay in section 8, which colours each crop by whether it was banked.
5. Read the funnel in section 11. A category that reaches zero cannot be used by Module 3.

`WRITE_DEBUG = False` by default. Turning it on writes DiffMask's seven intermediates per pair,
which is a debugging aid rather than an output.

## What to check before moving on

1. Every banked crop shows visible background in the section 5 overlay. A crop that is almost
   entirely defect will place badly in Module 3.
2. No category is empty. An empty category disappears silently from Module 3 rather than
   erroring.
3. `defect_type` is not the same label repeated across a category. Section 7 warns if it is.
4. If you changed the DiffMask flags, re-check every category.
