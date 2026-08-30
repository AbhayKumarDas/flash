# Module 3: Localisation and synthesis

Paper Stages 4 and 5. Notebook
[`notebooks/03_localisation_and_synthesis.ipynb`](../../notebooks/03_localisation_and_synthesis.ipynb).
Library: [`src/flash/obs.py`](../../src/flash/obs.py), [`src/flash/mrsp.py`](../../src/flash/mrsp.py),
[`src/flash/placement.py`](../../src/flash/placement.py),
[`src/flash/compositing.py`](../../src/flash/compositing.py).

## What this stage is for

The bank is built. This module spends it: a stored defect is localized on a fresh normal host and
composited into it, producing a synthetic anomaly and its pixel-accurate mask.

Nothing here loads a generative model. That is the whole point of the architecture. The defect
crop determines *what* is synthesized, the placement mask determines *where*, and one banked
defect can be replayed across many hosts, locations, orientations and sizes without touching a
generator again.

The bank is **loaded, not built**. Re-extracting crops here would bypass the Stage 3 gates and
silently put rejected crops back into the corpus.

## Stage 4: Object-aware localization

### OBS, finding the product

A noise field has no idea what is in the picture. Left alone it will put a scratch on the conveyor
belt, the tray, or the empty background. Object Boundary Suppression produces the region `Omega`
that prevents that. This is Algorithm 2 in the paper, and Eq. 2 is what it computes:

```
Omega = R( M_col^Otsu  OR  M_tex^Otsu )
```

Two independent cues, kept if **either** fires:

1. **Colour.** The background colour is the median of the pixels around the image border, since
   the frame edge is nearly always background. Distance from that colour in CIELAB is
   Otsu-thresholded.
2. **Local texture.** The standard deviation inside a window of `max(15, min(H, W) / 64)` pixels,
   also Otsu-thresholded. Products usually carry more fine detail than the surface they sit on.

Either cue alone fails somewhere. A product the same colour as its background is invisible to the
first, a smooth glass vial is invisible to the second, and OR-ing them recovers both. `R` is
morphological opening, closing and hole filling, then connected-component filtering that drops
anything under 0.0015 of the frame, keeping the largest component if that filter would empty the
mask.

**The degeneracy guard.** If coverage comes out below 0.03 or above 0.99, OBS has failed on that
image, and the honest fallback is to treat the whole frame as valid rather than confine every
defect to a sliver of a bad mask.

An **empty-frame gate** runs alongside it, using `MIN_OBJ_COV_REL = 0.40` and
`MIN_OBJ_COV_ABS = 0.02`. It is relative rather than absolute on purpose: across eight categories
normal OBS coverage runs from a few scattered wallplugs to a fabric sheet filling the frame, so
one fixed fraction would either pass every empty walnuts tray or reject every good vial.

### MRSP, choosing where inside it

Multi-Resolution Spectral Pyramid noise answers the remaining question: which patch of the product
should the defect land on? This is Algorithm 3, and Eq. 3 is the mask it produces.

The field is built from `MRSP_OCTAVES = 6` layers, coarse to fine. Each layer is synthesized in
the frequency domain with amplitudes falling off as `1/f^alpha` at `MRSP_ALPHA = 1.5` and a
uniformly random phase, transformed back with an inverse FFT, bilinearly resized to the host
resolution, and normalized. The layers are summed with persistence weighting
(`MRSP_PERSISTENCE = 0.8`, so layer `l` gets weight `0.8^l`) and the total is energy-normalized.
Coarse layers set the blob's overall shape, fine layers roughen its edges so the result looks
organic rather than circular. `MRSP_NOISE_SCALE = 14` warps the frequency axis and so sets the
characteristic blob size.

**The part that is easy to get wrong is how the field becomes a mask.** A threshold computed over
the whole image would happily pick bright spots on the background. Intersecting with the object
afterwards would chop the blob into fragments. So the threshold is taken from the field values
**inside the object only**:

```
M_f = L[ (F_MRSP > Q_{1-c}(F_MRSP[Omega > 0]))  AND  Omega ]
```

where `Q_{1-c}` is the `(1-c)`-quantile and `L` keeps the largest connected component. That single
choice is what makes `TARGET_COVERAGE = 0.018` mean "1.8 % of the product" rather than "1.8 % of
the picture".

Changing the seed moves the defect. Changing the coverage resizes it. Neither invokes a generator,
which is where the intra-image diversity comes from.

## Stage 5: adaptive synthesis

### The two placement modes

This is Algorithm 4. A corpus is built from a mix of both modes, `ADAPTIVE_FRACTION = 0.5`.

**`original`** reads a single number off the blob, its centroid, and drops the defect there at its
true relative size with a random rotation. Nothing is fitted, so nothing can fail, and there is no
containment figure to report.

**`adaptive`** reads four things off the blob instead of one:

| what | how |
|:--|:--|
| size | the defect is scaled so its area is `gamma` times the blob's area, `gamma` drawn per sample from `GAMMA_RANGE = (0.20, 0.60)` |
| orientation | the defect's principal axis is turned to line up with the blob's, from second central moments |
| anchor | the defect's centre of mass is put on the blob's centre of mass |
| containment | if the defect still pokes outside the blob, shrink by 0.85 and retry, up to 3 rounds |

Both modes scale the defect **isotropically only**, the same factor in x and y. Nothing here
stretches or squashes a real defect, so the morphology recovered in Module 2 survives placement
exactly.

Two details in `adaptive` look like paranoia and are not:

* **A centre of mass can fall outside its own shape.** One real `can` defect is 322 x 39 px and
  banana-shaped, so its centroid sits in empty space, and a ragged blob can do the same.
  Anchoring on a point that is not inside the region would put the defect off target, so both
  fall back to the deepest interior point from a distance transform, which is guaranteed to be
  inside.
* **Aligning two axes is ambiguous by 180 degrees.** All four candidate angles
  (`d`, `-d`, `d+180`, `-d+180`) are tried and the one with the best containment wins.

`CONTAINMENT_T = 0.90` is the target: shrink until 90 % of the defect lies inside the blob.
Measured across eight categories, `original` to `adaptive` moved EMD against real ground truth
from 0.905 to 0.705, kept from 0.848 to 0.970, and dissolved samples from 25.7 % to 8.3 %.

### Harmonisation

A crop cut from one image and dropped into another brings its original lighting with it.
`harmonise` transfers CIELAB statistics from the crop's **own** background to the host's **local**
background, background to background, never touching the defect pixels. The defect keeps its
appearance while its surroundings match the new image. The gain is clipped to `(0.75, 1.35)` so a
low-variance crop cannot be stretched into noise.

### Hybrid blending

Two operators and a rule for choosing between them. This is Eq. 4.

**Alpha blending** fades the defect in over a couple of pixels at its edge. Cheap, and it always
preserves the defect exactly, but on a smooth surface the join can still be visible.

**Poisson blending** (`cv2.seamlessClone`, `NORMAL_CLONE`) matches gradients across the boundary
so the join disappears. The catch is that it reconstructs the interior from the boundary values,
which means the boundary had better not be sitting on the defect. If it is, the solver rebuilds
the defect out of surrounding substrate and erases it.

**The substrate collar** is what prevents that. Before solving, the defect mask is dilated
outwards by `COLLAR_FRAC = 0.25` of its radius so the boundary the solver reads lands on clean
product surface rather than cutting through the defect. This one detail is the difference between
Poisson preserving a defect and destroying it.

The collar has a floor, `COLLAR_MIN_PX = 6`, so tiny defects still get a workable boundary. But a
floor applied blindly does the opposite of what it is for: a 6 px collar around a defect of radius
9.7 px eats 62 % of it. `COLLAR_KEEP_R = 0.5` caps the collar at half the defect's radius, so the
floor can never dissolve the defect it exists to protect.

**Routing.** Below `R_EQ_POISSON = 16.0` equivalent radius there is simply not enough defect to
give away, and those samples go to alpha instead. A dissolved sample is a normal image carrying a
defect label, which is worse for training than an imperfect seam.

**Why the defect thresholds are measured, not fixed.** `defect_thresholds` does not use a constant
cutoff to decide which pixels count as defect. On a granular product like rice the crop's grains
never line up with the host's, so the whole crop differs from the host and a fixed cutoff would
call the entire background a defect. The noise floor is measured instead, on a ring outside the
mask where by definition there is no defect, and the cutoff is placed above it.

### The retention check

Eq. 4 ends with "if Poisson blending fails, alpha blending is used as a fallback".
`compositing.hybrid` is what decides that a failure happened.

`cv2.seamlessClone` solves in the gradient domain. It keeps the source's gradients and discards
its absolute colour, so a defect whose mean colour differs strongly from the substrate can be
pulled back to substrate colour while the solver reports success. A composite that returns the
host is a failed blend whatever the solver says.

So retention is measured directly, in CIELAB over the defect pixels:

```
R = || out - host || / || patch - host ||
```

`R = 1` means the defect survived compositing intact, `R = 0` means the composite returned the
host. Below `RETENTION_MIN = 0.35` the Poisson result is treated as a failure.

The rescue is bounded by `R_EQ_ALPHA_MAX = 48.0`, three times `R_EQ_POISSON`. Alpha's feather is a
fixed 1.5 px, so the transition is equally sharp on a 30 px defect and a 300 px one, which means
proportionally the seam gets *worse* as the defect grows, and a hard edge is a shortcut feature a
detector will learn instead of the defect. Above that radius the Poisson result is kept even when
retention is poor: a faded defect is a weak sample, a seamed one is a corrupt label.

`DISSOLVED_T = 8.0` is a **reporting** threshold, not a filter. A composite whose ground-truth
region differs from the host by less than 8 of 255 on average carries no findable defect, and
those are counted and reported. Nothing is dropped. Poisson erasing a small defect is a property
of the operator, and removing those samples would quietly flatter it.

## Parameters

| name | value | controls |
|:--|:--|:--|
| `WORK_SIZE` | 1024 | resolution everything is composited at |
| `MRSP_NOISE_SCALE` | 14 | characteristic blob size |
| `MRSP_OCTAVES` | 6 | number of pyramid levels |
| `MRSP_PERSISTENCE` | 0.8 | weight decay across levels |
| `MRSP_ALPHA` | 1.5 | `1/f^alpha` spectral decay |
| `TARGET_COVERAGE` | 0.018 | blob size as a fraction of the detected object |
| `MIN_BLOB_PX` | 25 | floor on blob area |
| `MIN_OBJ_COV_REL` / `_ABS` | 0.40 / 0.02 | empty-frame gate |
| `ADAPTIVE_FRACTION` | 0.5 | share of samples using adaptive placement |
| `GAMMA_RANGE` | (0.20, 0.60) | defect's share of the blob |
| `CONTAINMENT_T` | 0.90 | containment target for adaptive placement |
| `BETA` | 1.0 | defect opacity. `beta < 1` was the cause of washed-out composites |
| `FEATHER_PX` | 1.5 | alpha seam feather, capped at 0.25 of the blob side |
| `COLLAR_FRAC` | 0.25 | substrate margin given to the Poisson solve |
| `COLLAR_MIN_PX` | 6 | collar floor |
| `COLLAR_KEEP_R` | 0.5 | collar ceiling, as a share of the defect radius |
| `R_EQ_POISSON` | 16.0 | below this radius, route to alpha |
| `RETENTION_MIN` | 0.35 | below this retention, the Poisson result is a failure |
| `R_EQ_ALPHA_MAX` | 48.0 | above this radius, keep Poisson regardless |
| `DISSOLVED_T` | 8.0 | reporting threshold for a dissolved composite |
| `N_SYNTH`, `SEEDS` | 24, (0, 1, 2) | corpus size |
| `N_HOSTS` | 32 | ceiling on the host pool per category |

Pixel-denominated values are tied to `WORK_SIZE`. If you change the resolution, rescale them. At
768 px a rice defect replays at about 22 px, which is below the scale at which a blending seam can
be judged at all.

## Note on the notebook and the library

`src/flash/compositing.py` carries the retention router described above, and
`src/flash/placement.py` uses the paper's `original` and `adaptive` names. The notebook was
written against the earlier revision: it routes on size alone and still refers to the modes as
`B` and `C`. The two agree on every sample where Poisson retains the defect, and differ only on
the rescue path. The library is the current behaviour.

## Input and output

**Input.** Module 2's defect bank and MVTec AD 2 normal images for the hosts, both as Kaggle
mounts. Donors are removed from the host pool, so a defect is never replayed onto the frame it
came from.

The bank's `manifest.csv` matters. If a bank ships without one, `defect_frac` has to be
reconstructed from `SOURCE_LONG_PX = 2448`, the AD 2 native long side, and the notebook says so
loudly, because a wrong value here places every defect at the wrong scale.

**Output.**

```
Final Synthetic Images/
  <category>/<seed>/<n>_<entry>.png          the synthetic anomaly
  <category>/<seed>/<n>_<entry>_mask.png     its ground-truth mask
  manifest.csv                               one row per image
  run_config.json                            the settings used
```

## Running it

1. Set the mount paths, `N_SYNTH` and `SEEDS` in section 1. Corpus size is
   `categories x SEEDS x N_SYNTH`.
2. Run all cells in order. Start with `N_SYNTH = 1` to check the setup before committing time.
3. Check the `replay@1024` column printed in section 2. It is the size each defect will appear at,
   and if those numbers look wrong the bank manifest is wrong.
4. Check the sanity figure in section 4, which shows the detected object region and the placement
   blob for one host per category.
5. Read the summary in section 7 and look at the images in section 9.

All three seeds are kept even at proof-of-concept size. The seed is what varies the blob, the bank
entry and the rotation, so dropping seeds would hide exactly the variation this is meant to show.

## What to check before using the corpus

1. The placement mode split is close to `ADAPTIVE_FRACTION`. A large skew means adaptive placement
   is failing its containment search and falling back.
2. Few samples routed to alpha. A large fraction means the bank's defects are mostly too small for
   Poisson blending, so the corpus never exercises it.
3. The dissolved count is low.
4. The seam check in section 9 passes by eye. No metric here catches a shortcut feature.
5. The reported generation time uses your measured Module 1 cost, not the placeholder.

## Cost

586 ms per synthetic image, and 112.8 s for a full category including the one-time Module 1 cost.
The comparison is 15.0 s per image and 1,348.1 s per category for per-sample generative synthesis.
The gap widens as the corpus grows, because the generative part of FLASH is fixed per category
while everything after it is procedural.
