## 1. Config

Set the mount paths and the output directory here.

### DiffMask flags

The same four flags run on every category. They were fitted on 12 pairs and are not the library
defaults, which fail on this data.

| flag | value | effect |
|---|---|---|
| `--w-sharp` | `0` | disables the sharpness cue, which fires on re-render differences rather than defects |
| `--shrink` | `0.4` | controls post-detection erosion; higher values fragment thin regions |
| `--norm-frac` | `0.10` | width of the local noise window |
| `--auto-low` | `2.5` | enables the low-frequency cue, needed for low-contrast foreign objects |

`--sat-guard` stays at `0.99`. It drops pixels that are clipped in every channel in either frame,
which removes false positives where one frame blows out to white and the other does not.

### Known limitation

DiffMask's score is sized for compact regions. A defect thinner than the score blur, or one that
differs from its background in only one colour channel, may not be recovered at all. Such a pair
produces an empty mask, is reported as `EMPTY` in section 4, and contributes nothing to the bank.

If you need thin defect recovery, raise `--combine-p` and lower `--smooth`, then re-check every
category: the change affects all of them.
