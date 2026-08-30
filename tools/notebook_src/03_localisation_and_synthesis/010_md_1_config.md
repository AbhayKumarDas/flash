## 1. Config

Set the mount paths, corpus size and output directory here. The values below are the published
configuration and are applied unchanged to every category.

| setting | value | controls |
|---|---|---|
| `WORK_SIZE` | 1024 | resolution everything is composited at |
| `AD2_COVERAGE` | 0.018 | placement blob size, as a fraction of the detected object |
| `COLLAR_FRAC` | 0.25 | width of the substrate margin given to the Poisson solve |
| `PLACEMENT_MIX` | 0.5 | fraction of samples using adaptive rather than original placement |
| `R_EQ_POISSON` | 16.0 | defects smaller than this radius composite with alpha instead |
| `N_SYNTH`, `SEEDS` | | corpus size |

Pixel-denominated values are tied to `WORK_SIZE`. If you change the resolution, rescale them.
