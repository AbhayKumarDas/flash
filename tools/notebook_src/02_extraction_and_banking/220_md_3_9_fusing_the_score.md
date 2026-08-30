### 3.9 Fusing the score

Combines luminance, chromaticity, gradient, low-frequency appearance and sharpness into a single
difference map. `--combine-p` controls the combination and defaults to a plain average.

Two consequences worth knowing. A defect that differs in only one channel is outvoted by the terms
blind to it. And `--smooth` blurs the score, so a defect narrower than the blur radius loses most
of its peak while broader background noise survives.

Both are levers for thin defect recovery, and both change behaviour on every category.
