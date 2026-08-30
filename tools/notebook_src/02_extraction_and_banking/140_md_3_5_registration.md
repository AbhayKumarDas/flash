### 3.5 Registration

Three routes are tried in order of cost: a coarse scale and translation search on edge maps, ECC
refinement on band-passed images, and an ORB with RANSAC fallback that can also handle rotation.

`align_score` decides between them. It is edge correlation over the overlap, discounted by the
overlap itself, so a warp that keeps only a corner of the frame cannot win by correlating well on
very little.

The printed `align` value is not a measure of warp quality. It tracks how much high-contrast
detail a frame carries, so a low value on a plain surface does not mean registration failed.
