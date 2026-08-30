## 5. Placement and blending

Two placement modes are used, mixed according to `PLACEMENT_MIX`.

Original placement puts the defect at the centre of the placement blob, at its true size relative
to the frame, with a random rotation.

Adaptive placement additionally scales the defect to a fraction of the blob area, aligns its
principal axis to the blob's, anchors it by centre of mass, and shrinks it until it sits inside the
blob. The defect is only ever scaled by the same factor in both directions, so its shape is
preserved.

Harmonisation then transfers CIELAB statistics from the crop's own background to the host's local
background, leaving the defect pixels untouched.

Blending uses Poisson cloning with a substrate collar: the mask is widened before the solve so the
boundary sits on clean product surface rather than through the defect. Defects below
`R_EQ_POISSON` cannot spare that margin and composite with alpha instead.
