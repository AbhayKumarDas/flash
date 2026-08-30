## 8. Contrast gate

A second, independent check. It measures the CIELAB distance from each masked pixel to the crop's
own background, and requires a minimum fraction of them to exceed `DEFECT_T`.

This catches a mask that landed on plain background. Such an entry would produce an image whose
ground truth marks a region where nothing changed, so a detector would be scored on finding
something that is not there.

Both gate results are recorded separately, so either can be disabled without re-running the other.
The verdict overlay below colours every crop by the outcome: green is banked, red is discarded,
with the gate that rejected it.
