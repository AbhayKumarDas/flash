### 3.7 Photometric matching and residuals

The pair differs in exposure as well as content, so the reference is matched to the test before
anything is subtracted. Both a whole-frame gain and offset fit and a local windowed fit are
computed, because the local fit is sharper but can erase a large flat defect.

`band_residual` then measures distance outside a tolerance band built from the reference's own
local range, rather than a plain absolute difference, so sub-pixel misregistration scores zero.
