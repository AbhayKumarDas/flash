## 3. Object detection and placement field

Two operators run per host image.

Object Boundary Suppression estimates the region the product occupies, using colour distance from
the border-median background combined with a local texture measure. Either cue alone fails on some
categories, so a pixel is kept when either fires. If the result covers almost none or almost all of
the frame, it falls back to the whole frame.

The placement field is multi-resolution spectral noise, thresholded to the target coverage. The
threshold is computed from the field values inside the detected object only, which is what makes
the coverage relative to the product rather than the frame.

An empty-frame gate then removes hosts whose detected object is far below the category median.
Those are frames where the product is absent, and a defect placed on one is a nonsense sample.
