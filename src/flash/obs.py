"""Object Boundary Suppression (OBS) -- finding the product in a host image.

A noise field has no idea what is in the picture. Left to itself it will put a scratch on the
conveyor belt, the tray, or the empty background. OBS produces the mask that prevents that: the
part of the frame actually occupied by the product.

It uses two independent cues and keeps a pixel if EITHER fires:

1. Colour. The background colour is estimated from the median of the pixels around the image
   border, since the frame edge is nearly always background. Anything far from that colour in
   CIELAB is probably the product.
2. Local texture. The standard deviation inside a small window. Products usually carry more fine
   detail than the surface they sit on.

Either cue alone fails somewhere. A product the same colour as its background is invisible to
the first; a smooth glass vial is invisible to the second. OR-ing them recovers both cases.

The last step is a degeneracy guard. If the result covers almost none of the frame, or almost
all of it, OBS has failed for that image. The honest fallback is to treat the whole frame as
valid rather than confine every defect to a sliver of a bad mask.

Provenance: these operators are a verbatim copy of the reference implementation the
paper's results were produced with. Comments and docstrings were added here; no function
body was edited. `tests/test_end_to_end.py` runs them end to end.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy import ndimage

__all__ = ["region_for"]


# `use_cues=False` disables the texture channel and leaves colour alone -- useful when
# debugging which of the two is responsible for a bad mask.
def _fg_robust(image, use_cues=True):
    img = (np.clip(image.permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]], 0)
    bg = np.median(border, 0)
    dist = np.sqrt(((lab - bg) ** 2).sum(2))
    du8 = (255 * dist / (dist.max() + 1e-8)).astype(np.uint8)
    _, cm = cv2.threshold(du8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cm > 0
    if use_cues:
        gray = img.mean(2).astype(np.float32) / 255.0
        k = max(15, round(min(img.shape[:2]) / 64))
        mean = ndimage.uniform_filter(gray, k)
        sq = ndimage.uniform_filter(gray * gray, k)
        std = np.sqrt(np.clip(sq - mean * mean, 0, None))
        su8 = (255 * std / (std.max() + 1e-8)).astype(np.uint8)
        _, tm = cv2.threshold(su8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = m | (tm > 0)
    m = ndimage.binary_opening(m, iterations=1)
    m = ndimage.binary_closing(m, iterations=2)
    m = ndimage.binary_fill_holes(m)
    lbl, n = ndimage.label(m)
    if n:
        sz = ndimage.sum(np.ones_like(lbl, dtype=np.float32), lbl, range(1, n + 1))
        keep = np.where(sz >= 0.0015 * m.size)[0] + 1
        if len(keep) == 0:
            keep = [int(sz.argmax()) + 1]
        m = np.isin(lbl, keep)
    return m.astype(np.float32)


def region_for(image):
    region = _fg_robust(image, use_cues=True); cov = region.mean()
    if cov < 0.03 or cov > 0.99:
        region = np.ones_like(region)
    return torch.from_numpy(region).float()
