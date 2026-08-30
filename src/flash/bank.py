"""The defect bank -- turning a recovered mask into a reusable, storable crop.

DiffMask gives us a mask over the whole generated frame. This module turns each connected region
of that mask into a bank ENTRY: a square crop of the defect together with the metadata needed to
replay it onto a different image later.

THE CROP KEEPS ITS BACKGROUND. This surprises people, because the obvious thing to store is a
cut-out of just the defect. But the harmoniser in `flash.placement` matches the crop's own
background against the host's background, and a cut-out has no background to measure. So the
crop is deliberately `CTX_EXPAND` times larger than the defect's bounding box, and typically
76-99% of it is ordinary product surface. The defect mask is stored ALONGSIDE the crop, never as
an alpha channel, for exactly this reason.

WHAT `defect_frac` IS FOR, and why it cannot be thrown away. It is the defect's long side as a
fraction of the ORIGINAL frame. A 322 px defect cut from a 2448 px photograph and a 322 px
defect cut from a 1024 px photograph are completely different sizes relative to the product, and
a crop PNG cannot tell you which it was. Without this number every defect is replayed at the
wrong scale.

THE CONTRAST GATE. A recovered mask can land on plain background -- DiffMask fires on
registration seams and texture shifts as well as on real defects. Such an entry is worse than
useless: replayed, it produces an image whose ground truth marks a region identical to the host,
so a detector is scored on finding something that is not there and every metric for that
category sinks. `cfrac` measures the fraction of masked pixels that actually stand off the
crop's own background by `DEFECT_T`, and entries below the floor are rejected before banking.

Provenance: these operators are a verbatim copy of the reference implementation the
paper's results were produced with. Comments and docstrings were added here; no function
body was edited. `tests/test_end_to_end.py` runs them end to end.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# Imported under the names the reference implementation uses, so the function bodies below
# stay byte-identical to the code the published results were produced with.
from flash.config import (
    ALPHA_SOFTEN,
    CTX_EXPAND,
    DEFECT_KIND,
    DEFECT_T,
    MIN_REGION_PX,
)

__all__ = ["regions_of", "extract_entry"]


# ----------------------------------------------------------------------------- regions
def regions_of(mask_u8, min_px=MIN_REGION_PX):
    """Connected components of a recovered mask, largest first, speckle removed."""
    lbl, n = ndimage.label(mask_u8 > 127)
    out = []
    for k in range(1, n + 1):
        b = lbl == k
        a = int(b.sum())
        if a < min_px:
            continue
        ys, xs = np.where(b)
        out.append(dict(bin=b, area=a, x=int(xs.min()), y=int(ys.min()),
                        w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1)))
    return sorted(out, key=lambda r: -r["area"])


# ----------------------------------------------------------------------------- entries
def extract_entry(donor_id, region, raw, ctx=CTX_EXPAND, soften=ALPHA_SOFTEN):
    """Substrate-retained square crop + soft defect alpha + scale metadata.

    `raw` maps a donor key to its frames, e.g.
    ``{"walnuts-015": {"anomaly": <HxWx3 uint8>, "cat": "walnuts", ...}}``. Keys are
    "<category>-<id>" because ids collide across categories -- rice/020 and sheet_metal/020 are
    unrelated images -- so a bare id is never a safe key.

    In the reference notebook this dict was a module-level global named RAW. A library function
    cannot reach a caller's global, so it is passed in explicitly; the body is otherwise
    unchanged.
    """
    d = raw[donor_id]
    anom = d["anomaly"]; m = region["bin"]
    H, W = anom.shape[:2]
    dh, dw = region["h"], region["w"]
    side = int(np.clip(round(max(dh, dw) * ctx), 16, min(H, W)))
    cy, cx = region["y"] + dh // 2, region["x"] + dw // 2
    ty = int(np.clip(cy - side // 2, 0, H - side))
    tx = int(np.clip(cx - side // 2, 0, W - side))

    rgb   = anom[ty:ty + side, tx:tx + side].copy()
    alpha = m[ty:ty + side, tx:tx + side].astype(np.float32).copy()
    if soften > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), soften)
    alpha = np.clip(alpha, 0, 1)

    # Internal contrast: how far the masked pixels sit from this crop's own substrate, in the
    # same Lab units as DEFECT_T. An entry whose mask covers plain substrate scores ~0 here, and
    # it is worse than useless -- replayed, it produces an image whose ground truth labels a
    # region where nothing changed, so every metric is scored against a defect that is not there.
    # That is what a mis-recovered diffmask looks like, and it has to be caught before the bank.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dsel, ssel = alpha > 0.5, alpha < 0.1
    if dsel.sum() >= 25 and ssel.sum() >= 25:
        ref  = lab[ssel].mean(0)
        dist = np.linalg.norm(lab[dsel] - ref, axis=1)
        contrast = float(np.median(dist))
        cfrac    = float((dist > DEFECT_T).mean())
    else:
        contrast, cfrac = 0.0, 0.0

    return dict(
        key=f"{donor_id}_{region['x']}_{region['y']}", donor=donor_id, cat=d["cat"],
        rgb=rgb, alpha=alpha, side=side,
        defect_long_px=int(max(dh, dw)),
        defect_frac=float(max(dh, dw) / max(H, W)),
        area_px=region["area"], area_frac=float(region["area"] / m.size),
        contrast=contrast, cfrac=cfrac,
        kind=DEFECT_KIND.get(d["cat"], "unknown"),
    )
