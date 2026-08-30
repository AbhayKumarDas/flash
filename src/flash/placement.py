"""Putting a retrieved defect onto a host image -- the two placement modes, and harmonisation.

By this point we know WHAT defect to use (a crop from the bank) and WHERE it may go (the MRSP
blob inside the OBS region). This module decides how it actually sits there: how big, which way
round, exactly where, and what colour.

The paper defines two placement modes, and a corpus is built from a mix of both.

ORIGINAL PLACEMENT reads a single number off the blob -- its centroid -- and drops the defect
there at its true relative size with a random rotation. Simple, and it never distorts anything.

ADAPTIVE PLACEMENT reads four things off the blob instead of one:

  size          the defect is scaled so its area is `gamma` times the blob's area
  orientation   the defect's long axis is turned to line up with the blob's long axis
  anchor        the defect's centre of mass is put on the blob's centre of mass
  containment   if the defect still pokes outside the blob, shrink and retry

Both modes only ever scale the defect ISOTROPICALLY -- the same factor in x and y. Nothing here
stretches or squashes a real defect, so the recovered shape survives placement exactly.

Two details that look like paranoia and are not:

  * A centre of mass can fall OUTSIDE its own shape. A curved sliver (one real defect is
    322x39 px and banana-shaped) has its centroid in empty space, and a ragged blob can too.
    Anchoring on a point that is not inside the region would put the defect off-target, so both
    fall back to the deepest interior point, which is guaranteed to be inside.
  * Aligning two axes is ambiguous by 180 degrees, so all four candidate angles are tried and
    the one with the best containment wins.

HARMONISATION is the last step and is easy to overlook. A crop cut from one image and dropped
into another carries its original lighting with it. `harmonise` transfers CIELAB statistics from
the crop's OWN background to the host's LOCAL background -- background to background, never
touching the defect pixels -- so the defect keeps its appearance while its surroundings match
the new image. The gain is clipped so a low-variance crop cannot be stretched into noise.

Provenance: these operators are a verbatim copy of the reference implementation the
paper's results were produced with. Comments and docstrings were added here; the only
edited body is `place_adaptive`, renamed from the reference's `place_mode_B` to match the
paper's terminology -- its def line and docstring changed, its logic did not.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

# Imported under the names the reference implementation uses, so the function bodies below
# stay byte-identical to the code the published results were produced with.
from flash.config import (
    CONTAINMENT_T as CONTAIN_T,
)
from flash.config import (
    GAMMA_SCALE_CLAMP as GAMMA_S_RANGE,
)

__all__ = ["target_long_px", "place_entry", "place_original", "place_adaptive", "harmonise"]


# ----------------------------------------------------------------------------- sizing
def target_long_px(entry, H, W):
    """real_scale: the defect replays at its true relative size."""
    return entry["defect_frac"] * max(H, W)


# ----------------------------------------------------------------------------- stamping
def place_entry(host, entry, center, long_px, theta_deg, flip, region):
    """Resize + optional flip/rotate the entry, stamp into a host-sized canvas initialised
    to the host, so the patch's own substrate is present for the band solve to work against."""
    _, H, W = host.shape
    rgb, a = entry["rgb"], entry["alpha"]
    if flip:
        rgb, a = rgb[:, ::-1].copy(), a[:, ::-1].copy()

    r = long_px / max(entry["defect_long_px"], 1)
    s = int(np.clip(round(entry["side"] * r), 8, max(H, W)))
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LANCZOS4
    rgb = cv2.resize(rgb, (s, s), interpolation=interp)
    a   = cv2.resize(a,   (s, s), interpolation=cv2.INTER_LINEAR)

    if theta_deg:
        p = int(math.ceil(s * (math.sqrt(2) - 1) / 2)) + 2
        rgb = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_REFLECT_101)
        a   = cv2.copyMakeBorder(a,   p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        s2 = rgb.shape[0]
        M = cv2.getRotationMatrix2D((s2 / 2, s2 / 2), theta_deg, 1.0)
        rgb = cv2.warpAffine(rgb, M, (s2, s2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        a   = cv2.warpAffine(a,   M, (s2, s2), flags=cv2.INTER_LINEAR, borderValue=0)
        s = s2

    cy, cx = center
    y0 = int(np.clip(cy - s // 2, 0, max(0, H - s))); x0 = int(np.clip(cx - s // 2, 0, max(0, W - s)))
    y1, x1 = min(H, y0 + s), min(W, x0 + s)
    sy, sx = y1 - y0, x1 - x0

    patch = host.clone()
    alpha = torch.zeros(H, W)
    pt = torch.from_numpy(rgb[:sy, :sx].astype(np.float32) / 255.0).permute(2, 0, 1)
    patch[:, y0:y1, x0:x1] = pt
    alpha[y0:y1, x0:x1] = torch.from_numpy(np.clip(a[:sy, :sx], 0, 1).astype(np.float32))
    alpha = alpha * region                     # never place off-object
    return patch, alpha, (y0, x0, sy, sx)


# ----------------------------------------------------------------------------- harmonisation
def harmonise(patch, alpha, host, box, std_clip=(0.75, 1.35)):
    y0, x0, sy, sx = box
    _, H, W = host.shape
    a = alpha.numpy()
    src_sel = np.zeros((H, W), bool); src_sel[y0:y0 + sy, x0:x0 + sx] = True
    src_sel &= (a < 0.10)
    pad = int(0.6 * max(sy, sx)) + 8
    ry0, ry1 = max(0, y0 - pad), min(H, y0 + sy + pad)
    rx0, rx1 = max(0, x0 - pad), min(W, x0 + sx + pad)
    dst_sel = np.zeros((H, W), bool); dst_sel[ry0:ry1, rx0:rx1] = True
    dst_sel &= (a < 0.10)
    if src_sel.sum() < 50 or dst_sel.sum() < 50:
        return patch
    p8 = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    h8 = (host.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    p_lab = cv2.cvtColor(p8, cv2.COLOR_RGB2LAB).astype(np.float32)
    h_lab = cv2.cvtColor(h8, cv2.COLOR_RGB2LAB).astype(np.float32)
    sm, ss = p_lab[src_sel].mean(0), p_lab[src_sel].std(0) + 1e-6
    dm, ds = h_lab[dst_sel].mean(0), h_lab[dst_sel].std(0) + 1e-6
    gain = np.clip(ds / ss, *std_clip)
    out = np.clip((p_lab - sm) * gain + dm, [0, 0, 0], [255, 255, 255]).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


# ----------------------------------------------------------------------------- geometry helpers
def _mask_centroid(a):
    m = a.numpy() > 0.5
    if m.sum() == 0:
        return None
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())


def _principal_angle(binary):
    """major-axis angle in degrees from second central moments; area comes free as m00"""
    m = cv2.moments(binary.astype(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        return None, 0.0
    mu20, mu11, mu02 = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    if abs(mu20 - mu02) < 1e-9 and abs(mu11) < 1e-9:
        return 0.0, m["m00"]
    return math.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02)), m["m00"]


# ----------------------------------------------------------------------------- placement modes
def place_original(blob_centroid, long_px, theta):
    """ORIGINAL placement. -> (centre, long_px, theta, containment).

    The defect goes at the MRSP blob's centroid, at its true relative size, turned by whatever
    random angle was drawn. Nothing is fitted to the blob, so there is nothing to fail and no
    containment figure to report -- hence the NaN, which downstream tables read as "not
    applicable" rather than "zero".

    This exists as a function purely so both modes have the same signature and the caller can
    treat them symmetrically; the reference implementation inlines these three values.
    """
    return blob_centroid, long_px, theta, float("nan")


def place_adaptive(host, entry, blob, region, bc, long_px, theta, flip, gamma):
    """ADAPTIVE placement. -> (centre, long_px, theta, containment).

    Falls back to the ORIGINAL mode's arguments when the blob or the defect mask is
    degenerate -- a zero-area blob has no orientation to align to and no area to scale
    against, so there is nothing to adapt and the original placement is the correct answer.
    """
    _, H, W = host.shape
    ones = torch.ones(H, W)
    bf = (blob.numpy() > 0.5).astype(np.float32)
    phi_b, area_b = _principal_angle(bf)
    _, af0, _ = place_entry(host, entry, bc, long_px, 0.0, flip, ones)
    phi_m, area_m = _principal_angle((af0.numpy() > 0.5).astype(np.uint8))
    if phi_b is None or phi_m is None or area_m <= 0 or area_b <= 0:
        return bc, long_px, theta, float("nan")

    # SIZE: the defect takes `gamma` of the blob; the rest is margin the collar can use.
    s = float(np.clip(math.sqrt(max(gamma * area_b, 1.0) / area_m), *GAMMA_S_RANGE))

    # ANCHOR: mask centroid on blob centroid. A bent sliver (can: 322x39) has its centroid OFF
    # its own mask, and a ragged blob can have its centroid outside itself -- in either case use
    # the deepest interior point, which is guaranteed to be inside.
    byi, bxi = int(np.clip(bc[0], 0, H - 1)), int(np.clip(bc[1], 0, W - 1))
    if bf[byi, bxi] > 0.5:
        anchor = (float(bc[0]), float(bc[1]))
    else:
        dt = cv2.distanceTransform((bf > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        yx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        anchor = (float(yx[0]), float(yx[1]))

    d = phi_b - phi_m                       # ORIENTATION: align the two principal axes
    best = None
    for _ in range(3):                      # CONTAINMENT: shrink until it fits
        lp = long_px * s
        for th in (d, -d, d + 180.0, -d + 180.0):
            th %= 360.0
            _, afp, _ = place_entry(host, entry, (H // 2, W // 2), lp, th, flip, ones)
            mc = _mask_centroid(afp)
            if mc is None:
                continue
            mb = (afp.numpy() > 0.5).astype(np.uint8)
            ay, ax = mc
            if mb[int(np.clip(ay, 0, H - 1)), int(np.clip(ax, 0, W - 1))] == 0:
                mdt = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
                ay, ax = np.unravel_index(int(np.argmax(mdt)), mdt.shape)
            c = (int(H // 2 + (anchor[0] - ay)), int(W // 2 + (anchor[1] - ax)))
            _, af, _ = place_entry(host, entry, c, lp, th, flip, ones)
            mm = af.numpy() > 0.5
            ct = float((mm & (bf > 0.5)).sum()) / max(mm.sum(), 1)
            if best is None or ct > best[0]:
                best = (ct, c, lp, th)
        if best and best[0] >= CONTAIN_T:
            break
        s = float(np.clip(s * 0.85, *GAMMA_S_RANGE))
    if best is None:
        return bc, long_px, theta, float("nan")
    ct, c, lp, th = best
    return c, lp, th, ct
