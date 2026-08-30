"""Hybrid blending -- merging the placed defect into the host without a visible seam.

Two operators, and a rule for choosing between them.

ALPHA BLENDING just fades the defect in over a couple of pixels at its edge. Cheap and it always
preserves the defect exactly, but on a smooth surface the join can still be visible.

POISSON BLENDING (`cv2.seamlessClone`) instead matches gradients across the boundary, so the
join disappears. The catch is that it reconstructs the interior from the boundary values, which
means the boundary had better not be sitting on the defect itself -- if it is, the solver
happily rebuilds the defect out of surrounding substrate and erases it.

That is what the SUBSTRATE COLLAR is for. Before solving, the defect mask is dilated outwards by
a few pixels, so the boundary the solver reads lands on clean product surface rather than
cutting through the defect. This one detail is the difference between Poisson preserving a
defect and destroying it.

The collar has a floor (`COLLAR_MIN_PX`) so tiny defects still get a workable boundary. But a
floor applied blindly does the opposite of what it is for: a 6 px collar around a defect of
radius 9.7 px eats 62% of it. `COLLAR_KEEP_R` caps the collar at half the defect's radius so the
floor can never dissolve the defect it exists to protect.

Below `R_EQ_POISSON` even that is not enough -- there is simply not enough defect to give away --
so those samples are routed to alpha blending instead. A dissolved sample is a normal image
carrying a defect label, which is worse for training than an imperfect seam.

WHY THE THRESHOLDS ARE MEASURED, NOT FIXED. `defect_thresholds` does not use a constant cutoff
to decide which pixels are "defect". On a granular product like rice, the crop's grains never
line up with the host's, so the whole crop differs from the host and a fixed cutoff would call
the entire background a defect. Instead the noise floor is measured on a ring OUTSIDE the mask,
where by definition there is no defect, and the cutoff is placed above it.

Provenance: these operators are a verbatim copy of the reference implementation the
paper's results were produced with. Comments and docstrings were added here; no function
body was edited. `tests/test_end_to_end.py` runs them end to end.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

# Imported under the names the reference implementation uses, so the function bodies below
# stay byte-identical to the code the published results were produced with.
from flash.config import (
    BETA,
    COLLAR_FRAC,
    COLLAR_KEEP_R,
    COLLAR_MIN_PX,
    DEFECT_T,
    FEATHER_PX,
    R_EQ_ALPHA_MAX,
    R_EQ_POISSON,
    RETENTION_MIN,
)
from flash.mrsp import mrsp_blob_stats

__all__ = ["alpha_composite", "collar_poisson", "composite", "hybrid", "retention",
           "defect_thresholds", "defect_core", "collar_for", "poisson_clone", "lab_of"]


# ----------------------------------------------------------------------------- alpha arm
def feather_mask(mask, sigma):
    return torch.from_numpy(cv2.GaussianBlur(mask.numpy().astype(np.float32), (0, 0), sigma)).clamp(0, 1)


def feather_sigma_for(mask, base_sigma=FEATHER_PX):
    """Cap the feather so it cannot fade the defect: a Gaussian with sigma comparable to the
    blob radius pulls the CORE below opacity 1.0, which is the bug this whole notebook avoids."""
    side, _ = mrsp_blob_stats(mask)
    if side <= 0:
        return float(base_sigma)
    return float(min(base_sigma, max(1.0, 0.25 * side)))


def alpha_composite(host, patch, alpha, region, beta=BETA, base_sigma=FEATHER_PX):
    sigma = feather_sigma_for(alpha, base_sigma)
    mf = (feather_mask(alpha, sigma) * region).unsqueeze(0)
    out = (host * (1 - mf) + (beta * patch + (1 - beta) * host) * mf).clamp(0, 1)
    # edit_mask = the region this operator actually modified. The seam metric is evaluated at
    # each arm's OWN boundary: the collar makes the Poisson arm edit a larger area, and scoring
    # both at the original mask would credit alpha for pixels it never touched.
    edit = ((mf[0] > 0.01).float().numpy() > 0).astype(np.uint8)
    return out, dict(sigma=sigma, edit_mask=edit)


# ----------------------------------------------------------------------------- measurement
def lab_of(t):
    return cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).astype(np.float32)


def defect_thresholds(host, patch, mask_bin, ring_px=4):
    """Per-composite Lab cutoffs separating "defect" from "substrate".

    A fixed cutoff does not survive contact with a granular category. On rice or walnuts the
    patch's grains never line up with the host's, so ||patch - host|| is large across the whole
    crop and a fixed 6-unit cutoff calls the entire substrate a defect. The cutoffs are
    therefore taken from the composite's own distribution: the substrate noise floor is
    measured on the ring OUTSIDE the mask, where by definition there is no defect, and the
    defect cutoff is placed above it.
    """
    k = np.ones((3, 3), np.uint8)
    m = (mask_bin.numpy() > 0.5).astype(np.uint8)
    outer = (cv2.dilate(m, k, iterations=3 * ring_px) - cv2.dilate(m, k, iterations=ring_px)) > 0
    d = np.linalg.norm(lab_of(patch) - lab_of(host), axis=2)
    if outer.sum() < 50:                       # degenerate placement; fall back to the constants
        return DEFECT_T, 4.0, float("nan")
    floor = float(np.quantile(d[outer], 0.90))  # what "same material, misaligned" looks like
    t_def = max(DEFECT_T, floor)                # defect must beat the misalignment floor
    t_sub = max(4.0, float(np.quantile(d[outer], 0.50)))
    return t_def, t_sub, floor


def defect_core(host, patch, mask_bin, T=None):
    """Pixels inside the mask that actually carry the defect. A recovered mask also contains a
    substrate margin, and the difference between the two is what the collar has to clear."""
    m = (mask_bin.numpy() > 0.5).astype(np.uint8)
    if T is None:
        T, _, _ = defect_thresholds(host, patch, mask_bin)
    d = np.linalg.norm(lab_of(patch) - lab_of(host), axis=2)
    k = np.ones((3, 3), np.uint8)
    c = ((d > T) & (m > 0)).astype(np.uint8)
    c = cv2.morphologyEx(c, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(c, cv2.MORPH_CLOSE, k, iterations=2)


# ----------------------------------------------------------------------------- retention
def retention(out, host, patch, mask_bin, t_def=None):
    """R = ||out-host|| / ||patch-host|| in Lab over defect pixels. 1 = present, 0 = dissolved."""
    m = (mask_bin.numpy() > 0.5)
    if m.sum() == 0:
        return dict(R=float("nan"), n_defect=0, t_def=float("nan"))
    if t_def is None:
        t_def, _, _ = defect_thresholds(host, patch, mask_bin)
    d_out = np.linalg.norm(lab_of(out) - lab_of(host), axis=2)
    d_pat = np.linalg.norm(lab_of(patch) - lab_of(host), axis=2)
    sel = m & (d_pat > t_def)
    if sel.sum() < 25:                        # too few defect pixels for the ratio to mean much
        return dict(R=float("nan"), n_defect=int(sel.sum()), t_def=float(t_def))
    return dict(R=float(np.clip(d_out[sel] / (d_pat[sel] + 1e-6), 0, 2).mean()),
                n_defect=int(sel.sum()), t_def=float(t_def))


# ----------------------------------------------------------------------------- poisson arm
def collar_for(core_np, frac=COLLAR_FRAC, min_px=COLLAR_MIN_PX, keep=COLLAR_KEEP_R):
    """Collar width in px, from the defect's equivalent radius. frac=0 means NO collar, and
    the floor does not apply -- otherwise the control sweep could never reach the vanilla
    NORMAL_CLONE it exists to compare against.

    COLLAR_MIN_PX is a floor meant to HELP small defects get a workable boundary, but on a
    defect smaller than ~2.5x the floor it does the opposite: a 6px collar on a defect of
    equivalent radius 9.7px takes 62% of the radius, and NORMAL_CLONE then rebuilds what is
    left from substrate. The cap keeps at least `keep` of the radius as defect, so the floor
    can never be the thing that dissolves the defect it was added to protect.
    """
    if frac <= 0:
        return 0
    a = float(core_np.sum())
    if a < 1:
        return int(min_px)
    r_eq = math.sqrt(a / math.pi)
    w = max(min_px, round(frac * r_eq))
    return int(max(1, min(w, math.floor((1.0 - keep) * r_eq))))


def poisson_clone(host, patch, mask_np):
    """cv2 NORMAL_CLONE on an explicit mask. Returns None if the mask is unusable."""
    mk = (mask_np > 0).astype(np.uint8) * 255
    mk[:3, :] = 0; mk[-3:, :] = 0; mk[:, :3] = 0; mk[:, -3:] = 0
    ys, xs = np.where(mk > 0)
    if len(xs) < 10:
        return None
    c = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    d = (host.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    s = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    try:
        out = cv2.seamlessClone(s, d, mk, c, cv2.NORMAL_CLONE)
    except cv2.error:
        return None
    return torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1).clamp(0, 1)


def collar_poisson(host, patch, alpha, region, frac=COLLAR_FRAC, min_px=COLLAR_MIN_PX):
    """Dilate the defect mask by a substrate collar, then solve. The collar moves the Dirichlet
    boundary off the defect and onto substrate, which is the whole mechanism."""
    hard = ((alpha > 0.5).float() * region)
    core_np = defect_core(host, patch, hard)
    if core_np.sum() < 10:                       # low-contrast defect: no content core to clear
        core_np = (hard.numpy() > 0.5).astype(np.uint8)
    w = collar_for(core_np, frac, min_px)
    k = np.ones((3, 3), np.uint8)
    comp = cv2.dilate((hard.numpy() > 0.5).astype(np.uint8), k, iterations=w) if w > 0 \
        else (hard.numpy() > 0.5).astype(np.uint8)
    comp = (comp * (region.numpy() > 0.5)).astype(np.uint8)   # never composite off-object
    info = dict(collar_px=w, core_px=int(core_np.sum()), comp_px=int(comp.sum()), fallback=None)

    info["edit_mask"] = comp
    out = poisson_clone(host, patch, comp)
    if out is None:
        info["fallback"] = "alpha (seamlessClone declined the mask)"
        out, ai = alpha_composite(host, patch, alpha, region)
        info["edit_mask"] = ai["edit_mask"]
    return out, info


# ----------------------------------------------------------------------------- one entry point
def composite(arm, host, patch, alpha, region, **kw):
    if arm == "alpha":
        return alpha_composite(host, patch, alpha, region, **kw)
    if arm == "poisson":
        return collar_poisson(host, patch, alpha, region, **kw)
    raise KeyError(arm)


# ----------------------------------------------------------------------------- hybrid routing
def hybrid(host, patch, alpha, region, r_eq, frac=COLLAR_FRAC, min_px=COLLAR_MIN_PX):
    """Paper Eq. 4, plus the failure check its fallback clause implies.

    Eq. 4 routes on SIZE: small placements alpha-blend, larger ones Poisson-blend with a
    substrate collar, "and if Poisson blending fails, alpha blending is used as a fallback".

    The size rule and the fallback are both unchanged here. What changes is what counts as a
    failure. Previously only a refusal by the solver -- a mask it would not accept -- was caught.
    But `cv2.seamlessClone` works in the gradient domain: it preserves the source's gradients
    and discards its absolute colour, reconstructing the interior as a harmonic function fitted
    to the boundary. For a defect whose mean colour differs strongly from the surrounding
    substrate, that boundary condition pulls the interior back toward the substrate and the
    defect is erased. Measured on one rice sample: the defect sat 66.9 from the host before
    compositing and 14.5 after -- 22% of its contrast survived. The solver reported success.

    A composite that returns the host is a failed blend whatever the solver says, so retention
    is measured and a genuine failure routes to alpha exactly as Eq. 4 already prescribes.

    THE UPPER BOUND MATTERS. Alpha's feather is a fixed ~1.5 px, so the transition is equally
    sharp whether the defect is 30 px across or 300 px; proportionally, the seam gets worse as
    the defect grows, and a hard edge is a shortcut feature a detector will learn instead of the
    defect. So the rescue applies only up to `R_EQ_ALPHA_MAX`. Above it Poisson is kept even
    when retention is poor: a faded defect is a weak sample, a seamed one is a corrupt label.

    Returns (image, info). `info["arm"]` is what actually ran, `info["route"]` says why.
    """
    hard = (alpha > 0.5).float() * region

    if r_eq < R_EQ_POISSON:                       # Eq. 4, first branch
        out, info = alpha_composite(host, patch, alpha, region)
        info.update(arm="alpha", route="small", retention=float("nan"))
        return out, info

    out, info = collar_poisson(host, patch, alpha, region, frac=frac, min_px=min_px)
    info.update(arm="poisson", route="large")

    if info.get("fallback"):                      # solver declined the mask; already alpha
        info.update(arm="alpha", route="solver-declined", retention=float("nan"))
        return out, info

    R = retention(out, host, patch, hard)["R"]
    info["retention"] = R

    # NaN means too few defect pixels for the ratio to mean anything -- not evidence of failure,
    # so it is left alone rather than triggering a fallback on no information.
    if np.isfinite(R) and R < RETENTION_MIN:
        if r_eq <= R_EQ_ALPHA_MAX:
            out, ai = alpha_composite(host, patch, alpha, region)
            info["edit_mask"] = ai["edit_mask"]
            info.update(arm="alpha", route="dissolved", collar_px=0)
        else:
            info["route"] = "large-kept"          # too big to alpha: a seam is worse than a fade
    return out, info
