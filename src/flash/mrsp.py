"""Multi-Resolution Spectral Pyramid (MRSP) noise -- deciding where a defect may go.

MRSP answers one question: given a host image and the region of it occupied by the product,
which patch of that product should the defect land on?

It builds a noise field by adding together several layers of random texture, coarse to fine.
The coarse layers set the overall blob shape; the fine layers roughen its edges so the result
looks organic instead of like a circle. Each layer is made in the frequency domain -- amplitudes
falling off as 1/f**alpha, with a random phase -- then transformed back into an image and mixed
in with a weight that shrinks for finer layers (that weight is `persistence`).

The part that matters, and the easy thing to get wrong, is how the field becomes a mask. A
threshold computed over the whole image would happily pick bright spots on the background;
intersecting with the object afterwards would chop the blob into fragments. So the threshold is
taken from the field values INSIDE THE OBJECT ONLY. That one choice is what makes a requested
coverage of 0.018 mean "1.8% of the product" rather than "1.8% of the picture".

Provenance: these operators are a verbatim copy of the reference implementation the
paper's results were produced with. Comments and docstrings were added here; no function
body was edited. `tests/test_end_to_end.py` runs them end to end.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# Imported under the names the reference implementation uses, so the function bodies below
# stay byte-identical to the code the published results were produced with.
from flash.config import (
    MIN_BLOB_PX,
    MRSP_ALPHA,
    SEED,
)
from flash.config import (
    MRSP_NOISE_SCALE as AD2_NOISE_SCALE,
)
from flash.config import (
    MRSP_OCTAVES as OCTAVES_FBM,
)
from flash.config import (
    MRSP_PERSISTENCE as PERSISTENCE,
)
from flash.config import (
    MRSP_SINGLE_BLOB as SINGLE_BLOB,
)
from flash.config import (
    TARGET_COVERAGE as AD2_COVERAGE,
)
from flash.obs import region_for

__all__ = ["generate_mrsp", "coverage_mask", "mrsp_blob_stats", "keep_largest_blob",
           "mrsp_mask_for"]


# ------------------------------------------------------------------------- the noise field
# `scale` warps the frequency axis and so sets the characteristic blob size; `alpha` controls
# how fast fine detail dies away. Callers pass both explicitly from flash.config -- the defaults
# here are the reference implementation's and are kept unchanged.
def generate_mrsp(h, w, scale=AD2_NOISE_SCALE, device="cpu", seed=None,
                  alpha=MRSP_ALPHA,
                  levels=OCTAVES_FBM, persistence=PERSISTENCE):
    if seed is not None:
        torch.manual_seed(int(seed))
    device = torch.device(device)
    field = torch.zeros(h, w, device=device, dtype=torch.float32)
    weight_sq_sum = 0.0
    for level in range(levels):
        factor = 2 ** (levels - 1 - level)
        h_l = max(8, h // factor); w_l = max(8, w // factor)
        fy = torch.fft.fftfreq(h_l, device=device).reshape(-1, 1)
        fx = torch.fft.rfftfreq(w_l, device=device).reshape(1, -1)
        f = torch.sqrt(fy * fy + fx * fx) * (1.0 + math.log2(max(1, int(scale))))
        amp = torch.where(f > 1e-6, f.pow(-alpha), torch.zeros_like(f))
        phase = 2 * math.pi * torch.rand(h_l, w_l // 2 + 1, device=device)
        spec = amp * torch.complex(torch.cos(phase), torch.sin(phase))
        noise_lr = torch.fft.irfft2(spec, s=(h_l, w_l))
        if (h_l, w_l) != (h, w):
            noise = F.interpolate(noise_lr.unsqueeze(0).unsqueeze(0), size=(h, w),
                                  mode="bilinear", align_corners=False).squeeze()
        else:
            noise = noise_lr
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        w_oct = persistence ** level
        field = field + w_oct * noise
        weight_sq_sum += w_oct ** 2
    field = field / math.sqrt(weight_sq_sum + 1e-8)
    return (field - field.mean()) / (field.std() + 1e-8)


# ------------------------------------------------------------------------- field -> mask
def coverage_mask(noise, region, target_coverage):
    region = region.to(noise.device)
    values = noise[region > 0]
    if values.numel() == 0:
        return torch.zeros_like(noise)
    thr = torch.quantile(values, 1.0 - target_coverage)
    return ((noise > thr) & (region > 0)).float()


def mrsp_blob_stats(mask, min_blob_px=MIN_BLOB_PX):
    """Area-weighted characteristic blob side + largest-blob centroid."""
    lbl, n = ndimage.label(mask.numpy() > 0.5)
    if n == 0:
        return 0.0, None
    areas = ndimage.sum(np.ones_like(lbl, dtype=np.float32), lbl, range(1, n + 1))
    big = areas[areas >= min_blob_px]
    if big.size == 0:
        big = areas
    char_side = math.sqrt(float((big ** 2).sum() / big.sum()))
    i = int(areas.argmax()) + 1
    ys, xs = np.where(lbl == i)
    return char_side, (int(ys.mean()), int(xs.mean()))


def keep_largest_blob(mask):
    lbl, k = ndimage.label(mask.numpy() > 0.5)
    if k <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl, np.float32), lbl, range(1, k + 1))
    return torch.from_numpy((lbl == int(sizes.argmax()) + 1).astype(np.float32))


def mrsp_mask_for(img, seed=SEED, region=None, coverage=None):
    _, H, W = img.shape
    if region is None:
        region = region_for(img)
    noise = generate_mrsp(H, W, scale=AD2_NOISE_SCALE, seed=seed).cpu()
    mask = coverage_mask(noise, region, AD2_COVERAGE if coverage is None else coverage)
    if SINGLE_BLOB:
        mask = keep_largest_blob(mask)
    return region, mask
