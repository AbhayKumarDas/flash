# ============================================================================
# diffmask (1).py lines 1-45 -- module docstring, imports, constants.
#
# The docstring is the six-step pipeline the paper's Algorithm 1 formalises. EPS is the shared
# divide-by-zero floor; every ratio in the file uses it rather than a local literal.
# ============================================================================
#!/usr/bin/env python3
"""
diffmask - find the extra object.

Takes a reference image and a defect image of the same scene and emits a binary
mask (white = object present only in the defect image) in the defect image's own
pixel frame.

The two inputs do not have to be aligned, the same size, or the same exposure.
Resizing them to a common size does not help and is not what the size mismatch
means: the subject sits at a different scale and offset inside each frame, so
matching the canvases still leaves the content tens of pixels apart.

The pipeline is:

  1. coarse scale + translation search on edge maps (multi-scale template match)
  2. ECC refinement to an affine or homography warp
  3. optional low-frequency dense flow to soak up non-rigid drift
  4. photometric-invariant difference: a tolerance band that forgives sub-pixel
     misregistration, over luma, chroma and gradient
  5. threshold against a local noise estimate, since a re-render is noisy in
     textured areas and silent on flat ones
  6. completion, which grows each region into the rest of the same object by
     matching the direction of its difference rather than the size

Usage:
    python diffmask.py ref.png defect.png -o mask.png
    python diffmask.py ref.png defect.png -o mask.png --overlay seen.png
    python diffmask.py ref.png defect.png -o mask.png --debug dbg/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EPS = np.float32(1e-6)

cv2.setUseOptimized(True)
