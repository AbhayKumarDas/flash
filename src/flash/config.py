"""Frozen FLASH configuration — the single source of truth for every hyperparameter.

The values here are fixed once and applied unchanged to every category. That is a claim, not a
convenience: the configuration was established on four development categories (rice, walnuts,
wallplugs, fruit_jelly) and applied without retuning to four held-out ones (can, fabric,
sheet_metal, vial). Any per-category override invalidates the claim and must be reported.

The paper's hyperparameter table is generated from this module (``tools/build_paper_table.py``)
rather than maintained beside it, which removes a whole class of paper-versus-code drift.

Absolute pixel values are bound to ``WORK_SIZE``. Changing the synthesis resolution requires
every pixel-denominated parameter to be rescaled with it.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- dataset
CATEGORIES: tuple[str, ...] = (
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
)

# The configuration was fitted on DEV_CATEGORIES and applied unchanged to HELDOUT_CATEGORIES.
# A mean over all eight is not a held-out result and must not be reported as one.
DEV_CATEGORIES: frozenset[str] = frozenset({"rice", "walnuts", "wallplugs", "fruit_jelly"})
HELDOUT_CATEGORIES: frozenset[str] = frozenset({"can", "fabric", "sheet_metal", "vial"})

# Coarse prior on defect appearance, supplied to VLM-2 as context so it judges "is this a
# plausible <kind> defect on <category>" rather than "is this visually interesting".
DEFECT_KIND: dict[str, str] = {
    "can": "surface",           "fabric": "surface",
    "fruit_jelly": "foreign_object", "rice": "foreign_object",
    "sheet_metal": "surface",   "vial": "foreign_object",
    "wallplugs": "foreign_object", "walnuts": "surface",
}

# --------------------------------------------------------------------------- resolution
# At 768 px a rice defect replays at ~22 px, below the scale at which a blending seam can be
# judged at all. Every MRSP parameter is scale-free or re-derived per image; only the pixel
# constants below are absolute, and they are sized against this value.
WORK_SIZE: int = 1024

# --------------------------------------------------------- Module 2: extraction and banking
DIFFMASK_FLAGS: tuple[str, ...] = (
    "--w-sharp", "0",        # the sharpness cue placed 29.9% of one frame inside the mask
    "--shrink", "0.4",       # 1.2 shattered an 1835 px seed into 13 fragments totalling 166 px
    "--norm-frac", "0.10",   # wider noise window; 0.04 is too local on flat substrate
    "--auto-low", "2.5",     # without it, low-contrast foreign objects do not detect at all
)

CTX_EXPAND: float = 1.8      # crop side = CTX_EXPAND x defect bbox long side, substrate retained
MIN_REGION_PX: int = 500     # recovered components below this are extraction speckle
ALPHA_SOFTEN: float = 1.0    # anti-alias sigma on the hard recovered mask edge
DEFECT_T: float = 6.0        # CIELAB distance above which a pixel counts as defect, not substrate
MIN_ENTRY_CFRAC: float = 0.10  # fraction of masked pixels that must clear DEFECT_T

# VLM-2 sees a slightly wider crop than the one that is banked, so clean substrate stays in
# frame to compare against. Every extra unit of context shrinks the defect inside a fixed visual
# token budget: defect size in the model's view ~= panel_px / VLM_CTX.
VLM_CTX: float = 2.2

# Gate on VLM-2's verdict, not its self-reported confidence. A 7B VLM's confidence is close to
# uncalibrated -- it clusters at 0.8-0.9 regardless of correctness -- so thresholding it mostly
# removes correct answers. The value is still recorded, and still breaks ties between two
# regions of the SAME donor, which is the one comparison it is fit for.
USE_VLM_CONFIDENCE: bool = False
VLM_MIN_CONF: float = 0.5

# Stage 1 introduces exactly one defect per generated image, so when several components are
# recovered from one donor at most one is that defect and the rest are extraction artifacts --
# most often a registration seam, which appears as a long thin strip.
ONE_DEFECT_PER_IMAGE: bool = True

# ------------------------------------------------- Module 3: localisation (OBS + MRSP)
MRSP_NOISE_SCALE: int = 14
MRSP_OCTAVES: int = 6
MRSP_PERSISTENCE: float = 0.8
MRSP_ALPHA: float = 1.5      # 1/f^alpha spectral decay
MRSP_SINGLE_BLOB: bool = True
MIN_BLOB_PX: int = 25
TARGET_COVERAGE: float = 0.018  # object-relative, not frame-relative -- see Eq. 3

# Empty-frame gate. Relative, not absolute: normal OBS coverage runs from a few scattered
# wallplugs to a fabric sheet filling the frame, so one fixed fraction would either pass every
# empty walnuts tray or reject every good vial.
MIN_OBJ_COV_REL: float = 0.40
MIN_OBJ_COV_ABS: float = 0.02

# ------------------------------------------------- Module 3: placement (paper 3.5)
# Two placement modes, named as in the paper:
#
#   ORIGINAL  places the defect at the MRSP blob centroid, preserving its source-relative
#             scale, with a random rotation. It reads ONE number off the blob.
#   ADAPTIVE  additionally uses the blob's area, principal orientation, centroid, coverage and
#             containment to fit the defect to the selected MRSP region. It reads FOUR.
#
# The defect mask is only ever scaled isotropically, so recovered morphology is preserved
# exactly under both modes; nothing here deforms a real defect.
#
# Measured across eight categories, ORIGINAL -> ADAPTIVE: EMD to real ground truth 0.905 ->
# 0.705, kept 0.848 -> 0.970, dissolved 25.7% -> 8.3%.
PLACEMENT_ORIGINAL = "original"
PLACEMENT_ADAPTIVE = "adaptive"
PLACEMENT_MODES: tuple[str, str] = (PLACEMENT_ORIGINAL, PLACEMENT_ADAPTIVE)

ADAPTIVE_FRACTION: float = 0.5      # share of samples generated by ADAPTIVE placement
GAMMA_RANGE: tuple[float, float] = (0.20, 0.60)  # defect's share of the blob, drawn per sample
GAMMA_SCALE_CLAMP: tuple[float, float] = (0.15, 2.50)
CONTAINMENT_T: float = 0.90         # shrink until this much of the defect is inside the blob
ROTATE: bool = True
PLACE_SCALE_MODE: str = "real_scale"  # the defect replays at its true relative size

# ------------------------------------------------- Module 3: hybrid blending
BETA: float = 1.0            # defect opacity. beta < 1 was the cause of washed-out composites
FEATHER_PX: float = 1.5      # alpha-arm seam feather, capped at 0.25 x blob side

# Substrate collar. The Poisson arm dilates the defect mask by this much before solving, so the
# Dirichlet boundary lands on substrate instead of cutting through the defect. This single
# number is the difference between Poisson preserving a defect and destroying it.
COLLAR_FRAC: float = 0.25
COLLAR_MIN_PX: int = 6
COLLAR_KEEP_R: float = 0.5   # the collar may never take more than half the defect's radius,
                             # whatever the floor says -- otherwise the floor destroys exactly
                             # the small defects it was added to protect

# Below this equivalent radius the collar takes more than the defect can spare and the Poisson
# solve rebuilds the interior from substrate. Those route to alpha blending: a dissolved sample
# is a normal image carrying a defect label, which is worse than an impure operator.
# ADAPTIVE placement only -- ORIGINAL is left exactly as published so it stays comparable.
R_EQ_POISSON: float = 16.0

# Paper Eq. 4 ends "if Poisson blending fails, alpha blending is used as a fallback". These two
# constants decide what counts as a failure, and how far the fallback is allowed to reach.
#
# RETENTION_MIN is a floor on R = ||out - host|| / ||patch - host|| measured in CIELAB over the
# defect pixels: 1.0 means the defect survived compositing intact, 0.0 means the composite
# returned the host. cv2.seamlessClone solves in the gradient domain -- it keeps the source's
# gradients and discards its absolute colour -- so a defect whose mean colour differs strongly
# from the substrate can be pulled back to substrate colour while the solver reports success.
# A composite that returns the host is a failed blend whatever the solver says.
RETENTION_MIN: float = 0.35
#
# R_EQ_ALPHA_MAX bounds the rescue. Alpha's feather is a fixed ~1.5 px, so the transition is
# equally sharp on a 30 px defect and a 300 px one -- proportionally the seam gets WORSE as the
# defect grows, and a hard edge is a shortcut feature a detector learns instead of the defect.
# Above this radius Poisson is kept even when retention is poor: a faded defect is a weak
# sample, a seamed one is a corrupt label. Set at 3x R_EQ_POISSON.
R_EQ_ALPHA_MAX: float = 48.0

# A composite whose ground-truth region differs from the host by less than this (mean
# |out - host| over the GT, 0-255) carries no findable defect. A REPORTING threshold: nothing is
# dropped. Poisson erasing a small defect is a property of the operator, and removing those
# samples would quietly flatter it.
DISSOLVED_T: float = 8.0

# --------------------------------------------------------------------------- VLM
VLM_MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_MAX_PIXELS: int = 2048 * 28 * 28   # visual-token cap; three side-by-side panels need this
VLM_MIN_PIXELS: int = 256 * 28 * 28
VLM_MAX_NEW_TOKENS: int = 220
VLM_TEMPERATURE: float = 0.0           # greedy: validation must reproduce exactly on re-run

# --------------------------------------------------------------------------- corpus
SEEDS: tuple[int, ...] = (0, 1, 2)
N_SYNTH: int = 24            # synthetic images per category per seed
N_HOSTS: int = 32            # ceiling on the host pool per category
SEED: int = 42


def summary() -> str:
    """One-line-per-parameter dump, for run logs and provenance files."""
    keys = [k for k in globals() if k.isupper() and not k.startswith("_")]
    width = max(len(k) for k in keys)
    return "\n".join(f"{k:<{width}}  {globals()[k]!r}" for k in sorted(keys))
