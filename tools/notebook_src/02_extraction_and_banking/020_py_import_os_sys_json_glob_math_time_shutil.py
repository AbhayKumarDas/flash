import os, sys, json, glob, math, time, shutil, zipfile, argparse, textwrap, random
import numpy as np
import cv2
from pathlib import Path

SEED = 0
random.seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------- mounts
# Both are searched if the literal path is absent, because Kaggle nests Dataset and Model
# mounts differently (models land under /kaggle/input/models/<owner>/<name>/<fw>/<var>/<ver>).
# Dataset ref OR literal path. A kaggle URL and an "owner/slug" both work: mount_candidates()
# in section 2 expands them, because Kaggle mounts the same dataset at different paths on
# different kernels (/kaggle/input/<slug> vs /kaggle/input/datasets/<owner>/<slug>).
PAIRS_ID = "abhaykdas/local-testing"              # <cat>/<id>_regular.png + <id>_anomaly.png
MODEL_ID = "/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/7b-instruct/2"
# TWO output folders, and nothing else.
#   BANK_DIR   accepted cropped defect patches only  -> Stages 4-5 read this
#   PAIRS_DIR  category-wise (recovered mask, generated anomaly) pairs, one per donor
OUT_ROOT  = "/kaggle/working"
BANK_DIR  = os.path.join(OUT_ROOT, "defect_bank")
PAIRS_DIR = os.path.join(OUT_ROOT, "Defect Masks and Anomalies")
WORK_DIR  = os.path.join(OUT_ROOT, "_work")       # diffmask scratch; not a deliverable

# diffmask's 7 intermediates per pair. Off by default -- they are a debugging aid, not output.
WRITE_DEBUG = False

CATS = ["can", "fabric", "fruit_jelly", "rice",
        "sheet_metal", "vial", "wallplugs", "walnuts"]

# Supplementary 3.A: the configuration was fixed on four development categories and applied
# unchanged to four held-out ones. Local Testing contains BOTH, so every table below marks
# which is which -- a mean over all eight is not a held-out result and must not be read as one.
DEV_CATS     = {"rice", "walnuts", "wallplugs", "fruit_jelly"}
HELDOUT_CATS = {"can", "fabric", "sheet_metal", "vial"}

# Coarse prior on what a defect looks like per category. Fed to VLM-2 as context so it judges
# "is this a plausible <kind> defect on <cat>" rather than "is this interesting".
DEFECT_KIND = {"rice": "foreign_object",   "walnuts": "surface",
               "wallplugs": "foreign_object", "fruit_jelly": "foreign_object",
               "can": "surface",           "fabric": "surface",
               "sheet_metal": "surface",   "vial": "foreign_object"}

# ---------------------------------------------------------------------- diffmask flags
# Frozen config from run_diffmask_batch.py, fixed on four development categories and applied
# unchanged to all eight. There is deliberately NO per-category override table: supplementary
# 3.A claims a fixed, category-agnostic configuration, and a rescue config for one category
# would contradict it. Do not edit without re-running every pair.
FLAGS = ["--w-sharp", "0", "--shrink", "0.4", "--norm-frac", "0.10", "--auto-low", "2.5"]

# ------------------------------------------------------------------------- bank gates
WORK_SIZE       = 1024   # paper Table 3, synthesis resolution
CTX_EXPAND      = 1.8    # crop side = CTX_EXPAND x defect bbox long side (substrate retained)
MIN_REGION_PX   = 500    # recovered components below this are diffmask speckle
ALPHA_SOFTEN    = 1.0    # anti-alias sigma on the hard recovered mask edge
DEFECT_T        = 6.0    # Lab distance above which a crop pixel counts as defect, not substrate
MIN_ENTRY_CFRAC = 0.10   # fraction of masked pixels that must clear DEFECT_T

# ------------------------------------------------------------------------------- VLM
LOAD_4BIT      = False           # False -> fp16 split across both T4s, no internet needed
# Visual-token cap. Raised from 1280 because section 7 now sends THREE panels side by side:
# at the old budget each panel got ~578px and the defect inside it ~180px, which is smaller
# than it was under the old single-panel prompt. Context is useless if it costs the acuity to
# see what you are judging. 2048 patches -> ~730px per panel.
MAX_PIXELS     = 2048 * 28 * 28
MIN_PIXELS     = 256 * 28 * 28
MAX_NEW_TOKENS = 220             # the JSON verdict is short; this is headroom
VLM_MIN_CONF   = 0.5             # accept threshold on VLM-2's own confidence

# Stage 1 introduces exactly ONE defect per generated image. So when DiffMask returns several
# components from one donor, at most one of them is that defect and the rest are extraction
# artifacts -- typically a registration seam, which shows up as a thin elongated strip.
# With this on, VLM-2 is told the region count and its rank, and at most one crop per donor is
# banked: the accepted one it was most confident about. Turn it off only if the generator is
# ever asked for scattered or multi-site defects, where several components are one defect.
ONE_DEFECT_PER_IMAGE = True

# VLM-2 sees a slightly wider crop than the one that gets banked, so some clean substrate stays
# in frame to compare against. This only affects what the model is SHOWN; the banked crop is
# still CTX_EXPAND. Kept modest on purpose: every extra unit of context shrinks the defect
# inside a fixed token budget, and 3.2 cost more acuity than the context was worth.
#   defect size in the model's view ~= (panel px) / VLM_CTX
VLM_CTX = 2.2

# Gate on VLM-2's verdict alone, not on its self-reported confidence. A 7B VLM's confidence is
# close to uncalibrated -- it clusters at 0.8-0.9 regardless of whether it is right -- so
# thresholding it mostly removes correct answers. The number is still recorded, and still used
# to break ties between two regions of the SAME donor (a relative comparison, which is the one
# thing it is fit for). Set True only after checking the confidence histogram is not flat.
USE_VLM_CONFIDENCE = False

for _d in (BANK_DIR, PAIRS_DIR, WORK_DIR):
    os.makedirs(_d, exist_ok=True)
print(f"cats={len(CATS)} ctx={CTX_EXPAND} min_region={MIN_REGION_PX}px "
      f"dE={DEFECT_T} cfrac>={MIN_ENTRY_CFRAC}")
print("flags:", " ".join(FLAGS), " (identical for every category)")
