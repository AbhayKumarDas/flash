"""End-to-end test: one synthetic anomaly per category, through every module in ``flash``.

WHAT THIS PROVES
    Unit tests show a function returns the right thing on its own. Only running the real chain
    shows that the modules agree with each other about shapes, dtypes, key names and call order.
    This walks a real image pair through all six operators and writes images you can look at.

        <pairs>/<category>/{<id>_regular.png, <id>_anomaly.png}
          -> flash.diffmask     recover the defect mask from the pair          (Stage 2)
          -> flash.bank         connected regions, then a substrate-kept crop  (Stage 3)
        <mvtec>/<category>/train/good/<host>.png   -- a DIFFERENT normal frame
          -> flash.obs          the part of the frame the product occupies     (Stage 4)
          -> flash.mrsp         a placement blob inside that region            (Stage 4)
          -> flash.placement    size / orientation / anchor, then harmonise    (Stage 5)
          -> flash.compositing  collar-Poisson, with alpha where Poisson fails (Stage 5)

    Not covered: the VLM validation gate. It needs a 7B model and a GPU, and it lives in the
    Module 2 notebook rather than in this package. This is the geometry and compositing chain.

WHAT YOU NEED BEFORE RUNNING IT
    Two directories. Neither ships with the repository -- they are data, not code.

    1. NORMAL/ANOMALY PAIRS, one folder per category::

           <pairs>/can/013_regular.png     a clean reference frame
           <pairs>/can/013_anomaly.png     the same scene with a defect in it

       These are Module 1's output. The two images do NOT need to be aligned, the same size, or
       the same exposure -- DiffMask registers them. Any pair of a normal frame and a defective
       version of it will do; you do not need our generated set to try this.

    2. MVTec AD 2 (or a reduced copy), for host images::

           <mvtec>/can/train/good/*.png

       https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2

    Either drop them into the repository, which needs no configuration at all::

        data/input/pairs/<category>/...
        data/input/mvtec_ad_2/<category>/train/good/...

    or leave them where they are and point at them::

        FLASH_PAIRS=/path/to/pairs FLASH_MVTEC=/path/to/mvtec_ad_2 python tests/test_end_to_end.py

    A third variable, FLASH_TEST_OUT, moves the output directory (default: <repo>/temp).

    If either directory is missing the test SKIPS rather than fails. An absent dataset is not a
    broken build.

HOW TO RUN IT

    For artifacts you can inspect::

        python tests/test_end_to_end.py

    For a pass/fail, e.g. in CI::

        pytest tests/test_end_to_end.py -v

WHAT IT WRITES

        temp/gallery.png              8 rows x 7 columns -- the whole pipeline at a glance
        temp/pairs/<cat>_anomaly.png  the synthetic anomaly
        temp/pairs/<cat>_mask.png     its ground-truth mask, same filename stem
        temp/_work/                   DiffMask intermediates, for debugging a bad mask

    Read a gallery row left to right and you see every stage for that category: the untouched
    host, the region OBS found, the blob MRSP picked inside it, the crop taken from the bank,
    the label that ships, the composite, and a zoom on the defect. If a defect lands on the
    background, or a mask disagrees with its image, it is visible there and nowhere else.

READING THE OUTPUT

        mode     `original` or `adaptive` -- the two placement modes from the paper
        arm      `poisson` or `alpha` -- which blend actually ran
        route    why that arm ran. `dissolved` means Poisson erased the defect and alpha
                 rescued it; `large-kept` means it was too big to alpha safely, so the fade
                 was accepted rather than risk a visible seam
        R        retention. 1.0 = the defect survived compositing intact, 0.0 = it was erased.
                 `nan` means too few defect pixels for the ratio to mean anything -- an absence
                 of evidence, not a failure, so nothing is done about it
        d_in     mean |composite - host| inside the ground truth, 0-255. Near zero would mean
                 the label marks a region identical to the host, i.e. an unfindable defect

A NOTE ON SEEDS
    Placement is stochastic on purpose. No single seed is best for every category -- on `vial`
    the blob lands high at some seeds and low at others, and on `fruit_jelly` the reverse. One
    draw is a spot check, not a distribution; a real corpus draws several seeds per category.
    `SEED` and `COVERAGE` below are one such draw, chosen so that both placement modes and both
    blending arms actually run.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flash import config  # noqa: E402
from flash.bank import extract_entry, regions_of  # noqa: E402
from flash.compositing import hybrid  # noqa: E402
from flash.diffmask import main as diffmask_main  # noqa: E402
from flash.mrsp import mrsp_blob_stats, mrsp_mask_for  # noqa: E402
from flash.obs import region_for  # noqa: E402
from flash.placement import (  # noqa: E402
    harmonise,
    place_adaptive,
    place_entry,
    place_original,
    target_long_px,
)


# --------------------------------------------------------------------------- where the data is
def _resolve(env_var, default):
    """The directory named by `env_var` if it is set, else `default`, else None.

    The default is inside the repository so that dropping the data into `data/input/` is enough
    to run this with no configuration. Anywhere else, set the environment variable.
    """
    p = Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else default
    return p if p.is_dir() else None


PAIRS = _resolve("FLASH_PAIRS", REPO / "data" / "input" / "pairs")
MVTEC = _resolve("FLASH_MVTEC", REPO / "data" / "input" / "mvtec_ad_2")
# Artifacts to look at, not repository content -- gitignored. Override with FLASH_TEST_OUT.
TEMP = Path(os.environ.get("FLASH_TEST_OUT", REPO / "temp"))

MISSING = [n for n, p in (("pairs (set FLASH_PAIRS)", PAIRS),
                          ("MVTec AD 2 (set FLASH_MVTEC)", MVTEC)) if p is None]

# --------------------------------------------------------------------------- this run's draw
WORK = config.WORK_SIZE
SEED = 11
# A test-only override, deliberately NOT a change to config.TARGET_COVERAGE (0.018): that value
# appears in the paper's hyperparameter table, and config.py is the single source that table is
# generated from. 0.012 is the low end of the reference's own coverage sweep, so it is a swept
# value rather than an invented one. A smaller blob places more tightly and -- because ADAPTIVE
# sizes the defect as gamma x blob area -- also makes adaptive defects smaller.
COVERAGE = 0.012

# Half the categories through each placement mode, so both paths are exercised. Left to the
# seeded RNG, eight samples could easily all take the same branch.
MODE_FOR = {
    "can": config.PLACEMENT_ORIGINAL,
    "fabric": config.PLACEMENT_ORIGINAL,
    "fruit_jelly": config.PLACEMENT_ORIGINAL,
    "rice": config.PLACEMENT_ORIGINAL,
    "sheet_metal": config.PLACEMENT_ADAPTIVE,
    "vial": config.PLACEMENT_ADAPTIVE,
    "wallplugs": config.PLACEMENT_ADAPTIVE,
    "walnuts": config.PLACEMENT_ADAPTIVE,
}

COLUMNS = ["(a) Host image", "(b) OBS region", "(c) Placement mask",
           "(d) Defect crop", "(e) Ground-truth mask", "(f) Synthetic anomaly",
           "(g) Defect, zoomed"]


# --------------------------------------------------------------------------- small helpers
def load_image(path, size=WORK):
    """Longest side -> `size`, aspect preserved, as a C x H x W float tensor in [0, 1]."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1).contiguous()


def as_np(t):
    return t.permute(1, 2, 0).numpy() if t.dim() == 3 else t.numpy()


# --------------------------------------------------------------------------- the pipeline
def recover_mask(ref, anomaly, out_png):
    """Stage 2. DiffMask is driven through its own CLI entry point, exactly as the notebooks do,
    so no default can quietly diverge from the published configuration."""
    argv = [str(ref), str(anomaly), "-o", str(out_png), *config.DIFFMASK_FLAGS]
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        diffmask_main(argv)
    return cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE), buf.getvalue()


def build_entry(cat, donor_id, anomaly_path, mask):
    """Stage 3. Largest recovered region -> a crop that keeps its own substrate."""
    anom = cv2.cvtColor(cv2.imread(str(anomaly_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    regions = regions_of(mask)
    if not regions:
        return None
    raw = {f"{cat}-{donor_id}": {"anomaly": anom, "cat": cat}}
    return extract_entry(f"{cat}-{donor_id}", regions[0], raw)


def synthesise(cat, entry, host_path, mode, seed=SEED):
    """Stages 4-5. Host -> OBS -> MRSP -> placement -> harmonise -> hybrid blend."""
    host = load_image(host_path)
    _, H, W = host.shape

    region = region_for(host)                                      # where the product is
    region, blob = mrsp_mask_for(host, seed=seed, region=region,   # where a defect may go
                                 coverage=COVERAGE)
    if blob.sum() < config.MIN_BLOB_PX:
        return None
    _, centre = mrsp_blob_stats(blob)
    if centre is None:
        return None

    long_px = target_long_px(entry, H, W)
    theta, flip, gamma = 37.0, False, 0.40      # fixed rather than drawn, so a rerun matches

    if mode == config.PLACEMENT_ADAPTIVE:
        centre, long_px, theta, contain = place_adaptive(
            host, entry, blob, region, centre, long_px, theta, flip, gamma)
    else:
        centre, long_px, theta, contain = place_original(centre, long_px, theta)

    patch, alpha, box = place_entry(host, entry, centre, long_px, theta, flip, region)
    if (alpha > 0.5).sum() < 20:
        return None
    ph = harmonise(patch, alpha, host, box)     # match the crop's colour to THIS host

    r_eq = float(np.sqrt(float((alpha > 0.5).sum()) / np.pi))
    out, info = hybrid(host, ph, alpha, region, r_eq, frac=config.COLLAR_FRAC)

    gt = (alpha > 0.5).float() * region
    g = (gt > 0.5).numpy()
    dif = np.abs(as_np(out) - as_np(host)).mean(2) * 255.0
    return dict(out=out, gt=gt, mode=mode, arm=info["arm"], route=info["route"],
                R=info.get("retention", float("nan")),
                contain=contain, r_eq=r_eq, collar=info.get("collar_px", 0),
                d_in=float(dif[g].mean()) if g.any() else 0.0,
                gt_px=int(g.sum()), host=Path(host_path).name,
                host_img=host, region=region, blob=blob, crop=entry["rgb"])


# --------------------------------------------------------------------------- the gallery
def _tint(img, mask, colour, strength, outline=False):
    """Blend `img` toward `colour` inside `mask`, and outline the boundary.

    Blending, not `max()` into a channel: `max()` does nothing wherever that channel is already
    bright, which is most of a pale product -- rice reads 0.63 red, fruit jelly 0.82 -- so an
    overlay drawn that way is invisible on exactly the images you most want to check.
    """
    m = (mask > 0.5).astype(np.float32)[..., None]
    out = img * (1 - strength * m) + np.array(colour, np.float32) * (strength * m)
    if outline:
        u8 = np.ascontiguousarray((out * 255).astype(np.uint8))
        cont, _ = cv2.findContours((m[..., 0] > 0.5).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(u8, cont, -1, tuple(int(255 * c) for c in colour), 2)
        out = u8 / 255.0
    return np.clip(out, 0, 1)


def _locate(img, mask, colour=(1.0, 0.0, 0.0), pad=14):
    """Draw a 1 px box around the defect on the full-frame composite.

    The files written to temp/pairs/ stay unmarked -- this is for the gallery only, where a
    30 px defect in a 1024 px frame cannot otherwise be found by eye.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    u8 = np.ascontiguousarray((img * 255).astype(np.uint8))
    H, W = u8.shape[:2]
    cv2.rectangle(u8, (max(0, xs.min() - pad), max(0, ys.min() - pad)),
                  (min(W - 1, xs.max() + pad), min(H - 1, ys.max() + pad)),
                  tuple(int(255 * c) for c in colour), 1)
    return u8 / 255.0


def _zoom(img, mask, pad=3.0):
    """Crop around the defect with its mask outlined.

    Columns (a)-(f) are full frames, and a defect placed at its true relative size is 1-3% of
    the frame width -- correct, and invisible at thumbnail scale. Without this column the
    gallery cannot be used to check anything.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    cy, cx = int(ys.mean()), int(xs.mean())
    H, W = img.shape[:2]
    side = min(max(int(max(ys.max() - ys.min(), xs.max() - xs.min()) * pad), 96), H, W)
    y0 = max(0, min(cy - side // 2, H - side))
    x0 = max(0, min(cx - side // 2, W - side))
    crop = np.ascontiguousarray((img[y0:y0 + side, x0:x0 + side] * 255).astype(np.uint8))
    mc = mask[y0:y0 + side, x0:x0 + side].astype(np.uint8)
    cont, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(crop, cont, -1, (255, 0, 0), max(1, side // 120))
    return crop / 255.0


def build_gallery(results, path):
    """One row per category, seven columns: the whole pipeline for that category, in order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, ax = plt.subplots(n, 7, figsize=(7 * 2.6, n * 2.6), squeeze=False)
    for r, res in enumerate(results):
        host = as_np(res["host_img"])
        gt = res["gt"].numpy() > 0.5
        panels = [host,
                  _tint(host, res["region"].numpy(), (0.90, 0.15, 0.15), 0.40, outline=True),
                  _tint(host, res["blob"].numpy(), (0.10, 0.95, 0.20), 0.75, outline=True),
                  res["crop"] / 255.0,
                  gt.astype(float),
                  _locate(as_np(res["out"]), gt),
                  _zoom(as_np(res["out"]), gt)]
        for c, img in enumerate(panels):
            a = ax[r][c]
            a.imshow(img, cmap="gray" if c == 4 else None,
                     vmin=0 if c == 4 else None, vmax=1 if c == 4 else None)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor("0.85")
            if r == 0:
                a.set_title(COLUMNS[c], fontsize=10, pad=8)
            if c == 6:
                a.set_xlabel(f"d_in {res['d_in']:.0f}   gt {res['gt_px']}px"
                             f"   R {res['R']:.2f}", fontsize=7)
        ax[r][0].set_ylabel(f"{res['cat'].replace('_', ' ').title()}\n"
                            f"{res['mode']} / {res['arm']}\n({res['route']})",
                            fontsize=8, rotation=0, ha="right", va="center", labelpad=42)

    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- driver
def run(verbose=True):
    """Synthesise one anomaly per category. Returns (results, failures)."""
    if MISSING:
        raise FileNotFoundError("missing input data: " + ", ".join(MISSING))

    if TEMP.exists():
        shutil.rmtree(TEMP)
    (TEMP / "pairs").mkdir(parents=True, exist_ok=True)
    (TEMP / "_work").mkdir(parents=True, exist_ok=True)

    results, failures, t0 = [], [], time.time()

    for cat in config.CATEGORIES:
        cdir, hdir = PAIRS / cat, MVTEC / cat / "train" / "good"
        if not cdir.is_dir() or not hdir.is_dir():
            failures.append((cat, "no pair or host directory"))
            continue

        donors = sorted(p.name[: -len("_anomaly.png")] for p in cdir.glob("*_anomaly.png")
                        if (cdir / f"{p.name[:-len('_anomaly.png')]}_regular.png").exists())
        host_pool = sorted(hdir.glob("*.png"))
        if not donors or not host_pool:
            failures.append((cat, "no complete pairs, or no hosts"))
            continue

        made = None
        for donor_id in donors:              # move on if a donor recovers nothing usable
            ref, anom = cdir / f"{donor_id}_regular.png", cdir / f"{donor_id}_anomaly.png"
            mask, _ = recover_mask(ref, anom, TEMP / "_work" / f"{cat}_{donor_id}_mask.png")
            if mask is None or (mask > 127).sum() == 0:
                continue
            entry = build_entry(cat, donor_id, anom, mask)
            if entry is None or entry["cfrac"] < config.MIN_ENTRY_CFRAC:
                continue
            # A host that is NOT this donor: donor and host pools must stay disjoint, or the
            # same frame ends up on both sides of the pipeline.
            host = next((h for h in host_pool
                         if h.stem not in (donor_id, f"{donor_id}_regular")), host_pool[0])
            made = synthesise(cat, entry, host, MODE_FOR.get(cat, config.PLACEMENT_ORIGINAL))
            if made:
                made.update(cat=cat, donor=donor_id, entry=entry["key"],
                            defect_px=entry["defect_long_px"], cfrac=entry["cfrac"])
                break

        if made is None:
            failures.append((cat, "no donor produced a usable sample"))
            continue

        img = (as_np(made["out"]) * 255).astype(np.uint8)
        cv2.imwrite(str(TEMP / "pairs" / f"{cat}_anomaly.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(TEMP / "pairs" / f"{cat}_mask.png"),
                    ((made["gt"].numpy() > 0.5) * 255).astype(np.uint8))
        results.append(made)
        if verbose:
            print(f"  {made['cat']:12s} {made['mode']:8s} {made['arm']:7s} {made['route']:<15s}"
                  f"R={made['R']:5.2f}  r_eq={made['r_eq']:5.1f} collar={made['collar']:2d} "
                  f"gt={made['gt_px']:6d}px d_in={made['d_in']:6.1f}")

    if results:
        build_gallery(results, TEMP / "gallery.png")
    if verbose:
        print(f"\n{len(results)}/{len(config.CATEGORIES)} categories in {time.time() - t0:.0f}s"
              f"   seed={SEED} coverage={COVERAGE} (config default {config.TARGET_COVERAGE})")
        print(f"  -> {TEMP}\n  -> {TEMP / 'gallery.png'}")
        for cat, why in failures:
            print(f"  FAILED  {cat}: {why}")
    return results, failures


def test_end_to_end():
    """Every category produces one synthetic anomaly carrying a real, findable defect."""
    import pytest

    if MISSING:
        pytest.skip("input data not found: " + ", ".join(MISSING)
                    + " -- see this file's docstring for what each directory needs.")

    results, failures = run(verbose=False)
    assert not failures, f"categories failed: {failures}"
    assert len(results) == len(config.CATEGORIES)
    for r in results:
        assert r["gt_px"] > 0, f"{r['cat']}: empty ground truth"
        assert r["d_in"] >= config.DISSOLVED_T, \
            f"{r['cat']}: defect dissolved (d_in={r['d_in']:.1f} < {config.DISSOLVED_T})"
    assert {r["mode"] for r in results} == set(config.PLACEMENT_MODES), \
        "both placement modes must run"


if __name__ == "__main__":
    if MISSING:
        print("Cannot run -- missing input data: " + ", ".join(MISSING))
        print("\nSet the paths and try again, for example:\n")
        print("  FLASH_PAIRS=/path/to/pairs FLASH_MVTEC=/path/to/mvtec_ad_2 \\")
        print("      python tests/test_end_to_end.py\n")
        print("See this file's docstring for what each directory must contain.")
        sys.exit(2)
    _, _fail = run()
    sys.exit(1 if _fail else 0)
