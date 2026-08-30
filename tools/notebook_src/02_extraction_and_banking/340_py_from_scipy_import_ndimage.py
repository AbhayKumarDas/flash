from scipy import ndimage
from PIL import Image

def load_u8(path):
    return np.array(Image.open(path).convert("RGB"))

def load_mask_u8(path):
    return np.array(Image.open(path).convert("L"))


# Rebuild flash_part1's RAW structure from what section 4 wrote, so regions_of and
# extract_entry below can be used exactly as they are in that notebook.
RAW, ids = {}, []
for r in _ok:
    k = f"{r['cat']}-{r['pid']}"                 # ids are NOT globally unique across categories
    # ref_aligned is the NORMAL image warped into I_a's own frame by diffmask's registration.
    # I_n and I_a are NOT aligned as they sit on disk -- different scale and offset -- so the raw
    # normal cannot be cropped at the same coordinates. The registered one can.
    _ra = os.path.join(r["dbg"], "01_ref_aligned.png")
    RAW[k] = dict(cat=r["cat"], did=r["pid"],
                  anomaly=load_u8(r["defect"]),
                  mask=load_mask_u8(r["mask"]),
                  normal=load_u8(r["ref"]),
                  ref_aligned=(load_u8(_ra) if os.path.exists(_ra) else None),
                  secs=r["secs"])
    ids.append(k)

_n_ra = sum(1 for k in ids if RAW[k]["ref_aligned"] is not None)
print(f"registered normal available for {_n_ra}/{len(ids)} donors"
      + ("" if _n_ra == len(ids) else "  <- the rest fall back to a 2-panel prompt"))


# --- Region filtering (verbatim from flash_part1) -----------------------------------
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


# flash_part1 filters known false positives here with a hand-maintained DROP_REGIONS blacklist.
# There is no blacklist in this notebook -- that is exactly the job VLM-2 does in section 7.
KEPT = {i: regions_of(RAW[i]["mask"]) for i in ids}
print(f"{'donor':20s} {'cat':13s} kept  dropped-speckle")
for i in ids:
    allr = regions_of(RAW[i]["mask"], min_px=1)
    print(f"{i:20s} {RAW[i]['cat']:13s} {len(KEPT[i]):4d}  {len(allr)-len(KEPT[i]):15d}   "
          + "  ".join(f"a={r['area']}({r['w']}x{r['h']})" for r in KEPT[i]))
print(f"\n{sum(len(v) for v in KEPT.values())} candidate regions from {len(ids)} donors")
