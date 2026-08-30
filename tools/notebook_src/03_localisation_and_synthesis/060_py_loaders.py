# --- Loaders ------------------------------------------------------------------------
def load_image(path, size=WORK_SIZE):
    """Longest side -> size (aspect preserved) -> C x H x W float in [0,1]."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = size / max(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return IMAGE_TO_TENSOR(im)

def as_np(t):
    return t.permute(1, 2, 0).numpy() if t.dim() == 3 else t.numpy()

def _imgs_in(d):
    out = []
    if os.path.isdir(d):
        for ext in _IMG_EXT:
            out += glob.glob(os.path.join(d, f"*.{ext}"))
    # set(): on a case-insensitive filesystem "*.png" and "*.PNG" both match every file, which
    # silently doubles the pool -- the same image becomes two hosts and the audit counts lie.
    return sorted(set(out))


def load_bank(root):
    """Read the Stage 2-3 bank. rgb and alpha stay SEPARATE, never merged to RGBA: harmonise()
    measures the crop's own substrate, and an RGBA cut-out has none.

    Driven by manifest.csv when present. Without it, side and defect_long_px are measured from
    the images and defect_frac is derived from SOURCE_LONG_PX -- which is an assumption, and is
    announced as one.
    """
    import csv as _csv
    mpath = os.path.join(root, "manifest.csv")
    meta = {}
    if os.path.exists(mpath):
        with open(mpath, newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                meta[r["key"]] = r
        print(f"  manifest.csv: {len(meta)} rows")
    else:
        print("  NO manifest.csv -- geometry measured from the images and defect_frac derived")
        print(f"     from SOURCE_LONG_PX={SOURCE_LONG_PX}. If the crops did not come from "
              f"{SOURCE_LONG_PX}px")
        print("     frames, every defect will be placed at the wrong scale.")

    bank, skipped = {}, []
    for ap in sorted(glob.glob(os.path.join(root, "*", "*_alpha.png"))):
        cat = os.path.basename(os.path.dirname(ap))
        key = os.path.basename(ap)[: -len("_alpha.png")]
        rp = os.path.join(os.path.dirname(ap), f"{key}.png")
        rgb = cv2.imread(rp, cv2.IMREAD_COLOR)
        alpha = cv2.imread(ap, cv2.IMREAD_GRAYSCALE)
        if rgb is None or alpha is None:
            skipped.append((key, "missing rgb or alpha")); continue

        m = meta.get(key)
        if m:
            if float(m.get("cfrac", 0) or 0) < MIN_ENTRY_CFRAC:
                skipped.append((key, f"cfrac {m.get('cfrac')} < {MIN_ENTRY_CFRAC}")); continue
            e = dict(m)
            e["side"] = int(float(m["side"]))
            e["defect_long_px"] = int(float(m["defect_long_px"]))
            e["defect_frac"] = float(m["defect_frac"])
            e["cat"] = m.get("cat") or cat
        else:
            ys, xs = np.nonzero(alpha > 127)
            if ys.size == 0:
                skipped.append((key, "empty alpha")); continue
            long_px = int(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
            e = dict(key=key, cat=cat, donor=key.rsplit("_", 2)[0],
                     kind=DEFECT_KIND.get(cat, "unknown"), defect_type="unknown")
            e["side"] = int(rgb.shape[0])
            e["defect_long_px"] = long_px
            e["defect_frac"] = long_px / float(SOURCE_LONG_PX)

        e["rgb"] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        e["alpha"] = alpha.astype(np.float32) / 255.0
        bank.setdefault(e["cat"], []).append(e)

    if skipped:
        print(f"  {len(skipped)} entr(y/ies) skipped: "
              + "; ".join(f"{k} ({w})" for k, w in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    return bank


BANK = load_bank(BANK_ROOT)
CATS_SEEN = sorted(BANK)
LIVE_CATS = [c for c in CATS_SEEN if BANK[c]]
DEAD_CATS = [c for c in CATS_SEEN if not BANK[c]]
if not LIVE_CATS:
    raise SystemExit(f"no usable bank entries under {BANK_ROOT}")

print(f"\n{sum(len(v) for v in BANK.values())} entries across {len(LIVE_CATS)} categories\n")
print(f"{'entry key':32s} {'cat':13s} {'crop':>8s} {'defect':>8s} {'frac':>8s}"
      f" {'replay@1024':>12s} {'subst%':>7s}")
for c in LIVE_CATS:
    for e in BANK[c]:
        print(f"{e['key']:32s} {e['cat']:13s} {e['side']:6d}px {e['defect_long_px']:6d}px"
              f" {e['defect_frac']:8.4f} {e['defect_frac']*WORK_SIZE:10.0f}px"
              f" {100*(e['alpha']<0.1).mean():6.1f}%")
if DEAD_CATS:
    print(f"\nDROPPED (no usable entry): {', '.join(DEAD_CATS)}")
