_IMG_EXT = ("png", "PNG", "jpg", "JPG", "jpeg", "JPEG")


def mount_candidates(hint):
    """Turn a dataset reference into the paths Kaggle might actually have mounted it at.

    Accepts a full URL, an "owner/slug" ref, or a literal path. Kaggle is not consistent about
    the mount point -- the same dataset appears as /kaggle/input/<slug> on some kernels and
    /kaggle/input/datasets/<owner>/<slug> on others -- so every plausible form is tried.
    """
    if not hint:
        return []
    h = hint.strip().rstrip("/\\")
    if os.path.isdir(h):                 # before the "/" test: a Windows path is absolute
        return [h]                       # without starting with "/"
    if h.startswith("http"):
        parts = [x for x in h.split("/") if x]
        if "datasets" in parts:
            k = parts.index("datasets")
            h = "/".join(parts[k + 1:k + 3])
    if os.path.isabs(h):
        # An absolute path that does not exist is still a strong hint about the slug, so fall
        # through to the slug forms rather than returning a path that is known to be missing.
        h = "/".join([p for p in h.split("/") if p][-2:])
    bits = [b for b in h.split("/") if b]
    slug, owner = bits[-1], (bits[-2] if len(bits) >= 2 else None)
    cands = [f"/kaggle/input/{slug}"]
    if owner:
        cands += [f"/kaggle/input/{owner}/{slug}", f"/kaggle/input/datasets/{owner}/{slug}"]
    return cands


def find_root(pred, hint, what, max_depth=8):
    """`hint` in any mount form if it satisfies `pred`, else the BEST match under /kaggle/input.

    Best, not first: a previous run's output can look like the real thing from the outside, and
    returning the first hit is how you end up synthesising from the wrong folder.
    """
    for cand in mount_candidates(hint):
        try:
            if os.path.isdir(cand) and pred(cand):
                print(f"  {what}: {cand}")
                return cand
        except OSError:
            pass
    if hint:
        print(f"  {what}: no match at any mount form of {hint!r} -- searching")
    base = "/kaggle/input" if os.path.isdir("/kaggle/input") else "."
    best, best_n = None, 0
    for root, dirs, _ in os.walk(base):
        if root.count(os.sep) - base.count(os.sep) > max_depth:
            dirs[:] = []; continue
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and not glob.glob(os.path.join(root, d, "*.safetensors"))]
        try:
            n = pred(root)
        except OSError:
            continue
        if n and n > best_n:
            best, best_n = root, int(n)
    if best is None:
        raise FileNotFoundError(f"could not locate {what} under {base}")
    print(f"  {what}: {best}  (score {best_n})")
    return best


def _score_bank(d):
    """Categories under `d` that hold at least one <key>.png + <key>_alpha.png pair.

    Scored on the images, not on manifest.csv, so a bank uploaded without one is still found --
    and so a stray manifest elsewhere in the tree cannot win.
    """
    n = 0
    for c in sorted(os.listdir(d)):
        p = os.path.join(d, c)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "*_alpha.png")):
            n += 1
    return n

def _score_ad2(d):
    return sum(os.path.isdir(os.path.join(d, c, "train", "good")) for c in CATS)


BANK_ROOT = find_root(_score_bank, BANK_ID, "defect bank")
AD2_ROOT  = find_root(_score_ad2, AD2_ID, "MVTec AD 2 reduced")
