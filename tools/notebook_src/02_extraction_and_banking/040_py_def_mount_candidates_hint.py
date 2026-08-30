def mount_candidates(hint):
    """Turn a dataset reference into the paths Kaggle might actually have mounted it at.

    Accepts a full URL, an "owner/slug" ref, or a literal path. Kaggle is not consistent about
    the mount point -- the same dataset appears as /kaggle/input/<slug> on some kernels and
    /kaggle/input/datasets/<owner>/<slug> on others -- so every plausible form is tried before
    falling back to a walk.
    """
    if not hint:
        return []
    h = hint.strip().rstrip("/\\")
    # A path that already exists wins outright. Checked BEFORE the "/" test, because a Windows
    # path (C:\...) is absolute without starting with "/" and would otherwise be mangled into
    # a bogus /kaggle/input/<drive> candidate.
    if os.path.isdir(h):
        return [h]
    if h.startswith("http"):                       # .../datasets/<owner>/<slug>[/...]
        parts = [x for x in h.split("/") if x]
        if "datasets" in parts:
            k = parts.index("datasets")
            parts = parts[k + 1:k + 3]
            h = "/".join(parts)
    if os.path.isabs(h):
        return [h]
    bits = [b for b in h.split("/") if b]
    slug = bits[-1]
    owner = bits[-2] if len(bits) >= 2 else None
    cands = [f"/kaggle/input/{slug}"]
    if owner:
        cands += [f"/kaggle/input/{owner}/{slug}",
                  f"/kaggle/input/datasets/{owner}/{slug}"]
    return cands


def find_pairs_root(hint=PAIRS_ID, max_depth=8):
    """Use `hint` if it holds category folders, else walk /kaggle/input for the best candidate.

    "Best" = the directory with the most immediate children that look like a category dir,
    i.e. contain at least one *_anomaly.png. Weight directories are pruned so the walk does
    not descend into model shards.
    """
    def score(root):
        """Number of COMPLETE pairs under `root`, not the number of folders holding anomalies.

        Counting folders picks the wrong directory whenever something else in the tree also
        holds *_anomaly.png -- a previous run's output, for instance, which has the anomaly
        copies but none of the _regular references and therefore yields zero usable pairs.
        """
        n = 0
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if not os.path.isdir(p):
                continue
            for a in glob.glob(os.path.join(p, "*_anomaly.png")):
                if os.path.exists(a.replace("_anomaly.png", "_regular.png")):
                    n += 1
        return n

    for cand in mount_candidates(hint):
        if os.path.isdir(cand) and score(cand):
            print(f"  pairs: {cand}")
            return cand
    if hint:
        print(f"  no complete pairs at any mount form of {hint!r} -- searching instead")

    best, best_n = None, 0
    base = "/kaggle/input" if os.path.isdir("/kaggle/input") else "."
    for root, dirs, _ in os.walk(base):
        if root.count(os.sep) - base.count(os.sep) > max_depth:
            dirs[:] = []; continue
        dirs[:] = [d for d in dirs if not d.startswith(".")
                   and not glob.glob(os.path.join(root, d, "*.safetensors"))]
        try:
            n = score(root)
        except OSError:
            continue
        if n > best_n:                    # best, not first: keep walking past a partial match
            best, best_n = root, n
    if best is None:
        raise FileNotFoundError(
            f"no <cat>/<id>_regular.png + <id>_anomaly.png pairs found under {base} -- attach "
            "the dataset, or point PAIRS_ID at it")
    print(f"  found {best_n} complete pairs under {best}")
    return best


PAIRS_ROOT = find_pairs_root()

PAIRS = []          # (cat, pid, ref_path, defect_path)
ORPHANS = []        # a prompt whose generation is missing -- reported, never silently dropped
for cat in sorted(os.listdir(PAIRS_ROOT)):
    cdir = os.path.join(PAIRS_ROOT, cat)
    if not os.path.isdir(cdir):
        continue
    for dfc in sorted(glob.glob(os.path.join(cdir, "*_anomaly.png"))):
        pid = os.path.basename(dfc)[: -len("_anomaly.png")]
        ref = os.path.join(cdir, f"{pid}_regular.png")
        (PAIRS if os.path.exists(ref) else ORPHANS).append((cat, pid, ref, dfc))

if not PAIRS:
    raise SystemExit(f"no complete pairs under {PAIRS_ROOT}")

_by_cat = {}
for cat, pid, _, _ in PAIRS:
    _by_cat.setdefault(cat, []).append(pid)
print(f"\n{len(PAIRS)} pairs across {len(_by_cat)} categories (root: {PAIRS_ROOT})")
for c in sorted(_by_cat):
    split = "dev" if c in DEV_CATS else ("held-out" if c in HELDOUT_CATS else "?")
    print(f"  {c:14s} {split:9s} {len(_by_cat[c]):3d}  {', '.join(_by_cat[c][:8])}"
          f"{' ...' if len(_by_cat[c]) > 8 else ''}")
if ORPHANS:
    print(f"\n{len(ORPHANS)} anomaly frames with no _regular reference:")
    for c, p, _, _ in ORPHANS[:12]:
        print(f"  {c}/{p}")
