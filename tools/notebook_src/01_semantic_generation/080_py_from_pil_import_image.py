from PIL import Image

random.seed(SEED)

# Pin exact normals per category once the bank is settled, e.g.
#   PINNED = {"rice": ["/kaggle/input/<ds>/rice/train/good/000_regular.png", ...]}
PINNED = {}


def find_model():
    """Use MODEL_ID if it exists on disk, else search /kaggle/input for the weights."""
    if MODEL_ID and os.path.isdir(MODEL_ID):
        return MODEL_ID
    if MODEL_ID:
        print(f"  MODEL_ID not on disk ({MODEL_ID}) -- searching instead")
    hits = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if not glob.glob(os.path.join(d, "*.safetensors")):
            continue                                   # config without weights -> skip
        try:
            arch = json.load(open(cfg)).get("architectures", [""])[0]
        except Exception:
            arch = ""
        if "Qwen2_5_VL" in arch or "qwen" in d.lower():
            hits.append((d, arch))
    if not hits:
        raise FileNotFoundError("no mounted Qwen model under /kaggle/input -- attach it, "
                                "or set MODEL_ID to a hub id")
    hits.sort(key=lambda t: t[0])
    for d, a in hits:
        print("  model candidate:", d, "|", a)
    return hits[-1][0]


def find_data_root(max_depth=8):
    """Walk /kaggle/input for the directory holding the most category folders.

    os.walk rather than a fixed glob depth: Kaggle nests mounts differently depending on
    whether something came in as a Dataset or a Model (e.g. models land under
    /kaggle/input/models/<owner>/<name>/<framework>/<variation>/<version>).
    Weight directories are pruned so the walk does not descend into model shards.
    """
    # If DATA_ROOT_ID exists, use it -- but still descend, because reduced copies often add
    # one wrapper folder between the mount point and the category folders.
    root = DATA_ROOT_ID if (DATA_ROOT_ID and os.path.isdir(DATA_ROOT_ID)) else "/kaggle/input"
    if DATA_ROOT_ID and not os.path.isdir(DATA_ROOT_ID):
        print(f"  DATA_ROOT_ID not on disk ({DATA_ROOT_ID}) -- searching /kaggle/input instead")
    best, best_hit = None, 0
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath[len(root):].count(os.sep) > max_depth:
            dirnames[:] = []                      # too deep, stop descending
            continue
        if any(f.endswith(".safetensors") or f.endswith(".bin") for f in filenames):
            dirnames[:] = []                      # this is a weights folder, not data
            continue
        hit = sum(d in CATS for d in dirnames)
        if hit > best_hit:
            best, best_hit = dirpath, hit
    if not best:
        print("  nothing matched. what IS mounted under /kaggle/input:")
        for d in sorted(glob.glob("/kaggle/input/*")):
            print("   ", d)
            for sub in sorted(glob.glob(os.path.join(d, "*")))[:8]:
                print("      ", os.path.basename(sub))
        raise FileNotFoundError(
            "no dataset root under /kaggle/input -- attach the dataset, or check that its "
            "category folder names match the keys in CONFIG")
    print(f"  data root: {best}  ({best_hit}/{len(CATS)} categories present)")
    return best


def normals_for(cat, root, k, rng):
    """k defect-free images for a category, sampled with the run seed.

    PINNED overrides sampling entirely -- set it once the bank is settled so the same
    images are used on every rerun.
    """
    if cat in PINNED:
        return PINNED[cat][:k]
    for sub in ("train/good", "validation/good", "test_public/good", "train", "good", ""):
        base = os.path.join(root, cat, sub) if sub else os.path.join(root, cat)
        hits = sorted(glob.glob(os.path.join(base, "*.png")) +
                      glob.glob(os.path.join(base, "*.jpg")))
        if hits:
            return sorted(rng.sample(hits, k=min(k, len(hits))))
    raise FileNotFoundError(f"no normal image for {cat} under {root}")


MODEL_PATH = find_model()
DATA_ROOT  = find_data_root()
CATS       = [c for c in CATS if os.path.isdir(os.path.join(DATA_ROOT, c))]

# Separate RNG stream for image choice, so changing N_IMAGES does not reshuffle families.
_pick_rng = random.Random(SEED)
NORMALS   = {c: normals_for(c, DATA_ROOT, N_IMAGES, _pick_rng) for c in CATS}

print("\nmodel:", MODEL_PATH)
print("\nnormal images sampled per category (each one gets its own prompt):")
for c, paths in NORMALS.items():
    for pth in paths:
        w, h = Image.open(pth).size
        print(f"  {c:12s} {w:5d}x{h:<5d}  {os.path.basename(pth)}")
n_prompts = sum(len(v) for v in NORMALS.values())
print(f"\n{len(CATS)} categories, {n_prompts} images -> {n_prompts} prompts to build")
