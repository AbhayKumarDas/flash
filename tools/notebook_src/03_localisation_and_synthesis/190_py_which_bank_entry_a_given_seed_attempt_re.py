# Which bank entry a given (seed, attempt) retrieves. Verbatim from flash_part1 -- the 7 keeps
# consecutive hosts from all drawing the same entry.
def _entry_for(cat, seed, idx):
    return BANK[cat][(seed * 7 + idx) % len(BANK[cat])]


SYN_DIR = FINAL_DIR
# Clear first. Your last run died mid-loop after writing one image; without this a re-run in the
# same session mixes the two corpora and the manifest disagrees with what is on disk.
if os.path.isdir(SYN_DIR):
    shutil.rmtree(SYN_DIR)
os.makedirs(SYN_DIR, exist_ok=True)

CORPUS = {}          # (cat, seed) -> list of records
t0 = time.time()
n_done, n_target = 0, sum(len(SEEDS) * N_SYNTH for _ in LIVE_CATS)

for cat in LIVE_CATS:
    for seed in SEEDS:
        set_seed(1000 * seed + 7)
        hosts = list(HOST_PATHS[cat])
        random.Random(SEED).shuffle(hosts)   # ONE fixed permutation, identical for every seed
        CORPUS[(cat, seed)] = []

        made, ai = 0, -1
        for ai, hp in enumerate(hosts):
            if made >= N_SYNTH:
                break
            e = _entry_for(cat, seed, ai)
            _ts = time.time()
            d = synthesize(cat, hp, e, seed=1000 * seed + ai)
            _secs = time.time() - _ts
            if d is None:
                continue                      # empty frame / degenerate blob / tiny alpha

            sub = os.path.join(SYN_DIR, cat, f"seed{seed}")
            os.makedirs(sub, exist_ok=True)
            stem = f"{made:03d}_{d['entry']}"
            ip = os.path.join(sub, stem + ".png")
            mp = os.path.join(sub, stem + "_mask.png")
            Image.fromarray((as_np(d["out"]) * 255).astype(np.uint8)).save(ip)
            Image.fromarray(((d["gt"].numpy() > 0.5) * 255).astype(np.uint8)).save(mp)

            # d_in: how much the composite actually differs from the host INSIDE the ground
            # truth, 0-255. This is the one number that says whether the sample is real -- near
            # zero means the GT marks a region identical to the host, so a detector would be
            # scored on finding a defect that is not in the image.
            _g = (d["gt"] > 0.5).numpy()
            _dif = np.abs(as_np(d["out"]) - as_np(d["host"])).mean(2) * 255.0
            d_in = float(_dif[_g].mean()) if _g.any() else 0.0

            CORPUS[(cat, seed)].append(dict(
                img=ip, mask=mp, host=os.path.basename(hp), entry=d["entry"], d_in=d_in,
                gt_frac=float((d["gt"] > 0.5).float().mean()),
                mode=d["mode"], gamma=d["gamma"], contain=d["contain"], r_eq=d["r_eq"],
                blob_px=d["blob_px"], arm_used=d["arm_used"],
                collar=d["collar_px"], secs=_secs))
            made += 1
            n_done += 1
            if n_done % 10 == 0 or n_done == 1:
                el = time.time() - t0
                print(f"    {n_done:4d}/{n_target}  {el:7.1f}s  {el/n_done:6.2f}s/img"
                      f"  eta {(n_target-n_done)*el/n_done:7.1f}s")

        recs = CORPUS[(cat, seed)]
        short = "" if made >= N_SYNTH else \
                f"   SHORT: {made}/{N_SYNTH} after exhausting all {len(hosts)} hosts"
        nb = sum(1 for r in recs if r["mode"] == "B")
        nal = sum(1 for r in recs if r["arm_used"] == "alpha")
        print(f"  {cat:12s} seed {seed}: {len(recs):3d} images   modeB={nb}/{len(recs)}"
              f"  routed_to_alpha={nal}   ({ai + 1} hosts tried){short}")

SYNTH_SECONDS = time.time() - t0
n_img = sum(len(v) for v in CORPUS.values())
print(f"\n{n_img} synthetic images in {SYNTH_SECONDS:.0f}s -> {SYN_DIR}")
