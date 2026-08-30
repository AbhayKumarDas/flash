# --- Folder 2: (recovered mask, generated anomaly) pairs, category-wise -------------
# Written by section 4 as diffmask produced them. Counted here, and pairs whose mask came back
# empty are removed: a mask with no defect in it is not a pair, it is a failure to recover one.
n_pairs = 0
for cat in sorted(os.listdir(PAIRS_DIR)):
    d = os.path.join(PAIRS_DIR, cat)
    if not os.path.isdir(d):
        continue
    for mp in sorted(glob.glob(os.path.join(d, "*_mask.png"))):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None or (m > 127).sum() == 0:
            os.remove(mp)
            ap = mp.replace("_mask.png", "_anomaly.png")
            if os.path.exists(ap):
                os.remove(ap)
        else:
            n_pairs += 1

print(f"Defect Masks and Anomalies/  {n_pairs} pairs")
for cat in sorted(os.listdir(PAIRS_DIR)):
    d = os.path.join(PAIRS_DIR, cat)
    if os.path.isdir(d):
        print(f"  {cat:14s} {len(glob.glob(os.path.join(d, '*_mask.png'))):3d}")

# Provenance, one file, beside the two folders rather than inside either.
json.dump(dict(seed=SEED, pairs_root=PAIRS_ROOT, model=MODEL_PATH,
               flags=" ".join(FLAGS), category_agnostic=True,
               ctx_expand=CTX_EXPAND, min_region_px=MIN_REGION_PX, alpha_soften=ALPHA_SOFTEN,
               defect_t=DEFECT_T, min_entry_cfrac=MIN_ENTRY_CFRAC, vlm_min_conf=VLM_MIN_CONF,
               work_size=WORK_SIZE, n_pairs=len(PAIRS), n_crops=len(CROPS),
               n_banked=n_written, n_mask_pairs=n_pairs,
               load_seconds=round(LOAD_SECONDS, 1),
               diffmask_seconds=round(sum(r["secs"] for r in RESULTS), 1),
               vlm_seconds=round(sum(e["vlm"]["secs"] for e in CROPS), 1)),
          open(os.path.join(OUT_ROOT, "stage23_run_config.json"), "w"), indent=2)
