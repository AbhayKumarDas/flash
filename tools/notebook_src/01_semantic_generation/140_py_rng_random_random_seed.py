rng = random.Random(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

manifest, rows = {}, []
run_t0 = time.time()

for cat in CATS:
    cat_dir = os.path.join(OUT_DIR, cat)
    os.makedirs(cat_dir, exist_ok=True)
    paths = NORMALS[cat]

    # One distinct family per image -- this is what stops a category's prompts converging.
    # Only families that can physically occur on this category. Without this you get
    # "walnut print defect" or "can missing part" -- the family pool is uniform otherwise.
    pool = CONFIG.get(cat, {}).get("families") or list(FAMILIES)
    fams = FAMILY_PLAN.get(cat) or rng.sample(pool, k=min(len(paths), len(pool)))

    print(f"[{cat}]")
    taken = []

    for i, (img, family) in enumerate(zip(paths, fams)):
        # Copy the image in; its filename is the tag used everywhere downstream.
        tag = f"Normal_{cat}_{os.path.basename(img)}"
        shutil.copy2(img, os.path.join(cat_dir, tag))

        size = rng.choice(SIZE_WORDS)
        # First image greedy for a stable anchor; later ones sampled to diverge.
        spec, raw, retries, secs = vlm1(cat, img, family, size, taken=taken,
                                        temperature=TEMP_GREEDY if i == 0 else TEMP_SAMPLE)
        taken.append(spec["defect_name"])
        para = build_paragraph(spec)

        rec = {"image": tag, "category": cat, "source_path": img,
               "spec": spec, "paragraph": para,
               "provenance": {"model": MODEL_REV, "model_path": MODEL_PATH,
                              "load_4bit": LOAD_4BIT, "seed": SEED,
                              "defect_source": DEFECT_SOURCE, "family": family,
                              "size_word": size, "retries": retries,
                              "gen_seconds": round(secs, 2), "raw_reply": raw}}
        manifest[tag] = rec
        rows.append({"category": cat, "image": tag, "family": family,
                     "defect_name": spec["defect_name"], "size": size,
                     "target": spec["target"], "defect": spec["defect"],
                     "where": spec["where"], "prompt": para,
                     "retries": retries, "gen_seconds": round(secs, 2)})

        print(f"    {tag}")
        print(f"      [{family:<22s}] {spec['defect_name']:<26s} {secs:6.1f}s"
              f"{'  (' + str(retries) + ' retries)' if retries else ''}")
    print()

RUN_SECONDS = time.time() - run_t0

# CSV is the durable artefact -- prompts survive a kernel reset even if nothing else does.
with open(os.path.join(OUT_DIR, "prompts.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"config": {"model": MODEL_REV, "model_path": MODEL_PATH, "seed": SEED,
                          "defect_source": DEFECT_SOURCE, "n_images": N_IMAGES,
                          "categories": CATS, "load_seconds": round(LOAD_SECONDS, 1),
                          "run_seconds": round(RUN_SECONDS, 1)},
               "entries": manifest}, f, indent=2)

print(f"{len(rows)} prompts over {len(CATS)} categories -> {OUT_DIR}")
print(f"  images   : {len(rows)} copied into per-category folders")
print(f"  prompts  : prompts.csv")
print(f"  manifest : manifest.json")
