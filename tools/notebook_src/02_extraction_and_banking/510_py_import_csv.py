import csv

# --- Folder 1: the defect bank. Accepted cropped patches, nothing else. -------------
if os.path.isdir(BANK_DIR):
    shutil.rmtree(BANK_DIR)          # a stale entry from a previous run is worse than no bank
os.makedirs(BANK_DIR, exist_ok=True)

MANIFEST_COLS = ["key", "cat", "donor", "kind", "defect_type", "side", "defect_long_px",
                 "defect_frac", "area_px", "area_frac", "substrate_frac", "contrast", "cfrac",
                 "vlm_conf", "vlm_changed", "vlm_contradiction", "vlm_what_changed", "vlm_has_ref",
                 "vlm_reason", "gate_vlm", "gate_contrast", "gate_primary",
                 "n_regions", "region_rank",
                 "diffmask_secs", "vlm_secs", "flags"]

def row_of(e):
    v = e["vlm"]
    return dict(key=e["key"], cat=e["cat"], donor=e["donor"], kind=e["kind"],
                defect_type=v["defect_type"], side=e["side"],
                defect_long_px=e["defect_long_px"], defect_frac=round(e["defect_frac"], 5),
                area_px=e["area_px"], area_frac=round(e["area_frac"], 6),
                substrate_frac=round(e["substrate_frac"], 4),
                contrast=round(e["contrast"], 2), cfrac=round(e["cfrac"], 4),
                vlm_conf=round(v["confidence"], 3), vlm_changed=v["changed"],
                vlm_contradiction=v.get("contradiction", False),
                vlm_what_changed=v["what_changed"], vlm_has_ref=v["has_ref"],
                vlm_reason=v["reason"],
                gate_vlm=e["gate_vlm"], gate_contrast=e["gate_contrast"],
                gate_primary=e["gate_primary"], n_regions=e.get("n_regions", 1),
                region_rank=e.get("region_rank", 1),
                diffmask_secs=round(e["diffmask_secs"], 2), vlm_secs=round(v["secs"], 2),
                flags=e["flags"])

n_written = 0
for cat in CATS:
    for e in BANK[cat]:
        d = os.path.join(BANK_DIR, cat); os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{e['key']}.png"),
                    cv2.cvtColor(e["rgb"], cv2.COLOR_RGB2BGR))
        # The alpha is NOT optional even though only "the patch" was asked for: Stages 4-5 call
        # place_entry(entry["rgb"], entry["alpha"], ...) and cannot composite without it. It is
        # a sibling file rather than an RGBA channel so the crop keeps its substrate -- an RGBA
        # cut-out leaves the harmoniser nothing to measure.
        cv2.imwrite(os.path.join(d, f"{e['key']}_alpha.png"),
                    np.clip(e["alpha"] * 255, 0, 255).astype(np.uint8))
        n_written += 1

with open(os.path.join(BANK_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
    w.writeheader()
    w.writerows([row_of(e) for c in CATS for e in BANK[c]])

# Rejection log. It sits at the ROOT, outside both deliverable folders, so defect_bank/ still
# holds only patches -- but "why did this category only produce two" has to be answerable after
# the run, not just from console output that scrolls away.
def _why(e):
    if not e["gate_vlm"]:
        if e["vlm"].get("contradiction"):
            return "vlm_contradiction"
        if not e["vlm"]["parse_ok"]:
            return "vlm_unparseable"
        if not e["vlm"]["changed"]:
            return "vlm_no_change"
        if USE_VLM_CONFIDENCE and e["vlm"]["confidence"] < VLM_MIN_CONF:
            return "vlm_low_confidence"
        return "vlm_not_defect"
    if not e["gate_contrast"]:
        return "low_contrast"
    if not e["gate_primary"]:
        return "not_primary_region"
    return "?"

_rej = [e for e in CROPS if not e["accepted"]]
with open(os.path.join(OUT_ROOT, "stage23_rejected.csv"), "w", newline="",
          encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["reason"] + MANIFEST_COLS)
    w.writeheader()
    for e in _rej:
        w.writerow(dict(reason=_why(e), **row_of(e)))

print(f"defect_bank/            {n_written} accepted patches")
if _rej:
    from collections import Counter as _C
    print(f"stage23_rejected.csv    {len(_rej)} discarded: "
          + ", ".join(f"{n} {r}" for r, n in _C(_why(e) for e in _rej).most_common()))
for c in CATS:
    if BANK[c]:
        print(f"  {c:14s} {len(BANK[c]):3d}")
