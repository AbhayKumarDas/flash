for e in CROPS:
    # Verdict alone unless USE_VLM_CONFIDENCE is switched on -- see the note in section 1.
    e["gate_vlm"]      = bool(e["vlm"]["is_defect"]) and (
        e["vlm"]["confidence"] >= VLM_MIN_CONF if USE_VLM_CONFIDENCE else True)
    e["gate_contrast"] = bool(e["cfrac"] >= MIN_ENTRY_CFRAC)
    e["gate_primary"]  = True                      # set below when ONE_DEFECT_PER_IMAGE is on
    e["accepted"]      = e["gate_vlm"] and e["gate_contrast"]

# Third gate: one defect per generated image. Among the crops from a donor that already passed
# both other gates, keep the one VLM-2 was most confident about and drop the rest. Ties break on
# area, because a seam artifact is usually the smaller of the two.
if ONE_DEFECT_PER_IMAGE:
    _cand = {}
    for e in CROPS:
        if e["accepted"]:
            _cand.setdefault(e["donor"], []).append(e)
    for d, rs in _cand.items():
        if len(rs) <= 1:
            continue
        rs.sort(key=lambda x: (-x["vlm"]["confidence"], -x["area_px"]))
        for e in rs[1:]:
            e["gate_primary"] = False
            e["accepted"] = False

BANK     = {c: [] for c in CATS}
REJECTED = []
for e in CROPS:
    (BANK[e["cat"]].append(e) if (e["accepted"] and e["cat"] in BANK) else REJECTED.append(e))

# Which gate did the work? If one gate never rejects anything the other one did not, it is not
# earning its place in the paper and 3.3 should say so.
_only_vlm  = [e for e in CROPS if not e["gate_vlm"] and e["gate_contrast"]]
_only_con  = [e for e in CROPS if e["gate_vlm"] and not e["gate_contrast"]]
_both_rej  = [e for e in CROPS if not e["gate_vlm"] and not e["gate_contrast"]]
_not_prim  = [e for e in CROPS if not e["gate_primary"]]

_n_bank = sum(len(v) for v in BANK.values())
print(f"accepted                    {_n_bank:3d}   from {len(ids)} anomaly images")
print(f"rejected by VLM-2 only      {len(_only_vlm):3d}   (contrast would have passed them)")
print(f"rejected by contrast only   {len(_only_con):3d}   (VLM-2 would have passed them)")
print(f"rejected by both            {len(_both_rej):3d}")
if ONE_DEFECT_PER_IMAGE:
    print(f"dropped as non-primary      {len(_not_prim):3d}   "
          f"(passed both gates, but their donor already had a better crop)")
    for e in _not_prim:
        best = max((x for x in CROPS if x["donor"] == e["donor"] and x["accepted"]),
                   key=lambda x: x["vlm"]["confidence"], default=None)
        print(f"    {e['key']:30s} conf {e['vlm']['confidence']:.2f} {e['area_px']:>7d}px"
              + (f"  <- kept {best['key'].split('_',1)[1]} instead" if best else ""))
    if _n_bank > len(ids):
        print("    NOTE: more crops than anomaly images -- ONE_DEFECT_PER_IMAGE did not bind.")
print()
for c in CATS:
    n = len(BANK[c])
    print(f"  {c:14s} {n:3d} entr{'y' if n == 1 else 'ies'}"
          f"{'   <- EMPTY, category unusable in Stages 4-5' if n == 0 else ''}")
