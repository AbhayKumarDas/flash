print(f"{'category':14s} {'split':9s} {'pairs':>6s} {'masks':>6s} {'crops':>6s} {'vlm':>5s}"
      f" {'banked':>7s} {'empty':>6s} {'fail':>5s}   why the crops were lost")
tot = dict(pairs=0, masks=0, crops=0, vlm=0, banked=0, empty=0, fail=0)
for c in sorted({r["cat"] for r in RESULTS}):
    rows  = [r for r in RESULTS if r["cat"] == c]
    crops = [e for e in CROPS if e["cat"] == c]
    row = dict(pairs=len(rows),
               masks=len([r for r in rows if r["err"] is None and r["mask_px"] > 0]),
               crops=len(crops),
               vlm=len([e for e in crops if e["gate_vlm"]]),
               banked=len(BANK.get(c, [])),
               empty=len([r for r in rows if r["err"] is None and r["mask_px"] == 0]),
               fail=len([r for r in rows if r["err"] is not None]))
    for k in tot:
        tot[k] += row[k]
    split = "dev" if c in DEV_CATS else ("held-out" if c in HELDOUT_CATS else "?")
    lost = [e for e in crops if not e["accepted"]]
    from collections import Counter as _C2
    why = "  ".join(f"{n}x{r}" for r, n in _C2(_why(e) for e in lost).most_common()) or "-"
    print(f"{c:14s} {split:9s} {row['pairs']:6d} {row['masks']:6d} {row['crops']:6d}"
          f" {row['vlm']:5d} {row['banked']:7d} {row['empty']:6d} {row['fail']:5d}   {why}"
          f"{'   <- EMPTY' if row['banked'] == 0 else ''}")
print(f"{'TOTAL':14s} {'':9s} {tot['pairs']:6d} {tot['masks']:6d} {tot['crops']:6d}"
      f" {tot['vlm']:5d} {tot['banked']:7d} {tot['empty']:6d} {tot['fail']:5d}")

# The split that matters. Held-out yield is the number supplementary 3.A's claim rests on:
# the same flags, never retuned, applied to categories they were not fitted on.
for name, group in (("dev", DEV_CATS), ("held-out", HELDOUT_CATS)):
    rows  = [r for r in RESULTS if r["cat"] in group]
    crops = [e for e in CROPS if e["cat"] in group]
    bank  = sum(len(BANK.get(c, [])) for c in group)
    if rows:
        print(f"{name:>10s}: {len(rows):3d} pairs -> {bank:3d} banked "
              f"({bank / max(len(rows), 1):.2f} entries/pair, "
              f"{len(crops)} crops)")
