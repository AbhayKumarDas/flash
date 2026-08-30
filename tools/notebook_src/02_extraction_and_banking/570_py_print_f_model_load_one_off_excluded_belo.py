print(f"model load (one-off, excluded below) : {LOAD_SECONDS:7.1f}s\n")

print(f"{'category':<14s}{'pairs':>7s}{'dm total':>10s}{'dm mean':>9s}"
      f"{'crops':>7s}{'vlm total':>11s}{'vlm mean':>10s}")
print("-" * 68)
_t = dict(pairs=0, dm=0.0, crops=0, vlm=0.0)
for c in sorted({r["cat"] for r in RESULTS}):
    rows  = [r for r in RESULTS if r["cat"] == c]
    crops = [e for e in CROPS if e["cat"] == c]
    dm  = sum(r["secs"] for r in rows)
    vl  = sum(e["vlm"]["secs"] for e in crops)
    _t["pairs"] += len(rows); _t["dm"] += dm
    _t["crops"] += len(crops); _t["vlm"] += vl
    print(f"{c:<14s}{len(rows):>7d}{dm:>10.1f}{dm / max(len(rows), 1):>9.1f}"
          f"{len(crops):>7d}{vl:>11.1f}{vl / max(len(crops), 1):>10.1f}")
print("-" * 68)
print(f"{'ALL':<14s}{_t['pairs']:>7d}{_t['dm']:>10.1f}{_t['dm'] / max(_t['pairs'], 1):>9.1f}"
      f"{_t['crops']:>7d}{_t['vlm']:>11.1f}{_t['vlm'] / max(_t['crops'], 1):>10.1f}")

# Warm figure: the first VLM call pays CUDA warmup that nothing after it does.
_warm = [e["vlm"]["secs"] for e in CROPS if not e["vlm"]["warmup"]]
_cold = [e["vlm"]["secs"] for e in CROPS if e["vlm"]["warmup"]]
print(f"\nStage 2  DiffMask  {_t['dm']:8.1f}s  over {_t['pairs']:3d} pairs"
      f"   {_t['dm'] / max(_t['pairs'], 1):6.2f}s/pair")
print(f"Stage 3  VLM-2     {_t['vlm']:8.1f}s  over {_t['crops']:3d} crops"
      f"   {_t['vlm'] / max(_t['crops'], 1):6.2f}s/crop")
if _warm:
    _cold_note = f"   (cold first call {_cold[0]:.1f}s)" if _cold else ""
    print(f"         VLM-2 warm  {len(_warm):8d} crops"
          f"   {np.mean(_warm):6.2f}s/crop  median {np.median(_warm):.2f}s{_cold_note}")
print(f"{'':9s}{'-' * 44}")
print(f"Stages 2-3 total   {_t['dm'] + _t['vlm']:8.1f}s"
      f"   (+ {LOAD_SECONDS:.0f}s one-off model load)")

_banked = sum(len(v) for v in BANK.values())
if _banked:
    print(f"\nAmortised cost per BANKED entry : "
          f"{(_t['dm'] + _t['vlm']) / _banked:.1f}s")
    print("Per banked entry, not per crop -- rejected crops cost time and yield nothing, so")
    print("they belong in the numerator. This is the number the paper should quote.")
