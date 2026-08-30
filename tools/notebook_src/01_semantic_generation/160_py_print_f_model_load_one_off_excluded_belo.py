print(f"model load (one-off, excluded below) : {LOAD_SECONDS:7.1f}s\n")
print(f"{'category':<14s}{'images':>8s}{'total s':>10s}{'mean s':>9s}"
      f"{'min s':>8s}{'max s':>8s}{'retries':>9s}")
print("-" * 66)

tot_s = tot_n = tot_r = 0
for cat in CATS:
    rs = [r for r in rows if r["category"] == cat]
    if not rs:
        continue
    ss = [r["gen_seconds"] for r in rs]
    r_ = sum(r["retries"] for r in rs)
    tot_s, tot_n, tot_r = tot_s + sum(ss), tot_n + len(rs), tot_r + r_
    print(f"{cat:<14s}{len(rs):>8d}{sum(ss):>10.1f}{sum(ss)/len(ss):>9.1f}"
          f"{min(ss):>8.1f}{max(ss):>8.1f}{r_:>9d}")

print("-" * 66)
print(f"{'ALL':<14s}{tot_n:>8d}{tot_s:>10.1f}{tot_s/max(tot_n,1):>9.1f}"
      f"{'':>8s}{'':>8s}{tot_r:>9d}")
print(f"\nmean per image (VLM generate only) : {tot_s/max(tot_n,1):.1f}s")
print(f"wall clock for the whole run       : {RUN_SECONDS:.1f}s")
