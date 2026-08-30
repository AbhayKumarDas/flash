# --- Empty-frame gate, measured per category ---------------------------------------
# On an empty frame OBS returns the tray, so the placement is on-region and still nonsense
# (a shell cavity floating on a conveyor). The gate is RELATIVE to the category's own median.
MIN_OBJ_COV_CAT, HOST_PATHS = {}, {}
_t0 = time.time()
for c in LIVE_CATS:
    covs = [(hp, float(host_and_region(hp)[1].mean())) for hp in HOST_POOL[c]]
    med = float(np.median([v for _, v in covs])) if covs else 0.0
    thr = max(MIN_OBJ_COV_ABS, MIN_OBJ_COV_REL * med)
    MIN_OBJ_COV_CAT[c] = thr
    HOST_PATHS[c] = [hp for hp, v in covs if v >= thr]
    dead = [(hp, v) for hp, v in covs if v < thr]
    print(f"  {c:14s} median cov {med:.3f}  gate {thr:.3f}  -> {len(HOST_PATHS[c])} hosts"
          + (f"  (dropped {len(dead)} empty)" if dead else ""))
print(f"\nOBS over {sum(len(v) for v in HOST_POOL.values())} hosts in {time.time()-_t0:.1f}s "
      f"(cached; not recomputed per seed)")
