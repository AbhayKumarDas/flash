# --- Host pool: donors removed -------------------------------------------------------
# The manifest stores donor as "<cat>-<id>" ("fruit_jelly-064") because ids collide across
# categories, but the normals on disk are named "064_regular.png". Comparing the two directly
# never matches, so the prefix comes off first -- and the drop count is printed, because a
# filter that silently removes nothing is worse than no filter.
HOST_POOL = {}
for c in LIVE_CATS:
    donors = set()
    for e in BANK[c]:
        d = str(e.get("donor", ""))
        donors.add(d[len(c) + 1:] if d.startswith(c + "-") else d)
    stems = set()
    for i in donors:
        stems |= {i, f"{i}_regular", (i.lstrip("0") or "0")}

    pool = _imgs_in(os.path.join(AD2_ROOT, c, "train", "good"))
    kept = [p for p in pool if os.path.splitext(os.path.basename(p))[0] not in stems]
    n_drop = len(pool) - len(kept)
    HOST_POOL[c] = kept[:N_HOSTS]
    print(f"  {c:14s} {len(pool):3d} normals - {n_drop} donor(s) = {len(kept):3d}"
          f" -> {len(HOST_POOL[c])} hosts")
    if n_drop == 0 and donors:
        print(f"      WARNING: no donor matched a filename in train/good "
              f"(ids {sorted(donors)[:3]}) -- a donor may also be used as a host")
    if len(HOST_POOL[c]) < N_SYNTH:
        print(f"      WARNING: fewer hosts than N_SYNTH={N_SYNTH}; hosts will be reused")
