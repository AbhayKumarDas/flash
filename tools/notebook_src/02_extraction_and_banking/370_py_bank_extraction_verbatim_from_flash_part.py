# --- Bank extraction (verbatim from flash_part1) ------------------------------------
def extract_entry(donor_id, region, ctx=CTX_EXPAND, soften=ALPHA_SOFTEN):
    """Substrate-retained square crop + soft defect alpha + scale metadata."""
    d = RAW[donor_id]
    anom = d["anomaly"]; m = region["bin"]
    H, W = anom.shape[:2]
    dh, dw = region["h"], region["w"]
    side = int(np.clip(round(max(dh, dw) * ctx), 16, min(H, W)))
    cy, cx = region["y"] + dh // 2, region["x"] + dw // 2
    ty = int(np.clip(cy - side // 2, 0, H - side))
    tx = int(np.clip(cx - side // 2, 0, W - side))

    rgb   = anom[ty:ty + side, tx:tx + side].copy()
    alpha = m[ty:ty + side, tx:tx + side].astype(np.float32).copy()
    if soften > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), soften)
    alpha = np.clip(alpha, 0, 1)

    # Internal contrast: how far the masked pixels sit from this crop's own substrate, in the
    # same Lab units as DEFECT_T. An entry whose mask covers plain substrate scores ~0 here, and
    # it is worse than useless -- replayed, it produces an image whose ground truth labels a
    # region where nothing changed, so every metric is scored against a defect that is not there.
    # That is what a mis-recovered diffmask looks like, and it has to be caught before the bank.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dsel, ssel = alpha > 0.5, alpha < 0.1
    if dsel.sum() >= 25 and ssel.sum() >= 25:
        ref  = lab[ssel].mean(0)
        dist = np.linalg.norm(lab[dsel] - ref, axis=1)
        contrast = float(np.median(dist))
        cfrac    = float((dist > DEFECT_T).mean())
    else:
        contrast, cfrac = 0.0, 0.0

    return dict(
        key=f"{donor_id}_{region['x']}_{region['y']}", donor=donor_id, cat=d["cat"],
        rgb=rgb, alpha=alpha, side=side,
        defect_long_px=int(max(dh, dw)),
        defect_frac=float(max(dh, dw) / max(H, W)),
        area_px=region["area"], area_frac=float(region["area"] / m.size),
        contrast=contrast, cfrac=cfrac,
        kind=DEFECT_KIND[d["cat"]],
    )


CROPS = []
for i in ids:
    for r in KEPT[i]:
        e = extract_entry(i, r)
        e["_region"] = r                 # kept for section 7's panels; not serialised
        e["substrate_frac"] = float((e["alpha"] < 0.1).mean())
        e["diffmask_secs"] = float(RAW[i]["secs"])
        e["flags"] = " ".join(FLAGS)
        CROPS.append(e)

# Rank each donor's regions by area so VLM-2 can be told "this is region k of n". The rank is
# a prior, not a decision -- the largest component is usually but not always the real defect.
_by_donor = {}
for e in CROPS:
    _by_donor.setdefault(e["donor"], []).append(e)
for _d, _rs in _by_donor.items():
    _rs.sort(key=lambda e: -e["area_px"])
    for _k, _e in enumerate(_rs):
        _e["n_regions"] = len(_rs)
        _e["region_rank"] = _k + 1

_multi = {d: r for d, r in _by_donor.items() if len(r) > 1}
print(f"{len(CROPS)} candidate crops from {len(ids)} masks")
if _multi:
    print(f"{len(_multi)} donor(s) returned more than one region -- at most one can be the "
          f"introduced defect:")
    for d, rs in sorted(_multi.items()):
        print(f"  {d:22s} " + "  ".join(f"{e['key'].split('_',1)[1]}({e['area_px']}px)"
                                        for e in rs))
print()
print(f"{'entry key':30s} {'cat':13s} {'crop':>9s} {'defect':>8s} {'frac':>7s}"
      f" {'subst%':>7s} {'contrast':>9s} {'cfrac':>6s}")
for e in CROPS:
    print(f"{e['key']:30s} {e['cat']:13s} {e['side']:7d}px {e['defect_long_px']:6d}px"
          f" {e['defect_frac']:7.4f} {100*e['substrate_frac']:6.1f}%"
          f" {e['contrast']:9.1f} {e['cfrac']:6.2f}")
