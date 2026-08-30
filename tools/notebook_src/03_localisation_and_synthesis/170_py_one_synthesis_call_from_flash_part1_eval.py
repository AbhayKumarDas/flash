# --- One synthesis call (from flash_part1, evaluation fields removed) ---------------
def synthesize(cat, host_path, entry, seed=SEED, arm="poisson", collar_frac=COLLAR_FRAC):
    """I_h + entry -> I_syn, M_gt. Returns None when the sample cannot be made.

    The RNG draws that pick the blob, the rotation, the flip and the C/B mode depend ONLY on
    the key string, so the corpus is reproducible: same seed -> same image, byte for byte."""
    host, region0 = host_and_region(host_path)
    _, H, W = host.shape
    if float(region0.mean()) < MIN_OBJ_COV_CAT[cat]:
        return None
    region, blob = mrsp_mask_for(host, seed=seed, region=region0)
    if blob.sum() < MIN_BLOB_PX:
        return None
    _, centre = mrsp_blob_stats(blob)
    if centre is None:
        return None

    # crc32 of the key string, NOT hash(): str hashing is salted per process unless
    # PYTHONHASHSEED is set, so hash() would give a different corpus on every session.
    key = f"{seed}|{cat}|{os.path.basename(host_path)}|{entry['key']}".encode()
    g = random.Random(zlib.crc32(key))
    theta = g.uniform(0, 360) if ROTATE else 0.0
    flip  = g.random() < 0.5
    long_px = target_long_px(entry, H, W)

    mode = "B" if g.random() < PLACEMENT_MIX else "C"
    gamma = g.uniform(*GAMMA_RANGE)
    if mode == "B":
        centre, long_px, theta, contain = place_mode_B(host, entry, blob, region, centre,
                                                       long_px, theta, flip, gamma)
    else:
        contain = float("nan")

    patch, alpha, box = place_entry(host, entry, centre, long_px, theta, flip, region)
    if (alpha > 0.5).sum() < 20:
        return None
    ph = harmonise(patch, alpha, host, box)

    # Below R_EQ_POISSON the collar takes more of the defect than it can spare and NORMAL_CLONE
    # rebuilds the interior from substrate -- a dissolved sample is a normal image carrying a
    # defect label. Those composite with alpha instead. MODE B ONLY.
    area_px = float((alpha > 0.5).float().sum())
    r_eq_px = math.sqrt(area_px / math.pi)
    use_arm = arm
    if mode == "B" and arm == "poisson" and r_eq_px < R_EQ_POISSON:
        use_arm = "alpha"
    out, info = composite(use_arm, host, ph, alpha, region,
                          **({"frac": collar_frac} if use_arm == "poisson" else {}))

    # GT is the warped DEFECT mask. Never the MRSP blob, and never the dilated collar mask:
    # the collar is a compositing device, and writing it to GT would inflate every target.
    gt = ((alpha > 0.5).float() * region)
    return dict(cat=cat, host_path=host_path, entry=entry["key"], seed=seed,
                mode=mode, gamma=gamma if mode == "B" else float("nan"),
                theta=theta, flip=flip, contain=contain, r_eq=r_eq_px,
                arm_used=use_arm, collar_px=info.get("collar_px", 0),
                blob_px=float(blob.sum()), gt_px=float((gt > 0.5).sum()),
                # host is returned because d_in measures |out - host| inside the GT, which is
                # the one number that says whether the composite carries a findable defect.
                host=host, out=out, gt=gt)

print("synthesize ready.")
