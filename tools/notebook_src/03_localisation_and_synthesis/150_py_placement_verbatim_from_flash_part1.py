# --- Placement (verbatim from flash_part1) ------------------------------------------
def target_long_px(entry, H, W):
    """real_scale: the defect replays at its true relative size."""
    return entry["defect_frac"] * max(H, W)

def place_entry(host, entry, center, long_px, theta_deg, flip, region):
    """Resize + optional flip/rotate the entry, stamp into a host-sized canvas initialised
    to the host, so the patch's own substrate is present for the band solve to work against."""
    _, H, W = host.shape
    rgb, a = entry["rgb"], entry["alpha"]
    if flip:
        rgb, a = rgb[:, ::-1].copy(), a[:, ::-1].copy()

    r = long_px / max(entry["defect_long_px"], 1)
    s = int(np.clip(round(entry["side"] * r), 8, max(H, W)))
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LANCZOS4
    rgb = cv2.resize(rgb, (s, s), interpolation=interp)
    a   = cv2.resize(a,   (s, s), interpolation=cv2.INTER_LINEAR)

    if theta_deg:
        p = int(math.ceil(s * (math.sqrt(2) - 1) / 2)) + 2
        rgb = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_REFLECT_101)
        a   = cv2.copyMakeBorder(a,   p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        s2 = rgb.shape[0]
        M = cv2.getRotationMatrix2D((s2 / 2, s2 / 2), theta_deg, 1.0)
        rgb = cv2.warpAffine(rgb, M, (s2, s2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        a   = cv2.warpAffine(a,   M, (s2, s2), flags=cv2.INTER_LINEAR, borderValue=0)
        s = s2

    cy, cx = center
    y0 = int(np.clip(cy - s // 2, 0, max(0, H - s))); x0 = int(np.clip(cx - s // 2, 0, max(0, W - s)))
    y1, x1 = min(H, y0 + s), min(W, x0 + s)
    sy, sx = y1 - y0, x1 - x0

    patch = host.clone()
    alpha = torch.zeros(H, W)
    pt = torch.from_numpy(rgb[:sy, :sx].astype(np.float32) / 255.0).permute(2, 0, 1)
    patch[:, y0:y1, x0:x1] = pt
    alpha[y0:y1, x0:x1] = torch.from_numpy(np.clip(a[:sy, :sx], 0, 1).astype(np.float32))
    alpha = alpha * region                     # never place off-object
    return patch, alpha, (y0, x0, sy, sx)


# --- Harmonisation (verbatim from flash_part1) --------------------------------------
def harmonise(patch, alpha, host, box, std_clip=(0.75, 1.35)):
    y0, x0, sy, sx = box
    _, H, W = host.shape
    a = alpha.numpy()
    src_sel = np.zeros((H, W), bool); src_sel[y0:y0 + sy, x0:x0 + sx] = True
    src_sel &= (a < 0.10)
    pad = int(0.6 * max(sy, sx)) + 8
    ry0, ry1 = max(0, y0 - pad), min(H, y0 + sy + pad)
    rx0, rx1 = max(0, x0 - pad), min(W, x0 + sx + pad)
    dst_sel = np.zeros((H, W), bool); dst_sel[ry0:ry1, rx0:rx1] = True
    dst_sel &= (a < 0.10)
    if src_sel.sum() < 50 or dst_sel.sum() < 50:
        return patch
    p8 = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    h8 = (host.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    p_lab = cv2.cvtColor(p8, cv2.COLOR_RGB2LAB).astype(np.float32)
    h_lab = cv2.cvtColor(h8, cv2.COLOR_RGB2LAB).astype(np.float32)
    sm, ss = p_lab[src_sel].mean(0), p_lab[src_sel].std(0) + 1e-6
    dm, ds = h_lab[dst_sel].mean(0), h_lab[dst_sel].std(0) + 1e-6
    gain = np.clip(ds / ss, *std_clip)
    out = np.clip((p_lab - sm) * gain + dm, [0, 0, 0], [255, 255, 255]).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


# --- Mode B: MRSP sets size, orientation, anchor and containment --------------------
def _mask_centroid(a):
    m = a.numpy() > 0.5
    if m.sum() == 0:
        return None
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())

def _principal_angle(binary):
    """major-axis angle in degrees from second central moments; area comes free as m00"""
    m = cv2.moments(binary.astype(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        return None, 0.0
    mu20, mu11, mu02 = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    if abs(mu20 - mu02) < 1e-9 and abs(mu11) < 1e-9:
        return 0.0, m["m00"]
    return math.degrees(0.5 * math.atan2(2.0 * mu11, mu20 - mu02)), m["m00"]

def place_alpha_only(host_shape, entry, center, long_px, theta_deg, flip):
    """The alpha half of place_entry, without the RGB half.

    place_mode_B calls place_entry up to 25 times per sample and throws the RGB patch away every
    time -- it only ever reads the returned alpha. Each of those calls was cloning the 3x1024x1024
    host (12 MB) and resizing + warping the crop's RGB for nothing. This does the identical alpha
    arithmetic and skips both, which is where most of mode B's cost was.

    Output is bit-identical to place_entry(...)[1] with region=ones: same resize interpolation,
    same padding, same rotation matrix, same clipping.
    """
    H, W = host_shape
    a = entry["alpha"]
    if flip:
        a = a[:, ::-1].copy()

    r = long_px / max(entry["defect_long_px"], 1)
    s = int(np.clip(round(entry["side"] * r), 8, max(H, W)))
    a = cv2.resize(a, (s, s), interpolation=cv2.INTER_LINEAR)

    if theta_deg:
        p = int(math.ceil(s * (math.sqrt(2) - 1) / 2)) + 2
        a = cv2.copyMakeBorder(a, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        s2 = a.shape[0]
        M = cv2.getRotationMatrix2D((s2 / 2, s2 / 2), theta_deg, 1.0)
        a = cv2.warpAffine(a, M, (s2, s2), flags=cv2.INTER_LINEAR, borderValue=0)
        s = s2

    cy, cx = center
    y0 = int(np.clip(cy - s // 2, 0, max(0, H - s))); x0 = int(np.clip(cx - s // 2, 0, max(0, W - s)))
    y1, x1 = min(H, y0 + s), min(W, x0 + s)
    sy, sx = y1 - y0, x1 - x0

    alpha = torch.zeros(H, W)
    alpha[y0:y1, x0:x1] = torch.from_numpy(np.clip(a[:sy, :sx], 0, 1).astype(np.float32))
    return alpha


def place_mode_B(host, entry, blob, region, bc, long_px, theta, flip, gamma):
    """-> (centre, long_px, theta, containment). Falls back to mode C's arguments if degenerate."""
    _, H, W = host.shape
    bf = (blob.numpy() > 0.5).astype(np.float32)
    phi_b, area_b = _principal_angle(bf)
    af0 = place_alpha_only((H, W), entry, bc, long_px, 0.0, flip)
    phi_m, area_m = _principal_angle((af0.numpy() > 0.5).astype(np.uint8))
    if phi_b is None or phi_m is None or area_m <= 0 or area_b <= 0:
        return bc, long_px, theta, float("nan")

    # SIZE: the defect takes `gamma` of the blob; the rest is margin the collar can use.
    s = float(np.clip(math.sqrt(max(gamma * area_b, 1.0) / area_m), *GAMMA_S_RANGE))

    # ANCHOR: mask centroid on blob centroid. A bent sliver (can: 322x39) has its centroid OFF
    # its own mask, and a ragged blob can have its centroid outside itself -- in either case use
    # the deepest interior point, which is guaranteed to be inside.
    byi, bxi = int(np.clip(bc[0], 0, H - 1)), int(np.clip(bc[1], 0, W - 1))
    if bf[byi, bxi] > 0.5:
        anchor = (float(bc[0]), float(bc[1]))
    else:
        dt = cv2.distanceTransform((bf > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        yx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        anchor = (float(yx[0]), float(yx[1]))

    d = phi_b - phi_m                       # ORIENTATION: align the two principal axes
    best = None
    for _ in range(3):                      # CONTAINMENT: shrink until it fits
        lp = long_px * s
        for th in (d, -d, d + 180.0, -d + 180.0):
            th %= 360.0
            afp = place_alpha_only((H, W), entry, (H // 2, W // 2), lp, th, flip)
            mc = _mask_centroid(afp)
            if mc is None:
                continue
            mb = (afp.numpy() > 0.5).astype(np.uint8)
            ay, ax = mc
            if mb[int(np.clip(ay, 0, H - 1)), int(np.clip(ax, 0, W - 1))] == 0:
                mdt = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
                ay, ax = np.unravel_index(int(np.argmax(mdt)), mdt.shape)
            c = (int(H // 2 + (anchor[0] - ay)), int(W // 2 + (anchor[1] - ax)))
            af = place_alpha_only((H, W), entry, c, lp, th, flip)
            mm = af.numpy() > 0.5
            ct = float((mm & (bf > 0.5)).sum()) / max(mm.sum(), 1)
            if best is None or ct > best[0]:
                best = (ct, c, lp, th)
        if best and best[0] >= CONTAIN_T:
            break
        s = float(np.clip(s * 0.85, *GAMMA_S_RANGE))
    if best is None:
        return bc, long_px, theta, float("nan")
    ct, c, lp, th = best
    return c, lp, th, ct

print("placement + harmonisation ready (mode C + mode B).")
