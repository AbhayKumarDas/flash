# --- Compositing (verbatim from flash_part1) ----------------------------------------
def feather_mask(mask, sigma):
    return torch.from_numpy(cv2.GaussianBlur(mask.numpy().astype(np.float32), (0, 0), sigma)).clamp(0, 1)

def feather_sigma_for(mask, base_sigma=FEATHER_PX):
    """Cap the feather so it cannot fade the defect: a Gaussian with sigma comparable to the
    blob radius pulls the CORE below opacity 1.0, which is the bug this whole notebook avoids."""
    side, _ = mrsp_blob_stats(mask)
    if side <= 0:
        return float(base_sigma)
    return float(min(base_sigma, max(1.0, 0.25 * side)))

def alpha_composite(host, patch, alpha, region, beta=BETA, base_sigma=FEATHER_PX):
    sigma = feather_sigma_for(alpha, base_sigma)
    mf = (feather_mask(alpha, sigma) * region).unsqueeze(0)
    out = (host * (1 - mf) + (beta * patch + (1 - beta) * host) * mf).clamp(0, 1)
    # edit_mask = the region this operator actually modified. The seam metric is evaluated at
    # each arm's OWN boundary: the collar makes the Poisson arm edit a larger area, and scoring
    # both at the original mask would credit alpha for pixels it never touched.
    edit = ((mf[0] > 0.01).float().numpy() > 0).astype(np.uint8)
    return out, dict(sigma=sigma, edit_mask=edit)


def lab_of(t):
    return cv2.cvtColor((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB).astype(np.float32)

def defect_thresholds(host, patch, mask_bin, ring_px=4):
    """Per-composite Lab cutoffs separating "defect" from "substrate".

    A fixed cutoff does not survive contact with a granular category. On rice or walnuts the
    patch's grains never line up with the host's, so ||patch - host|| is large across the whole
    crop and a fixed 6-unit cutoff calls the entire substrate a defect. The cutoffs are
    therefore taken from the composite's own distribution: the substrate noise floor is
    measured on the ring OUTSIDE the mask, where by definition there is no defect, and the
    defect cutoff is placed above it.
    """
    k = np.ones((3, 3), np.uint8)
    m = (mask_bin.numpy() > 0.5).astype(np.uint8)
    outer = (cv2.dilate(m, k, iterations=3 * ring_px) - cv2.dilate(m, k, iterations=ring_px)) > 0
    d = np.linalg.norm(lab_of(patch) - lab_of(host), axis=2)
    if outer.sum() < 50:                       # degenerate placement; fall back to the constants
        return DEFECT_T, 4.0, float("nan")
    floor = float(np.quantile(d[outer], 0.90))  # what "same material, misaligned" looks like
    t_def = max(DEFECT_T, floor)                # defect must beat the misalignment floor
    t_sub = max(4.0, float(np.quantile(d[outer], 0.50)))
    return t_def, t_sub, floor

def defect_core(host, patch, mask_bin, T=None):
    """Pixels inside the mask that actually carry the defect. A recovered mask also contains a
    substrate margin, and the difference between the two is what the collar has to clear."""
    m = (mask_bin.numpy() > 0.5).astype(np.uint8)
    if T is None:
        T, _, _ = defect_thresholds(host, patch, mask_bin)
    d = np.linalg.norm(lab_of(patch) - lab_of(host), axis=2)
    k = np.ones((3, 3), np.uint8)
    c = ((d > T) & (m > 0)).astype(np.uint8)
    c = cv2.morphologyEx(c, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(c, cv2.MORPH_CLOSE, k, iterations=2)

def collar_for(core_np, frac=COLLAR_FRAC, min_px=COLLAR_MIN_PX, keep=COLLAR_KEEP_R):
    """Collar width in px, from the defect's equivalent radius. frac=0 means NO collar, and
    the floor does not apply -- otherwise the control sweep could never reach the vanilla
    NORMAL_CLONE it exists to compare against.

    COLLAR_MIN_PX is a floor meant to HELP small defects get a workable boundary, but on a
    defect smaller than ~2.5x the floor it does the opposite: a 6px collar on a defect of
    equivalent radius 9.7px takes 62% of the radius, and NORMAL_CLONE then rebuilds what is
    left from substrate. The cap keeps at least `keep` of the radius as defect, so the floor
    can never be the thing that dissolves the defect it was added to protect.
    """
    if frac <= 0:
        return 0
    a = float(core_np.sum())
    if a < 1:
        return int(min_px)
    r_eq = math.sqrt(a / math.pi)
    w = max(min_px, round(frac * r_eq))
    return int(max(1, min(w, math.floor((1.0 - keep) * r_eq))))

def poisson_clone(host, patch, mask_np):
    """cv2 NORMAL_CLONE on an explicit mask. Returns None if the mask is unusable."""
    mk = (mask_np > 0).astype(np.uint8) * 255
    mk[:3, :] = 0; mk[-3:, :] = 0; mk[:, :3] = 0; mk[:, -3:] = 0
    ys, xs = np.where(mk > 0)
    if len(xs) < 10:
        return None
    c = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    d = (host.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    s = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    try:
        out = cv2.seamlessClone(s, d, mk, c, cv2.NORMAL_CLONE)
    except cv2.error:
        return None
    return torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1).clamp(0, 1)

def collar_poisson(host, patch, alpha, region, frac=COLLAR_FRAC, min_px=COLLAR_MIN_PX):
    """Dilate the defect mask by a substrate collar, then solve. The collar moves the Dirichlet
    boundary off the defect and onto substrate, which is the whole mechanism."""
    hard = ((alpha > 0.5).float() * region)
    core_np = defect_core(host, patch, hard)
    if core_np.sum() < 10:                       # low-contrast defect: no content core to clear
        core_np = (hard.numpy() > 0.5).astype(np.uint8)
    w = collar_for(core_np, frac, min_px)
    k = np.ones((3, 3), np.uint8)
    comp = cv2.dilate((hard.numpy() > 0.5).astype(np.uint8), k, iterations=w) if w > 0 \
        else (hard.numpy() > 0.5).astype(np.uint8)
    comp = (comp * (region.numpy() > 0.5)).astype(np.uint8)   # never composite off-object
    info = dict(collar_px=w, core_px=int(core_np.sum()), comp_px=int(comp.sum()), fallback=None)

    info["edit_mask"] = comp
    out = poisson_clone(host, patch, comp)
    if out is None:
        info["fallback"] = "alpha (seamlessClone declined the mask)"
        out, ai = alpha_composite(host, patch, alpha, region)
        info["edit_mask"] = ai["edit_mask"]
    return out, info

def composite(arm, host, patch, alpha, region, **kw):
    if arm == "alpha":
        return alpha_composite(host, patch, alpha, region, **kw)
    if arm == "poisson":
        return collar_poisson(host, patch, alpha, region, **kw)
    raise KeyError(arm)

print("compositing operators ready.")
