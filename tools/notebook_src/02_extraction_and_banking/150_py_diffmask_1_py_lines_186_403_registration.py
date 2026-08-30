# ============================================================================
# diffmask (1).py lines 186-403 -- registration.
#
# Alg.1 step 1: coarse_similarity, a scale-translation template search on edge maps.
# Alg.1 step 2: ecc_refine, then feature_similarity as the rotation-capable challenger.
#
# register() ties them together. The coarse+ECC path is what runs whenever it works, which is the
# common case; the rotation paths are tried only when align_score comes back below --rot-trigger,
# and a challenger must win by --rot-margin to replace the incumbent.
# ============================================================================
# ------------------------------------------------------------------ registration


def coarse_similarity(ref_gray, test_gray, work=224, smin=0.30, smax=3.0, steps=37):
    """
    Brute force scale + translation search of reference against defect.

    Returns (s, tx, ty, ncc) describing  p_test = s * p_ref + (tx, ty).
    """
    ft = fit_long_side(test_gray.shape, work)
    t_small = edge_map(resize_f(test_gray, ft))
    r_base = edge_map(resize_f(ref_gray, ft))

    best = (1.0, 0.0, 0.0, -2.0)
    for s in np.geomspace(smin, smax, steps):
        tmpl = resize_f(r_base, float(s))
        th, tw = tmpl.shape[:2]
        if th < 8 or tw < 8 or th > 4000 or tw > 4000:
            continue
        py, px = th // 3, tw // 3  # allow overhang, but demand real overlap
        padded = cv2.copyMakeBorder(t_small, py, py, px, px, cv2.BORDER_CONSTANT, value=0.0)
        if padded.shape[0] < th or padded.shape[1] < tw:
            continue
        res = cv2.matchTemplate(padded, tmpl, cv2.TM_CCOEFF_NORMED)

        # A featureless template sitting on featureless padding correlates
        # perfectly and means nothing, so require the matched window to carry
        # a comparable amount of edge energy to the template itself.
        tmpl_energy = float(tmpl.mean())
        if tmpl_energy < 1e-4:
            continue
        ii = cv2.integral(padded)
        win = (ii[th:, tw:] - ii[:-th, tw:] - ii[th:, :-tw] + ii[:-th, :-tw]) / float(th * tw)
        res = np.where(win >= 0.35 * tmpl_energy, res, -1.0).astype(np.float32)

        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[3]:
            best = (float(s), (loc[0] - px) / ft, (loc[1] - py) / ft, float(mx))
    return best


def similarity_to_warp(s, tx, ty) -> np.ndarray:
    """Invert p_test = s*p_ref + t into a test -> ref warp."""
    inv = 1.0 / max(s, 1e-6)
    return np.array([[inv, 0.0, -tx * inv], [0.0, inv, -ty * inv]], np.float32)


def ecc_refine(ref_gray, test_gray, W, homography=False, work=640,
               levels=(0.25, 0.5, 1.0), iters=100, eps=1e-6):
    """
    Coarse-to-fine ECC on band-passed images, so exposure differences are
    irrelevant. W maps test -> ref, expressed at full resolution, in and out.
    """
    mode = cv2.MOTION_HOMOGRAPHY if homography else cv2.MOTION_AFFINE
    if homography:
        W = to3x3(W)

    base_t = fit_long_side(test_gray.shape, work)
    base_r = fit_long_side(ref_gray.shape, work * 2)

    for lvl in levels:
        ft, fr = base_t * lvl, base_r * lvl
        t_s = edge_map(resize_f(test_gray, ft), 1.2)
        r_s = edge_map(resize_f(ref_gray, fr), 1.2)
        if min(t_s.shape[:2]) < 32 or min(r_s.shape[:2]) < 32:
            continue
        Wl = np.ascontiguousarray(rewarp(W, ft, fr), dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        try:
            _, Wl = cv2.findTransformECC(t_s, r_s, Wl, mode, crit, None, 5)
        except cv2.error:
            continue  # diverged at this level, keep what we had
        cand = rewarp(Wl, 1.0 / ft, 1.0 / fr)
        if np.isfinite(cand).all():
            W = cand.astype(np.float32)
    return W


def align_score(ref_gray, test_gray, W, work=256, min_cov=0.30):
    """
    How well a warp actually lines the pair up, as one number.

    Edge correlation over the overlap, so exposure is irrelevant, discounted by
    the overlap itself: a warp that keeps only a corner of the frame can
    correlate perfectly and still be wrong. Cheap enough (a few ms) to act as
    the referee between competing registrations.
    """
    if W is None or not np.isfinite(W).all():
        return 0.0
    f = fit_long_side(test_gray.shape, work)
    t_s = resize_f(test_gray, f)
    h, w = t_s.shape[:2]
    Ww = rewarp(W, f, 1.0)
    if not np.isfinite(Ww).all():
        return 0.0
    r_a = warp_into(ref_gray, Ww, (w, h))
    cov = warp_into(np.full(ref_gray.shape[:2], 255, np.uint8), Ww, (w, h))
    v = cv2.erode(cov, np.ones((5, 5), np.uint8)) > 0
    if int(v.sum()) < 500:
        return 0.0
    ea = edge_map(r_a)[v].ravel()
    eb = edge_map(t_s)[v].ravel()
    c = float(np.corrcoef(ea, eb)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return max(0.0, c) * min(1.0, float(v.mean()) / max(min_cov, 1e-6))


def feature_similarity(ref_gray, test_gray, work=768, nfeat=3000, ratio=0.78):
    """
    Rotation-capable registration from ORB correspondences and RANSAC.

    The coarse search covers scale and translation only, by construction: it
    slides a template. Rotation needs either an extra search dimension or a
    representation that does not care, and matched keypoints are the cheap
    version of the latter, because an ORB descriptor is already rotation
    invariant. Fitted as a 4-dof similarity rather than a full affine, since
    that is the transform the pair actually differs by and the extra freedom
    only buys RANSAC ways to be confidently wrong.

    Returns (W test -> ref at full resolution, inlier count), or (None, 0).
    """
    ft = fit_long_side(test_gray.shape, work)
    fr = fit_long_side(ref_gray.shape, work)
    a = np.clip(resize_f(test_gray, ft) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(resize_f(ref_gray, fr) * 255.0, 0, 255).astype(np.uint8)
    if min(a.shape[:2]) < 40 or min(b.shape[:2]) < 40:
        return None, 0

    # Local contrast equalisation first: the pair differs in exposure, and FAST
    # corner detection is a plain intensity threshold.
    clahe = cv2.createCLAHE(2.5, (8, 8))
    a, b = clahe.apply(a), clahe.apply(b)

    orb = cv2.ORB_create(nfeat, scaleFactor=1.15, nlevels=14, fastThreshold=7,
                         edgeThreshold=19, patchSize=31)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pr in bf.knnMatch(da, db, k=2):
        if len(pr) == 2 and pr[0].distance < ratio * pr[1].distance:
            good.append(pr[0])
    if len(good) < 12:
        return None, 0

    pa = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=4000,
                                         confidence=0.995, refineIters=20)
    if M is None or not np.isfinite(M).all():
        return None, 0
    n_in = int(inl.sum()) if inl is not None else 0
    if n_in < 10:
        return None, 0
    # Reject a degenerate fit outright rather than letting ECC chase it.
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.15 < sc < 6.0):
        return None, 0
    return rewarp(M.astype(np.float32), 1.0 / ft, 1.0 / fr), n_in


def register(ref_gray, test_gray, args):
    """
    Full registration, with a rotation-capable fallback.

    The coarse-plus-ECC path stays exactly as it was and is still what runs
    whenever it works, which is the common case. Only when its own alignment
    score comes back poor are the rotation-capable paths tried, and an
    alternative has to beat the incumbent by a margin to replace it, so a pair
    that was already registered correctly cannot be talked out of it.
    """
    s, tx, ty, cscore = coarse_similarity(ref_gray, test_gray)
    W = similarity_to_warp(s, tx, ty)
    if not args.no_ecc:
        W = ecc_refine(ref_gray, test_gray, W, homography=args.homography,
                       work=args.ecc_work)
    if args.no_rot_search:
        return W, s, cscore, "coarse", -1.0

    a_best = align_score(ref_gray, test_gray, W)
    if a_best >= args.rot_trigger:
        return W, s, cscore, "coarse", a_best

    def consider(Wc, tag, best):
        if Wc is None:
            return best
        if args.homography:
            Wc = to3x3(Wc)
        cands = [Wc]
        if not args.no_ecc:
            try:
                cands.append(ecc_refine(ref_gray, test_gray, Wc,
                                        homography=args.homography, work=args.ecc_work))
            except cv2.error:
                pass
        for c in cands:
            a = align_score(ref_gray, test_gray, c)
            if a > best[0] + args.rot_margin:
                best = (a, c, tag)
        return best

    best = (a_best, W, "coarse")
    Wf, _ = feature_similarity(ref_gray, test_gray)
    best = consider(Wf, "orb", best)
    # Report the scale of the warp that won. The coarse estimate is the number a
    # user reads to sanity check registration, so it must not describe a warp
    # that was discarded.
    Wb = best[1]
    s_eff = s if best[2] == "coarse" else float(
        1.0 / max(np.sqrt(abs(np.linalg.det(to3x3(Wb)[:2, :2]))), 1e-9))
    return best[1], s_eff, cscore, best[2], best[0]
