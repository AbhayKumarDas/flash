# ============================================================================
# diffmask (1).py lines 433-579 -- photometric fitting and the first residual terms.
#
# Alg.1 step 4: global_match / photometric_fit build the matched reference; band_residual turns
# the comparison into a tolerance band so sub-pixel misregistration costs nothing.
# Alg.1 step 5 (part): low_freq_residual and low_freq_disagreement are the low-frequency
# appearance cue, gated behind --w-low / --auto-low.
#
# --auto-low exists because some defects only show up as a broad appearance shift with no edge:
# image 010's wallplug is the case that forced it.
# ============================================================================
# --------------------------------------------------------------------- differing


def global_match(ref_c, test_c, valid, gain=True, lo=0.5, hi=2.0):
    """
    Match the reference channel to the test channel over the whole frame.

    Unlike the local fit this cannot erase a defect, however large, because a
    single gain and offset has nowhere to hide one. Robust statistics keep the
    defect itself from dragging the fit.
    """
    mr, sr = robust_scale(ref_c, valid, 1e-3)
    mt, st = robust_scale(test_c, valid, 1e-3)
    g = float(np.clip(st / (sr + 1e-6), lo, hi)) if gain else 1.0
    return (ref_c - mr) * g + mt


def photometric_fit(ref_c, test_c, win, gain=True, lo=0.5, hi=2.0, rng=None, w=None):
    """
    Match the reference channel to the test channel on a local window.

    Note this necessarily erases any difference that is smooth across the
    window, a large flat object included. That is why the caller also keeps a
    globally matched copy and scores the low frequencies separately.
    """
    if w is None:
        mr, mt = box(ref_c, win), box(test_c, win)
    else:
        # Only pixels the two frames actually share may inform the fit. Without
        # this the window reaches past the overlap, and past the frame edge,
        # where the box filter reflects content back in: the two images then get
        # windows holding different scenes and the fit answers a question nobody
        # asked. That is why the outermost band of a re-render lights up.
        den = box(w, win) + EPS
        mr, mt = box(ref_c * w, win) / den, box(test_c * w, win) / den
    if not gain:
        out = ref_c - mr + mt
    else:
        if w is None:
            vr = np.maximum(box(ref_c * ref_c, win) - mr * mr, 0.0)
            vt = np.maximum(box(test_c * test_c, win) - mt * mt, 0.0)
        else:
            vr = np.maximum(box(ref_c * ref_c * w, win) / den - mr * mr, 0.0)
            vt = np.maximum(box(test_c * test_c * w, win) / den - mt * mt, 0.0)
        g = np.clip(np.sqrt((vt + 0.25) / (vr + 0.25)), lo, hi)
        out = mt + g * (ref_c - mr)
    # The fit is estimated from misaligned content, so near a strong edge it can
    # predict a value the channel cannot physically take. The test image then
    # disagrees with the prediction by the whole overshoot and no tolerance band
    # can forgive it, because the fitted reference holds nothing the test could
    # have matched. Clamping to the channel's own range removes that alone.
    if rng is not None:
        out = np.clip(out, rng[0], rng[1])
    return out


def band_residual(ref_c, test_c, radius):
    """
    How far the test channel falls outside the range the reference takes within
    `radius` pixels. Sub-pixel misregistration and small non-rigid drift move a
    value around inside the band and cost nothing, while genuinely new content
    has no nearby reference value to explain it.
    """
    if radius < 1:
        d = test_c - ref_c
        return np.maximum(d, 0.0) + np.maximum(-d, 0.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    lo = cv2.erode(ref_c, k)
    hi = cv2.dilate(ref_c, k)
    return np.maximum(test_c - hi, 0.0) + np.maximum(lo - test_c, 0.0)


def low_freq_residual(glob, test, norm_win, tol_r):
    """
    The low frequency difference term: band residuals of the globally matched
    reference against the test, both blurred past the local fit window.

    It lives on its own because two callers have to measure the exact same
    quantity. difference_score scores it, and low_freq_disagreement decides
    whether scoring it is safe at all. If the decision and the term ever drifted
    apart, the gate would still return a number and would no longer be gating
    anything.
    """
    Lg, Ag, Bg = glob
    Lt, At, Bt = test
    lw = max(3, norm_win // 2)
    return band_residual(box(Lg, lw), box(Lt, lw), tol_r) + 0.7 * (
        band_residual(box(Ag, lw), box(At, lw), tol_r)
        + band_residual(box(Bg, lw), box(Bt, lw), tol_r))


def low_freq_disagreement(ref_bgr, test_bgr, valid, norm_win, tol_r, q=90.0, work=512):
    """
    How far apart the pair's low frequencies still are after a global fit, in
    Lab units, at the `q`th percentile over the frame.

    Measured on a downscaled copy, with both window sizes scaled to match. This
    is a statistic about low frequencies by construction, so resolution buys it
    nothing, and at working resolution it cost 188 ms of a 1000 ms run - 19% of
    the total, spent deciding whether to switch on a term that is usually left
    off anyway. Downscaling to a 512 px long side makes it ~12 ms.

    This is exactly the quantity that makes the low frequency term unsafe by
    default. That term is the only thing that can see the inside of an object
    wider than the photometric fit window, because the local fit has agreed
    with that interior and erased it; the reason it is off is that it also
    reports uneven illumination, and on a re-rendered pair the illumination is
    uneven everywhere. A percentile says which of the two is happening: a
    defect occupies a small part of the frame and leaves the percentile at the
    floor, while drifting illumination lifts the whole distribution.
    """
    f = fit_long_side(ref_bgr.shape, work)
    if f < 1.0:
        ref_bgr, test_bgr = resize_f(ref_bgr, f), resize_f(test_bgr, f)
        valid = resize_f(valid.astype(np.uint8) * 255, f) > 127
        norm_win = max(3, int(norm_win * f) | 1)
        tol_r = max(1, int(round(tol_r * f)))

    Lr, Ar, Br = cv2.split(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2Lab))
    Lt, At, Bt = cv2.split(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2Lab))
    glob = (global_match(Lr, Lt, valid, gain=True),
            global_match(Ar, At, valid, gain=False),
            global_match(Br, Bt, valid, gain=False))
    d = low_freq_residual(glob, (Lt, At, Bt), norm_win, tol_r)
    v = d[valid]
    if v.size < 1024:
        return 1e9
    return float(np.percentile(v[:: max(1, v.size // 400_000)], q))


def big_blur(img, sigma, cap=8.0):
    """
    Gaussian blur with a large sigma, evaluated on a downsampled copy.

    A kernel that wide carries no detail worth resolving at full resolution,
    and doing it directly would dominate the runtime.
    """
    if sigma <= cap:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    f = cap / sigma
    sw, sh = max(8, int(round(w * f))), max(8, int(round(h * f)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma * sw / float(w)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
