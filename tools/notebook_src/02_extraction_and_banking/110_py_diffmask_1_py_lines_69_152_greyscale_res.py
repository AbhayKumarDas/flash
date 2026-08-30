# ============================================================================
# diffmask (1).py lines 69-152 -- greyscale, resize, edge maps, robust statistics.
#
# Two functions here carry real decisions:
#
# edge_pair vs edge_map. edge_map normalises each image by its OWN 99.5th percentile, which is
# what makes it tone invariant -- correct when the two maps are never compared. When they are
# subtracted it is wrong: the warped reference carries a hard step where its coverage runs out,
# that step sets its percentile, and every real edge in the reference comes out 2-3x weaker than
# the identical edge in the test. The whole silhouette then reads as a difference. edge_pair uses
# one normaliser measured only over the overlap.
#
# robust_scale's `floor`. Without a physically meaningful noise floor, a near-identical pair
# drives sigma to zero and every rounding error becomes an enormous z-score.
# ============================================================================
# ------------------------------------------------------------------ small utils


def to_gray32(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def fit_long_side(shape, target: int) -> float:
    h, w = shape[:2]
    return min(1.0, target / float(max(h, w))) if target > 0 else 1.0


def resize_f(img: np.ndarray, f: float) -> np.ndarray:
    if abs(f - 1.0) < 1e-9:
        return img
    interp = cv2.INTER_LINEAR if f > 1.0 else cv2.INTER_AREA
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                      interpolation=interp)


def edge_map(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Normalised gradient magnitude. Tone invariant, so good for registration."""
    g = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    m = cv2.magnitude(gx, gy)
    hi = float(np.percentile(m[:: max(1, m.size // 200_000)], 99.5))
    return np.clip(m / (hi + EPS), 0.0, 1.0)


def edge_pair(a: np.ndarray, b: np.ndarray, mask, sigma: float = 1.0):
    """
    Gradient magnitude of two images on one shared scale.

    edge_map normalises each image by its own 99.5th percentile, which is what
    makes it tone invariant and is right when the two maps are never compared.
    Here they are subtracted, and then it is wrong: the warped reference carries
    the hard step where its coverage runs out, that step sets its percentile, and
    every real edge in the reference comes out two or three times weaker than the
    identical edge in the test. The whole silhouette of the scene then reads as a
    difference. One normaliser, measured only where the two frames overlap, is
    what makes the subtraction mean anything.
    """
    def mag(x):
        g = cv2.GaussianBlur(x, (0, 0), sigma)
        return cv2.magnitude(cv2.Scharr(g, cv2.CV_32F, 1, 0),
                             cv2.Scharr(g, cv2.CV_32F, 0, 1))
    ma, mb = mag(a), mag(b)
    va = ma[mask] if mask is not None else ma.ravel()
    vb = mb[mask] if mask is not None else mb.ravel()
    if va.size < 64:
        hi = max(float(ma.max()), float(mb.max()))
    else:
        st = max(1, va.size // 100_000)
        hi = float(np.percentile(np.concatenate([va[::st], vb[::st]]), 99.5))
    hi = max(hi, 1e-6)
    return np.clip(ma / hi, 0.0, 1.0), np.clip(mb / hi, 0.0, 1.0)


def box(img: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)
    return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def robust_scale(x: np.ndarray, mask: np.ndarray, floor: float = 1e-5) -> tuple[float, float]:
    """
    Median and MAD-derived sigma over the masked pixels.

    `floor` must be a physically meaningful noise level for the quantity being
    measured. Without it, a pair of near-identical images drives sigma to zero
    and every rounding error becomes an enormous z-score.
    """
    v = x[mask] if mask is not None else x.ravel()
    if v.size < 64:
        return 0.0, max(floor, 1e-5)
    if v.size > 400_000:
        v = v[:: max(1, v.size // 400_000)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, max(mad * 1.4826, floor, 1e-5)
