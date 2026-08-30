# ============================================================================
# diffmask (1).py lines 153-185 -- warp conversion and application.
#
# Convention, and it is the one thing to get right when reading the rest: a warp W maps
# DESTINATION coordinates to SOURCE coordinates, which is what cv2.warp*(..., WARP_INVERSE_MAP)
# consumes. rewarp re-expresses W between resolutions, which is why registration can run at 640px
# and be applied at 1024px without re-solving.
# ============================================================================
# -------------------------------------------------------------------- warp math
# Convention: a warp W maps DESTINATION coordinates to SOURCE coordinates, which
# is what cv2.warp*(..., WARP_INVERSE_MAP) consumes.


def to3x3(W: np.ndarray) -> np.ndarray:
    if W.shape[0] == 3:
        return W.astype(np.float32)
    return np.vstack([W, np.array([[0.0, 0.0, 1.0]], np.float32)]).astype(np.float32)


def rewarp(W: np.ndarray, f_dst: float, f_src: float) -> np.ndarray:
    """
    Re-express W for images resampled by f_dst (destination) and f_src (source).
    Returns the same shape (2x3 or 3x3) as the input.
    """
    H = to3x3(W)
    S = np.diag([f_src, f_src, 1.0]).astype(np.float32)
    Si = np.diag([1.0 / f_dst, 1.0 / f_dst, 1.0]).astype(np.float32)
    R = (S @ H @ Si).astype(np.float32)
    R /= R[2, 2]
    return R if W.shape[0] == 3 else R[:2].copy()


def warp_into(img, W, size_wh, value=0):
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if W.shape[0] == 3:
        return cv2.warpPerspective(img, W, size_wh, flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=value)
    return cv2.warpAffine(img, W, size_wh, flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=value)
