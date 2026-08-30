# ============================================================================
# diffmask (1).py lines 404-432 -- low-frequency dense flow on the aligned reference.
#
# Deliberately computed small and smoothed hard: it must absorb slow geometric drift from a
# re-render, NOT deform itself around the defect. --flow-smooth exposes the sigma that used to be
# hardcoded here.
#
# Known limit: wallplugs are 3D objects at different depths, so each shifts differently under one
# affine warp and this is smoothed too hard to follow them. That is why image 010 needs
# --auto-low to detect at all. Lowering -k or --min-score does not help -- it surfaces four plugs
# of near-equal strength, i.e. parallax residual, not the defect.
# ============================================================================
def flow_refine(ref_aligned, test, max_px=18.0, work=320, smooth=0.06):
    """
    Low-frequency dense flow applied to the already aligned reference.

    Deliberately computed small and smoothed hard: it should absorb slow
    geometric drift from a re-render, not deform itself around the defect.
    """
    h, w = test.shape[:2]
    f = fit_long_side((h, w), work)
    a = np.ascontiguousarray(resize_f(cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY), f))
    b = np.ascontiguousarray(resize_f(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY), f))

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis.setUseSpatialPropagation(True)
    flow = dis.calc(b, a, None)  # where in the reference each defect pixel comes from

    flow = cv2.GaussianBlur(flow, (0, 0), max(1.0, smooth * max(flow.shape[:2])))
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / f)

    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    if (mag > max_px).any():
        flow *= np.where(mag > max_px, max_px / np.maximum(mag, EPS), 1.0).astype(np.float32)[..., None]

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(ref_aligned, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out, float(mag.mean())
