# ============================================================================
# diffmask (1).py lines 797-1009 -- from thresholded Z to final components.
#
# Alg.1 step 7: hysteresis grows each strong seed through the connected weak region around it,
# so faint defect extensions survive while isolated weak noise does not.
# Alg.1 step 8: select_components ranks by accumulated evidence and keeps by --keep-ratio /
# --top-k / --min-area.
# Alg.1 step 9: complete_object recovers weak continuations by matching the direction of the
# difference vector rather than its magnitude.
#
# shrink_keep_thin is the --shrink implementation and it is the one to watch. At the stock 1.2 it
# shattered image 015 into 13 fragments -- 166px surviving from an 1835px seed -- because a
# uniform erosion is fatal to any region narrower than twice the radius. 0.4 is the frozen value.
#
# --keep-ratio 0.50 drops a thin region even once it is the strongest candidate, because a thin
# region's MEAN strength is diluted by its own length. Left at 0.50 for every category; that is
# a known cost of one configuration for all eight, and it surfaces as an EMPTY mask in section 4.
# ============================================================================
def hysteresis(strong, weak, max_grow=12.0):
    """
    Grow each strong seed through the connected weak region around it.

    A translucent object only differs strongly at its edges, so seeding on the
    strong evidence and growing through weak evidence recovers the whole shape.
    A component that balloons far beyond its seed is treated as a leak and
    falls back to the seed.
    """
    weak_u8 = weak.astype(np.uint8)
    n, lab = cv2.connectedComponents(weak_u8, connectivity=8)
    if n <= 1:
        return strong.astype(np.uint8) * 255

    seed_area = np.bincount(lab[strong].ravel(), minlength=n).astype(np.float64)
    full_area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    keep = seed_area > 0
    keep[0] = False
    keep &= full_area <= max_grow * np.maximum(seed_area, 1.0)

    out = keep[lab]
    return (out | strong).astype(np.uint8) * 255


def fill_holes(mask):
    """Fill background regions fully enclosed by the mask."""
    h, w = mask.shape
    inv = cv2.copyMakeBorder(cv2.bitwise_not(mask), 1, 1, 1, 1,
                             cv2.BORDER_CONSTANT, value=255)
    # Foreground is treated as 8-connected, so the background must be 4-connected
    # for a hole to count as enclosed.
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    if n <= 1:
        return mask
    outside = lab[0, 0]
    holes = ((lab != outside) & (inv > 0))[1:h + 1, 1:w + 1]
    out = mask.copy()
    out[holes] = 255
    return out


def within(mask, radius):
    """Pixels no further than `radius` from the mask. A big structuring element
    would do the same thing, at many times the cost."""
    if radius < 1:
        return mask > 0
    d = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return d <= radius


def shrink_keep_thin(mask, radius):
    """
    Pull every boundary in by `radius` without amputating slender parts.

    The distance transform gives each pixel's local half width, so eroding is
    just `dist > radius`. Anything narrower than that would vanish entirely, so
    those parts keep their medial ridge, detected as a local maximum of the
    distance transform.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    eroded = dist > radius
    if not eroded.any():
        return mask

    # Thin means narrower than the shrink, so the erosion would lose it whole.
    # Judged per part rather than per component, so a slender tail on a solid
    # body is treated as thin even though its component survives.
    thin = (mask > 0) & ~within(eroded.astype(np.uint8) * 255, radius)

    out = eroded.astype(np.uint8) * 255
    if thin.any():
        ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.float32)) - 1e-4) & (dist > 1.5)
        out |= (ridge & (mask > 0)).astype(np.uint8) * 255
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        out = cv2.bitwise_and(out, mask)
        out[eroded] = 255
    return out


def object_edges(test_bgr, lo=10, hi=30, thicken=True):
    """
    Silhouette of whatever is in the defect image, tuned to be over-inclusive.

    The asymmetry matters: a spurious edge only stops the growth below early,
    while a missing one lets it escape the object entirely. So this errs
    sensitive, and the result is thickened so barriers are watertight.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(gray.astype(np.uint8), lo, hi)
    if thicken:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    return edges


def complete_object(mask, dvec, valid, radius, frac=0.30, noise_k=3.0,
                    max_grow=5.0, edges=None):
    """
    Grow each region into the rest of the same object.

    Differencing responds where an object differs most, which for a translucent
    one is its densest part. A shape tapering to a point barely differs from
    what it covers near the tip, so the tip is missed however low a magnitude
    threshold is set, and growing on magnitude alone just bleeds into whatever
    else happens to be noisy.

    Direction is the discriminator that survives. One object changes the image
    the same way throughout, only more faintly where it is thin, so projecting
    the local difference onto the region's own mean direction separates the rest
    of that object from unrelated texture by a wide margin. The measured margin
    on translucent plastic over jelly is about 3.5 against 0.0.

    `edges`, when given, stops growth crossing a hard silhouette.
    """
    # Fragments lying close together are parts of one object, not separate
    # objects. Grouping them first matters: thin extremities often survive
    # detection only as a dotted line, and each dot on its own is both too small
    # to estimate a signature from and disconnected from the body it belongs to.
    grp = within(mask, max(1, radius // 2)).astype(np.uint8)
    n, glab, stats, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
    if n <= 1:
        return mask

    H, W = mask.shape
    out = np.zeros_like(mask)
    pad = int(2.0 * radius) + 4  # growth zone, plus a ring to measure noise in

    for i in range(1, n):
        # Everything below is local to the object, so crop to its neighbourhood
        # rather than sweeping the whole frame once per object.
        bx, by, bw, bh = stats[i, :4]
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)

        seed = (glab[y0:y1, x0:x1] == i) & (mask[y0:y1, x0:x1] > 0)
        seed_area = int(seed.sum())
        if seed_area == 0:
            continue
        d = dvec[y0:y1, x0:x1]

        sig = d[seed].mean(axis=0)
        norm = float(np.linalg.norm(sig))
        if norm < 1e-6:
            out[y0:y1, x0:x1][seed] = 255
            continue
        unit = sig / norm

        sub_valid = valid[y0:y1, x0:x1]
        reachable = within(seed.astype(np.uint8) * 255, radius) & sub_valid
        proj = d @ unit

        # Measure noise in the ring just outside the growth zone: near enough to
        # describe this part of the image, far enough that the missing part of
        # the object cannot inflate it and threshold itself away.
        med, sd = robust_scale(proj, sub_valid & ~reachable, 1e-3)
        thr = max(frac * float(proj[seed].mean()), med + noise_k * sd)

        cand = ((proj > thr) & reachable) | seed
        if edges is not None:
            cand &= (edges[y0:y1, x0:x1] == 0) | seed
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        nc, lc = cv2.connectedComponents(cand, connectivity=8)
        touched = np.unique(lc[seed & (cand > 0)])
        keep = np.zeros(nc, bool)
        keep[touched[touched != 0]] = True
        grown = keep[lc] | seed

        out[y0:y1, x0:x1][grown if grown.sum() <= max_grow * seed_area else seed] = 255

    return fill_holes(out)


def select_components(binary, min_area, keep_ratio, score, k, top_k=0):
    """
    Rank regions by total evidence above threshold, not by area alone.

    Also returns the regions that cleared --min-area and were then dropped for
    being weaker than keep_ratio times the best one. That rule assumes the frame
    holds one dominant defect, and when it does not - two defects of unequal
    strength, or anything spurious that outranks the real one - it discards the
    answer without saying so. Measured on fabric 134, whose reference carries a
    speck of lint the defect frame does not: the speck scores 1.000 and the
    thread that is actually the defect scores 0.201, so the thread was dropped
    and the mask sat on blank fabric. Handing the dropped regions back lets the
    caller draw them, which is the difference between a wrong answer and a
    wrong answer you can see.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary), 0, [], np.zeros_like(binary)
    excess = np.clip(score - k, 0.0, None)
    cand = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cand.append((float(excess[labels == i].sum()), i, area))
    if not cand:
        return np.zeros_like(binary), 0, [], np.zeros_like(binary)

    cand.sort(reverse=True)
    top = cand[0][0]
    keep = cand[:top_k] if top_k > 0 else [c for c in cand if c[0] >= keep_ratio * top]
    kept_ids = {i for _, i, _ in keep}
    out, dropped = np.zeros_like(binary), np.zeros_like(binary)
    for _, i, _ in cand:
        (out if i in kept_ids else dropped)[labels == i] = 255
    report = [(a, s / max(top, 1e-9), stats[i, :4].tolist()) for s, i, a in cand]
    return out, len(keep), report, dropped
