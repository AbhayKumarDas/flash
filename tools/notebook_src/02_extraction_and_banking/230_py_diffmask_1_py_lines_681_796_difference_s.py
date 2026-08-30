# ============================================================================
# diffmask (1).py lines 681-796 -- difference_score, the fusion of Alg.1 step 5.
#
# Returns (S, dvec, z_sharp). dvec is the per-channel signed Lab difference and it matters later:
# complete_object grows a region by matching the DIRECTION of dvec, which is how a partially
# detected defect is completed without also swallowing whatever else is nearby.
#
# The term weights (--w-luma/--w-chroma/--w-grad/--w-low/--w-sharp) were function arguments only
# until they were exposed as CLI flags. They are left at their defaults here -- reweighting them
# per category is exactly the category-specific tuning supplementary 3.A says FLASH does not do.
# ============================================================================
def difference_score(ref_bgr, test_bgr, valid, norm_win, tol_r, smooth=0.45,
                     w_luma=1.0, w_chroma=0.7, w_grad=0.9,
                     w_low=0.0, w_sharp=0.0, p=1.0,
                     fit_clamp=False, fit_mask=False, shared_edge=False,
                     sharp_win=21, sharp_beta=0.35, sharp_sd=0.022, sharp_open=12):
    """
    Photometric-invariant dissimilarity map, in units of robust sigmas.

    Every term is a band residual computed after a local photometric fit, so
    exposure drift and small geometric error are absorbed rather than reported.
    """
    Lr, Ar, Br = cv2.split(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2Lab))
    Lt, At, Bt = cv2.split(cv2.cvtColor(test_bgr, cv2.COLOR_BGR2Lab))

    # Globally matched copies, which keep large flat differences intact. Only
    # built when something actually needs them.
    Lg = Ag = Bg = None
    if w_low > 0:
        Lg = global_match(Lr, Lt, valid, gain=True)
        Ag = global_match(Ar, At, valid, gain=False)
        Bg = global_match(Br, Bt, valid, gain=False)

    # Kept before the local fit, because the sharpness term below must not see a
    # reference whose contrast has already been pulled toward the blurred test.
    Lr_raw = Lr

    # Locally matched copies, which are immune to uneven tone but blind to
    # anything smooth across the fit window.
    rL, rC = ((0.0, 100.0), (-128.0, 127.0)) if fit_clamp else (None, None)
    fw_ = valid.astype(np.float32) if fit_mask else None
    Lr = photometric_fit(Lr, Lt, norm_win, gain=True, rng=rL, w=fw_)
    Ar = photometric_fit(Ar, At, norm_win, gain=False, rng=rC, w=fw_)
    Br = photometric_fit(Br, Bt, norm_win, gain=False, rng=rC, w=fw_)

    d_luma = band_residual(Lr, Lt, tol_r)
    d_chroma = band_residual(Ar, At, tol_r) + band_residual(Br, Bt, tol_r)

    if shared_edge:
        gr, gt = edge_pair(Lr / 100.0, Lt / 100.0, valid, 1.0)
    else:
        gr, gt = edge_map(Lr / 100.0, 1.0), edge_map(Lt / 100.0, 1.0)
    d_grad = band_residual(gr, gt, tol_r)

    # Low frequencies, scored on the globally matched copies. This is what sees
    # an object bigger than the local fit window, whose interior the local fit
    # has quietly agreed with. Off by default: it also sees uneven illumination.
    d_low = None
    if w_low > 0:
        d_low = low_freq_residual((Lg, Ag, Bg), (Lt, At, Bt), norm_win, tol_r)

    # Local high frequency energy. A region that was blurred, or sharpened,
    # keeps its colours and its local mean and shows up in nothing else.
    # Computed as a z-score in its own right and folded in by the caller with a
    # maximum, not averaged in as one more term: it is a specialist that is
    # silent on every other defect, and an average would let the silent
    # majority veto it.
    z_sharp = None
    if w_sharp > 0:
        z_sharp = sharpness_z(Lr_raw, Lt, valid, max(3, tol_r | 1),
                              max(5, int(sharp_win) | 1), sharp_beta,
                              floor_sd=sharp_sd, open_r=sharp_open)

    # Floors are in each term's own units, and set the smallest difference
    # worth calling real.
    terms = [(d_luma, w_luma, 0.25), (d_chroma, w_chroma, 0.40),
             (d_grad, w_grad, 0.010)]
    if d_low is not None:
        terms.append((d_low, w_low, 0.25))

    # Combined as a power mean. p=1 is the plain average and is the default,
    # because it is what the thresholds downstream are calibrated against.
    # Raising p moves toward a maximum, which suits a mix of specialist terms:
    # a translucent film answers in chroma and gradient and nowhere else, a
    # large flat object only in the low frequencies, a blurred patch only in
    # sharpness. Under an average the silent terms outvote the one term that
    # can actually see the defect, and each term added makes that worse.
    total_w, acc = 0.0, None
    for arr, wgt, floor in terms:
        if wgt <= 0:
            continue
        med, sig = robust_scale(arr, valid, floor)
        z = np.clip((arr - med) / sig, 0.0, None)
        contrib = wgt * (z if p == 1.0 else np.power(z, p))
        acc = contrib if acc is None else acc + contrib
        total_w += wgt

    score = acc / max(total_w, 1e-6)
    if p != 1.0:
        score = np.power(score, 1.0 / p)
    if smooth > 0:
        score = cv2.GaussianBlur(score, (0, 0), max(0.6, tol_r * smooth))
    score[~valid] = 0.0

    # Returned alongside the score rather than mixed into it, so that every
    # statistic the caller derives from the score - its global median, its
    # global sigma, its local mean and spread - is bit for bit what it was
    # before this term existed. The caller folds it in at the point of
    # thresholding, which is the only place it can add anything.
    if z_sharp is not None:
        z_sharp = cv2.GaussianBlur(z_sharp, (0, 0), max(0.6, tol_r * 0.5))
        z_sharp = (w_sharp * z_sharp).astype(np.float32)
        z_sharp[~valid] = 0.0

    # The signed Lab difference is returned too. Its magnitude is what scores
    # above, but its direction is what identifies an object, and the two parts
    # of one object agree on direction long after the fainter part has dropped
    # below any magnitude threshold. Measured against the local fit normally,
    # which isolates the object's own effect; against the global fit when the
    # low frequency term is on, since the local fit erases a large interior.
    if Lg is None:
        dvec = cv2.merge([Lt - Lr, At - Ar, Bt - Br])
    else:
        dvec = cv2.merge([Lt - Lg, At - Ag, Bt - Bg])
    return score.astype(np.float32), dvec, z_sharp
