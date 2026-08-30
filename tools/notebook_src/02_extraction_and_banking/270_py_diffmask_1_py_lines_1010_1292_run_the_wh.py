# ============================================================================
# diffmask (1).py lines 1010-1292 -- run(), the whole pipeline end to end.
#
# Reads the pair, registers, flows, scores, thresholds, completes, writes the mask and optionally
# the overlay and the 7 debug intermediates. Returns 0.
#
# This is the function the paper's Algorithm 1 describes. Everything above is what it calls.
# ============================================================================
# -------------------------------------------------------------------------- main


def run(args) -> int:
    t0 = time.perf_counter()
    ref_full = imread(args.reference)
    test_full = imread(args.defect)
    Ht, Wt = test_full.shape[:2]

    ref_g = to_gray32(ref_full)
    test_g = to_gray32(test_full)

    t_reg = time.perf_counter()
    W, s, cscore, reg_mode, _ = register(ref_g, test_g, args)
    t_reg = time.perf_counter() - t_reg

    # ---- render the reference into the defect frame at working resolution --
    fw = fit_long_side((Ht, Wt), args.work)
    test_w = resize_f(test_full, fw)
    hw, ww = test_w.shape[:2]

    Ww = rewarp(W, fw, 1.0)
    ref_aligned = warp_into(ref_full, Ww, (ww, hw))
    cov = warp_into(np.full(ref_full.shape[:2], 255, np.uint8), Ww, (ww, hw))

    # Re-renders and rescales always disagree along the frame edge, so drop both
    # the seam where the warped reference runs out and a margin of the frame
    # itself. cv2.erode leaves the image border alone, hence the explicit trim.
    b = int(args.border * max(hw, ww))
    er = max(3, b * 2 + 1)
    valid = cv2.erode(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    if b > 0:
        valid[:b, :] = 0
        valid[-b:, :] = 0
        valid[:, :b] = 0
        valid[:, -b:] = 0

    flow_px = 0.0
    if not args.no_flow:
        ref_aligned, flow_px = flow_refine(ref_aligned, test_w, max_px=args.flow_max * fw,
                                           smooth=args.flow_smooth)
        alive = (cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        valid = cv2.bitwise_and(valid, cv2.erode(alive, np.ones((5, 5), np.uint8)))

    # A clipped pixel holds no information. Whatever was there is gone, the other
    # frame has nothing to be compared against, and the photometric fit cannot
    # recover it either since there is nothing left to fit. Measured on the jelly
    # pairs, whose reference blows out along the cup's right wall while the defect
    # frame still resolves it: the wall then reads as new content for its whole
    # length. No weighting of the terms removes that, because the difference is
    # real in pixels and meaningless in fact - which is exactly the case validity
    # is for. Both frames are tested, since either one clipping is enough to make
    # the comparison empty.
    if args.sat_guard > 0:
        lim = args.sat_guard * 255.0
        sat = ((ref_aligned.min(axis=2) >= lim) | (test_w.min(axis=2) >= lim)).astype(np.uint8)
        sat = cv2.dilate(sat, np.ones((5, 5), np.uint8))  # resampling spreads it
        valid[sat > 0] = 0

    valid_b = valid > 0
    if valid_b.mean() < 0.05:
        print("diffmask: warning: registration overlap is tiny, result is unreliable",
              file=sys.stderr)

    # Every stage downstream assumes the two frames actually correspond. When
    # they do not, the output is not a smaller mask, it is a meaningless one, so
    # say so rather than returning quiet nonsense.
    ea = edge_map(to_gray32(ref_aligned))[valid_b].ravel()
    eb = edge_map(to_gray32(test_w))[valid_b].ravel()
    step = max(1, ea.size // 200_000)
    align = float(np.corrcoef(ea[::step], eb[::step])[0, 1]) if ea.size > 1000 else 0.0
    if not np.isfinite(align):
        align = 0.0
    if align < args.min_align:
        print(f"diffmask: warning: the two images align poorly (edge correlation "
              f"{align:.2f}, expected above {args.min_align:.2f}). every stage after "
              f"this assumes they show the same objects. if they show different "
              f"instances of the same kind of object, or differ by a rotation the "
              f"coarse search does not cover, then everything differs and no mask "
              f"here means anything. treat the output as unreliable.", file=sys.stderr)

    # ---- difference --------------------------------------------------------
    L = max(hw, ww)
    norm_win = max(5, int(args.norm_frac * L) | 1)
    tol_r = max(1, int(round(args.tol_frac * L)))
    ref_d = ref_aligned.astype(np.float32) / 255.0
    test_d = test_w.astype(np.float32) / 255.0

    # The low frequency term is the only one that can see the middle of an
    # object wider than the fit window, and it is off by default because it
    # also reports uneven illumination. Measure whether this pair has any, and
    # switch the term on only when it does not.
    w_low, lf_q = args.w_low, -1.0
    if args.auto_low > 0 and args.w_low <= 0:
        lf_q = low_freq_disagreement(ref_d, test_d, valid_b, norm_win, tol_r)
        if lf_q <= args.auto_low:
            w_low = args.auto_low_w

    score, dvec, z_sharp = difference_score(
        ref_d, test_d, valid_b, norm_win, tol_r, args.smooth,
        w_luma=args.w_luma, w_chroma=args.w_chroma, w_grad=args.w_grad,
        w_low=w_low, w_sharp=args.w_sharp, p=args.combine_p,
        fit_clamp=not args.no_fit_clamp,
        fit_mask=not args.no_fit_mask,
        shared_edge=not args.no_shared_edge,
        sharp_win=max(5, int(args.sharp_frac * L) | 1),
        sharp_beta=args.sharp_beta, sharp_sd=args.sharp_sd,
        sharp_open=int(args.sharp_open * L))

    # The re-render noise floor varies a lot across the frame, so judge every
    # pixel against its own neighbourhood rather than against a global constant.
    med, sig = robust_scale(score, valid_b, 0.10)
    z = local_zscore(score, valid_b, max(4.0, args.adapt_frac * L), max(0.5 * sig, 0.10))

    # Folded in here, after med, sig and the local statistics have all been
    # taken from the untouched score, so nothing the sharpness term does can
    # move an existing detection. Only into z, which is already in sigmas: the
    # score carries the other terms' units, and mixing a sharpness sigma into
    # it would make -k and --abs-k retune this term as a side effect.
    if z_sharp is not None:
        # Only the gained-detail half stands back where the ordinary terms
        # already have an answer. This term is measured over a window and then
        # opened, so it locates a defect to within about thirty pixels and no
        # better. That is fine for a blurred patch, which nothing else sees at
        # all. It is not fine for a three pixel fibre: a fibre is new detail, so
        # it reads as a local gain in sharpness, and folding that in inflates
        # the line into a thirty pixel band and throws away the precision the
        # other terms had earned. The lost-detail half is not restrained the
        # same way, because losing detail is not something adding an object
        # does, so it has no thin defect to smear.
        lost, gained = z_sharp[..., 0], z_sharp[..., 1]
        if args.sharp_avoid > 0:
            near = within((z > args.k).astype(np.uint8) * 255,
                          int(args.sharp_avoid * L))
            gained = np.where(near, 0.0, gained).astype(np.float32)
        z_sharp = np.maximum(lost, gained)
        z = np.maximum(z, z_sharp)
    z[~valid_b] = 0.0

    # ---- threshold, at full resolution -------------------------------------
    z_full = cv2.resize(z, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    score_full = cv2.resize(score, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    valid_full = cv2.resize(valid, (Wt, Ht), interpolation=cv2.INTER_NEAREST) > 0
    # An absolute floor as well as a relative one, so a pair that simply does
    # not differ anywhere yields an empty mask instead of amplified noise.
    thr = max(med + args.abs_k * sig, args.min_score)
    abs_ok = score_full > thr
    weak_abs = score_full > thr * args.low_ratio
    # The sharpness channel clears the absolute floor on its own scale instead
    # of on `thr`, which is built from the other terms' median and sigma. A
    # sharpness sigma and a score are different quantities, and letting this
    # term borrow thr made -k, --abs-k and --min-score silently retune blur
    # sensitivity while being changed for unrelated reasons.
    if z_sharp is not None:
        zs_full = cv2.resize(z_sharp, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
        abs_ok |= zs_full > args.sharp_floor
        weak_abs |= zs_full > args.sharp_floor * args.low_ratio
    floor_ok = abs_ok & valid_full
    strong = (z_full > args.k) & floor_ok
    weak = (z_full > args.k * args.low_ratio) & weak_abs & valid_full

    r = max(3, int(0.004 * max(Ht, Wt)) | 1)
    strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))) > 0

    binary = hysteresis(strong, weak, args.max_grow)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 3, r * 3)))
    binary = fill_holes(binary)

    # Decide which regions are defects before touching their boundaries. Doing
    # it the other way round lets the shrink below sever a slender part, which
    # then gets discarded here as a separate weak region.
    min_area = max(24, int(args.min_area * Ht * Wt))
    mask, n_kept, allc, dropped = select_components(
        binary, min_area, args.keep_ratio, z_full, args.k * args.low_ratio, args.top_k)

    # The tolerance band and the score smoothing both widen every boundary by a
    # known amount, so take that much back.
    shrink = int(round(tol_r * args.shrink / max(fw, 1e-6)))
    if shrink >= 1 and mask.any():
        mask = shrink_keep_thin(mask, shrink)

    # Complete the shape, so a tapering end that barely differs from what it
    # covers still comes out whole.
    # With the low frequency term active the mask arrives as a sparse core of a
    # much larger shape, since a wide object differs most at its rim and its
    # near-uniform interior is what the local noise estimate flattens hardest,
    # so the completion is given a wider radius and a far looser area bound.
    snap_r, snap_grow = args.snap, args.snap_grow
    if w_low > 0:
        snap_r, snap_grow = args.low_snap, args.low_snap_grow
    if snap_r > 0 and mask.any():
        d_full = cv2.resize(dvec, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        edges_full = object_edges(test_full, args.canny_lo, args.canny_hi) \
            if args.snap_edges else None
        mask = complete_object(mask, d_full, valid_full,
                               max(2, int(snap_r * max(Ht, Wt))),
                               args.snap_frac, args.snap_noise, snap_grow,
                               edges_full)

    if args.dilate:
        d = max(1, int(args.dilate)) | 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))

    # select_components already enforced --min-area, but two stages have run
    # since: shrink_keep_thin, which can sever a region into ridge fragments,
    # and complete_object, which can split one and leave debris behind. Neither
    # is filtered, so single pixel regions reach the output - rice 126 shipped
    # two of them. The rule is applied a second time here rather than moved,
    # because the reason it runs before the shrink in the first place (a severed
    # slender arm must not then be discarded as a separate weak region) still
    # holds.
    if mask.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        drop = np.zeros(max(n, 1), bool)
        for i in range(1, n):
            drop[i] = st[i, cv2.CC_STAT_AREA] < min_area
        if drop.any():
            mask[drop[lab]] = 0

    imwrite(args.out, mask)
    dt = time.perf_counter() - t0

    # Count what was actually written. n_kept is decided before the completion
    # stage, which can split a region or leave fragments behind, so reporting it
    # describes an intermediate the user never sees.
    area = int((mask > 0).sum())
    n_out = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)[0] - 1
    print(f"register  scale={s:.4f} coarse_ncc={cscore:.3f} align={align:.2f} "
          f"via={reg_mode} "
          f"{'homography' if args.homography else 'affine'} "
          f"flow={flow_px:.2f}px  {t_reg * 1000:.0f}ms")
    print(f"lowfreq   low_freq_p90={lf_q:.2f} w_low={w_low:.2f}")
    print(f"threshold median={med:.3f} sigma={sig:.3f} z>{args.k} and score>{thr:.3f}")
    n_drop = cv2.connectedComponents((dropped > 0).astype(np.uint8), connectivity=8)[0] - 1
    print(f"mask      {n_out} region(s) {area}px "
          f"({100.0 * area / (Ht * Wt):.3f}% of image) -> {args.out}"
          + (f"  [{n_kept} before completion]" if n_out != n_kept else ""))
    if n_drop:
        # Said out loud rather than left to -v: a dropped region is the shape a
        # missed defect takes, and it costs nothing to mention.
        print(f"dropped   {n_drop} region(s) {int((dropped > 0).sum())}px below "
              f"--keep-ratio {args.keep_ratio}, outlined in red on the overlay")
    if args.verbose:
        for j, (a, rel, bb) in enumerate(allc):
            print(f"  [{'x' if j < n_kept else ' '}] strength={rel:.3f} "
                  f"raw_area={a:7d} bbox={bb}")
    print(f"total     {dt * 1000:.0f} ms")

    if args.overlay or args.debug:
        # Outline only by default: a filled overlay hides the very thing you are
        # trying to check the boundary against.
        ov = test_full.copy()
        if area and args.overlay_fill > 0:
            f = min(max(args.overlay_fill, 0.0), 1.0)
            ov[mask > 0] = ((1 - f) * ov[mask > 0] + f * np.array([0, 0, 255])).astype(np.uint8)
        # Discarded candidates first, so a kept region drawn over one stays
        # readable. Red is not "a second detection", it is "this cleared every
        # bar except being half as strong as the winner".
        if not args.no_show_dropped and dropped.any():
            dc, _ = cv2.findContours(dropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, dc, -1, (0, 0, 255), args.overlay_width)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (255, 255, 0), args.overlay_width)
        if args.overlay:
            imwrite(args.overlay, ov)
            print(f"overlay   -> {args.overlay}")

    if args.debug:
        d = Path(args.debug)
        imwrite(str(d / "01_ref_aligned.png"), ref_aligned)
        imwrite(str(d / "02_test_work.png"), test_w)
        imwrite(str(d / "03_blend.png"), cv2.addWeighted(ref_aligned, 0.5, test_w, 0.5, 0))
        sn = np.clip(score / max(float(score.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
        imwrite(str(d / "04_score.png"), cv2.applyColorMap(sn, cv2.COLORMAP_TURBO))
        zn = np.clip(z / max(args.k, 1e-6) * 160, 0, 255).astype(np.uint8)
        imwrite(str(d / "04b_zscore.png"), cv2.applyColorMap(zn, cv2.COLORMAP_TURBO))
        imwrite(str(d / "05_overlay.png"), ov)
        imwrite(str(d / "06_valid.png"), valid)
    return 0
