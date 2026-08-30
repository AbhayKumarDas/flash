# ============================================================================
# diffmask (1).py lines 580-680 -- the sharpness term and Z(p).
#
# sharpness_z is the fifth cue and it is DISABLED in the frozen config (--w-sharp 0). It put
# 29.9% of image 030 into the mask: a re-render is sharper or softer than its source over large
# areas for reasons that have nothing to do with a defect, so the term answers strongly and
# wrongly across whole objects.
#
# local_zscore is Eq.1 in the paper. Note it is a large-neighbourhood z-score with a variance
# floor (floor_sd), not a global threshold: a re-render is noisy in textured areas and silent on
# flat ones, so a fixed threshold is either deaf on texture or hallucinating on flat substrate.
# ============================================================================
def sharpness_z(Lr, Lt, valid, hw, agg, beta=0.35, cfloor=0.15, floor_sd=0.022,
                open_r=12, coarse=3.5):
    """
    Regional loss (or gain) of high frequency detail, in calibrated sigmas.

    A blurred patch keeps its colours and its local mean, so it is invisible to
    every other term. What it does lose is high frequency energy, and the honest
    way to ask about that is a ratio rather than a difference: the same blur
    costs 3 L-units on coarse fabric and 0.05 on a soft gradient, so an absolute
    residual is uncomparable across a frame and unthresholdable across scenes.

    Four things the earlier prototype got wrong, all measured on the photo pair:

      * it compared the high frequency maps through band_residual, whose erode
        takes the minimum of the reference over the tolerance disc. High
        frequency energy is spiky, so that minimum sits near zero almost
        everywhere and it threw away most of the signal: the true drop inside
        the patch is 0.17 L and the prototype reported 0.09, against an outside
        99th percentile of 0.11.
      * it measured against the locally gain-fitted reference, and that fit is
        driven by exactly the local variance blurring destroys, so it had
        already pulled the reference partway toward the blurred test. On the
        texture scene the fitted gain inside the patch was 0.57.
      * it scored an absolute L-unit residual against a fixed 0.15 floor. On the
        photo scenes the entire signal is 0.09, so the z-score came out below 1
        and could never have survived the thresholds downstream.
      * it was averaged into the score with the other terms, which are all silent
        on a blur, so they outvoted it three to one.

    What is measured instead: high frequency energy aggregated over a window,
    which makes it insensitive to the sub-pixel shifts a pointwise comparison
    chokes on, compared as a normalised ratio, at two scales.

    The two scales are the discriminator. A real blur is band limited: it empties
    the fine band and leaves the coarse band nearly alone. Resampling the
    reference through the warp aliases fine detail, which also empties the fine
    band, but it disturbs the coarse band by a comparable fraction because it
    moves edges rather than softening them. Subtracting the coarse ratio from the
    fine one keeps the blur and cancels the artefact. On the photo scene, whose
    packaging lettering is the worst offender, this cuts the outside 99.9th
    percentile from 0.66 to 0.44 while the inside stays three quarters of its
    value.

    The two directions are scored separately rather than folded together,
    because the artefact is not symmetric - the aliased reference reads as
    spuriously sharp - so a combined measure buries the blur direction under the
    noise of the other one. Scoring them apart also means sharpening is caught
    on its own terms and not merely as an unsigned discrepancy.
    """
    def ratio(k):
        a = box(np.abs(Lr - box(Lr, k)), k)
        b = box(np.abs(Lt - box(Lt, k)), k)
        A, B = box(a, agg), box(b, agg)
        s = A + B
        # The additive constant keeps a ratio meaningful where there is barely
        # any detail to lose. Scaled to the frame's own high frequency level so
        # it means the same thing on coarse fabric and on a soft gradient, with
        # an absolute floor for a frame that is nearly featureless.
        c = max(cfloor, beta * float(np.mean(s[valid])) if valid.any() else cfloor)
        return (A - B) / (s + c)

    r_fine = ratio(hw)
    r_coarse = ratio(max(9, int(coarse * hw) | 1))

    # `floor_sd` is not a safety net here, it is the calibration. The ratio is
    # dimensionless and bounded by one, so what counts as a real loss of detail
    # is a fixed fraction and not whatever the MAD of this particular frame
    # happens to be. Left to the MAD the scale collapses: on a frame that
    # matches almost everywhere the MAD is zero, and a ratio of 0.09, which is
    # nothing, came out at 22 sigmas and swallowed the whole image.
    out = []
    for sgn in (1.0, -1.0):
        d = np.maximum(np.maximum(sgn * r_fine, 0.0)
                       - np.maximum(sgn * r_coarse, 0.0), 0.0)
        med, sig = robust_scale(d, valid, floor_sd)
        out.append(np.clip((d - med) / max(sig, floor_sd), 0.0, None))

    # A sharpness defect is a region. Anything narrower than the window the
    # measurement was made over is not a measurement of sharpness at all, it is
    # an edge that moved, and on the photo scene that means every letter of the
    # packaging text. A grey opening removes exactly those: a plateau wider than
    # the element keeps its value, a ridge thinner than it drops to its
    # surroundings.
    if open_r >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(open_r) + 1,) * 2)
        out = [cv2.morphologyEx(x, cv2.MORPH_OPEN, k) for x in out]
    # Returned as two channels, lost detail and gained detail, because the
    # caller treats them differently.
    return cv2.merge([out[0].astype(np.float32), out[1].astype(np.float32)])


def local_zscore(score, valid, sigma, floor_sd):
    """Z-score against a large local neighbourhood, ignoring invalid pixels."""
    v = valid.astype(np.float32)
    den = big_blur(v, sigma) + EPS
    m = big_blur(score * v, sigma) / den
    m2 = big_blur(score * score * v, sigma) / den
    sd = np.sqrt(np.maximum(m2 - m * m, 0.0))
    return (score - m) / np.maximum(sd, floor_sd)
