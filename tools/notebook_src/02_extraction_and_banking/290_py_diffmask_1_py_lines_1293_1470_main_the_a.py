# ============================================================================
# diffmask (1).py lines 1293-1470 -- main(), the argument parser.
#
# The __main__ guard (lines 1472-1474) is deliberately excluded: __name__ == "__main__" is TRUE
# in a notebook, so it would fire on cell execution.
#
# main(argv) both parses and runs -- the last line is `return run(p.parse_args(argv))` -- which is
# exactly how run_diffmask_batch.py drives it. The notebook does the same rather than
# reconstructing an argparse.Namespace by hand, so no default can silently diverge.
# ============================================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="diffmask",
        description="Binary mask of the object present only in the defect image.")
    p.add_argument("reference", help="clean reference image")
    p.add_argument("defect", help="image containing the extra object")
    p.add_argument("-o", "--out", default="mask.png", help="output mask path")
    p.add_argument("--overlay", metavar="PNG", help="also write the mask outlined on the defect image")
    p.add_argument("--overlay-fill", type=float, default=0.0,
                   help="tint the interior by this much, 0 = outline only (default)")
    p.add_argument("--overlay-width", type=int, default=2, help="outline thickness in px")
    p.add_argument("--debug", metavar="DIR", help="write intermediate images here")

    p.add_argument("--work", type=int, default=1024, help="working long side (default 1024)")
    p.add_argument("--ecc-work", type=int, default=640, help="registration long side")
    p.add_argument("--homography", action="store_true", help="8-dof warp instead of affine")
    p.add_argument("--no-ecc", action="store_true", help="skip ECC refinement")
    p.add_argument("--no-flow", action="store_true", help="skip dense flow refinement")
    p.add_argument("--flow-max", type=float, default=18.0, help="flow cap, in output px")
    p.add_argument("--flow-smooth", type=float, default=0.06,
                   help="how hard the dense flow is smoothed, as a fraction of "
                        "the flow field's own long side. the default is "
                        "deliberately heavy so the flow absorbs global drift "
                        "without deforming around the defect. lower it when the "
                        "scene is several rigid objects at different depths, "
                        "where each one shifts differently and one affine warp "
                        "cannot follow them; the cost is that a large soft "
                        "defect starts being absorbed too")
    p.add_argument("--no-rot-search", action="store_true",
                   help="never try the rotation-capable registration fallback")
    p.add_argument("--rot-trigger", type=float, default=0.60,
                   help="try the rotation fallback when the coarse+ECC warp "
                        "scores below this alignment")
    p.add_argument("--rot-margin", type=float, default=0.02,
                   help="alignment the fallback must gain before it replaces the "
                        "coarse warp")
    p.add_argument("--min-align", type=float, default=0.75,
                   help="warn below this edge correlation between the aligned pair. "
                        "measured: 0.92 and 0.96 on pairs that differ by one added "
                        "object, 0.59 on a pair whose objects are different objects")
    p.add_argument("--sat-guard", type=float, default=0.99,
                   help="treat a pixel as invalid when either frame is clipped "
                        "at or above this fraction of full scale in every "
                        "channel. a blown highlight has lost what was there, so "
                        "the frames cannot be compared and any difference found "
                        "is an artefact of one of them clipping first. 0 disables")
    p.add_argument("--border", type=float, default=0.015,
                   help="ignore this fraction of the long side around the overlap seam")

    p.add_argument("--norm-frac", type=float, default=0.04,
                   help="photometric fit window, as a fraction of the long side")
    p.add_argument("--tol-frac", type=float, default=0.006,
                   help="geometric tolerance band radius, fraction of the long side")
    p.add_argument("--adapt-frac", type=float, default=0.15,
                   help="local noise estimation radius, fraction of the long side")
    p.add_argument("--smooth", type=float, default=0.60,
                   help="score smoothing, as a multiple of the tolerance radius")
    p.add_argument("--w-luma", type=float, default=1.0,
                   help="weight of the L band residual")
    p.add_argument("--w-chroma", type=float, default=0.7,
                   help="weight of the a+b band residual")
    p.add_argument("--w-grad", type=float, default=0.9,
                   help="weight of the gradient band residual. this is the term "
                        "a thin high-contrast structure answers in, so lower it "
                        "when the frame contains one that registration cannot "
                        "settle - a specular rim on transparent plastic being "
                        "the case it was measured on")
    p.add_argument("--w-low", type=float, default=0.0,
                   help="weight for the low frequency term; needed for defects "
                        "wider than the fit window, but also sees uneven light")
    p.add_argument("--auto-low", type=float, default=0.0,
                   help="turn the low frequency term on by itself when the "
                        "pair's low frequencies already agree to within this "
                        "many Lab units over 90%% of the frame (0 = never). "
                        "2.5 is the calibrated value and is what to pass to "
                        "enable this; it is off by default because the term is "
                        "a low pass, so it widens every boundary it touches by "
                        "its own radius, which measured -0.13 IoU on a real "
                        "pair whose defect is smaller than the fit window")
    p.add_argument("--auto-low-w", type=float, default=0.6,
                   help="weight the low frequency term gets when --auto-low fires")
    p.add_argument("--w-sharp", type=float, default=1.0,
                   help="gain on the sharpness term; needed for blur or "
                        "sharpening defects, which no other term can see. NOTE "
                        "this is not the averaged-in band residual it named "
                        "before: it is a maximum-folded two-scale ratio, folded "
                        "in after every other statistic has been taken. 0 "
                        "disables it. it is not purely additive: raising z also "
                        "widens the hysteresis growth of regions the other "
                        "terms found, so existing detections can change shape. "
                        "it resolves to about 30px, so it is a regional change "
                        "detector, not only a blur detector")
    p.add_argument("--sharp-frac", type=float, default=0.021,
                   help="window the sharpness term aggregates over, as a "
                        "fraction of the long side")
    p.add_argument("--sharp-beta", type=float, default=0.35,
                   help="sharpness ratio softening, as a fraction of the "
                        "frame's own mean high frequency energy")
    p.add_argument("--sharp-avoid", type=float, default=0.05,
                   help="ignore GAINED-detail sharpness evidence within this "
                        "fraction of the long side of a region the other terms "
                        "already found, since they localise it far better")
    p.add_argument("--sharp-open", type=float, default=0.012,
                   help="drop sharpness evidence narrower than this fraction "
                        "of the long side; a blur is a region, a moved edge is "
                        "a filament")
    p.add_argument("--sharp-sd", type=float, default=0.022,
                   help="what counts as one sigma of sharpness ratio; this is "
                        "a calibration, not a floor, because the ratio is "
                        "dimensionless. it sets the scale that -k and "
                        "--sharp-floor are then read in, so lowering it makes "
                        "the term more sensitive at both of those gates at once")
    p.add_argument("--sharp-floor", type=float, default=4.0,
                   help="absolute gate the sharpness term must clear, in its "
                        "own sigmas; the other terms use --min-score and "
                        "--abs-k, which are not in these units")
    p.add_argument("--no-shared-edge", action="store_true",
                   help="normalise each gradient map by its own percentile "
                        "instead of one shared over the overlap")
    p.add_argument("--no-fit-mask", action="store_true",
                   help="let the local photometric fit read pixels outside the "
                        "overlap and outside the frame")
    p.add_argument("--no-fit-clamp", action="store_true",
                   help="let the local photometric fit predict values outside "
                        "the channel range")
    p.add_argument("--combine-p", type=float, default=1.0,
                   help="power mean exponent, 1 = average; raise toward 3 when "
                        "extra terms are enabled so specialists are not outvoted")
    p.add_argument("--shrink", type=float, default=1.20,
                   help="boundary shrink, as a multiple of the tolerance radius")
    p.add_argument("--snap", type=float, default=0.035,
                   help="complete each region to the whole object within this "
                        "fraction of the long side (0 disables)")
    p.add_argument("--snap-frac", type=float, default=0.30,
                   help="keep pixels matching this fraction of the region's own signature")
    p.add_argument("--snap-noise", type=float, default=2.0,
                   help="floor for the completion threshold, in robust sigmas")
    p.add_argument("--snap-grow", type=float, default=5.0,
                   help="max area the completion may add, relative to the seed")
    p.add_argument("--low-snap", type=float, default=0.08,
                   help="completion radius used instead of --snap while the "
                        "low frequency term is active")
    p.add_argument("--low-snap-grow", type=float, default=120.0,
                   help="completion area allowance used instead of --snap-grow "
                        "while the low frequency term is active")
    p.add_argument("--snap-edges", action="store_true",
                   help="also stop completion at hard silhouette edges")
    p.add_argument("--canny-lo", type=int, default=20, help="edge detector low threshold")
    p.add_argument("--canny-hi", type=int, default=60, help="edge detector high threshold")
    p.add_argument("-k", type=float, default=5.0, help="threshold in local sigmas")
    p.add_argument("--abs-k", type=float, default=3.0,
                   help="absolute floor in global robust sigmas")
    p.add_argument("--min-score", type=float, default=4.0,
                   help="hard minimum score, so matching pairs return an empty mask")
    p.add_argument("--low-ratio", type=float, default=0.35,
                   help="hysteresis growth threshold, relative to -k")
    p.add_argument("--max-grow", type=float, default=12.0,
                   help="max area a region may gain relative to its seed")
    p.add_argument("--min-area", type=float, default=2e-4,
                   help="min region area as a fraction of the image")
    p.add_argument("--keep-ratio", type=float, default=0.50,
                   help="keep regions this strong relative to the best one")
    p.add_argument("--top-k", type=int, default=0,
                   help="keep exactly the N strongest regions (0 = use --keep-ratio)")
    p.add_argument("--dilate", type=int, default=0, help="grow the final mask by N px")
    p.add_argument("--no-show-dropped", action="store_true",
                   help="do not outline the regions --keep-ratio discarded. by "
                        "default the overlay draws them in red, because that "
                        "rule silently throws away a region whenever something "
                        "else in the frame outranks it, and when the winner is "
                        "spurious the answer disappears with no trace. on "
                        "fabric 134 the mask sits on a speck of lint that is on "
                        "the reference and not on the defect frame, while the "
                        "thread that is the actual defect scored 0.201 and was "
                        "dropped. the mask is unchanged either way - this only "
                        "decides whether you can see what was discarded")
    p.add_argument("-v", "--verbose", action="store_true", help="list every candidate region")
    return run(p.parse_args(argv))
