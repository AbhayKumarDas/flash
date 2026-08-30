import io as _io, contextlib, traceback

RESULTS = []   # one row per pair

for cat, pid, ref, dfc in PAIRS:
    # The recovered mask is written straight into the deliverable folder, beside a copy of the
    # anomaly frame it came from. That pair IS the Stage 2 output; nothing else needs keeping.
    odir = os.path.join(PAIRS_DIR, cat); os.makedirs(odir, exist_ok=True)
    mask_p = os.path.join(odir, f"{pid}_mask.png")
    shutil.copy2(dfc, os.path.join(odir, f"{pid}_anomaly.png"))

    # --debug is always on now, and not for debugging: 01_ref_aligned.png is the NORMAL image
    # registered into I_a's frame, which is what section 7 shows VLM-2 as the before-panel.
    # Without it the model is asked to spot an anomaly from a single image, which is not a
    # question that can be answered. The files land in _work/, which is not a deliverable.
    dbg_p = os.path.join(WORK_DIR, cat, f"{pid}_debug")
    argv = [ref, dfc, "-o", mask_p, "--debug", dbg_p, "-v", *FLAGS]
    if WRITE_DEBUG:
        argv += ["--overlay", os.path.join(WORK_DIR, cat, f"{pid}_overlay.png")]
    buf, t0, err = _io.StringIO(), time.time(), None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = main(argv)                     # diffmask's own entry point, unmodified
    except SystemExit as e:                     # imread failure calls sys.exit
        rc, err = int(e.code or 1), "SystemExit"
    except Exception as e:
        rc, err = 1, f"{type(e).__name__}: {e}"
        buf.write("\n" + traceback.format_exc())
    secs = time.time() - t0

    m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_p) else None
    px = int((m > 127).sum()) if m is not None else 0
    RESULTS.append(dict(cat=cat, pid=pid, ref=ref, defect=dfc, mask=mask_p, dbg=dbg_p,
                        rc=rc, err=err, secs=secs, mask_px=px,
                        log=buf.getvalue()))
    print(f"{cat:13s} {pid:>6s}  {secs:5.1f}s  {px:7d}px"
          f"{'  FAIL ' + str(err) if err else ''}"
          f"{'  EMPTY' if err is None and px == 0 else ''}")

_ok    = [r for r in RESULTS if r["err"] is None and r["mask_px"] > 0]
_empty = [r for r in RESULTS if r["err"] is None and r["mask_px"] == 0]
_fail  = [r for r in RESULTS if r["err"] is not None]
print(f"\n{len(_ok)}/{len(RESULTS)} recovered a non-empty mask "
      f"| {len(_empty)} empty | {len(_fail)} failed "
      f"| {sum(r['secs'] for r in RESULTS):.0f}s total")
if _empty:
    print("empty (the mask is the label -- an empty one means no usable defect was recovered):")
    for r in _empty:
        print(f"  {r['cat']}/{r['pid']}")
