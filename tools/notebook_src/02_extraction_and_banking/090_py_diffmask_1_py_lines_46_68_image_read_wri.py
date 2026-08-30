# ============================================================================
# diffmask (1).py lines 46-68 -- image read/write.
#
# Goes through np.fromfile/imdecode rather than cv2.imread because cv2's own path handling is
# byte-oriented and breaks on non-ascii Windows paths. Irrelevant on Kaggle, kept because
# diverging from the file under test to save two lines is a bad trade.
# ============================================================================

# --------------------------------------------------------------------------- io


def imread(path: str) -> np.ndarray:
    """Read via numpy so non-ascii Windows paths work."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"diffmask: cannot read image: {path}")
    return img


def imwrite(path: str, img: np.ndarray) -> None:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", img)
    if not ok:
        sys.exit(f"diffmask: cannot encode: {path}")
    buf.tofile(str(p))
