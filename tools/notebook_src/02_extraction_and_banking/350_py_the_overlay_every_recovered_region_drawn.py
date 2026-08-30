# --- THE OVERLAY. Every recovered region drawn on its own anomaly frame. -------------
# This replaces diffmask's --overlay file: same information, rendered at a size you can judge,
# and one panel per REGION rather than one per image, which is the unit the bank is built from.
# Adapted from flash_part1's region-QA figure.
import matplotlib.pyplot as plt

_qa = [(i, r) for i in ids for r in KEPT[i]]
if _qa:
    _ncol = 6
    _nrow = int(math.ceil(len(_qa) / _ncol))
    fig, ax = plt.subplots(_nrow, _ncol, figsize=(2.6 * _ncol, 2.9 * _nrow), squeeze=False)
    for k, (i, r) in enumerate(_qa):
        pad = int(0.6 * max(r["w"], r["h"])) + 12
        H, W = RAW[i]["mask"].shape[:2]
        y0, y1 = max(0, r["y"] - pad), min(H, r["y"] + r["h"] + pad)
        x0, x1 = max(0, r["x"] - pad), min(W, r["x"] + r["w"] + pad)
        crop = np.ascontiguousarray(RAW[i]["anomaly"][y0:y1, x0:x1].copy())
        cnts, _ = cv2.findContours(r["bin"][y0:y1, x0:x1].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, cnts, -1, (0, 255, 0), 2)
        a = ax[divmod(k, _ncol)]
        a.imshow(crop)
        a.set_title(f"{i}\n a={r['area']} ({r['w']}x{r['h']})", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout(); plt.show()
    print("Every candidate region, contoured on the frame it was recovered from.")
    print("Anything here that is NOT a defect is what VLM-2 has to reject in section 7.")
else:
    print("no regions survived MIN_REGION_PX -- nothing to show")
