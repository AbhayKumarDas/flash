# --- The recovered masks themselves, whole-frame, next to their anomaly --------------
# The per-region view above cannot show a mask that landed somewhere absurd. This can.
_show = ids[:8]
fig, ax = plt.subplots(len(_show), 3, figsize=(11, 3.6 * len(_show)), squeeze=False)
for r_i, i in enumerate(_show):
    d = RAW[i]
    ov = d["anomaly"].copy()
    cnts, _ = cv2.findContours((d["mask"] > 127).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cnts, -1, (255, 0, 0), 3)
    ax[r_i, 0].imshow(d["anomaly"]);          ax[r_i, 0].set_title(f"{i} - I_a", fontsize=9)
    ax[r_i, 1].imshow(d["mask"], cmap="gray"); ax[r_i, 1].set_title("M_d recovered", fontsize=9)
    ax[r_i, 2].imshow(ov);                     ax[r_i, 2].set_title("overlay", fontsize=9)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.show()
print(f"First {len(_show)} of {len(ids)} donors. This is the pair written to")
print(f"  {PAIRS_DIR}/<category>/")
