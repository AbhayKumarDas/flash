# --- Sanity: host -> OBS region -> MRSP blob ----------------------------------------
_show = LIVE_CATS[:4]
fig, ax = plt.subplots(len(_show), 3, figsize=(13, 4.3 * len(_show)), squeeze=False)
for r, cat in enumerate(_show):
    img, reg = host_and_region(HOST_PATHS[cat][0])
    _, mask = mrsp_mask_for(img, seed=SEED, region=reg)
    base = as_np(img)
    ov_r = base.copy(); ov_r[..., 0] = np.maximum(ov_r[..., 0], 0.45 * reg.numpy())
    ov_m = base.copy(); ov_m[..., 1] = np.maximum(ov_m[..., 1], 0.85 * mask.numpy())
    side, _ = mrsp_blob_stats(mask)
    ax[r, 0].imshow(base); ax[r, 0].set_title(f"{cat} - host normal", fontsize=9)
    ax[r, 1].imshow(ov_r); ax[r, 1].set_title(f"Ω  ({100*reg.mean():.0f}% of frame)", fontsize=9)
    ax[r, 2].imshow(ov_m); ax[r, 2].set_title(f"M_f - char side {side:.0f}px", fontsize=9)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.show()      # shown, not saved: the output dir holds the corpus only
