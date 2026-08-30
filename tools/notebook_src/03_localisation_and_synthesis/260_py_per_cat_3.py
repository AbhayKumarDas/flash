_per_cat = 3
for cat in LIVE_CATS:
    sel = MANI[MANI.cat == cat].head(_per_cat)
    if sel.empty:
        continue
    fig, ax = plt.subplots(2, len(sel), figsize=(4.2 * len(sel), 8.4), squeeze=False)
    for j, (_, r) in enumerate(sel.iterrows()):
        img = np.array(Image.open(r["img"]).convert("RGB"))
        m8 = (np.array(Image.open(r["mask"]).convert("L")) > 127).astype(np.uint8)
        ax[0, j].imshow(img)
        ax[0, j].set_title(f"{cat} · mode {r['mode']} · {r['arm_used']}\n"
                           f"r_eq {r['r_eq']:.0f}px · collar {r['collar']}px · "
                           f"d_in {r['d_in']:.0f}", fontsize=8)
        marked = img.copy()
        cont, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(marked, cont, -1, (255, 0, 0), 3)
        ax[1, j].imshow(marked)
        ax[1, j].set_title(f"GT - {int(m8.sum())}px", fontsize=8)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    plt.tight_layout(); plt.show()
