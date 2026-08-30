# --- THE VERDICT OVERLAY. Every crop, contoured, coloured by what the gates decided. --
# Green = banked, red = discarded. This is the figure to read before trusting the bank: the
# green panels are what Stages 4-5 will composite, and anything green that is not a defect
# becomes a wrong label in every downstream metric.
_ncol = 6
_nrow = int(math.ceil(len(CROPS) / _ncol))
fig, ax = plt.subplots(_nrow, _ncol, figsize=(2.7 * _ncol, 3.1 * _nrow), squeeze=False)
for k, e in enumerate(CROPS):
    crop = np.ascontiguousarray(e["rgb"].copy())
    cnts, _ = cv2.findContours((e["alpha"] > 0.5).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    col = (0, 255, 0) if e["accepted"] else (255, 0, 0)
    cv2.drawContours(crop, cnts, -1, col, 2)
    a = ax[divmod(k, _ncol)]
    a.imshow(crop)
    why = "KEEP" if e["accepted"] else (
        ("vlm" if not e["gate_vlm"] else "")
        + ("+contrast" if not e["gate_contrast"] else "")
        + ("+non-primary" if not e["gate_primary"] else ""))
    a.set_title(f"{e['key'][-18:]}\n{why}  conf {e['vlm']['confidence']:.2f}  cf {e['cfrac']:.2f}",
                fontsize=7, color="green" if e["accepted"] else "red")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.show()
print("Green = goes into defect_bank/. Red = discarded, with the gate that rejected it.")
print("Anything green that is not a defect belongs in the red set -- lower VLM_MIN_CONF or")
print("raise MIN_ENTRY_CFRAC. Anything red that IS a defect is a gate that is too strict.")
