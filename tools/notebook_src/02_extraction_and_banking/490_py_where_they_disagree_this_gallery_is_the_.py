# --- Where they disagree. This gallery is the deliverable, whichever way the numbers fall. --
if AGREE:
    dis = AGREE["fn"] + AGREE["fp"]
    if not dis:
        print("no disagreements")
    else:
        fig, ax = plt.subplots(1, len(dis), figsize=(2.1 * len(dis), 2.6), squeeze=False)
        for j, e in enumerate(dis):
            marked = np.array(vlm_panels(e)[0])
            ax[0, j].imshow(marked); ax[0, j].axis("off")
            side = "VLM kept, human dropped" if e in AGREE["fn"] else "VLM dropped, human kept"
            ax[0, j].set_title(f"{e['key'][-14:]}\n{side}", fontsize=7)
        plt.tight_layout(); plt.show()
        for e in dis:
            print(f"{e['key']:38s} conf {e['vlm']['confidence']:.2f}  "
                  f"{e['vlm']['artifact_kind']:8s} {e['vlm']['reason']}")
