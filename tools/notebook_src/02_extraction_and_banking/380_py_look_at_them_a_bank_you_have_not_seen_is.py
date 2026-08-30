# --- Look at them. A bank you have not seen is a bank you cannot defend. -------------
import matplotlib.pyplot as plt

_n = min(len(CROPS), 12)
if _n:
    fig, ax = plt.subplots(3, _n, figsize=(1.7 * _n, 5.6))
    ax = np.atleast_2d(ax)
    for j, e in enumerate(CROPS[:_n]):
        ax[0, j].imshow(e["rgb"]);                ax[0, j].set_title(e["key"][-16:], fontsize=7)
        ax[1, j].imshow(e["alpha"], cmap="gray"); ax[1, j].set_title("alpha (stored apart)", fontsize=7)
        ring = np.zeros((*e["alpha"].shape, 3), np.float32)
        ring[..., 1] = e["alpha"]                 # green = defect
        ring[..., 2] = (e["alpha"] < 0.10)        # blue  = substrate the harmoniser samples
        ax[2, j].imshow(ring)
        ax[2, j].set_title(f"dE {e['contrast']:.0f} / cf {e['cfrac']:.2f}", fontsize=7)
        for r in range(3):
            ax[r, j].axis("off")
    plt.tight_layout(); plt.show()
    print("Green = recovered defect. Blue = the substrate Stage 5's harmoniser measures.")
    print("A crop with almost no blue will place badly however good its mask is.")
