_s = MANI.secs.tolist()
print(f"{'category':14s}{'images':>8s}{'total s':>10s}{'mean s':>9s}{'min s':>8s}{'max s':>8s}")
print("-" * 57)
for c in LIVE_CATS:
    ss = MANI[MANI.cat == c].secs.tolist()
    if ss:
        print(f"{c:14s}{len(ss):>8d}{sum(ss):>10.1f}{np.mean(ss):>9.3f}"
              f"{min(ss):>8.3f}{max(ss):>8.3f}")
print("-" * 57)
print(f"{'ALL':14s}{len(_s):>8d}{sum(_s):>10.1f}{np.mean(_s):>9.3f}"
      f"{min(_s):>8.3f}{max(_s):>8.3f}")

print(f"\nwall clock              {SYNTH_SECONDS:8.1f}s   (serial)")
print(f"per image (wall)        {SYNTH_SECONDS/max(n_img,1):8.3f}s")
print(f"MRSP device             {MRSP_DEVICE:>8s}")

for mode in ("C", "B"):
    ss = MANI[MANI["mode"] == mode].secs.tolist()
    if ss:
        print(f"  mode {mode:2s} {len(ss):4d} images  {np.mean(ss):.3f}s each"
              + ("   (containment search costs the difference)" if mode == "B" else ""))
for arm in ("poisson", "alpha"):
    ss = MANI[MANI.arm_used == arm].secs.tolist()
    if ss:
        print(f"  {arm:8s} {len(ss):4d} images  {np.mean(ss):.3f}s each")

GEN_SECONDS = 60.0      # per-call cost of the Stage 1 generator; set to YOUR measured value
print(f"\nAgainst per-sample generative synthesis at {GEN_SECONDS:.0f}s/image:")
print(f"  {n_img} images generatively  {n_img * GEN_SECONDS:10.0f}s")
print(f"  {n_img} images via FLASH 4-5 {SYNTH_SECONDS:10.1f}s")
print(f"  ratio                        {n_img * GEN_SECONDS / max(SYNTH_SECONDS, 1e-6):10.1f}x")
print("\nStages 1-3 are NOT in this ratio -- they are the fixed cost this notebook amortises.")
print("Quote the end-to-end number (Stages 1-3 + this) in the paper, not this ratio alone.")
