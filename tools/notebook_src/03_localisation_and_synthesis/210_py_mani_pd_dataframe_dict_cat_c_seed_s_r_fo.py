MANI = pd.DataFrame([dict(cat=c, seed=s, **r) for (c, s), v in CORPUS.items() for r in v])
if MANI.empty:
    raise SystemExit("no images were synthesised -- see the per-category lines above")
MANI["dissolved"] = MANI.d_in < DISSOLVED_T
MANI.to_csv(os.path.join(SYN_DIR, "manifest.csv"), index=False)

json.dump(dict(seed=SEED, bank_root=BANK_ROOT, ad2_root=AD2_ROOT,
               cats=LIVE_CATS, seeds=SEEDS, n_synth=N_SYNTH, n_images=int(n_img),
               work_size=WORK_SIZE, coverage=AD2_COVERAGE, noise_scale=AD2_NOISE_SCALE,
               octaves=OCTAVES_FBM, persistence=PERSISTENCE, mrsp_alpha=MRSP_ALPHA,
               placement_mix=PLACEMENT_MIX, gamma_range=list(GAMMA_RANGE),
               collar_frac=COLLAR_FRAC, r_eq_poisson=R_EQ_POISSON, dissolved_t=DISSOLVED_T,
               mrsp_device=MRSP_DEVICE, synth_seconds=round(SYNTH_SECONDS, 1)),
          open(os.path.join(SYN_DIR, "run_config.json"), "w"), indent=2)

print("per category:")
print(MANI.groupby("cat").agg(n=("d_in", "size"), d_in=("d_in", "mean"),
                              gt_frac=("gt_frac", "mean"), r_eq=("r_eq", "mean"),
                              dissolved=("dissolved", "sum")).round(3).to_string())

print(f"\nplacement mix: modeB={int((MANI['mode']=='B').sum())}/{len(MANI)}"
      f"  ({100*(MANI['mode']=='B').mean():.0f}%, target {100*PLACEMENT_MIX:.0f}%)")
print(f"routed to alpha: {int((MANI.arm_used=='alpha').sum())}/{len(MANI)}"
      f"  (mode B defects below r_eq {R_EQ_POISSON}px)")

n_bad = int(MANI.dissolved.sum())
if n_bad:
    print(f"\n{n_bad}/{len(MANI)} composites are dissolved (d_in < {DISSOLVED_T}): the ground "
          "truth marks a region the operator left identical to the host.")
    print(MANI[MANI.dissolved].groupby(["cat", "mode"]).size().to_string())
    print("  Kept on purpose. If one mode dominates this count, that is a result about the")
    print("  operator, not a bug to tune away.")
else:
    print(f"\nno dissolved composites (all d_in >= {DISSOLVED_T})")
