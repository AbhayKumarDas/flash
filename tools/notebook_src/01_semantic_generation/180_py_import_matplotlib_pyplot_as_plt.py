import matplotlib.pyplot as plt
import textwrap

for cat in CATS:
    recs = [r for r in manifest.values() if r["category"] == cat]
    if not recs:
        continue
    print("=" * 100)
    print(f"  {cat.upper()}   ({len(recs)} images)")
    print("=" * 100)

    for r in recs:
        p = r["provenance"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.imshow(Image.open(r["source_path"]).convert("RGB"))
        ax.set_title(r["image"], fontsize=9)      # image tag == filename on disk
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"  image  : {r['image']}")
        print(f"  family : {p['family']}   |   defect: {r['spec']['defect_name']}   "
              f"|   {p['gen_seconds']}s"
              f"{'   (' + str(p['retries']) + ' retries)' if p['retries'] else ''}")
        print("  prompt :")
        for line in textwrap.wrap(r["paragraph"].split("\n\nHARD RULES")[0], 92):
            print("           " + line)
        print()

# -----------------------------------------------------------------------------------------
# HANDOFF -- MANUAL AT PRESENT.
# Stage 1 ends here. Every normal image above now has its Anomaly Prompt. The prompt and its
# matching normal image are currently pasted into ChatGPT BY HAND, one pair at a time, and
# the returned anomaly image I_a is saved back for Stage 2 (DiffMask).
#
# Cost of that manual step: roughly 50-60 s per image, so ~24 min for the 24-image bank.
# This is the 60 s/call figure the paper's efficiency analysis assumes.
#
# Nothing in this notebook calls a generator. Moving the handoff onto the OpenAI Images API
# would cut the per-image time substantially -- no browser round trip, no copy-paste, and
# calls can run concurrently -- as well as recording the model version, which the manual
# route cannot. That module reads manifest.json and writes one I_a per row.
# -----------------------------------------------------------------------------------------
