# Keys the human rejected, translated from build_flash_replay_nb.py's DROP_REGIONS.
# Its format is (f"{cat}-{donor}", x, y); this notebook's key is f"{cat}_{donor}_{x}_{y}".
#
# THE WHOLE LIST IS ONE ENTRY. That is the honest state of the human label set today, and it
# is not enough to support a paper claim -- one artifact cannot establish a recall. Growing
# this set is the prerequisite for section 3.3, not an optional extra.
HUMAN_DROP = {
    "fruit_jelly_064_789_916",      # cup base, not a defect -- documented in _command.txt
}
HUMAN_LABELLED = set()      # every key the human actually looked at; defaults to all crops
MIN_LABELS_FOR_RATE = 8     # below this, print counts but refuse to quote a rate


def agreement_report(crops, human_drop, human_seen):
    seen = human_seen or {e["key"] for e in crops}
    scored = [e for e in crops if e["key"] in seen]
    covered = human_drop & {e["key"] for e in scored}
    if not scored or not covered:
        print("No crop in this run matches a key in HUMAN_DROP -- cannot conclude.")
        print(f"HUMAN_DROP holds {len(human_drop)} key(s); none of them were recovered here.")
        print("Either this run used different pairs, or the region coordinates shifted.")
        return None

    tp = [e for e in scored if e["key"] in human_drop and not e["gate_vlm"]]   # both reject
    fn = [e for e in scored if e["key"] in human_drop and e["gate_vlm"]]       # VLM accepted an artifact
    fp = [e for e in scored if e["key"] not in human_drop and not e["gate_vlm"]]
    tn = [e for e in scored if e["key"] not in human_drop and e["gate_vlm"]]

    n, n_pos = len(scored), len(tp) + len(fn)
    print(f"scored {n} crops against {n_pos} human rejection(s)\n")
    print(f"{'':22s}{'VLM-2 reject':>14s}{'VLM-2 accept':>14s}")
    print(f"{'human reject':22s}{len(tp):>14d}{len(fn):>14d}")
    print(f"{'human keep':22s}{len(fp):>14d}{len(tn):>14d}")
    print()
    if n_pos < MIN_LABELS_FOR_RATE:
        print(f"NOT ENOUGH LABELS. {n_pos} human rejection(s) covered; "
              f"{MIN_LABELS_FOR_RATE} is the floor for quoting a rate.")
        print("The counts above are real. A recall computed from them is not, and must not")
        print("go in the paper -- one artifact caught or missed swings it by 100 points.")
        print("\nTo make section 3.3 measurable: label the regions this run recovered, add the")
        print("rejected keys to HUMAN_DROP, and put every key you looked at in HUMAN_LABELLED.")
    else:
        print(f"agreement                  {(len(tp) + len(tn)) / n:.1%}")
        print(f"recall on known artifacts  {len(tp) / n_pos:.1%}"
              f"   <- the number that matters")
        if (len(fp) + len(tn)):
            print(f"false-reject rate          {len(fp) / (len(fp) + len(tn)):.1%}"
                  f"   (costs bank size, not correctness)")
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, n=n, n_pos=n_pos,
                conclusive=n_pos >= MIN_LABELS_FOR_RATE)


AGREE = agreement_report(CROPS, HUMAN_DROP, HUMAN_LABELLED)
