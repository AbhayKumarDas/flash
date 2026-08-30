# NEUTRAL. An earlier version of this prompt told the model that most regions are not defects
# and that "no change" is the answer to reach for first. That is a thumb on the scale, and on a
# 7B model it collapsed the accept rate -- trading the original false-accept problem for a
# false-reject one. State the task and the failure modes; do not state a base rate.
SYS_VLM2 = (
    "You are a quality-control inspector for industrial visual inspection. "
    "You are shown a BEFORE and an AFTER photograph of the same product, plus a candidate "
    "region. Your job is to say what changed inside that region, and whether the change is a "
    "genuine manufacturing defect or an artifact of the imaging and extraction process. "
    "Judge only on the evidence in the two photographs. Both answers are equally acceptable: "
    "report a defect when you can see one, and report no change when the two photographs look "
    "the same inside the outline. "
    "Reply with a single JSON object and no other text."
)

# Three-panel prompt. The before-panel is what makes this answerable: "is this walnut crevice
# damage or normal shell?" cannot be decided from one image, and a model forced to decide it
# anyway falls back on "does this look irregular", which is why plain texture was being accepted.
Q_VLM2_3 = """Panel 1 (LEFT) is a reference photograph of a {cat} BEFORE any change.
Panel 2 (MIDDLE) is the SAME view of the SAME object AFTER an edit was applied.
Panel 3 (RIGHT) is panel 2 with a candidate region outlined in red.
{multi}
Work in this order:

STEP 1. Compare panel 1 and panel 2 INSIDE the outlined area only. State what is different.
        If they look the same there, the answer is no change -- say so and stop.
STEP 2. Only if something changed, decide whether that change is a genuine defect on a {cat}
        ({kind_h}), or one of these extraction artifacts:
          - normal product texture, grain, weave or print that shifted between the two shots
          - a shadow, highlight, reflection, or overall exposure difference
          - a registration seam or edge halo along the object's own boundary
            (typically a long thin strip)
          - blur, noise, or a rendering artifact with no physical cause
          - a normal design feature: a hole, rim, edge, seam or printed mark that is
            SUPPOSED to be there and is present in panel 1 too

Reply with exactly this JSON and nothing else:
{{"changed": true or false,
  "what_changed": "<what differs between panel 1 and panel 2, or \"nothing\">",
  "is_defect": true or false,
  "defect_type": "<short noun phrase, or \"none\">",
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence, max 25 words>",
  "artifact_kind": "<one of: texture, shadow, seam, blur, design_feature, none>"}}"""

# Fallback when registration produced no usable before-panel. Same task, weaker evidence --
# and it says so, so the model is not misled into thinking it has a reference it does not have.
Q_VLM2_2 = """Panel 1 (LEFT) is a crop from a photograph of a {cat} ({kind_h}).
Panel 2 (RIGHT) is the same crop with a candidate region outlined in red.

No before-image is available for this crop, so judge the outlined region on its own.
The region was proposed by an automatic difference detector, so check whether it is instead
normal texture, a shadow, a registration seam, or a design feature that is supposed to be
there -- but call it a defect if that is what it is.
{multi}
Reply with exactly this JSON and nothing else:
{{"changed": true or false,
  "what_changed": "<what looks anomalous, or \"nothing\">",
  "is_defect": true or false,
  "defect_type": "<short noun phrase, or \"none\">",
  "confidence": <0.0 to 1.0>,
  "reason": "<one sentence, max 25 words>",
  "artifact_kind": "<one of: texture, shadow, seam, blur, design_feature, none>"}}"""

MULTI_CLAUSE = """
NOTE: exactly ONE defect was introduced into this image, but {n} separate regions were
detected in it. This is region {k} of {n}, ranked {k} by area. At most one of the {n} can be
the introduced defect -- the others are extraction artifacts. A long thin strip is very
likely a registration seam rather than a defect. Judge THIS region on its own merits and be
correspondingly stricter.
"""

KIND_HUMAN = {"foreign_object": "defects here are usually foreign objects or contamination",
              "surface":        "defects here are usually surface damage, marks, or deformation",
              "unknown":        "defect appearance varies"}


def vlm_panels(entry, ctx=VLM_CTX):
    """[registered normal | anomaly | anomaly + outline] at identical coordinates.

    Returns (PIL image, has_reference). The crop is WIDER than the banked one (VLM_CTX vs
    CTX_EXPAND) so clean substrate stays in frame for the model to compare against.
    """
    d = RAW[entry["donor"]]; r = entry["_region"]
    anom, msk = d["anomaly"], d["mask"]
    H, W = anom.shape[:2]
    dh, dw = r["h"], r["w"]
    side = int(np.clip(round(max(dh, dw) * ctx), 48, min(H, W)))
    cy, cx = r["y"] + dh // 2, r["x"] + dw // 2
    ty = int(np.clip(cy - side // 2, 0, H - side))
    tx = int(np.clip(cx - side // 2, 0, W - side))

    a_crop = np.ascontiguousarray(anom[ty:ty + side, tx:tx + side])
    m_crop = (msk[ty:ty + side, tx:tx + side] > 127).astype(np.uint8)

    # The registered normal lives at diffmask's WORKING resolution, not the full frame, so the
    # box has to be rescaled before it can be cut from it.
    n_crop, ref = None, d.get("ref_aligned")
    if ref is not None and ref.size:
        fy, fx = ref.shape[0] / float(H), ref.shape[1] / float(W)
        ry, rx = int(round(ty * fy)), int(round(tx * fx))
        rh, rw = max(8, int(round(side * fy))), max(8, int(round(side * fx)))
        cand = ref[ry:ry + rh, rx:rx + rw]
        # Registration leaves black where the reference did not cover the frame. A mostly-black
        # before-panel is worse than none: it invites the model to call the difference a change.
        if cand.size and float((cand.reshape(-1, 3).max(1) > 8).mean()) > 0.90:
            n_crop = cv2.resize(cand, (side, side), interpolation=cv2.INTER_LINEAR)

    marked = a_crop.copy()
    cont, _ = cv2.findContours(m_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marked, cont, -1, (255, 0, 0), max(2, side // 160))

    panels = ([n_crop] if n_crop is not None else []) + [a_crop, marked]
    gap = np.full((side, 12, 3), 255, np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.concatenate([out, gap, p], axis=1)
    return Image.fromarray(out), (n_crop is not None)


def vlm2(entry):
    """Semantic validation of one crop. Never raises -- a bad reply is a reject with a reason."""
    img, has_ref = vlm_panels(entry)
    multi = ""
    if ONE_DEFECT_PER_IMAGE and entry.get("n_regions", 1) > 1:
        multi = MULTI_CLAUSE.format(n=entry["n_regions"], k=entry["region_rank"])
    tmpl = Q_VLM2_3 if has_ref else Q_VLM2_2
    q = tmpl.format(cat=entry["cat"].replace("_", " "), multi=multi,
                    kind_h=KIND_HUMAN.get(entry["kind"], KIND_HUMAN["unknown"]))
    raw, secs = vlm(img, SYS_VLM2, q, temperature=0.0)
    try:
        v = grab_json(raw)
        changed = bool(v.get("changed", True))
        raw_def = bool(v.get("is_defect", False))
        # A reply that says "nothing changed" and then ticks is_defect has contradicted itself.
        # Rejecting is the safe reading, but it is logged as its own class rather than folded
        # into "not a defect" -- if this fires often the prompt is confusing the model, which is
        # a prompt bug and not evidence about the crop.
        is_def = raw_def and changed
        contradiction = raw_def and not changed
        return dict(changed=changed, is_defect=is_def, contradiction=contradiction,
                    what_changed=str(v.get("what_changed", ""))[:120],
                    defect_type=str(v.get("defect_type", "none"))[:60],
                    confidence=float(v.get("confidence", 0.0)),
                    reason=str(v.get("reason", ""))[:200],
                    artifact_kind=str(v.get("artifact_kind", "none"))[:20],
                    has_ref=has_ref, raw=raw, secs=secs, parse_ok=True)
    except Exception as e:
        # An unparseable reply is a reject, not a crash, and it is recorded as its own class so
        # section 9 can separate "the model said no" from "the model said something unusable".
        return dict(changed=False, is_defect=False, contradiction=False, what_changed="",
                    defect_type="none", confidence=0.0,
                    reason=f"unparseable reply: {type(e).__name__}", artifact_kind="none",
                    has_ref=has_ref, raw=raw, secs=secs, parse_ok=False)


# The first call is timed like the rest but flagged: it carries CUDA warmup no later call pays.
for i, e in enumerate(CROPS):
    e["vlm"] = vlm2(e)
    e["vlm"]["warmup"] = (i == 0)

_pass = [e for e in CROPS if e["vlm"]["is_defect"] and e["vlm"]["confidence"] >= VLM_MIN_CONF]
_bad  = [e for e in CROPS if not e["vlm"]["parse_ok"]]
_secs = [e["vlm"]["secs"] for e in CROPS]
print(f"VLM-2: {len(_pass)}/{len(CROPS)} accepted at confidence >= {VLM_MIN_CONF} "
      f"| {len(_bad)} unparseable | {sum(_secs):.0f}s total\n")

_n_ref = sum(1 for e in CROPS if e["vlm"]["has_ref"])
print(f"{_n_ref}/{len(CROPS)} crops judged WITH a before-panel; "
      f"{len(CROPS)-_n_ref} fell back to the 2-panel prompt\n")
print(f"{'key':30s} {'ref':>4s} {'chg':>4s} {'acc':>4s} {'conf':>5s} {'secs':>6s} "
      f"{'artifact':14s} what changed")
for e in CROPS:
    v = e["vlm"]
    print(f"{e['key']:30s} {'3p' if v['has_ref'] else '2p':>4s} "
          f"{'yes' if v['changed'] else 'NO':>4s} "
          f"{'YES' if v['is_defect'] else 'no':>4s} {v['confidence']:5.2f} "
          f"{v['secs']:6.1f}{'*' if v['warmup'] else ' '} "
          f"{v['artifact_kind'][:14]:14s} {(v['what_changed'] or v['reason'])[:52]}")

# "nothing changed" is the answer the old single-image prompt could never give. If it never
# fires, the before-panel is not doing its job and the gate is still guessing from texture.
_nochg = [e for e in CROPS if not e["vlm"]["changed"]]
print(f"\n{len(_nochg)} crop(s) reported NO CHANGE between the before and after panels"
      + ("  <- these are the false positives the 3-panel prompt exists to catch"
         if _nochg else "  <- suspicious: check the before-panel is really being shown"))
for e in _nochg:
    print(f"    {e['key']:30s} {e['vlm']['reason'][:70]}")
if _secs:
    print(f"\n* first call, includes CUDA warmup")
    print(f"per crop: mean {np.mean(_secs):.1f}s | median {np.median(_secs):.1f}s "
          f"| min {min(_secs):.1f}s | max {max(_secs):.1f}s")

# Is the confidence number worth thresholding on? If it is flat, it is not, and
# USE_VLM_CONFIDENCE should stay off. Printed rather than assumed.
_cf = [e["vlm"]["confidence"] for e in CROPS]
if _cf:
    import numpy as _np
    _h, _ = _np.histogram(_cf, bins=[0, .2, .4, .6, .8, 1.01])
    print("\nconfidence distribution  "
          + "  ".join(f"{lo:.1f}-{hi:.1f}:{n}" for (lo, hi), n in
                      zip([(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0)], _h)))
    print(f"  spread {min(_cf):.2f}-{max(_cf):.2f}"
          + ("   <- flat: do NOT gate on it (USE_VLM_CONFIDENCE stays False)"
             if max(_cf) - min(_cf) < 0.25 else "   <- has spread; gating on it is defensible"))

# What the accepted crops were CALLED. This is the bank's second index axis, not decoration --
# section 10 writes entries under <category>/<defect_type>/, per paper 3.3.
from collections import Counter
_types = Counter(e["vlm"]["defect_type"].strip().lower() for e in _pass)
print(f"\ndefect types named by VLM-2 across {len(_pass)} accepted crops:")
for t, n in _types.most_common():
    print(f"  {n:3d}  {t}")
if len(_types) == 1 and len(_pass) > 3:
    print("  -- one label for every crop. VLM-2 is acting as a pure yes/no gate here; the")
    print("     defect_type axis is carrying no information and the bank index is flat.")
