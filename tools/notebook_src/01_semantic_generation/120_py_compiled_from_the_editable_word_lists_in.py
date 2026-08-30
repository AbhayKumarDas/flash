# Compiled from the editable word lists in the config cell.
_rx = lambda words: re.compile(r"\b(" + "|".join(words) + r")\b", re.I)
VAGUE              = _rx(VAGUE_WORDS)
VAGUE_UNLESS_STAIN = _rx(STAIN_ONLY_WORDS)
NORMAL             = _rx(NORMAL_WORDS)


def validate(spec, cat, family, taken):
    """Raise if the reply is unusable. Every raise triggers a retry with feedback."""
    missing = [f for f in FIELDS if not spec.get(f)]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    if spec["defect_name"] in taken:
        raise ValueError("duplicate defect_name: " + spec["defect_name"])

    d = spec["defect"]

    # Under DEFECT_SOURCE == "config" the VLM may not invent a defect: what it returns has to
    # correspond to one of the entries listed for this category.
    if DEFECT_SOURCE == "config":
        allowed_defects = CONFIG.get(cat, {}).get("defects", [])
        if allowed_defects:
            words = set(re.findall(r"[a-z]{4,}", d.lower()))
            hit = any(words & set(re.findall(r"[a-z]{4,}", a.lower()))
                      for a in allowed_defects)
            if not hit:
                raise ValueError(f"{d!r} is not one of the listed defects for {cat}")

    if NORMAL.search(d) or NORMAL.search(spec.get("defect_name", "")):
        raise ValueError(f"names normal variation, not a defect: {d!r}")
    if VAGUE.search(d):
        raise ValueError(f"too vague, name the physical fault: {d!r}")
    if family != STAIN_FAMILY and VAGUE_UNLESS_STAIN.search(d):
        raise ValueError(f"too vague for this family, name the physical fault: {d!r}")
    if len(d.split()) < 2:
        raise ValueError(f"defect is one word, needs a noun phrase: {d!r}")

    # The object must be called what it is -- "the drink" for a tub of jelly is a mis-read.
    # The object must be called what it is. "the drink" for a tub of jelly is a mis-read and
    # every prompt built from it would be wrong. Matching is on 3+ char stems so "can" and
    # "vial" survive, and plurals match ("walnut" ~ "walnuts"). The category name is folded in
    # so "the wall plug at the left" matches the wallplugs category.
    stop = set(STOP_WORDS)

    def stems(txt):
        return {w[:5] for w in re.findall(r"[a-z]{3,}", txt.lower()) if w not in stop}

    allowed = stems(CONFIG.get(cat, {}).get("object", "")) | stems(cat.replace("_", " "))
    if allowed and not (stems(spec["target"]) & allowed):
        raise ValueError(f"target {spec['target']!r} does not name the object "
                         f"({CONFIG.get(cat, {}).get('object', cat).split(',')[0]})")
    return spec


def _fallback(cat, family, size, last_raw):
    """Used only after every retry failed. Built from CONFIG so it is still category-valid,
    and it keeps the last raw reply so the failure can be diagnosed from the JSON."""
    cfg = CONFIG.get(cat, {})
    defect = (cfg.get("defects") or ["surface fault"])[0]
    target = "the " + (cfg.get("object", "object").split(" with ")[0]
                       .replace("a printed ", "").replace("a sealed ", "")
                       .replace("a single sealed ", "").replace("a flat sheet of ", "")
                       .replace("a loose pile of ", "").replace("several ", "")
                       .replace("loose ", "").split(",")[0].split(" lying")[0]
                       .split(" filling")[0].strip())
    return {"defect_name": defect.split()[0] + " fault", "target": target,
            "defect": defect, "where": "on its visible surface",
            "size": size, "family": family, "FALLBACK": True,
            "last_raw_reply": last_raw}


def _constraints(cat):
    """Config hints injected into the ask. Empty when the config has nothing to say."""
    cfg, out = CONFIG.get(cat, {}), ""
    if DEFECT_SOURCE != "derive" and cfg.get("defects"):
        out += "The defect must be one of: " + ", ".join(cfg["defects"]) + ".\n"
    if cfg.get("notes"):
        out += "Note: " + cfg["notes"] + ".\n"
    return out


def vlm1(cat, img, family, size, taken=(), temperature=None, tries=None):
    """One Anomaly Prompt spec. Returns (spec, raw_reply, retries, generate seconds).

    A rejected reply is retried WITH the reason appended, so the model is told what was
    wrong rather than being asked the same question again. Sampling is turned on from the
    second attempt so a greedy failure does not simply repeat.
    """
    tries = TRIES if tries is None else tries
    temperature = TEMP_GREEDY if temperature is None else temperature
    mp    = CONFIG.get(cat, {}).get("max_pixels", MAX_PIXELS)
    obj   = CONFIG.get(cat, {}).get("object", "an industrial part")
    avoid = ("Do NOT reuse any of these defect names: " + ", ".join(taken) + ".\n") if taken else ""
    base  = ASK_VLM1.format(dataset_context=DATASET_CONTEXT, object=obj, family=family,
                            family_hint=FAMILIES[family], size=size,
                            constraints=_constraints(cat), avoid=avoid)
    gen_secs, raw, why = 0.0, "", None
    for t in range(tries):
        q = base if why is None else (
            base + f"\n\nYour previous answer was rejected: {why}. Fix exactly that and "
                   f"answer again, JSON only.")
        raw, secs = vlm(img, SYS_VLM1, q, max_new_tokens=MAX_NEW_TOKENS,
                        temperature=temperature if t == 0 else TEMP_SAMPLE, max_pixels=mp)
        gen_secs += secs
        try:
            spec = validate(grab_json(raw), cat, family, taken)
            spec["size"], spec["family"] = size, family
            return spec, raw, t, gen_secs
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
            print(f"      retry {t + 1}/{tries} -- {why}")
    print(f"      !! FALLBACK for {cat} / {family} -- last reply: {raw[:120]!r}")
    return _fallback(cat, family, size, raw), raw, tries, gen_secs
