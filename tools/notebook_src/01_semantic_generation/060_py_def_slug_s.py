def _slug(s):
    """'shell crack' -> 'shell_crack'. Used for filenames and defect-bank keys."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:32]


def build_paragraph(spec):
    """VLM-1 JSON -> the text handed to the image generator.

    Pure string substitution: no model involved, so the same spec always gives the same
    prompt.
    """
    body = TEMPLATE.format(target=spec["target"], size=spec["size"],
                           defect=spec["defect"], where=spec["where"].rstrip("."))
    return body + "\n\n" + HARD_RULES
