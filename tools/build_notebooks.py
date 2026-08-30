#!/usr/bin/env python
"""Build the module notebooks from editable source fragments.

WHY THE NOTEBOOKS ARE GENERATED

    A .ipynb is a JSON file with source split across lists of strings. Editing one directly
    produces diffs nobody can review, and two people touching the same notebook produces a merge
    conflict nobody can resolve. So the reviewable form of each notebook lives in
    tools/notebook_src/ as ordinary .md and .py files, and this script assembles them.

    Edit the fragments, run this script, commit both.

LAYOUT

    tools/notebook_src/<notebook name>/
        000_md_title.md          becomes a markdown cell
        010_py_imports.py        becomes a code cell
        020_md_config.md         ...

    Fragments are ordered by the leading number, which is spaced in tens so a cell can be
    inserted without renumbering the rest. The `md` or `py` in the name decides the cell type;
    the text after it is a label for humans and is otherwise ignored.

COMMANDS

    python tools/build_notebooks.py --build            write notebooks/*.ipynb
    python tools/build_notebooks.py --check            verify they match the fragments
    python tools/build_notebooks.py --extract          refresh fragments FROM the notebooks
    python tools/build_notebooks.py --build --only 02  restrict to one notebook

    --check is the one to run in CI. It fails if a notebook has been edited directly, which is
    the failure mode this whole arrangement exists to prevent.

    --extract is a repair tool, not part of the normal loop. It overwrites the fragments with
    whatever is currently in the notebooks. Use it once when adopting this layout, or after
    someone has edited a notebook directly and you want to keep their changes.

OUTPUTS ARE NEVER WRITTEN

    Generated cells carry no execution counts and no outputs, which is the same state nbstripout
    leaves a notebook in on commit. Run the notebook to see results; do not commit them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "tools" / "notebook_src"
OUT = REPO / "notebooks"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

FRAGMENT = re.compile(r"^(\d+)_(md|py)_(.+)\.(md|py)$")


def _lines(text: str) -> list[str]:
    """Notebook JSON stores source as a list of lines, each keeping its newline except the last."""
    text = text.rstrip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _slug(text: str, kind: str) -> str:
    """A short filename label taken from the cell's own first meaningful line."""
    for line in text.split("\n"):
        line = line.strip().lstrip("#").lstrip("=").strip()
        if line and not set(line) <= set("-=_ "):
            break
    else:
        line = kind
    line = re.sub(r"[^a-z0-9]+", "_", line.lower()).strip("_")
    return (line or kind)[:40]


# --------------------------------------------------------------------------------- build
def build_one(name: str) -> dict:
    """Assemble one notebook from its fragment directory."""
    folder = SRC / name
    if not folder.is_dir():
        sys.exit(f"no fragments for {name}: expected {folder}")

    frags = []
    for p in sorted(folder.iterdir()):
        m = FRAGMENT.match(p.name)
        if not m:
            if p.name != "README.md":
                print(f"  skipping unrecognised file: {p.name}")
            continue
        order, kind, _, ext = m.groups()
        if (kind == "md") != (ext == "md"):
            sys.exit(f"{p.name}: cell type '{kind}' does not match extension '.{ext}'")
        frags.append((int(order), kind, p))

    if not frags:
        sys.exit(f"{folder} contains no fragments")

    cells = []
    for _, kind, p in sorted(frags):
        text = p.read_text(encoding="utf-8")
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _lines(text)})
    return {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}


def notebook_names() -> list[str]:
    if not SRC.is_dir():
        sys.exit(f"no fragment directory: {SRC}\nRun with --extract first to create it.")
    return sorted(d.name for d in SRC.iterdir() if d.is_dir())


def cmd_build(only: str | None) -> int:
    for name in notebook_names():
        if only and only not in name:
            continue
        nb = build_one(name)
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  wrote {path.relative_to(REPO)}  {len(nb['cells'])} cells "
              f"({n_md} markdown, {len(nb['cells']) - n_md} code)")
    return 0


# --------------------------------------------------------------------------------- check
def cmd_check(only: str | None) -> int:
    """Fail if any notebook on disk differs from what the fragments produce."""
    bad = 0
    for name in notebook_names():
        if only and only not in name:
            continue
        path = OUT / f"{name}.ipynb"
        if not path.exists():
            print(f"  MISSING  {path.relative_to(REPO)}")
            bad += 1
            continue
        want = build_one(name)["cells"]
        have = json.loads(path.read_text(encoding="utf-8"))["cells"]
        if len(want) != len(have):
            print(f"  DIFFERS  {name}: {len(have)} cells on disk, {len(want)} from fragments")
            bad += 1
            continue
        # Compare source only. Outputs and execution counts are stripped on commit and are not
        # the builder's business.
        diff = [i for i, (w, h) in enumerate(zip(want, have, strict=True))
                if "".join(w["source"]) != "".join(h["source"])
                or w["cell_type"] != h["cell_type"]]
        if diff:
            print(f"  DIFFERS  {name}: cells {diff[:8]}"
                  f"{' ...' if len(diff) > 8 else ''}")
            bad += 1
        else:
            print(f"  ok       {name}")
    if bad:
        print(f"\n{bad} notebook(s) do not match their fragments.")
        print("Either rebuild them:        python tools/build_notebooks.py --build")
        print("or keep the direct edits:   python tools/build_notebooks.py --extract")
    return 1 if bad else 0


# --------------------------------------------------------------------------------- extract
def cmd_extract(only: str | None) -> int:
    """Write fragments FROM the notebooks. Overwrites whatever is in notebook_src."""
    for path in sorted(OUT.glob("*.ipynb")):
        name = path.stem
        if only and only not in name:
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        folder = SRC / name
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            if FRAGMENT.match(old.name):
                old.unlink()

        for i, cell in enumerate(nb["cells"]):
            text = "".join(cell["source"])
            kind = "md" if cell["cell_type"] == "markdown" else "py"
            ext = "md" if kind == "md" else "py"
            fname = f"{i * 10:03d}_{kind}_{_slug(text, kind)}.{ext}"
            (folder / fname).write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        print(f"  {name}: {len(nb['cells'])} fragments -> "
              f"{folder.relative_to(REPO)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="build_notebooks",
        description="Assemble the module notebooks from tools/notebook_src/.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="write notebooks from fragments")
    g.add_argument("--check", action="store_true",
                   help="verify notebooks match fragments; exits 1 if not")
    g.add_argument("--extract", action="store_true",
                   help="rewrite fragments from the notebooks (repair tool)")
    ap.add_argument("--only", metavar="SUBSTRING",
                    help="restrict to notebooks whose name contains this")
    a = ap.parse_args(argv)

    if a.extract:
        return cmd_extract(a.only)
    if a.check:
        return cmd_check(a.only)
    return cmd_build(a.only)


if __name__ == "__main__":
    raise SystemExit(main())
