#!/usr/bin/env python
"""Regenerate the directory tree printed at the bottom of README.md.

WHY

    A hand-written layout section is wrong within a week. This script reads the repository as
    it actually is on disk and rewrites the block between the two markers in README.md:

        <!-- BEGIN DIRECTORY TREE -->
        <!-- END DIRECTORY TREE -->

    Add a file, delete a folder, rename a notebook: run this and the README follows.

WHAT IS SHOWN

    Everything git would track, so ignored files (caches, virtualenvs, run outputs) never
    appear. Git decides that, not this script, which means the tree and .gitignore can never
    drift apart. If git is unavailable the script falls back to a small built-in skip list.

    Two readability rules are applied on top:

      * Directories listed in COLLAPSE print a one-line summary instead of their contents.
        The notebook fragment folders hold roughly 60 files each and listing them would bury
        everything else.
      * Paths listed in NOTES get a short trailing comment. Keep these to a few words.

USAGE

    python tools/update_readme_tree.py            rewrite the block in README.md
    python tools/update_readme_tree.py --check    exit 1 if the block is out of date
    python tools/update_readme_tree.py --print    write nothing, print the tree

    --check is the pre-commit form. It fails when someone adds or removes a file without
    refreshing the README, and the fix is to run the script with no arguments.
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

BEGIN = "<!-- BEGIN DIRECTORY TREE -->"
END = "<!-- END DIRECTORY TREE -->"

ROOT_LABEL = "flash/"

# Directories whose contents are summarised rather than listed, keyed by path relative to the
# repository root. The value is the text that replaces the listing.
COLLAPSE = {
    "tools/notebook_src/01_semantic_generation": "{n} cell fragments",
    "tools/notebook_src/02_extraction_and_banking": "{n} cell fragments",
    "tools/notebook_src/03_localisation_and_synthesis": "{n} cell fragments",
}

# Short trailing comments, keyed by path relative to the repository root.
NOTES = {
    "notebooks": "the pipeline, run in order",
    "notebooks/01_semantic_generation.ipynb": "Stage 1",
    "notebooks/02_extraction_and_banking.ipynb": "Stages 2 and 3",
    "notebooks/03_localisation_and_synthesis.ipynb": "Stages 4 and 5",
    "src/flash": "operators the notebooks import",
    "src/flash/config.py": "every threshold, one file",
    "src/flash/diffmask.py": "Stage 2, mask recovery",
    "src/flash/obs.py": "Stage 4, foreground region",
    "src/flash/mrsp.py": "Stage 4, placement field",
    "src/flash/placement.py": "Stage 5, original and adaptive modes",
    "src/flash/compositing.py": "Stage 5, hybrid blending",
    "src/flash/bank.py": "defect bank read and write",
    "data/input": "what each module consumes",
    "data/output": "what each module produced, shipped via Git LFS",
    "evaluation": "detector-side calibration study",
    "docs/modules": "one reference page per module",
    "docs/paper": "the paper",
    "docs/assets": "figures used by this README",
    "tools/build_notebooks.py": "notebooks are generated, not edited",
    "tests/test_end_to_end.py": "one synthetic image per category",
}

# Used only when git is not available.
FALLBACK_SKIP = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".venv", "venv", "temp", "*.egg-info", "*.pyc",
}


def _git_ignored(paths: list[Path]) -> set[Path]:
    """Ask git which of these paths it would ignore. Empty set if git cannot answer."""
    if not paths:
        return set()
    payload = "\0".join(str(p.relative_to(REPO)).replace("\\", "/") for p in paths)
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            input=payload, capture_output=True, text=True, cwd=REPO, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    # Exit code 128 means git itself failed, for example outside a work tree. 1 means nothing
    # matched, which is a normal answer.
    if proc.returncode not in (0, 1):
        return set()
    return {REPO / line for line in proc.stdout.split("\0") if line}


def _fallback_ignored(paths: list[Path]) -> set[Path]:
    out = set()
    for p in paths:
        if any(fnmatch.fnmatch(p.name, pat) for pat in FALLBACK_SKIP):
            out.add(p)
    return out


def _children(folder: Path, use_git: bool) -> list[Path]:
    """Visible children of a folder, directories first, each group alphabetical."""
    entries = [p for p in folder.iterdir() if p.name != ".git"]
    ignored = _git_ignored(entries) if use_git else _fallback_ignored(entries)
    entries = [p for p in entries if p not in ignored]
    if use_git:
        # A folder holding nothing but ignored files is noise. Empty folders that are genuinely
        # part of the layout, such as a placeholder for reports, are kept.
        entries = [p for p in entries if not (p.is_dir() and _is_empty_of_content(p))]
    return sorted(entries, key=lambda p: (not p.is_dir(), p.name.lower()))


def _is_empty_of_content(folder: Path) -> bool:
    """True when a folder holds files but every one of them is ignored."""
    kids = [p for p in folder.iterdir() if p.name != ".git"]
    if not kids:
        return False  # genuinely empty, keep it
    return len(_git_ignored(kids)) == len(kids)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("\\", "/")


def _render(folder: Path, prefix: str, lines: list[str], use_git: bool) -> None:
    kids = _children(folder, use_git)
    for i, path in enumerate(kids):
        last = i == len(kids) - 1
        stem = "`-- " if last else "|-- "
        name = path.name + ("/" if path.is_dir() else "")
        note = NOTES.get(_rel(path))
        lines.append(f"{prefix}{stem}{name}" + (f"    # {note}" if note else ""))

        if not path.is_dir():
            continue
        deeper = prefix + ("    " if last else "|   ")
        summary = COLLAPSE.get(_rel(path))
        if summary is not None:
            n = sum(1 for p in path.iterdir() if p.is_file())
            lines.append(f"{deeper}`-- ... {summary.format(n=n)}")
        else:
            _render(path, deeper, lines, use_git)


def build_tree() -> str:
    use_git = bool(_git_ignored([REPO / "pyproject.toml"]) or _git_available())
    lines = [ROOT_LABEL]
    _render(REPO, "", lines, use_git)
    return "\n".join(lines)


def _git_available() -> bool:
    try:
        proc = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, cwd=REPO, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def render_block(tree: str) -> str:
    return f"{BEGIN}\n\n```text\n{tree}\n```\n\n{END}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="update_readme_tree",
        description="Rewrite the directory tree block in README.md from the repository itself.")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the README block is out of date")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the tree and exit")
    a = ap.parse_args(argv)

    tree = build_tree()
    if a.show:
        print(tree)
        return 0

    text = README.read_text(encoding="utf-8")
    start, stop = text.find(BEGIN), text.find(END)
    if start == -1 or stop == -1 or stop < start:
        sys.exit(f"markers not found in {README.name}. Expected {BEGIN} and {END}.")

    updated = text[:start] + render_block(tree) + text[stop + len(END):]
    if updated == text:
        print("README directory tree is up to date.")
        return 0
    if a.check:
        print("README directory tree is out of date.")
        print("Run: python tools/update_readme_tree.py")
        return 1
    README.write_text(updated, encoding="utf-8")
    print(f"updated {README.name}: {len(tree.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
