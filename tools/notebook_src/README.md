# Notebook source fragments

The `.ipynb` files in `notebooks/` are generated from the fragments here. Edit these, not the
notebooks.

Each subdirectory is one notebook. Each file is one cell:

```
000_md_title.md        a markdown cell
010_py_imports.py      a code cell
020_md_config.md       ...
```

Order comes from the leading number, spaced in tens so a cell can be inserted without renumbering
everything after it. The `md` or `py` decides the cell type. The rest of the name is a label and
has no effect.

## Working on a notebook

```
python tools/build_notebooks.py --build     rebuild after editing a fragment
python tools/build_notebooks.py --check     confirm notebooks match fragments
```

Commit the fragments and the rebuilt notebook together.

If someone edits a notebook directly and you want to keep their changes, run
`python tools/build_notebooks.py --extract` to fold them back into the fragments. That overwrites
everything here, so check the diff before committing.
