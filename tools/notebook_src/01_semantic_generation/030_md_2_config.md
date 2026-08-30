## 2. Config

The only cell you need to edit. Nothing tunable is hidden in the cells below.

| block | contains |
|---|---|
| run knobs | `SEED`, mount paths, `LOAD_4BIT`, pixel caps, `N_IMAGES`, `DEFECT_SOURCE`, output dir |
| dataset context | one paragraph telling the model what kind of images these are |
| per-category `CONFIG` | `object`, `families`, `defects`, `notes`, optional `max_pixels` |
| prompt text | `FAMILIES`, `SIZE_WORDS`, `TEMPLATE`, `HARD_RULES` |
| model wording | `SYS_VLM1` and `ASK_VLM1` |

The per-category entries are hints rather than rules. `defects` seeds the vocabulary and `notes`
warns about scenes that are commonly misread, but the model still looks at the image.

To add a category, add an entry to `CONFIG` and list it in `CATS`.
