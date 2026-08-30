# Module outputs

The results of a full run, shipped so the published numbers can be checked without re-running the
pipeline. Archives are stored with Git LFS; see `../input/README.md` for how to fetch them.

| file | produced by | contents |
|---|---|---|
| `module1_prompts.csv` | Module 1 | one row per image: tag, defect family, defect, prompt, seconds |
| `module2_defect_bank.zip` | Module 2 | the defect bank: crops, masks, manifest |
| `module3_final_synthetic_images.zip` | Module 3 | synthetic anomalies and their ground-truth masks |

Everything here is reproducible by running the notebooks in order. Rerunning will overwrite these
files, so keep a copy if you want to compare against the shipped versions.

`module2_defect_bank.zip` is the same archive as `../input/module3_defect_bank.zip`. It is stored
in both places so each module can be run standalone.
