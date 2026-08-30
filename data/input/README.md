# Module inputs

These archives ship with the repository so a module can be run without running the ones before
it. They are stored with Git LFS.

| file | consumed by | contents |
|---|---|---|
| `General_Input.md` | all modules | dataset and model attachment instructions |
| `module2_generated_anomalies.zip` | Module 2 | the normal and anomaly image pairs from Module 1 |
| `module3_defect_bank.zip` | Module 3 | the validated defect bank from Module 2 |

## Getting them

Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/AbhayKumarDas/flash
```

If you cloned first, or you see small text files where the archives should be, run:

```bash
git lfs pull
```

A pointer file is a few hundred bytes and starts with `version https://git-lfs.github.com/spec/v1`.
If that is what you have, Git LFS was not installed when you cloned.

## Using them

Unzip into this directory, or attach the archive directly as a Kaggle Dataset and point the
notebook's mount path at it. Each module's first section says which path it reads.

Module 3 can take either `module3_defect_bank.zip` from here or `module2_defect_bank.zip` from
`data/output/`. They are the same archive: Module 2's output is Module 3's input.
