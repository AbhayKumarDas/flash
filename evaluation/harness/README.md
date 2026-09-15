# Paper evaluation harness

This directory contains the detector-side evaluation code used for the FLASH calibration
study. It is separate from the generation pipeline under `src/flash/`: the generation
pipeline creates the synthetic images, while this harness trains an Anomalib detector and
measures how well a threshold calibrated on synthetic anomalies transfers to real MVTec AD 2
test anomalies.

The published comparison uses five detectors across eight MVTec AD 2 categories and three
seeds. The paper's Real setting is the oracle reference: its threshold is calibrated with
real test anomalies. The Perlin, FLASH, and AnoStyler settings calibrate thresholds with
held-out normals plus generated anomalies, then transfer those thresholds to the same real
test set. The optional synthetic-set diagnostic is kept only to characterize calibration
set difficulty; it is not a headline benchmark setting.

## Installation

Install Git LFS before obtaining the evaluation artifacts. The evaluation dependency is pinned
to the anomalib commit used by the reported experiments:

```bash
git lfs install
pip install -r requirements-eval.txt
```

Do not substitute a floating anomalib branch or a newer PyPI release when reproducing the
published values. The pinned tree includes the MVTec AD 2 test-split handling and the exact
SuperADD implementation used for these runs. The models are imported from anomalib; this
repository does not vendor a copy of anomalib.

## Evaluation data layout

Set `FLASH_EVAL_DATA_ROOT` to a directory with the following layout:

```text
<FLASH_EVAL_DATA_ROOT>/
|-- MVTec_AD_2/
|-- dtd/
`-- SynthetciGenMVAD2/
    |-- MVTec_AD_2_hybrid_0/
    |-- MVTec_AD_2_hybrid_1/
    |-- MVTec_AD_2_hybrid_2/
    |-- MVTec_AD_2_anostyler_0/
    |-- MVTec_AD_2_anostyler_1/
    `-- MVTec_AD_2_anostyler_2/
```

`MVTec_AD_2` is the official dataset. `dtd` is used by the stock Perlin baseline. The
`SynthetciGenMVAD2` directories are the exact three-seed, evaluation-ready FLASH and AnoStyler
calibration sets used by the paper tables. They are separate from the small illustrative
`data/output/module3_final_synthetic_images.zip` export in this repository; that sample is not
sufficient to reproduce the reported detector results.

The full evaluation-ready archive is intentionally not duplicated in this source tree yet.
Once it is hosted, place its download URL and checksum here and extract it under the layout
above.

## Reproducing the runs

From the repository root, set the data location and run the three paper comparisons:

```bash
export FLASH_EVAL_DATA_ROOT="$PWD/data/evaluation"

python -m evaluation.harness.sweep \
    --experiment cross_model_comparison --gpus 0 1 --procs-per-gpu 1
python -m evaluation.harness.sweep \
    --experiment anostyler_comparison --gpus 0 1 --procs-per-gpu 1
python -m evaluation.harness.sweep \
    --experiment dinomaly_comparison --gpus 0 1 --procs-per-gpu 1
```

Runs are resumable: completed jobs are stored as JSON under
`evaluation/harness/results/raw/`. Rebuild the aggregate CSV without running jobs with:

```bash
python -m evaluation.harness.sweep --aggregate-only
```

For a small single-job smoke test:

```bash
python -m evaluation.harness.run_one \
    --experiment cross_model_comparison \
    --dataset mvtec2 \
    --category rice \
    --model patchcore \
    --seed 1 \
    --sources Perlin \
    --calibration heldout
```

The reference aggregate containing only the rows used by the paper's final comparison is
`evaluation/results/paper_table1_results.csv`. The comparison-to-table mapping and the
published summary statistics are documented in `paper_accuracy_results.md`.

## Reproducibility boundary

This harness reproduces the detector-side calibration experiment when the same anomalib
commit, MVTec AD 2 release, pregenerated calibration sets, seeds, and runtime settings are
used. The generation notebooks under `notebooks/` remain the reference implementation for
recreating the synthetic data itself. Re-running the generation pipeline can produce different
images if the attached generative model or prompt service changes; the evaluation-ready
three-seed calibration sets are therefore required for byte-for-byte reproduction of the
reported table.
