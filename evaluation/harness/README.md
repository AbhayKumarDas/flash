# Paper evaluation harness

This harness trains Anomalib detectors and measures transfer from synthetic calibration
anomalies to real MVTec AD 2 test anomalies. The paper compares Real, Perlin, FLASH, and
AnoStyler calibration across five detectors, eight categories, and three seeds.

## Installation

Install Git LFS and the pinned evaluation dependencies:

```bash
git lfs install
pip install -r requirements-eval.txt
```

The anomalib pin preserves the MVTec AD 2 split handling and SuperADD implementation used for
the reported values. Anomalib is not vendored here.

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

`MVTec_AD_2` is the official dataset; `dtd` supports Perlin. The `SynthetciGenMVAD2`
directories contain the three-seed FLASH and AnoStyler calibration sets. The small sample
archive under `data/output/` is not sufficient for detector reproduction. The full archive is
not included yet; add its URL and checksum here when hosted.

## Reproducing the runs

From the repository root:

```bash
export FLASH_EVAL_DATA_ROOT="$PWD/data/evaluation"

python -m evaluation.harness.sweep \
    --experiment cross_model_comparison --gpus 0 1 --procs-per-gpu 1
python -m evaluation.harness.sweep \
    --experiment anostyler_comparison --gpus 0 1 --procs-per-gpu 1
python -m evaluation.harness.sweep \
    --experiment dinomaly_comparison --gpus 0 1 --procs-per-gpu 1
```

Completed jobs are stored under `evaluation/harness/results/raw/` and can be resumed. Aggregate
existing rows with:

```bash
python -m evaluation.harness.sweep --aggregate-only
```

Smoke test:

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

The paper rows are in `evaluation/results/paper_table1_results.csv`; the table mapping is in
`evaluation/harness/paper_accuracy_results.md`.

## Reproducibility boundary

Exact table reproduction requires the pinned anomalib commit, MVTec AD 2 release, pre-generated
sets, seeds, and runtime settings. The notebooks under `notebooks/` generate new sets, but
model or prompt-service changes can alter their contents.
