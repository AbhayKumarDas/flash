# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enumerate and schedule experiment runs across multiple GPUs.

Each run is launched as an isolated subprocess pinned to a single GPU via
``CUDA_VISIBLE_DEVICES``. Completed runs (existing result JSON) are skipped, so
the sweep is resumable. Results are aggregated into a single CSV.

Usage:
    python -m evaluation.harness.sweep --phase phase11_clean_baseline --gpus 0 1
    python -m evaluation.harness.sweep --phase phase14_anostyler --gpus 0 1
    python -m evaluation.harness.sweep --phase phase15_dinomaly --gpus 0 1
    python -m evaluation.harness.sweep --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from queue import Queue

import pandas as pd

from evaluation.harness.harness import DATASETS, MVTEC_CATEGORIES, JobConfig

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# Each job trains one model once and emits arm A + arm B per pipeline (+ arm C per
# pipeline when ``include_c`` is set), so arms share the identical trained model.
PHASES: dict[str, dict] = {
    "phase0": {
        "datasets": ["mvtec"],
        "categories": ["bottle", "cable", "hazelnut"],
        "models": ["patchcore", "draem"],
        "pipelines": ["P1", "P3"],
        "seeds": [1],
        "include_c": True,
    },
    "phase1": {
        "datasets": ["mvtec"],
        "categories": MVTEC_CATEGORIES,
        "models": ["patchcore", "padim", "efficient_ad"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": False,
    },
    # Phase 2 - grand comparison across all val-only models and both datasets.
    # The three implemented pipelines (P1/P2/P3) serve as the shortlist.
    "phase2": {
        "datasets": ["mvtec", "visa"],
        "categories": None,  # use each dataset's full category list
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": False,
    },
    # Phase 3 - does the area-routed hybrid blend (P4) beat its two parents?
    # P2/P3/P4 are evaluated on the identical trained model, so the comparison is
    # paired. Arm C is included to test whether P4 lands closer to the real-test
    # difficulty than the too-hard P3.
    "phase3": {
        "datasets": ["mvtec", "visa"],
        "categories": None,
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P2", "P3", "P4"],
        "seeds": [1],
        "include_c": True,
    },
    # Phase 4 - is the P3/P4 tie fundamental, or is 2500 px simply a mistuned knob?
    # Sweeps the hybrid routing threshold on a category subset chosen from the blob-area
    # census to span the full range of "fraction of mask area below threshold":
    # carpet 68.7% / screw 32.3% / pcb1 32.3% / bottle 27.3% / macaroni1 17.8% /
    # cashew 15.9% / hazelnut 12.4% / candle 7.5%. P3 is re-run inside the same jobs
    # as a paired anchor.
    "phase4": {
        "datasets": ["mvtec", "visa"],
        "categories": {
            "mvtec": ["bottle", "hazelnut", "screw", "carpet"],
            "visa": ["candle", "macaroni1", "pcb1", "cashew"],
        },
        "models": ["patchcore", "padim", "efficient_ad", "anomaly_dino", "dinomaly"],
        "pipelines": ["P3", "P4a", "P4b", "P4", "P4c", "P4d"],
        "seeds": [1],
        "include_c": False,
    },
    # Phase 5 - MVTec AD 2 pilot. Unlike AD 1 / VisA this benchmark is not saturated
    # (SOTA SegF1 ~57%), its official metric is pixel F1 at a single label-free
    # threshold, and it ships a normal-only ``validation/`` split -- so the arm-B
    # calibration negatives are drawn from a genuinely test-disjoint pool
    # (``calibration="heldout"``) rather than the real test normals.
    # Runs at 448 px (see ``RESOLUTIONS``); ``draem`` is included here because it was
    # only ever smoke-tested at phase 0 despite being a synthetic-anomaly model itself.
    "phase5": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["padim", "dinomaly", "anomaly_dino", "efficient_ad", "draem", "patchcore"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 6 - closes the gap to published MVTec AD 2 SOTA (SuperADD 57.42% SegF1,
    # RoBiS 51.00%). Phase 5's oracle SegF1 (0.08-0.32) sits far below that because a
    # single 448-resize discards 80-95% of native pixels (2.3-5.0 MP -> 0.2 MP). Here,
    # training uses a random tile_size crop of the NATIVE-resolution image every step
    # (see tiled_harness.RandomCrop), and evaluation tiles the native image with
    # anomalib's own Tiler, scores each tile, and stitches the anomaly map back to full
    # resolution -- so pixel metrics are computed against native-resolution masks, not
    # a 448 downsize. One seed initially: tiled scoring is ~5-7s/image (vs <1s at 448),
    # so a full 6-model x 8-category x 3-seed grid would cost several days; add seeds
    # once the single-seed numbers show the gap is actually closing.
    "phase6": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["padim", "efficient_ad", "patchcore", "dinomaly", "anomaly_dino", "draem"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1],
        "include_c": True,
        "calibration": "heldout",
        "tiled": True,
    },
    # Phase 7 - SuperADD (anomalib-native, PR #3628) on MVTec AD 2 at 448 px, calibrated
    # with the pre-generated "semantic defect bank" pipelines (P5=alpha, P6=poisson;
    # see the generation notebook) instead of the live P1/P2/P3
    # generators. Only 4/8 categories had donor-bank coverage at the time this phase
    # ran. Split into two phases (rather than one phase with pipelines=["P5","P6"]) so
    # alpha results land as a complete, immediately-usable file before poisson starts,
    # per an explicit "alpha first, then later poisson" request -- SuperADD's training
    # is a single embedding-collection + coreset-subsampling pass (no gradient
    # descent), so retraining once per wave is cheap, unlike doubling a
    # gradient-trained model.
    # NOTE: these 24 jobs are already complete and used backbone "large" (303M) -- the
    # pilot choice at the time, lighter than the paper's default "huge_plus" (840M).
    # SUPERADD_BACKBONE has since been changed to "huge_plus" for phase 8 below; do not
    # delete these result files and re-run this phase expecting "large" results, since
    # doing so would now build with "huge_plus" instead. Also note: the on-disk dataset
    # was later regenerated (2026-08-25) with the "alpha" variant removed entirely and
    # "poisson" renamed to "hybrid" -- `PREGENERATED_PIPELINES` no longer resolves
    # "P5", so this phase spec would now raise `KeyError` if re-run from scratch.
    "phase7_alpha": {
        "datasets": ["mvtec2"],
        "categories": {"mvtec2": ["rice", "walnuts", "wallplugs", "fruit_jelly"]},
        "models": ["superadd"],
        "pipelines": ["P5"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    "phase7_poisson": {
        "datasets": ["mvtec2"],
        "categories": {"mvtec2": ["rice", "walnuts", "wallplugs", "fruit_jelly"]},
        "models": ["superadd"],
        "pipelines": ["P6"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 8 - re-run of phase 7 with two upgrades: (1) the generation notebook
    # regenerated SynthetciGenMVAD2 with a donor bank covering all 8 categories (was 4),
    # in place -- the phase 7 categories now score against different, newer synthetic
    # images than when phase 7 ran; (2) backbone upgraded to the paper's actual
    # "huge_plus" (840M) now that the pipeline is validated (see SUPERADD_BACKBONE).
    # Deliberately a NEW phase, not a re-run of phase7_alpha/phase7_poisson in place: job
    # keys are phase-qualified, so reusing "phase7_*" would silently skip every category
    # phase 7 already has a result file for (stale "large"-backbone/old-dataset numbers)
    # instead of re-scoring them with the new backbone and images. This keeps both
    # phase 7 (large, 4 categories, old images) and phase 8 (huge_plus, 8 categories, new
    # images) results available side by side for comparison.
    "phase8_alpha": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd"],
        "pipelines": ["P5"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    "phase8_poisson": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd"],
        "pipelines": ["P6"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 9 - does the semantic-defect-bank calibration source (P5/P6, real donor
    # patches) generalize beyond SuperADD? Re-runs the phase8 calibration source across
    # the other model families already benchmarked with the live Perlin pipelines in
    # phase5/6, to decompose whether phase8's numbers come from SuperADD's architecture
    # or from the calibration source itself. `_pregenerated_eval_set` dispatch in
    # `run_job` is keyed only on ``pipeline in PREGENERATED_PIPELINES``, independent of
    # ``job.model`` -- no harness changes needed, just this phase spec.
    # `dinomaly` and `efficient_ad` are deliberately excluded here: both need an
    # epoch-count decision (val-only models like these three default to few/no epochs;
    # dinomaly/efficient_ad are full gradient-trained and need tuning first) -- planned
    # as a follow-up phase once that's settled. `draem` is also excluded: it trains ON
    # synthetic Perlin anomalies internally (a different mechanism than the val-only
    # threshold-calibration protocol here) and needs its own consideration.
    "phase9_crossmodel": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["anomaly_dino", "padim", "patchcore"],
        "pipelines": ["P5", "P6"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 10 - completes the Perlin-vs-semantic-defect-bank comparison symmetrically
    # for SuperADD, mirroring phase5's live-pipeline sweep (P1/P2/P3) for the other
    # models. SuperADD was never included in phase5/6's ``models`` list -- it only ever
    # ran the pregenerated pipelines (phase7/8). `_synthetic_eval_set` (live P1-P4
    # generation) is model-agnostic in the same way `_pregenerated_eval_set` (P5/P6) is,
    # so this needed only a new phase spec, no harness changes. Uses the same
    # `huge_plus` backbone as phase8 (global `SUPERADD_BACKBONE`), so results are
    # directly comparable to phase8's semantic-bank numbers.
    "phase10_superadd_perlin": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd"],
        "pipelines": ["P1", "P2", "P3"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 11 - clean, honest three-way comparison per model: (arm A) real MVTec AD 2
    # anomalies, (arm B/P0) anomalib's stock, UNMODIFIED `PerlinAnomalyGenerator`
    # (no region/source/blend customization from this project at all -- see the P0
    # docstring in harness.py's `PIPELINES`/`make_generator`), and (arm B/P6) our
    # semantic-defect-bank pipeline (real donor patches, poisson-hybrid blend). P5
    # (pure alpha) is deliberately omitted here: phase 8/9's alpha-vs-poisson-hybrid
    # comparison found the two statistically indistinguishable (paired Wilcoxon,
    # p >= 0.08 for image_F1 on every model), so P6 alone stands in for "the
    # semantic-bank method" without re-litigating that comparison.
    "phase11_clean_baseline": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd", "padim", "patchcore", "anomaly_dino"],
        "pipelines": ["P0", "P6"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 12 - SuperADD backbone-size ablation: keep everything identical to the
    # phase-11 SuperADD arm (same 8 categories, 3 seeds, P0 stock-Perlin vs P6
    # semantic-bank hybrid, arm A oracle, arm C diagnostic, heldout calibration) and
    # swap only the DINOv3 backbone. `vit_huge_plus_patch16_dinov3` (840M) is the
    # paper default already captured by phase 11, so this phase adds small (22M),
    # base (86M) and large (300M) to complete a 4-point capacity curve. `backbones`
    # is keyed into JobConfig so each (backbone, seed, category) is a distinct job.
    "phase12_superadd_backbone_ablation": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd"],
        "backbones": [
            "vit_small_patch16_dinov3",
            "vit_base_patch16_dinov3",
            "vit_large_patch16_dinov3",
        ],
        "pipelines": ["P0", "P6"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 13 - AnomalyAny (CVPR 2025) generative baseline. Mirrors phase11 exactly
    # (same 4 models, 8 categories, 3 seeds, heldout calibration, arm A oracle + arm C
    # diagnostic) but swaps the pregenerated pipeline from P6 (our semantic-defect-bank
    # hybrid) to P7 (AnomalyAny diffusion generation). P0 (stock Perlin) is kept as the
    # paired anchor so P0-vs-P7 is compared on the identical trained model. P7 images
    # are image-only (no masks), so the calibration set is classification-only and arm
    # B transfers only the image threshold to the real test (which ships its own GT
    # masks); pixel metrics on the synthetic set are skipped -- a stated limitation of
    # the generative baseline (AnomalyAny emits no defect-level GT).
    "phase13_anomalyany": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd", "padim", "patchcore", "anomaly_dino"],
        "pipelines": ["P0", "P7"],
        # Only generation seed 0 is produced initially (see the phase-13 plan, §4b --
        # one seed first to get a result in ~11 h instead of ~34 h); sweep seed 1 maps
        # onto gen_seed 0 via ``(seed - 1) % 3``. Add seeds [2, 3] once gen_seeds 1/2
        # are generated.
        "seeds": [1],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 14 - AnoStyler (AAAI 2026) generative baseline. Same structure as phase13
    # (4 models, 8 categories, heldout calibration, arm A + arm C) but swaps P7
    # (AnomalyAny, image-only -> classification fallback) for P8 (AnoStyler). AnoStyler
    # emits masks, so the calibration set is full segmentation and arm B fits both the
    # image and pixel thresholds from the synthetic set (unlike P7's image-only path).
    "phase14_anostyler": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd", "padim", "patchcore", "anomaly_dino"],
        "pipelines": ["P0", "P8"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
    # Phase 15 - Dinomaly on all four anomaly sources. Dinomaly is the strongest
    # reconstruction-based model in the library (MVTec AD 2 SegF1 far above PaDiM, which
    # is essentially at noise on this benchmark). Runs it on the full 4-way comparison in
    # ONE phase so all four methods share the identical trained model (paired): arm A
    # (real oracle), B/P0 (Perlin), B/P6 (ours), B/P8 (AnoStyler), plus arm C diagnostic.
    # max_steps=1000 (MODEL_TRAINER["dinomaly"]) -- the paper's full recipe is 5000, but
    # 1000 is the loss's p_schedule_steps and a pragmatic budget for the comparison.
    "phase15_dinomaly": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["dinomaly"],
        "pipelines": ["P0", "P6", "P8"],
        "seeds": [1, 2, 3],
        "include_c": True,
        "calibration": "heldout",
    },
}

# Column order for the aggregated CSV.
METRIC_COLUMNS = [
    "image_AUROC", "image_F1Score", "image_AUPR", "image_BinaryPrecision", "image_BinaryRecall",
    "pixel_AUROC", "pixel_F1Score", "pixel_AUPR", "pixel_AUPRO", "pixel005_AUPRO",
]
COLUMN_ORDER = [
    "phase", "dataset", "category", "model", "backbone", "pipeline", "arm", "seed",
    *METRIC_COLUMNS,
    "n_train", "n_val", "n_test", "image_threshold", "normalized_image_threshold",
    "calibration", "resolution", "fit_seconds", "test_seconds", "anomalib_version", "timestamp",
]


def enumerate_jobs(phase: str) -> list[JobConfig]:
    """Expand a phase specification into individual training jobs.

    One job is emitted per (dataset, category, model, seed); each job trains once
    and internally evaluates all arms/pipelines from the identical model.
    """
    spec = PHASES[phase]
    pipelines = tuple(spec["pipelines"])
    include_c = spec["include_c"]
    calibration = spec.get("calibration", "test_normals")
    backbones = spec.get("backbones", [None])
    jobs: list[JobConfig] = []
    for dataset in spec["datasets"]:
        subset = spec["categories"]
        if isinstance(subset, dict):
            subset = subset.get(dataset)
        categories = subset or DATASETS[dataset][2]
        for category, model, seed, backbone in product(
            categories, spec["models"], spec["seeds"], backbones
        ):
            jobs.append(
                JobConfig(phase, dataset, category, model, seed, pipelines, include_c, calibration, backbone),
            )
    return jobs


def run_job(config: JobConfig, gpu: int, cpu_threads: int | None = None) -> None:
    """Launch a single job as a subprocess pinned to ``gpu``.

    Args:
        config (JobConfig): Job specification.
        gpu (int): CUDA device index to pin this job to.
        cpu_threads (int | None): If set, caps the BLAS/OpenMP thread pool size for this
            subprocess via environment variables (must be set before the interpreter
            starts, since numpy/torch/OpenCV size their thread pools at import time).
            Needed because torch defaults to using ALL logical cores per process
            (``torch.get_num_threads()``); running several concurrent jobs without this
            cap causes severe CPU oversubscription -- observed directly on an 8-GPU/
            112-core machine running 8 concurrent tiled jobs: each process defaulted to
            56 threads (8 x 56 = 448 threads for 112 cores), driving load average to
            ~400 and one job to run for 22+ hours doing what takes under 2 hours
            uncontended. Confirmed the fix works: setting ``OMP_NUM_THREADS`` in the
            subprocess env constrains ``torch.get_num_threads()`` to match.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    if cpu_threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(cpu_threads)
    cmd = [
        sys.executable, "-m", "evaluation.harness.run_one",
        "--phase", config.phase, "--dataset", config.dataset, "--category", config.category,
        "--model", config.model, "--seed", str(config.seed),
        "--pipelines", *config.pipelines,
        "--calibration", config.calibration,
    ]
    if config.include_c:
        cmd.append("--include-c")
    if config.backbone:
        cmd.extend(["--backbone", config.backbone])
    if PHASES[config.phase].get("tiled", False):
        cmd.append("--tiled")
    subprocess.run(cmd, env=env, check=False)  # noqa: S603  # fixed internal command, no shell


def schedule(jobs: list[JobConfig], gpus: list[int], procs_per_gpu: int) -> None:
    """Run jobs concurrently, pinning each to a free GPU slot.

    Caps each job's CPU thread pool to ``cpu_count() // concurrency`` (see
    :func:`run_job`) so ``concurrency`` simultaneous jobs cannot oversubscribe the
    machine's cores between them -- without this, every job independently defaults to
    using every logical core, and concurrency multiplies that default rather than
    dividing it.
    """
    concurrency = len(gpus) * procs_per_gpu
    cpu_threads = max(1, (os.cpu_count() or concurrency) // concurrency)
    slots: Queue[int] = Queue()
    for gpu in gpus:
        for _ in range(procs_per_gpu):
            slots.put(gpu)

    def worker(config: JobConfig) -> None:
        gpu = slots.get()
        try:
            run_job(config, gpu, cpu_threads)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, jobs))


def aggregate() -> int:
    """Aggregate all raw result JSONs into a single CSV. Returns row count.

    Each JSON holds a list of rows (one job emits several arm/pipeline rows).
    """
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        rows.extend(data if isinstance(data, list) else [data])
    if not rows:
        print("No results to aggregate.")
        return 0
    frame = pd.DataFrame(rows)
    # Defensive rename: a metric's reported column name is derived from its class
    # identity (see AnomalibMetric.name), so a rename made only to fix pickling has
    # silently changed a column name before (image_Precision vs image_BinaryPrecision).
    # Both names can appear together across phases run before/after that fix, so a
    # plain ``rename`` would create duplicate columns instead of merging them -- coalesce
    # explicitly (first non-null wins) so aggregation cannot corrupt the schema again.
    for old, new in (("image_Precision", "image_BinaryPrecision"), ("image_Recall", "image_BinaryRecall")):
        if old in frame.columns:
            if new in frame.columns:
                frame[new] = frame[new].combine_first(frame[old])
            else:
                frame[new] = frame[old]
            frame = frame.drop(columns=[old])
    ordered = [c for c in COLUMN_ORDER if c in frame.columns]
    remainder = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + remainder]
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_CSV, index=False)
    print(f"Aggregated {len(frame)} runs -> {RESULTS_CSV}")
    return len(frame)


def pending(jobs: list[JobConfig]) -> list[JobConfig]:
    """Filter out jobs whose result JSON already exists."""
    return [job for job in jobs if not (RAW_DIR / f"{job.key}.json").exists()]


def parse_args() -> argparse.Namespace:
    """Parse sweep command-line arguments."""
    parser = argparse.ArgumentParser(description="Schedule synthetic-anomaly experiments.")
    parser.add_argument("--phase", choices=list(PHASES))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--models", nargs="+", help="Restrict the phase to these models.")
    parser.add_argument("--exclude-models", nargs="+", help="Skip these models in the phase.")
    parser.add_argument("--categories", nargs="+", help="Restrict the phase to these categories.")
    parser.add_argument("--backbones", nargs="+", help="Restrict the phase to these backbones.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Restrict the phase to these seeds.")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point: schedule a phase and aggregate results."""
    args = parse_args()
    if args.aggregate_only:
        aggregate()
        return
    if not args.phase:
        print("Specify --phase or --aggregate-only.")
        return

    jobs = enumerate_jobs(args.phase)
    if args.models:
        jobs = [job for job in jobs if job.model in set(args.models)]
    if args.exclude_models:
        jobs = [job for job in jobs if job.model not in set(args.exclude_models)]
    if args.categories:
        jobs = [job for job in jobs if job.category in set(args.categories)]
    if args.backbones:
        jobs = [job for job in jobs if job.backbone in set(args.backbones)]
    if args.seeds:
        jobs = [job for job in jobs if job.seed in set(args.seeds)]
    todo = pending(jobs)
    print(f"{args.phase}: {len(jobs)} jobs, {len(todo)} pending, {len(jobs) - len(todo)} done.")
    schedule(todo, args.gpus, args.procs_per_gpu)
    aggregate()


if __name__ == "__main__":
    main()
