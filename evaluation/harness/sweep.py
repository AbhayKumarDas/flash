# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Enumerate and schedule the paper's calibration comparisons across GPUs.

Completed jobs are skipped, so a comparison can be resumed after interruption. Each
job trains one detector once and evaluates the Real, Perlin, FLASH, and/or AnoStyler
calibration settings from the same fitted model.

Usage:
    python -m evaluation.harness.sweep --experiment cross_model_comparison --gpus 0 1
    python -m evaluation.harness.sweep --experiment anostyler_comparison --gpus 0 1
    python -m evaluation.harness.sweep --experiment dinomaly_comparison --gpus 0 1
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

from evaluation.harness.harness import CALIBRATION_SOURCES, DATASETS, JobConfig

RESULTS_DIR = Path(__file__).parent / "results"
RAW_DIR = RESULTS_DIR / "raw"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# These are the three comparisons used by the paper's MVTec AD 2 results.
EXPERIMENTS: dict[str, dict] = {
    "cross_model_comparison": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd", "padim", "patchcore", "anomaly_dino"],
        "pipelines": ["Perlin", "FLASH"],
        "seeds": [1, 2, 3],
        "include_diagnostic": True,
        "calibration": "heldout",
    },
    "anostyler_comparison": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["superadd", "padim", "patchcore", "anomaly_dino"],
        "pipelines": ["Perlin", "AnoStyler"],
        "seeds": [1, 2, 3],
        "include_diagnostic": True,
        "calibration": "heldout",
    },
    "dinomaly_comparison": {
        "datasets": ["mvtec2"],
        "categories": None,
        "models": ["dinomaly"],
        "pipelines": ["Perlin", "FLASH", "AnoStyler"],
        "seeds": [1, 2, 3],
        "include_diagnostic": True,
        "calibration": "heldout",
    },
}

METRIC_COLUMNS = [
    "image_AUROC",
    "image_F1Score",
    "image_AUPR",
    "image_BinaryPrecision",
    "image_BinaryRecall",
    "pixel_AUROC",
    "pixel_F1Score",
    "pixel_AUPR",
    "pixel_AUPRO",
    "pixel005_AUPRO",
]
COLUMN_ORDER = [
    "experiment",
    "dataset",
    "category",
    "model",
    "backbone",
    "calibration_source",
    "evaluation_set",
    "seed",
    *METRIC_COLUMNS,
    "n_train",
    "n_val",
    "n_test",
    "image_threshold",
    "normalized_image_threshold",
    "calibration",
    "resolution",
    "fit_seconds",
    "test_seconds",
    "anomalib_version",
    "timestamp",
]


def enumerate_jobs(experiment: str) -> list[JobConfig]:
    """Expand a named comparison into individual detector training jobs."""
    spec = EXPERIMENTS[experiment]
    pipelines = tuple(spec["pipelines"])
    include_diagnostic = spec["include_diagnostic"]
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
                JobConfig(
                    experiment,
                    dataset,
                    category,
                    model,
                    seed,
                    pipelines,
                    include_diagnostic,
                    calibration,
                    backbone,
                ),
            )
    return jobs


def run_job(config: JobConfig, gpu: int, cpu_threads: int | None = None) -> None:
    """Launch one training job as a subprocess pinned to ``gpu``."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    if cpu_threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(cpu_threads)
    sources = [CALIBRATION_SOURCES[pipeline] for pipeline in config.pipelines]
    cmd = [
        sys.executable,
        "-m",
        "evaluation.harness.run_one",
        "--experiment",
        config.experiment,
        "--dataset",
        config.dataset,
        "--category",
        config.category,
        "--model",
        config.model,
        "--seed",
        str(config.seed),
        "--sources",
        *sources,
        "--calibration",
        config.calibration,
    ]
    if config.include_diagnostic:
        cmd.append("--include-diagnostic")
    if config.backbone:
        cmd.extend(["--backbone", config.backbone])
    if EXPERIMENTS[config.experiment].get("tiled", False):
        cmd.append("--tiled")
    subprocess.run(cmd, env=env, check=False)  # noqa: S603  # fixed internal command, no shell


def schedule(jobs: list[JobConfig], gpus: list[int], procs_per_gpu: int) -> None:
    """Run jobs concurrently while limiting CPU oversubscription."""
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
    """Aggregate raw result JSON files into the public CSV and return its row count."""
    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        rows.extend(data if isinstance(data, list) else [data])
    if not rows:
        print("No results to aggregate.")
        return 0

    frame = pd.DataFrame(rows)
    for old, new in (("image_Precision", "image_BinaryPrecision"), ("image_Recall", "image_BinaryRecall")):
        if old in frame.columns:
            if new in frame.columns:
                frame[new] = frame[new].combine_first(frame[old])
            else:
                frame[new] = frame[old]
            frame = frame.drop(columns=[old])
    ordered = [column for column in COLUMN_ORDER if column in frame.columns]
    remainder = [column for column in frame.columns if column not in ordered]
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
    parser = argparse.ArgumentParser(description="Schedule synthetic-anomaly calibration experiments.")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--procs-per-gpu", type=int, default=2)
    parser.add_argument("--models", nargs="+", help="Restrict the comparison to these models.")
    parser.add_argument("--exclude-models", nargs="+", help="Skip these models.")
    parser.add_argument("--categories", nargs="+", help="Restrict the comparison to these categories.")
    parser.add_argument("--backbones", nargs="+", help="Restrict the comparison to these backbones.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Restrict the comparison to these seeds.")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Schedule a comparison and aggregate its results."""
    args = parse_args()
    if args.aggregate_only:
        aggregate()
        return
    if not args.experiment:
        print("Specify --experiment or --aggregate-only.")
        return

    jobs = enumerate_jobs(args.experiment)
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
    print(f"{args.experiment}: {len(jobs)} jobs, {len(todo)} pending, {len(jobs) - len(todo)} done.")
    schedule(todo, args.gpus, args.procs_per_gpu)
    aggregate()


if __name__ == "__main__":
    main()
