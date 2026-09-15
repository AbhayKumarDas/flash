# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run a single training job and persist its result rows as JSON.

One job trains a model once and emits several rows (arm A + arm B per pipeline,
plus arm C per pipeline when requested). Each job writes ``results/raw/<key>.json``
as a list of rows, so jobs are independent and the sweep is resumable (an existing
file means the job is skipped). Aggregation into a CSV is handled by ``sweep.py``.

Usage:
    python -m evaluation.harness.run_one \
        --phase phase11_clean_baseline --dataset mvtec2 --category rice \
        --model patchcore --seed 1 --pipelines P0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.harness.harness import JobConfig, run_job
from evaluation.harness.tiled_harness import run_tiled_job

RAW_DIR = Path(__file__).parent / "results" / "raw"
SCORES_DIR = Path(__file__).parent / "results" / "scores"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single job."""
    parser = argparse.ArgumentParser(description="Run a single synthetic-anomaly training job.")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pipelines", nargs="+", default=["P1", "P2", "P3"])
    parser.add_argument("--include-c", action="store_true")
    parser.add_argument(
        "--calibration",
        default="test_normals",
        choices=["test_normals", "heldout"],
        help="Source of calibration negatives; 'heldout' is leakage-free.",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Backbone override for models that accept one (e.g. SuperADD).",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help=(
            "Train on native-resolution random crops and evaluate with tiled inference "
            "(anomalib's Tiler) instead of a single whole-image resize. See tiled_harness.py."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Execute one job and write its result JSON (unless already present)."""
    args = parse_args()
    job = JobConfig(
        phase=args.phase,
        dataset=args.dataset,
        category=args.category,
        model=args.model,
        seed=args.seed,
        pipelines=tuple(args.pipelines),
        include_c=args.include_c,
        calibration=args.calibration,
        backbone=args.backbone,
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{job.key}.json"
    if out_path.exists():
        print(f"[skip] {job.key}")
        return

    rows, scores = (run_tiled_job if args.tiled else run_job)(job)
    out_path.write_text(json.dumps(rows, indent=2))
    (SCORES_DIR / f"{job.key}.json").write_text(json.dumps(scores))
    print(f"[done] {job.key} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
