# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run one calibration job and save its result rows as JSON.

Existing result files are skipped; ``sweep.py`` aggregates the rows into CSV.

Usage:
    python -m evaluation.harness.run_one \
        --experiment cross_model_comparison --dataset mvtec2 --category rice \
        --model patchcore --seed 1 --sources Perlin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.harness.harness import CALIBRATION_PIPELINES, JobConfig, run_job

RAW_DIR = Path(__file__).parent / "results" / "raw"
SCORES_DIR = Path(__file__).parent / "results" / "scores"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a single job."""
    parser = argparse.ArgumentParser(description="Run a single synthetic-anomaly calibration job.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(CALIBRATION_PIPELINES),
        default=["Perlin", "FLASH", "AnoStyler"],
        help="Synthetic calibration sources to evaluate.",
    )
    parser.add_argument("--include-diagnostic", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    """Execute one job and write its result JSON (unless already present)."""
    args = parse_args()
    job = JobConfig(
        experiment=args.experiment,
        dataset=args.dataset,
        category=args.category,
        model=args.model,
        seed=args.seed,
        pipelines=tuple(CALIBRATION_PIPELINES[source] for source in args.sources),
        include_diagnostic=args.include_diagnostic,
        calibration=args.calibration,
        backbone=args.backbone,
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{job.key}.json"
    if out_path.exists():
        print(f"[skip] {job.key}")
        return

    rows, scores = run_job(job)
    out_path.write_text(json.dumps(rows, indent=2))
    (SCORES_DIR / f"{job.key}.json").write_text(json.dumps(scores))
    print(f"[done] {job.key} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
