# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native-resolution tiled training and evaluation for MVTec AD 2.

Training uses random native-resolution crops. Evaluation scores overlapping tiles and
stitches their anomaly maps with anomalib's ``Tiler``.
"""

from __future__ import annotations

import copy
import tempfile
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from anomalib.data.utils import read_image, read_mask, split_by_label
from anomalib.data.utils.tiler import Tiler
from anomalib.engine import Engine
from anomalib.metrics import F1AdaptiveThreshold
from torchvision.transforms.v2 import RandomCrop

from .harness import (
    CALIBRATION_SOURCES,
    DEFAULT_BATCH,
    MODEL_EVAL_BATCH,
    MODEL_TRAINER,
    JobConfig,
    _calibration_negatives,
    _effective_input_size,
    _synthetic_eval_set,
    build_model,
    make_generator,
)

if TYPE_CHECKING:
    import pandas as pd
    from anomalib.data.utils.synthetic import SyntheticAnomalyDataset

# Match the model input size.
TILE_SIZE = (448, 448)
# 25% overlap.
TILE_STRIDE = (336, 336)
# Per-model tile batches.
SUB_BATCH = {"draem": 2, "dinomaly": 4, "anomaly_dino": 4, "patchcore": 4}
DEFAULT_SUB_BATCH = 8
# Bound pixel-level curve metrics.
PIXEL_THRESHOLD_BINS = 200
# Exclude edge artifacts from image-level scores.
IMAGE_SCORE_BORDER_MARGIN = 32


def build_datamodule_tiled(job: JobConfig):  # noqa: ANN201
    """Build a datamodule for native-resolution tiled training."""
    from .harness import DATASETS  # local import to avoid a cycle at module load time

    dataset_cls, root, _ = DATASETS[job.dataset]
    train_batch = {"draem": 2, "efficient_ad": 1}.get(job.model, DEFAULT_BATCH)
    kwargs: dict = {
        "root": root,
        "category": job.category,
        "train_batch_size": train_batch,
        "eval_batch_size": MODEL_EVAL_BATCH.get(job.model, DEFAULT_BATCH),
        "num_workers": 4,
        "seed": job.seed,
        "train_augmentations": RandomCrop(TILE_SIZE),
    }
    datamodule = dataset_cls(**kwargs)
    datamodule.prepare_data()
    datamodule.setup()
    datamodule.calibration_normals = datamodule.val_data
    datamodule.val_data = copy.deepcopy(datamodule.test_data)
    datamodule._is_setup = True  # noqa: SLF001
    return datamodule


def _load_native(image_path: str, mask_path: str | float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Load an image and its mask (or an all-zero mask) at native resolution."""
    image = read_image(image_path, as_tensor=True)
    if isinstance(mask_path, str) and mask_path:
        mask = read_mask(mask_path, as_tensor=True)
    else:
        mask = torch.zeros(image.shape[-2:], dtype=torch.uint8)
    return image, mask


@torch.no_grad()
def tiled_forward(
    model: object,
    image: torch.Tensor,
    sub_batch: int = DEFAULT_SUB_BATCH,
) -> torch.Tensor:
    """Score one native-resolution image with overlapping tiles.

    Args:
        model (object): A trained anomaly model (already in eval mode).
        image (torch.Tensor): Native-resolution image, shape ``(3, H, W)``, in ``[0, 1]``.
        sub_batch (int): Number of tiles forwarded through the model at once.

    Returns:
        torch.Tensor: Stitched anomaly map with shape ``(H, W)``.
    """
    tiler = Tiler(tile_size=TILE_SIZE, stride=TILE_STRIDE)
    tiles = tiler.tile(image.unsqueeze(0))
    device = next(model.parameters()).device
    maps = []
    for start in range(0, tiles.shape[0], sub_batch):
        chunk = tiles[start : start + sub_batch].to(device)
        output = model(chunk)
        amap = output.anomaly_map
        if amap.dim() == 3:
            amap = amap.unsqueeze(1)
        maps.append(amap.detach().to(dtype=torch.float32, device="cpu"))
    all_maps = torch.cat(maps, dim=0)
    return tiler.untile(all_maps).squeeze(0).squeeze(0)


def _crop_border(tensor: torch.Tensor, margin: int = IMAGE_SCORE_BORDER_MARGIN) -> torch.Tensor:
    """Crop a fixed margin from a 2-D ``(H, W)`` tensor's border, if it is large enough."""
    h, w = tensor.shape[-2:]
    if h <= 2 * margin or w <= 2 * margin:
        return tensor
    return tensor[..., margin : h - margin, margin : w - margin]


def score_dataset_tiled(model: object, samples: pd.DataFrame, sub_batch: int) -> list[dict]:
    """Score each sample and crop the unreliable image border.

    Returns:
        list[dict]: Records with ``label``, ``score``, ``map``, and ``mask``.
    """
    records = []
    for row in samples.itertuples(index=False):
        image, mask = _load_native(row.image_path, getattr(row, "mask_path", None))
        full_map = tiled_forward(model, image, sub_batch=sub_batch)
        cropped_map = _crop_border(full_map)
        cropped_mask = _crop_border(mask)
        records.append({
            "label": int(row.label_index),
            "score": float(cropped_map.max()),
            "map": cropped_map,
            "mask": cropped_mask,
        })
    return records


def _fit_thresholds(records: list[dict]) -> tuple[float, float, float, float, float, float]:
    """Fit image and pixel thresholds and record score ranges.

    Pixel thresholds use the observed range; curve metrics use normalized scores.
    """
    image_min, image_max = float("inf"), float("-inf")
    pixel_min, pixel_max = float("inf"), float("-inf")
    for record in records:
        image_min, image_max = min(image_min, record["score"]), max(image_max, record["score"])
        pixel_min = min(pixel_min, float(record["map"].min()))
        pixel_max = max(pixel_max, float(record["map"].max()))

    image_metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    # Use the observed range, not the default [0, 1] grid.
    pixel_thresholds = torch.linspace(pixel_min, pixel_max, PIXEL_THRESHOLD_BINS)
    pixel_metric = F1AdaptiveThreshold(fields=["anomaly_map", "gt_mask"], thresholds=pixel_thresholds)
    for record in records:
        image_metric.update(SimpleNamespace(
            pred_score=torch.tensor([record["score"]]),
            gt_label=torch.tensor([record["label"]]),
        ))
        pixel_metric.update(SimpleNamespace(
            anomaly_map=record["map"].unsqueeze(0),
            gt_mask=record["mask"].unsqueeze(0),
        ))
    image_threshold = float(image_metric.compute())
    pixel_threshold = float(pixel_metric.compute())
    return image_threshold, pixel_threshold, image_min, image_max, pixel_min, pixel_max


def _normalize(value: torch.Tensor, threshold: float, vmin: float, vmax: float) -> torch.Tensor:
    """Normalize a raw score or map into ``[0, 1]``."""
    scale = vmax - vmin
    if scale <= 0:
        return torch.zeros_like(value)
    return ((value - threshold) / scale + 0.5).clamp(0.0, 1.0)


def _evaluate(
    records: list[dict],
    image_threshold: float,
    pixel_threshold: float,
    image_min: float,
    image_max: float,
    pixel_min: float,
    pixel_max: float,
) -> dict[str, float]:
    """Compute the standard metric suite at fixed thresholds."""
    from anomalib.metrics import AUPR, AUPRO, AUROC, F1Score

    from .harness import Precision, Recall

    metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        AUPR(fields=["pred_score", "gt_label"], prefix="image_"),
        Precision(fields=["pred_label", "gt_label"], prefix="image_"),
        Recall(fields=["pred_label", "gt_label"], prefix="image_"),
        AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", thresholds=PIXEL_THRESHOLD_BINS),
        F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_", thresholds=PIXEL_THRESHOLD_BINS),
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.05, prefix="pixel005_"),
    ]
    for record in records:
        raw_score, raw_map = record["score"], record["map"]
        # Apply thresholds before normalization.
        pred_label = torch.tensor([int(raw_score > image_threshold)])
        pred_mask = (raw_map > pixel_threshold).to(torch.uint8).unsqueeze(0)
        batch = SimpleNamespace(
            pred_score=_normalize(torch.tensor([raw_score]), image_threshold, image_min, image_max),
            pred_label=pred_label,
            gt_label=torch.tensor([record["label"]]),
            anomaly_map=_normalize(raw_map, pixel_threshold, pixel_min, pixel_max).unsqueeze(0),
            pred_mask=pred_mask,
            gt_mask=record["mask"].unsqueeze(0),
        )
        for metric in metrics:
            metric.update(batch)
    results: dict[str, float] = {}
    for metric in metrics:
        try:
            results[metric.name] = float(metric.compute())
        except (RuntimeError, ValueError, IndexError) as exc:  # noqa: PERF203
            # AUPRO can exceed metric-library limits at native resolution.
            print(f"[tiled] metric {metric.name} failed to compute, reporting NaN: {exc}", flush=True)
            results[metric.name] = float("nan")
    return results


def run_tiled_job(job: JobConfig) -> tuple[list[dict], dict[str, dict[str, list]]]:
    """Train with native-resolution crops and evaluate with tiled inference."""
    model = build_model(job.model, TILE_SIZE, tiled=True)
    datamodule = build_datamodule_tiled(job)
    n_train = len(datamodule.train_data)
    real_test = datamodule.test_data
    n_real_test = len(real_test)
    sub_batch = SUB_BATCH.get(job.model, DEFAULT_SUB_BATCH)

    with tempfile.TemporaryDirectory(prefix="anomalib_tiled_run_") as scratch:
        engine = Engine(
            accelerator="gpu",
            devices=1,
            logger=False,
            default_root_dir=scratch,
            limit_val_batches=0,  # Memory-bank finalization still runs after training.
            num_sanity_val_steps=0,
            **MODEL_TRAINER[job.model],
        )

        start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = round(time.time() - start, 2)

    model.eval()
    eff = _effective_input_size(model)
    resolution = "x".join(str(v) for v in eff) if eff else "model_default"

    rows: list[dict] = []
    start = time.time()
    real_records = score_dataset_tiled(model, real_test.samples, sub_batch)
    test_seconds = round(time.time() - start, 2)

    (
        image_threshold_a,
        pixel_threshold_a,
        img_min_a,
        img_max_a,
        pix_min_a,
        pix_max_a,
    ) = _fit_thresholds(real_records)
    metrics_a = _evaluate(
        real_records, image_threshold_a, pixel_threshold_a, img_min_a, img_max_a, pix_min_a, pix_max_a,
    )
    rows.append({
        "experiment": job.experiment, "dataset": job.dataset, "category": job.category, "model": job.model,
        "calibration_source": "Real", "evaluation_set": "Real test set", "seed": job.seed,
        **metrics_a,
        "n_train": n_train, "n_val": n_real_test, "n_test": n_real_test,
        "image_threshold": image_threshold_a, "normalized_image_threshold": None,
        "calibration": job.calibration, "resolution": resolution,
        "fit_seconds": fit_seconds, "test_seconds": test_seconds,
    })

    test_normals, test_anomalies = split_by_label(real_test)
    n_anomalies = len(test_anomalies)
    negatives = _calibration_negatives(job, datamodule, test_normals)
    scores: dict[str, dict[str, list]] = {
        "real_test": {
            "scores": [r["score"] for r in real_records],
            "labels": [r["label"] for r in real_records],
        },
    }

    keep_alive: list[SyntheticAnomalyDataset] = []
    for pipeline in job.pipelines:
        calibration = _synthetic_eval_set(
            negatives=negatives,
            source_normals=datamodule.train_data,
            n_anomalies=n_anomalies,
            augmenter=make_generator(pipeline),
            seed=job.seed,
        )
        keep_alive.append(calibration)
        n_calib = len(calibration)

        start = time.time()
        calib_records = score_dataset_tiled(model, calibration.samples, sub_batch)
        calib_seconds = round(time.time() - start, 2)
        scores[pipeline] = {
            "scores": [r["score"] for r in calib_records],
            "labels": [r["label"] for r in calib_records],
        }

        image_threshold_b, pixel_threshold_b, img_min_b, img_max_b, pix_min_b, pix_max_b = _fit_thresholds(
            calib_records,
        )
        metrics_b = _evaluate(
            real_records, image_threshold_b, pixel_threshold_b, img_min_b, img_max_b, pix_min_b, pix_max_b,
        )
        rows.append({
            "experiment": job.experiment,
            "dataset": job.dataset,
            "category": job.category,
            "model": job.model,
            "calibration_source": CALIBRATION_SOURCES.get(pipeline, pipeline),
            "evaluation_set": "Real test set", "seed": job.seed,
            **metrics_b,
            "n_train": n_train, "n_val": n_calib, "n_test": n_real_test,
            "image_threshold": image_threshold_b, "normalized_image_threshold": None,
            "calibration": job.calibration, "resolution": resolution,
            "fit_seconds": fit_seconds, "test_seconds": calib_seconds,
        })

        if job.include_diagnostic:
            metrics_c = _evaluate(
                calib_records,
                image_threshold_b,
                pixel_threshold_b,
                img_min_b,
                img_max_b,
                pix_min_b,
                pix_max_b,
            )
            rows.append({
                "experiment": job.experiment,
                "dataset": job.dataset,
                "category": job.category,
                "model": job.model,
                "calibration_source": CALIBRATION_SOURCES.get(pipeline, pipeline),
                "evaluation_set": "Synthetic calibration set", "seed": job.seed,
                **metrics_c,
                "n_train": n_train, "n_val": n_calib, "n_test": n_calib,
                "image_threshold": image_threshold_b, "normalized_image_threshold": None,
                "calibration": job.calibration, "resolution": resolution,
                "fit_seconds": fit_seconds, "test_seconds": calib_seconds,
            })

    return rows, scores
