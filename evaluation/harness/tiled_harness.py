# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native-resolution tiled training and evaluation for MVTec AD 2.

At 448x448 our models see a 5-40x downsized version of MVTec AD 2's native images
(2.3-5.0 MP), which is the most plausible reason our oracle numbers sit far below the
published leaderboard (SuperADD, RoBiS) -- those methods run tiled/patch-based inference
at native resolution. This module closes that gap without modifying model internals:

- **Training**: samples a random ``tile_size`` crop from the native-resolution image
  each step (a torchvision ``RandomCrop`` augmentation), so the model sees genuine local
  detail rather than a globally-downsized image. No custom training loop is needed --
  this is a plain data augmentation, so ``Engine.fit`` is used unchanged.
- **Evaluation**: bypasses ``Engine.test``/``Engine.validate`` entirely. Each image is
  loaded at native resolution, split into overlapping ``tile_size`` tiles with
  anomalib's own :class:`~anomalib.data.utils.tiler.Tiler`, scored tile-by-tile through
  the trained model (still expecting ``tile_size`` input), and the per-tile anomaly maps
  are stitched back into a native-resolution map. Thresholds and metrics reuse anomalib's
  own metric classes directly (``F1AdaptiveThreshold``, and the same test metrics as
  :func:`harness.build_evaluator`), fed via a lightweight batch-like namespace -- so the
  only new logic is the tile/untile bookkeeping, not the metrics themselves.

This works for any of the six models: the tiler operates purely on the model's forward
output (``anomaly_map``), not on its internals, so it does not depend on a model
exposing its own ``self.tiler`` attribute (only PaDiM and PatchCore do).
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

# Tile size matches the model's own input resolution so a tile is a single forward pass.
TILE_SIZE = (448, 448)
# 25% overlap (112 px). Widened from an initial 50% overlap (224 stride) after
# measuring that tiled inference is the dominant cost (patchcore/efficient_ad:
# 60-70 s/image on a 4.5 MP category) -- overlap directly multiplies tile count and
# therefore wall-clock time. 25% still avoids hard seams at tile borders.
TILE_STRIDE = (336, 336)
# Number of tiles forwarded through the model at once; bounded to fit models with the
# heaviest per-sample memory footprint (draem, dinomaly, anomaly_dino) at native tile size.
SUB_BATCH = {"draem": 2, "dinomaly": 4, "anomaly_dino": 4, "patchcore": 4}
DEFAULT_SUB_BATCH = 8
# Thresholds for pixel-level curve metrics: bounded/binned mode so accumulating across
# native-resolution images (up to ~5M px each) stays O(bins), not O(all pixels).
PIXEL_THRESHOLD_BINS = 200
# The TRUE image border (not tile-to-tile seams -- remove_border_count handles those,
# but has no effect here since no overlapping neighbour tile exists at the outer edge
# to compensate) gets a convolutional edge-effect artifact: verified directly, two
# unrelated normal images produced the exact same score (97.2359) at the exact same
# pixel (row 0, col W-3). At margin=0 every image's max-based score collapses to one
# constant value regardless of true content (image_AUROC == 0.500 on every completed
# job). Excluding a 32 px border from the image-level score (NOT from the returned
# pixel map, which pixel metrics still use in full) restores clean separation: on a
# 4-normal/4-anomalous check, margin=16 already gives normal=[55.9-58.9] vs
# anomalous=[70.8-81.6] with zero overlap, stable through margin=128; 32 adds headroom.
IMAGE_SCORE_BORDER_MARGIN = 32


def build_datamodule_tiled(job: JobConfig):  # noqa: ANN201
    """Construct the datamodule for native-resolution tiled training.

    Unlike :func:`harness.build_datamodule`, images are never resized: training uses a
    random ``TILE_SIZE`` crop (so batches have a uniform shape while still exposing
    native-resolution local detail); validation/test augmentations are left at ``None``
    since evaluation is scored manually by :func:`score_dataset_tiled`, not through
    ``Engine.test``.
    """
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
    """Score one native-resolution image by tiling it through a fixed-input-size model.

    Args:
        model (object): A trained anomaly model (already in eval mode).
        image (torch.Tensor): Native-resolution image, shape ``(3, H, W)``, in ``[0, 1]``.
        sub_batch (int): Number of tiles forwarded through the model at once.

    Returns:
        torch.Tensor: The stitched native-resolution anomaly map, shape ``(H, W)``.
        The true image border still carries a tile-boundary convolutional artifact (see
        ``IMAGE_SCORE_BORDER_MARGIN``); callers should crop it before using the map for
        either image- or pixel-level scoring. Not cropped here so the map returned
        stays at the image's true resolution for callers that need that invariant.
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
    """Tiled-score every row of a ``samples`` dataframe.

    The true image border is cropped by ``IMAGE_SCORE_BORDER_MARGIN`` from both the
    anomaly map and the ground-truth mask before being stored, so image- and
    pixel-level metrics are computed consistently on the same (artifact-free) region --
    see ``tiled_forward`` for why the border is unreliable.

    Returns:
        list[dict]: One record per row with keys ``label``, ``score``, ``map`` (the
        border-cropped anomaly map) and ``mask`` (the correspondingly cropped GT mask).
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
    """Fit image/pixel F1-adaptive thresholds plus min-max normalization stats.

    Mirrors anomalib's ``OneClassPostProcessor``, which always min-max normalizes
    ``pred_score``/``anomaly_map`` into ``[0, 1]`` before any of its rank-based metrics
    (``AUROC``, ``AUPR``) see them. Skipping that step -- as bypassing ``Engine.test``
    does -- is silently catastrophic in two distinct, unrelated ways, both triggered by
    the same root cause (our scores are raw/unbounded, not normalized to ``[0, 1]``):

    1. **AUROC/AUPR sigmoid saturation.** torchmetrics' ``BinaryROC``/
       ``BinaryPrecisionRecallCurve`` apply ``sigmoid()`` to any ``preds`` tensor found
       outside ``[0, 1]``, assuming it must be a logit. Our raw, unbounded map-derived
       scores saturate completely under sigmoid (``sigmoid(30) ~= 1.0``), collapsing
       every score to the same value and every downstream AUROC/AUPR to exactly 0.5 --
       confirmed directly: sklearn's AUROC on the same raw scores from a completed job
       was 0.616, while anomalib's ``AUROC`` metric gave 0.500 on identical data.
       Anomalib's own ``BinaryPrecisionRecallCurve`` override (see
       ``anomalib/metrics/precision_recall_curve.py``) explicitly disables this sigmoid
       step, so ``F1AdaptiveThreshold``'s *image*-level fit (non-binned mode, no
       ``thresholds=`` arg -- exact candidate values come straight from the observed
       scores) is unaffected. AUROC/AUPR still need the normalized view, applied in
       :func:`_evaluate`.
    2. **Pixel F1AdaptiveThreshold bin-range mismatch (distinct bug, same symptom
       class).** The *pixel* fit passes ``thresholds=PIXEL_THRESHOLD_BINS`` (a bare
       int) to bound memory at native resolution. torchmetrics' ``_adjust_threshold_arg``
       converts any bare int into ``torch.linspace(0, 1, n)`` -- a FIXED [0, 1] grid --
       regardless of the actual score range, and anomalib's sigmoid-disabling override
       does NOT correspondingly rescale this grid. Since our raw pixel scores sit far
       above 1 almost everywhere, every one of the 200 candidate thresholds ends up
       below virtually the entire score distribution, so the "F1-maximizing" search
       degenerates to classifying nearly all pixels positive. Reproduced synthetically:
       a clearly-separable 1%-defect map (background ~30-50, defect ~90-110) gave a
       fitted threshold of 0.0 and F1 = 0.02 with a bare-int ``thresholds=200``, vs. a
       threshold of 50.1 and F1 = 1.0 once ``thresholds`` is an explicit
       ``linspace(pixel_min, pixel_max, 200)`` spanning the true score range. This is
       the reason patchcore/padim/efficient_ad's tiled pixel-F1 looked catastrophically
       worse than 448 px despite healthy pixel AUROC/AUPRO (which don't depend on this
       threshold) -- not a real capability regression.

    Both fixes only touch how AUROC/AUPR/F1 candidates are generated -- they are
    invariant, order-preserving transforms of the ranking variable, so they cannot
    change which threshold is "best", only whether the library can find it.
    """
    image_min, image_max = float("inf"), float("-inf")
    pixel_min, pixel_max = float("inf"), float("-inf")
    for record in records:
        image_min, image_max = min(image_min, record["score"]), max(image_max, record["score"])
        pixel_min = min(pixel_min, float(record["map"].min()))
        pixel_max = max(pixel_max, float(record["map"].max()))

    image_metric = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    # Explicit range-aware thresholds (fix #2 above) instead of a bare int, which
    # anomalib/torchmetrics would silently reinterpret as linspace(0, 1, n).
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
    """Min-max normalize a raw score/map into ``[0, 1]``, mirroring ``OneClassPostProcessor._normalize``.

    Only needed to keep AUROC/AUPR out of torchmetrics' sigmoid-on-out-of-range-preds path
    (see :func:`_fit_thresholds`) -- it does not change their *value*, since both metrics are
    invariant to this affine, order-preserving transform of the ranking variable.
    """
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
    """Compute the standard test-metric suite from tiled-scored records at fixed thresholds.

    ``image_min``/``image_max``/``pixel_min``/``pixel_max`` should come from the same record
    set used to fit ``image_threshold``/``pixel_threshold`` (see :func:`_fit_thresholds`) --
    they are used only to normalize ``pred_score``/``anomaly_map`` into ``[0, 1]`` before
    AUROC/AUPR, avoiding the sigmoid-saturation bug described there.
    """
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
        # pred_label/pred_mask compare the RAW score/map against the RAW threshold, which is
        # mathematically equivalent to comparing the normalized versions against 0.5 (the
        # normalization is centered on the threshold) -- so these stay on raw values.
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
            # AUPRO's connected-component analysis accumulates every native-resolution
            # mask (up to ~5M px each) across the whole record set and labels regions
            # over the full batch at once; on some image/mask combinations this blows up
            # to an unallocatable label count (RuntimeError) or an index that overflows
            # the accumulated tensor's size (IndexError -- observed directly: an index
            # of 281475190239367 against a tensor of size 237153280, i.e. an internal
            # index/count computation overflowing at this scale). Both are downstream
            # metric-library edge cases at native resolution, not something to risk
            # losing an entire job's already-completed (and expensive: 30-70 s/image)
            # tiled scoring pass over. Report NaN for just this metric and keep the rest
            # of the row.
            print(f"[tiled] metric {metric.name} failed to compute, reporting NaN: {exc}", flush=True)
            results[metric.name] = float("nan")
    return results


def run_tiled_job(job: JobConfig) -> tuple[list[dict], dict[str, dict[str, list]]]:
    """Train at native resolution (random-crop augmented) and evaluate with tiled inference.

    Mirrors :func:`harness.run_job`'s arm structure (A/B/C) but every evaluation is a
    manual tiled scoring pass rather than ``Engine.test``/``Engine.validate``.
    """
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
            limit_val_batches=0,  # validation would run at native resolution -- skip it;
            # memory-bank finalisation (subsample_embedding, etc.) still fires from
            # on_train_epoch_end regardless of whether validation ran.
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
        "phase": job.phase, "dataset": job.dataset, "category": job.category, "model": job.model,
        "pipeline": "-", "arm": "A", "seed": job.seed,
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
            "phase": job.phase, "dataset": job.dataset, "category": job.category, "model": job.model,
            "pipeline": pipeline, "arm": "B", "seed": job.seed,
            **metrics_b,
            "n_train": n_train, "n_val": n_calib, "n_test": n_real_test,
            "image_threshold": image_threshold_b, "normalized_image_threshold": None,
            "calibration": job.calibration, "resolution": resolution,
            "fit_seconds": fit_seconds, "test_seconds": calib_seconds,
        })

        if job.include_c:
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
                "phase": job.phase, "dataset": job.dataset, "category": job.category, "model": job.model,
                "pipeline": pipeline, "arm": "C", "seed": job.seed,
                **metrics_c,
                "n_train": n_train, "n_val": n_calib, "n_test": n_calib,
                "image_threshold": image_threshold_b, "normalized_image_threshold": None,
                "calibration": job.calibration, "resolution": resolution,
                "fit_seconds": fit_seconds, "test_seconds": calib_seconds,
            })

    return rows, scores
