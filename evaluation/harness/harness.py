# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core harness for synthetic-anomaly threshold calibration.

Each job fits one detector, then evaluates Real, Perlin, FLASH, and AnoStyler calibration.
"""

from __future__ import annotations

import copy
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import anomalib
import pandas as pd
import torch
from anomalib.data import MVTecAD, MVTecAD2, Visa
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import Split, ValSplitMode, split_by_label
from anomalib.data.utils.generators import SyntheticAnomalyGenerator
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.data.utils.synthetic import SyntheticAnomalyDataset
from anomalib.engine import Engine
from anomalib.metrics import AUPR, AUPRO, AUROC, Evaluator, F1Score, create_anomalib_metric
from anomalib.models import AnomalyDINO, Dinomaly, Draem, EfficientAd, Padim, Patchcore, SuperADD
from anomalib.post_processing import PostProcessor
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchvision.transforms.v2 import CenterCrop, Resize

EVAL_DATA_ROOT = Path(os.environ.get("FLASH_EVAL_DATA_ROOT", "./data/evaluation"))

BinaryPrecisionMetric = create_anomalib_metric(BinaryPrecision)
BinaryRecallMetric = create_anomalib_metric(BinaryRecall)

# Make dynamic metrics importable for checkpointing and keep their output names.
for _name, _cls in (("BinaryPrecision", BinaryPrecisionMetric), ("BinaryRecall", BinaryRecallMetric)):
    _cls.__name__ = _cls.__qualname__ = _name
    _cls.__module__ = __name__
globals()["BinaryPrecision"] = BinaryPrecisionMetric
globals()["BinaryRecall"] = BinaryRecallMetric
Precision = BinaryPrecisionMetric
Recall = BinaryRecallMetric

# Dataset registry: name -> (class, root, categories).
MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
MVTEC2_CATEGORIES = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
]
DATASETS = {
    "mvtec": (MVTecAD, EVAL_DATA_ROOT / "MVTecAD", MVTEC_CATEGORIES),
    "visa": (Visa, EVAL_DATA_ROOT / "visa", VISA_CATEGORIES),
    "mvtec2": (MVTecAD2, EVAL_DATA_ROOT / "MVTec_AD_2", MVTEC2_CATEGORIES),
}

# MVTec AD 2 uses explicit 448x448 inputs.
RESOLUTIONS: dict[str, tuple[int, int] | None] = {
    "mvtec": None,
    "visa": None,
    "mvtec2": (448, 448),
}

# Live synthetic baseline.
PIPELINES = {"Perlin": None}

# Pre-generated calibration sources.
PREGENERATED_PIPELINES = {"FLASH": "hybrid", "AnoStyler": "anostyler"}
CALIBRATION_SOURCES = {
    "-": "Real",
    "Perlin": "Perlin",
    "FLASH": "FLASH",
    "AnoStyler": "AnoStyler",
}
CALIBRATION_PIPELINES = {
    source: source for source in CALIBRATION_SOURCES.values() if source != "Real"
}
# Categories covered by the pre-generated sets.
PREGENERATED_CATEGORIES = (
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
)
SYNTHETIC_GEN_ROOT = EVAL_DATA_ROOT / "SynthetciGenMVAD2"
# Pre-generated seed ids; sweep seeds 1, 2, 3 map to 0, 1, 2.
PREGENERATED_SEEDS = (0, 1, 2)
# SuperADD backbone used for the paper results.
SUPERADD_BACKBONE = "vit_huge_plus_patch16_dinov3"


# Per-model trainer settings and train batch size.
MODEL_TRAINER = {
    "patchcore": {"max_epochs": 1},
    "padim": {"max_epochs": 1},
    "efficient_ad": {"max_epochs": 20},
    "draem": {"max_epochs": 15},
    # Dinomaly stops at max_steps.
    "dinomaly": {"max_steps": 1000, "max_epochs": 1000},
    "anomaly_dino": {"max_epochs": 1},
    # SuperADD is training-free but still needs a trainer entry.
    "superadd": {"max_epochs": 1},
}
MODEL_BATCH = {"efficient_ad": 1, "draem": 2, "superadd": 4}
MODEL_EVAL_BATCH = {"draem": 2, "superadd": 4}
DEFAULT_BATCH = 8


@dataclass
class JobConfig:
    """Configuration for a single training job.

    A job fits one model and evaluates each configured calibration source.

    Args:
        experiment (str): Named comparison this job belongs to.
        dataset (str): Dataset key (``"mvtec"`` or ``"visa"``).
        category (str): Category within the dataset.
        model (str): Model name.
        seed (int): Random seed for the datamodule and synthetic sampling.
        pipelines (tuple[str, ...]): Synthetic sources to evaluate.
        include_diagnostic (bool): Whether to emit synthetic-set diagnostic rows.
        calibration (str): Normal pool: ``"test_normals"`` or test-disjoint ``"heldout"``.
        backbone (str | None): Optional backbone override.
    """

    experiment: str
    dataset: str
    category: str
    model: str
    seed: int = 1
    pipelines: tuple[str, ...] = ("Perlin", "FLASH", "AnoStyler")
    include_diagnostic: bool = False
    calibration: str = "test_normals"
    backbone: str | None = None

    @property
    def key(self) -> str:
        """Unique, filesystem-safe identifier for this job."""
        parts = [self.experiment, self.dataset, self.category, self.model]
        if self.backbone:
            parts.append(self.backbone)
        parts.append(f"s{self.seed}")
        return "_".join(parts)


def build_evaluator() -> Evaluator:
    """Return the image- and pixel-level evaluator used by the comparisons."""
    image = {"prefix": "image_"}
    pixel = {"prefix": "pixel_", "strict": False}
    val_metrics = [
        AUROC(fields=["pred_score", "gt_label"], **image),
        AUROC(fields=["anomaly_map", "gt_mask"], **pixel),
    ]
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], **image),
        F1Score(fields=["pred_label", "gt_label"], **image),
        AUPR(fields=["pred_score", "gt_label"], **image),
        Precision(fields=["pred_label", "gt_label"], **image),
        Recall(fields=["pred_label", "gt_label"], **image),
        AUROC(fields=["anomaly_map", "gt_mask"], **pixel),
        F1Score(fields=["pred_mask", "gt_mask"], **pixel),
        AUPR(fields=["anomaly_map", "gt_mask"], **pixel),
        AUPRO(fields=["anomaly_map", "gt_mask"], **pixel),
        AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.05, prefix="pixel005_", strict=False),
    ]
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)


def build_model(
    name: str,
    resolution: tuple[int, int] | None = None,
    *,
    tiled: bool = False,
    backbone: str | None = None,
) -> object:
    """Instantiate a detector with the evaluation metrics.

    Args:
        name (str): Model key.
        resolution (tuple[int, int] | None): Requested input size.
        tiled (bool): Whether tiled inference is used.
        backbone (str | None): Optional backbone override.

    Raises:
        RuntimeError: If the requested resolution is not the one the model will use.
    """
    evaluator = build_evaluator()
    factories = {
        "patchcore": Patchcore,
        "padim": Padim,
        "efficient_ad": EfficientAd,
        "draem": Draem,
        "dinomaly": Dinomaly,
        "anomaly_dino": AnomalyDINO,
        "superadd": SuperADD,
    }
    model_cls = factories[name]
    kwargs: dict = {"evaluator": evaluator}
    if name == "superadd":
        # Use the common thresholding rule for all detectors.
        kwargs["post_processor"] = PostProcessor()
        kwargs["backbone"] = backbone or SUPERADD_BACKBONE
    if name == "anomaly_dino":
        # Limit the memory bank at 448 px.
        kwargs["coreset_subsampling"] = True
        kwargs["sampling_ratio"] = 0.1
    if resolution is not None:
        if name == "dinomaly" and tiled:
            # Disable Dinomaly's center crop so tile and map sizes match.
            kwargs["pre_processor"] = model_cls.configure_pre_processor(resolution, crop_size=resolution[0])
        else:
            kwargs["pre_processor"] = model_cls.configure_pre_processor(resolution)
    model = model_cls(**kwargs)
    if resolution is not None:
        requested = _resize_size(model)
        if requested is not None and tuple(requested) != tuple(resolution):
            msg = (
                f"Model '{name}' would resize to {tuple(requested)} instead of the requested "
                f"{tuple(resolution)}; its pre-processor overrode the configured resolution."
            )
            raise RuntimeError(msg)
    return model


def _resize_size(model: object) -> tuple[int, int] | None:
    """Return the ``Resize`` target in the model's pre-processor."""
    return _transform_size(model, Resize)


def _effective_input_size(model: object) -> tuple[int, int] | None:
    """Return the model's effective input size."""
    return _transform_size(model, CenterCrop) or _resize_size(model)


def _transform_size(model: object, kind: type) -> tuple[int, int] | None:
    """Return the ``size`` of the last transform of ``kind`` in the pre-processor."""
    transform = getattr(getattr(model, "pre_processor", None), "transform", None)
    if transform is None:
        return None
    found = None
    for step in getattr(transform, "transforms", [transform]):
        size = getattr(step, "size", None)
        if isinstance(step, kind) and size is not None:
            found = tuple(size) if isinstance(size, list | tuple) else (size, size)
    return found


def make_generator(pipeline: str) -> SyntheticAnomalyGenerator | PerlinAnomalyGenerator:
    """Build the paper's live synthetic-anomaly generator."""
    if pipeline == "Perlin":
        return PerlinAnomalyGenerator(
            anomaly_source_path=EVAL_DATA_ROOT / "dtd",
            probability=1.0,
            blend_factor=(0.01, 0.2),
        )
    preset, overrides = PIPELINES[pipeline]
    return SyntheticAnomalyGenerator.from_preset(preset, probability=1.0, **overrides)


def _take_normals(dataset: object, count: int, seed: int) -> object:
    """Return ``count`` sampled normal rows, with replacement if needed."""
    subset = copy.copy(dataset)
    available = len(dataset.samples)
    replace = count > available
    subset.samples = (
        dataset.samples.sample(count, replace=replace, random_state=seed).reset_index(drop=True)
        if count > 0
        else dataset.samples.head(0).copy()
    )
    return subset


def _synthetic_eval_set(
    negatives: object,
    source_normals: object,
    n_anomalies: int,
    augmenter: SyntheticAnomalyGenerator | PerlinAnomalyGenerator,
    seed: int,
) -> SyntheticAnomalyDataset:
    """Build an evaluation set from real normals and synthetic anomalies."""
    source = _take_normals(source_normals, max(2 * n_anomalies, 4), seed)
    synthetic = SyntheticAnomalyDataset.from_dataset(source, augmenter=augmenter)
    anomalies = synthetic.samples[synthetic.samples.label_index == 1].head(n_anomalies).copy()
    negative_samples = negatives.samples.copy()
    negative_samples["split"] = Split.VAL
    anomalies["split"] = Split.VAL
    synthetic.samples = pd.concat([negative_samples, anomalies], ignore_index=True)
    synthetic.samples.attrs["task"] = "segmentation"
    return synthetic


def _pregenerated_eval_set(
    negatives: object,
    category: str,
    pipeline: str,
    n_anomalies: int,
    seed: int,
) -> AnomalibDataset:
    """Build an evaluation set from pre-generated images and masks.

    Args:
        negatives (object): Normal-sample pool.
        category (str): MVTec AD 2 category.
        pipeline (str): ``"FLASH"`` or ``"AnoStyler"``.
        n_anomalies (int): Number of synthetic anomalies.
        seed (int): Sweep seed, mapped to the pre-generated seed ids.

    Raises:
        FileNotFoundError: If no pre-generated images are found.
    """
    source_dir = PREGENERATED_PIPELINES[pipeline]
    gen_seed = PREGENERATED_SEEDS[(seed - 1) % len(PREGENERATED_SEEDS)]
    root = SYNTHETIC_GEN_ROOT / f"MVTec_AD_2_{source_dir}_{gen_seed}" / category / "test_public"
    bad_dir, mask_dir = root / "bad", root / "ground_truth" / "bad"
    image_paths = sorted(bad_dir.glob("*.png"))
    if not image_paths:
        msg = f"No pre-generated images found under {bad_dir}"
        raise FileNotFoundError(msg)
    anomalies = pd.DataFrame([
        {
            "image_path": str(image_path),
            "label": "abnormal",
            "label_index": 1,
            "mask_path": (
                str(mask_path)
                if (mask_path := mask_dir / f"{image_path.stem}_mask.png").exists()
                else None
            ),
            "split": Split.VAL,
        }
        for image_path in image_paths
    ]).head(n_anomalies)
    negative_samples = negatives.samples.copy()
    negative_samples["split"] = Split.VAL
    dataset = AnomalibDataset(augmentations=getattr(negatives, "augmentations", None))
    dataset.samples = pd.concat([negative_samples, anomalies], ignore_index=True)
    # Use classification when generated masks are absent.
    has_masks = anomalies["mask_path"].notna().any()
    dataset.samples.attrs["task"] = "segmentation" if has_masks else "classification"
    return dataset


def build_datamodule(job: JobConfig) -> object:
    """Build the datamodule and preserve a test-disjoint calibration pool."""
    dataset_cls, root, _ = DATASETS[job.dataset]
    train_batch = MODEL_BATCH.get(job.model, DEFAULT_BATCH)
    eval_batch = MODEL_EVAL_BATCH.get(job.model, DEFAULT_BATCH)
    kwargs: dict = {
        "root": root,
        "category": job.category,
        "train_batch_size": train_batch,
        "eval_batch_size": eval_batch,
        "num_workers": 4,
        "seed": job.seed,
    }
    resolution = RESOLUTIONS.get(job.dataset)
    if resolution is not None:
        kwargs["augmentations"] = Resize(resolution, antialias=True)

    has_native_val = job.dataset == "mvtec2"
    if not has_native_val:
        kwargs["val_split_mode"] = ValSplitMode.SAME_AS_TEST

    datamodule = dataset_cls(**kwargs)
    datamodule.prepare_data()
    datamodule.setup()

    if has_native_val:
        datamodule.calibration_normals = datamodule.val_data
        datamodule.val_data = copy.deepcopy(datamodule.test_data)
    else:
        datamodule.calibration_normals = None

    # Keep the injected splits.
    datamodule._is_setup = True  # noqa: SLF001
    return datamodule


def _set_test_data(datamodule: object, dataset: object) -> None:
    """Point both MVTec AD 2 test datasets at ``dataset``."""
    datamodule.test_data = dataset
    if hasattr(datamodule, "test_public_data"):
        datamodule.test_public_data = dataset


def _calibration_negatives(job: JobConfig, datamodule: object, test_normals: object) -> object:
    """Select the normal pool for synthetic calibration."""
    if job.calibration == "test_normals":
        return test_normals
    if job.calibration == "heldout":
        pool = getattr(datamodule, "calibration_normals", None)
        if pool is None or len(pool.samples) == 0:
            msg = (
                f"calibration='heldout' needs a test-disjoint normal pool, but dataset "
                f"'{job.dataset}' does not provide one."
            )
            raise ValueError(msg)
        return pool
    msg = f"Unknown calibration mode: {job.calibration!r}"
    raise ValueError(msg)


def _thresholds(model: object) -> tuple[float | None, float | None]:
    """Best-effort extraction of the fitted (raw, normalized) image thresholds."""
    try:
        post_processor = model.post_processor
        return float(post_processor.image_threshold), float(post_processor.normalized_image_threshold)
    except (AttributeError, RuntimeError, ValueError, TypeError):
        return None, None


def _metrics(results: list | None) -> dict[str, float]:
    """Flatten the first test-result dict into ``{metric: float}``."""
    return {k: float(v) for k, v in (results[0] if results else {}).items()}


def _reset_metrics(model: object) -> None:
    """Reset evaluator metrics so successive validate/test calls do not accumulate."""
    for metric in (*model.evaluator.val_metrics, *model.evaluator.test_metrics):
        metric.reset()


def _reset_post_processor(model: object) -> None:
    """Clear fitted threshold and normalization buffers."""
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return
    for name in ("_image_threshold", "_pixel_threshold", "image_min", "image_max", "pixel_min", "pixel_max"):
        buffer = getattr(post_processor, name, None)
        if isinstance(buffer, torch.Tensor):
            buffer.fill_(float("nan"))


@torch.no_grad()
def collect_raw_scores(model: object, dataset: object, batch_size: int = 8) -> dict[str, list]:
    """Collect raw per-image scores and labels.

    Scores bypass post-processing so calibration settings remain comparable.

    Args:
        model (object): A trained anomaly model exposing ``forward``.
        dataset (object): Dataset to score.
        batch_size (int): Batch size for the scoring pass.

    Returns:
        dict[str, list]: ``{"scores": [...], "labels": [...]}``.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    scores: list[float] = []
    labels: list[int] = []
    for batch in loader:
        output = model(batch.image.to(device))
        pred = getattr(output, "pred_score", None)
        if pred is None:  # models that only emit an anomaly map
            pred = output.anomaly_map.flatten(1).amax(dim=1)
        scores.extend(pred.detach().cpu().flatten().tolist())
        gt = batch.gt_label
        labels.extend(gt.detach().cpu().flatten().int().tolist() if gt is not None else [-1] * len(pred))

    if was_training:
        model.train()
    return {"scores": scores, "labels": labels}


def _row(
    job: JobConfig,
    calibration_source: str,
    evaluation_set: str,
    metrics: dict[str, float],
    sizes: tuple[int, int, int],
    thresholds: tuple[float | None, float | None],
    timings: tuple[float, float],
    effective_size: str,
) -> dict:
    """Assemble a single flat result row."""
    n_train, n_val, n_test = sizes
    image_threshold, normalized_threshold = thresholds
    fit_seconds, test_seconds = timings
    resolution = effective_size
    return {
        "experiment": job.experiment,
        "dataset": job.dataset,
        "category": job.category,
        "model": job.model,
        "backbone": job.backbone,
        "calibration_source": calibration_source,
        "evaluation_set": evaluation_set,
        "seed": job.seed,
        **metrics,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "image_threshold": image_threshold,
        "normalized_image_threshold": normalized_threshold,
        "calibration": job.calibration,
        "resolution": resolution,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
        "anomalib_version": anomalib.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_job(job: JobConfig) -> tuple[list[dict], dict[str, dict[str, list]]]:
    """Fit one model and evaluate all configured calibration sources.

    Returns:
        tuple: Result rows and raw score records by source.
    """
    model = build_model(job.model, RESOLUTIONS.get(job.dataset), backbone=job.backbone)
    eff_size = _effective_input_size(model)
    effective_size = "x".join(str(v) for v in eff_size) if eff_size else "model_default"
    datamodule = build_datamodule(job)
    n_train = len(datamodule.train_data)
    real_test = datamodule.test_data
    n_real_test = len(real_test)

    with tempfile.TemporaryDirectory(prefix="anomalib_run_") as scratch:
        engine = Engine(
            accelerator="gpu",
            devices=1,
            logger=False,
            default_root_dir=scratch,
            **MODEL_TRAINER[job.model],
        )

        # Fit once; Real calibration is the oracle reference.
        start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = round(time.time() - start, 2)

        rows: list[dict] = []

        _reset_metrics(model)
        start = time.time()
        results_a = engine.test(model=model, datamodule=datamodule, verbose=False)
        test_seconds = round(time.time() - start, 2)
        rows.append(
            _row(job, "Real", "Real test set", _metrics(results_a), (n_train, n_real_test, n_real_test),
                 _thresholds(model), (fit_seconds, test_seconds), effective_size),
        )

        test_normals, test_anomalies = split_by_label(real_test)
        n_anomalies = len(test_anomalies)
        negatives = _calibration_negatives(job, datamodule, test_normals)

        eval_batch = MODEL_EVAL_BATCH.get(job.model, DEFAULT_BATCH)
        scores: dict[str, dict[str, list]] = {
            "real_test": collect_raw_scores(model, real_test, batch_size=eval_batch),
        }

        keep_alive: list[SyntheticAnomalyDataset | AnomalibDataset] = []  # hold temp dirs until job ends
        for pipeline in job.pipelines:
            if pipeline in PREGENERATED_PIPELINES:
                calibration = _pregenerated_eval_set(
                    negatives=negatives,
                    category=job.category,
                    pipeline=pipeline,
                    n_anomalies=n_anomalies,
                    seed=job.seed,
                )
            else:
                calibration = _synthetic_eval_set(
                    negatives=negatives,
                    source_normals=datamodule.train_data,
                    n_anomalies=n_anomalies,
                    augmenter=make_generator(pipeline),
                    seed=job.seed,
                )
            keep_alive.append(calibration)
            n_calib = len(calibration)
            scores[pipeline] = collect_raw_scores(model, calibration, batch_size=eval_batch)

            calibration_source = CALIBRATION_SOURCES.get(pipeline, pipeline)

            datamodule.val_data = calibration
            _set_test_data(datamodule, real_test)
            _reset_metrics(model)
            _reset_post_processor(model)
            engine.validate(model=model, datamodule=datamodule, verbose=False)

            _reset_metrics(model)
            start = time.time()
            results_b = engine.test(model=model, datamodule=datamodule, verbose=False)
            test_seconds = round(time.time() - start, 2)
            rows.append(
                _row(job, calibration_source, "Real test set", _metrics(results_b),
                     (n_train, n_calib, n_real_test),
                     _thresholds(model), (fit_seconds, test_seconds), effective_size),
            )

            if job.include_diagnostic:
                _set_test_data(datamodule, calibration)
                _reset_metrics(model)
                start = time.time()
                results_c = engine.test(model=model, datamodule=datamodule, verbose=False)
                test_seconds = round(time.time() - start, 2)
                rows.append(
                    _row(job, calibration_source, "Synthetic calibration set", _metrics(results_c),
                         (n_train, n_calib, n_calib),
                         _thresholds(model), (fit_seconds, test_seconds), effective_size),
                )
                _set_test_data(datamodule, real_test)

    return rows, scores
