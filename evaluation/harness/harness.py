# Copyright (C) 2020-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core harness for the synthetic-anomaly experiments.

A single job trains one model **once** on all train normals and then derives every
evaluation arm from that identical model, so the arms differ only in the threshold
source (never in the underlying weights). This removes the model-variance confound
that separate per-arm training introduced.

Arms (all share one trained model; full official test for A and B):
    A: oracle. Threshold from the real test (fit-time val = SAME_AS_TEST) -> oracle F1-max.
    B: proxy. Threshold refit on {real test normals + N synthetic anomalies}
       (N = real anomaly count), then transferred to the full real test. Differs from
       A only in real->synthetic anomalies (same normals, same model).
    C: diagnostic. Threshold refit on, and evaluated on, the synthetic set itself
       (is the synthetic set too easy/hard relative to the real test?).
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

# ``create_anomalib_metric`` builds the class with ``type()``, so ``__module__``
# resolves to ``abc`` and pickle cannot find it again -- which makes checkpointing fail
# for any dynamically created metric. Repoint the identity at this module (using the
# original torchmetrics class names, so the reported column stays ``image_BinaryPrecision``
# / ``image_BinaryRecall`` and matches every earlier phase) so the classes are
# importable by their bound names.
for _name, _cls in (("BinaryPrecision", BinaryPrecisionMetric), ("BinaryRecall", BinaryRecallMetric)):
    _cls.__name__ = _cls.__qualname__ = _name
    _cls.__module__ = __name__
globals()["BinaryPrecision"] = BinaryPrecisionMetric
globals()["BinaryRecall"] = BinaryRecallMetric
Precision = BinaryPrecisionMetric
Recall = BinaryRecallMetric

# Dataset registry: name -> (datamodule class, root, categories).
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

# Input resolution per dataset. MVTec AD 2 ships 2.3-5.0 MP images with non-uniform
# aspect ratios, so it must be resized explicitly or the models run out of memory.
RESOLUTIONS: dict[str, tuple[int, int] | None] = {
    "mvtec": None,
    "visa": None,
    "mvtec2": (448, 448),
}

# Pipeline registry: id -> (preset, generator overrides).
PIPELINES = {
    # Stock, unmodified anomalib `PerlinAnomalyGenerator` (the literal class DRAEM's
    # own training loop uses, and `make_synthetic_dataset`'s own no-argument default)
    # -- NOT built from our composable SyntheticAnomalyGenerator framework at all. This
    # is the genuine "vanilla anomalib Perlin, zero customization from this project"
    # baseline: whole-image region, DTD texture content, alpha blend, all baked into
    # one class with no separate mask/region/source/blend abstraction. Kwargs mirror
    # `make_synthetic_dataset`'s own default (see `synthetic.py`), which is anomalib's
    # literal out-of-the-box behavior when no custom augmenter is supplied.
    "P0": None,
    "P1": ("texture_alpha", {"source_kwargs": {"texture_path": EVAL_DATA_ROOT / "dtd"}}),
    "P2": ("self_alpha", {}),
    "P3": ("self_poisson", {}),
    # Area-routed blend: alpha keeps small components visible, Poisson keeps large
    # ones seamless. Threshold is in pixels at the model's input resolution.
    "P4": ("self_hybrid", {"blend_kwargs": {"area_threshold": 2500}}),
    # Threshold ablation around P4. Lower values route more area through Poisson
    # (approaching P3); higher values route more through alpha (approaching P2).
    "P4a": ("self_hybrid", {"blend_kwargs": {"area_threshold": 500}}),
    "P4b": ("self_hybrid", {"blend_kwargs": {"area_threshold": 1000}}),
    "P4c": ("self_hybrid", {"blend_kwargs": {"area_threshold": 6000}}),
    "P4d": ("self_hybrid", {"blend_kwargs": {"area_threshold": 12000}}),
}

# Pre-generated synthetic-anomaly pipelines: id -> semantic-defect-bank blend arm.
# Unlike PIPELINES above (live augmentation via SyntheticAnomalyGenerator), these read
# already-rendered (image, mask) pairs from disk -- produced once by
# the generation notebook, which cuts real defect patches from a
# small donor bank and replays them onto held-out train/good hosts at MRSP/OBS-chosen
# sites (see that notebook's own docs for the full method). See
# ``_pregenerated_eval_set`` for how these are turned into an evaluable dataset.
#
# As of the 2026-08-25 regeneration (`flash-part1.ipynb`), the on-disk directories were
# renamed and consolidated: `MVTec_AD_2_alpha_*` was removed and `MVTec_AD_2_poisson_*`
# was renamed to `MVTec_AD_2_hybrid_*` to correct the earlier terminology -- what this
# harness always called "poisson" (`P6`) was, per the notebook's own labeling, already
# a poisson+alpha hybrid blend, not pure Poisson; there is now only one pregenerated
# variant on disk. `P5` ("alpha") is therefore no longer resolvable and is not used
# from phase 11 onward; `P6` now maps to the `hybrid` directory name.
PREGENERATED_PIPELINES = {"P6": "hybrid", "P7": "anomalyany", "P8": "anostyler"}
# The generation notebook regenerated SynthetciGenMVAD2 in place with a
# donor bank covering all 8 MVTec AD 2 categories (was 4: rice/walnuts/wallplugs/
# fruit_jelly, from the earlier `semantic_bank_blend.ipynb` pipeline) -- 3 donor pairs
# per category, 24 pairs total. Verified live against the 4 newly-covered categories
# (can, fabric, sheet_metal, vial): counts match each category's real test_public/bad
# count exactly, same as before.
PREGENERATED_CATEGORIES = (
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
)
SYNTHETIC_GEN_ROOT = EVAL_DATA_ROOT / "SynthetciGenMVAD2"
# The notebook only rendered 3 fixed seeds (0, 1, 2), unlike the live generators above
# which accept any seed -- job.seed (sweep convention: 1, 2, 3) is remapped onto this
# range by subtracting 1 in ``_pregenerated_eval_set``.
PREGENERATED_SEEDS = (0, 1, 2)
# Backbone for the SuperADD MVTec AD 2 sweep. `large` (303M params) was used as a
# lighter pilot to validate the pipeline (see phase7); this is now the paper's actual
# `huge_plus` (840M) config. Verified via an isolated memory/timing test: batch=4 at
# 448x448 peaks at ~4.1 GB GPU memory across a full train-forward + subsample_embedding
# + inference cycle -- comfortably fits alongside another concurrent job per GPU.
SUPERADD_BACKBONE = "vit_huge_plus_patch16_dinov3"


# Per-model trainer settings and train batch size.
MODEL_TRAINER = {
    "patchcore": {"max_epochs": 1},
    "padim": {"max_epochs": 1},
    "efficient_ad": {"max_epochs": 20},
    "draem": {"max_epochs": 15},
    # Dinomaly is a full gradient-trained reconstruction model. Its trainer_arguments
    # intentionally drops ``max_steps`` so the engine/user controls the budget, and its
    # ``configure_optimizers`` reads ``max_steps`` from the trainer to size the LR
    # schedule (the paper uses 5000 steps; 1000 is the loss's ``p_schedule_steps`` and
    # the value requested for the MVTec AD 2 comparison). ``max_epochs`` is set loose
    # so the ``max_steps`` limit is the actual stopper.
    "dinomaly": {"max_steps": 1000, "max_epochs": 1000},
    "anomaly_dino": {"max_epochs": 1},
    # SuperADD is a training-free memory-bank method: its own trainer_arguments
    # hardcodes max_epochs=1 regardless of what is requested here (Engine always
    # defers to the model's trainer_arguments over any user-supplied value), so this
    # entry exists only to document that and avoid a KeyError below.
    "superadd": {"max_epochs": 1},
}
MODEL_BATCH = {"efficient_ad": 1, "draem": 2, "superadd": 4}
MODEL_EVAL_BATCH = {"draem": 2, "superadd": 4}
DEFAULT_BATCH = 8


@dataclass
class JobConfig:
    """Configuration for a single training job.

    One job trains a model once and emits several result rows: arm A (oracle) plus
    arm B for each pipeline (and arm C for each pipeline when ``include_c`` is set).

    Args:
        phase (str): Sweep phase this job belongs to.
        dataset (str): Dataset key (``"mvtec"`` or ``"visa"``).
        category (str): Category within the dataset.
        model (str): Model name.
        seed (int): Random seed for the datamodule and synthetic sampling.
        pipelines (tuple[str, ...]): Synthetic-anomaly pipeline ids to evaluate for arm B/C.
        include_c (bool): Whether to also emit the diagnostic arm C.
        calibration (str): Source of the negative (normal) samples in the arm-B/C
            calibration set. ``"test_normals"`` reuses the real test normals, which
            leaks: the threshold is fitted not to flag the very images it is then
            evaluated on. ``"heldout"`` instead draws them from a normal pool that is
            disjoint from the test set (MVTec AD 2's native ``validation/`` split).
        backbone (str | None): Backbone override for models that accept one (e.g.
            SuperADD). ``None`` keeps the model's default (huge_plus for SuperADD).
    """

    phase: str
    dataset: str
    category: str
    model: str
    seed: int = 1
    pipelines: tuple[str, ...] = ("P1", "P2", "P3")
    include_c: bool = False
    calibration: str = "test_normals"
    backbone: str | None = None

    @property
    def key(self) -> str:
        """Unique, filesystem-safe identifier for this job."""
        parts = [self.phase, self.dataset, self.category, self.model]
        if self.backbone:
            parts.append(self.backbone)
        parts.append(f"s{self.seed}")
        return "_".join(parts)


def build_evaluator() -> Evaluator:
    """Return an evaluator with image and pixel metrics.

    ``pixel_F1Score`` is computed from ``pred_mask`` against ``gt_mask`` with a single
    global threshold and micro-averaged pixel counts, which is exactly MVTec AD 2's
    SegF1 -- so no separate SegF1 metric is needed. ``AUPRO`` is reported at both the
    conventional FPR<=0.3 and MVTec AD 2's stricter FPR<=0.05. ``AUPR`` at pixel level
    is added because pixel AUROC is inflated by class imbalance.
    """
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
    """Instantiate a model by name with the rich evaluator.

    Args:
        name (str): Model key.
        resolution (tuple[int, int] | None): Input size for the model's pre-processor.
            The model's own transform decides the effective input size, so resizing
            only the datamodule is silently ignored. ``None`` keeps the model default.
        tiled (bool): Whether the model will be run through tiled inference (see
            ``tiled_harness.py``). Currently only affects Dinomaly, whose published
            392 px center-crop must be disabled under tiling (see inline comment) --
            kept as a general parameter so future tiled-specific adjustments have a
            place to hook in.
        backbone (str | None): Backbone override for models that accept one (e.g.
            SuperADD). ``None`` keeps the model's default.

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
        # SuperADD's own configure_post_processor() returns a percentile-based
        # threshold fit on normal-only validation scores (no F1AdaptiveThreshold),
        # specifically because that adaptive threshold degenerates to the max
        # validation score on datasets whose validation split is normal-only (like
        # MVTec AD 2) -- see SuperADDPostProcessor's docstring. Force the standard
        # F1AdaptiveThreshold-based PostProcessor instead so arms A/B/C are computed
        # identically to the other 6 models (apples-to-apples comparison); SuperADD's
        # own built-in threshold is a candidate follow-up "arm D" but is not wired in
        # here.
        kwargs["post_processor"] = PostProcessor()
        kwargs["backbone"] = backbone or SUPERADD_BACKBONE
    if name == "anomaly_dino":
        # At 448 px the per-category patch bank (n_train x patches/image) is large
        # enough that the all-pairs query/bank distance matrix OOMs on every MVTec AD 2
        # category except the smallest (sheet_metal, 137 train images). Coreset
        # subsampling shrinks the bank itself rather than the resolution.
        kwargs["coreset_subsampling"] = True
        kwargs["sampling_ratio"] = 0.1
    # NOTE: an earlier version shrank PatchCore's coreset_sampling_ratio to 0.01 for
    # tiled inference to cut its ~63 s/image cost. Verified (empirically, on normal vs
    # anomalous scores) that this drops the memory bank to ~43 vectors, which destroys
    # discriminative power: normal-image scores (57.9-59.1) and anomalous-image scores
    # (62.0-63.5) barely overlapped at the DEFAULT ratio 0.1, but became indistinguishable
    # (61.4-62.7 vs 62.3-63.9) at 0.01, producing a near-chance AUROC. Reverted -- do not
    # shrink PatchCore's bank for speed; the wider tile stride below is the safe lever.
    if resolution is not None:
        if name == "dinomaly" and tiled:
            # Dinomaly's published recipe resizes to 448 then center-crops to 392 (see
            # configure_pre_processor's docstring); that crop lives INSIDE the model's
            # pre-processor, so it fires on every forward pass, including each tile's.
            # Under tiled inference each tile is already exactly `resolution` (448x448)
            # and anomalib's `Tiler.untile()` expects the model to return a map the same
            # shape as its input tile -- but the 392px crop shrinks the output map to
            # 392x392, corrupting the stitched map (confirmed: every dinomaly phase 6
            # job crashed with a tensor-size mismatch when fitting the pixel threshold,
            # since the corrupted map's element count no longer matched the mask's).
            # Fix: pass crop_size == image_size so the crop is a no-op under tiling; the
            # non-tiled 448 px path (phase 5) is unaffected since it never sets `tiled`.
            kwargs["pre_processor"] = model_cls.configure_pre_processor(resolution, crop_size=resolution[0])
        else:
            # Use the model class's own factory: several models constrain their
            # transform (EfficientAd and DRAEM reject a Normalize step because they
            # normalise inside the forward pass), so the generic AnomalibModule
            # pre-processor is rejected.
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
    """Return the spatial size the network actually receives.

    Some models crop after resizing (Dinomaly's published recipe is resize 448 then
    center-crop 392), so the resize target alone overstates the true input size. The
    crop is part of the model's official configuration and is deliberately preserved --
    overriding it would deviate from the published recipe and weaken the baseline.
    """
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
    """Build the synthetic-anomaly generator for a pipeline id.

    ``"P0"`` returns anomalib's stock, unmodified ``PerlinAnomalyGenerator`` directly
    (not our composable ``SyntheticAnomalyGenerator`` framework) -- see ``PIPELINES``.
    """
    if pipeline == "P0":
        return PerlinAnomalyGenerator(
            anomaly_source_path=EVAL_DATA_ROOT / "dtd",
            probability=1.0,
            blend_factor=(0.01, 0.2),
        )
    preset, overrides = PIPELINES[pipeline]
    return SyntheticAnomalyGenerator.from_preset(preset, probability=1.0, **overrides)


def _take_normals(dataset: object, count: int, seed: int) -> object:
    """Return a shallow copy of a normal dataset limited to ``count`` random rows.

    Samples with replacement when ``count`` exceeds the pool size, so a small source
    pool can still seed the requested number of anomalies (each draw gets an
    independent mask, so duplicated sources still yield distinct anomalies).
    """
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
    """Build an eval dataset of ``{real normal negatives + N synthetic anomalies}``.

    The negatives are used as-is (e.g. the real test normals) so the set differs from
    the real test only in the anomaly source. Synthetic anomalies are generated from
    ``source_normals`` (train normals); the generated normal rows are discarded.
    """
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
    """Build an eval dataset from a pre-generated (semantic-defect-bank) pipeline.

    Unlike :func:`_synthetic_eval_set`, the (image, mask) pairs already exist on disk
    under ``SYNTHETIC_GEN_ROOT`` -- rendered once by the generation notebook -- so this
    only assembles the samples
    ``DataFrame``, it does not augment anything itself.

    Args:
        negatives (object): Normal-sample pool (e.g. the heldout ``validation/`` set);
            used as-is, matching :func:`_synthetic_eval_set`'s convention.
        category (str): MVTec AD 2 category; must be one of ``PREGENERATED_CATEGORIES``.
        pipeline (str): One of ``PREGENERATED_PIPELINES`` (``"P6"``).
        n_anomalies (int): Number of anomalous rows to keep (matches the real test
            anomaly count, for parity with the live-generator pipelines).
        seed (int): Sweep seed (1, 2, 3 convention); remapped onto the notebook's fixed
            ``PREGENERATED_SEEDS`` (0, 1, 2) via ``(seed - 1) % 3``.

    Raises:
        FileNotFoundError: If no pre-generated images are found for this
            (category, pipeline, seed) combination.
    """
    arm = PREGENERATED_PIPELINES[pipeline]
    gen_seed = PREGENERATED_SEEDS[(seed - 1) % len(PREGENERATED_SEEDS)]
    root = SYNTHETIC_GEN_ROOT / f"MVTec_AD_2_{arm}_{gen_seed}" / category / "test_public"
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
    # AnomalyAny (P7) emits image-only rows (no ``*_mask.png``). ``AnomalibDataset``
    # would crash on ``read_mask(None)`` for an abnormal row in segmentation mode, so
    # fall back to classification when no masks are present. Pixel metrics on this set
    # are then skipped (they are never evaluated on the synthetic rows anyway -- arm B
    # transfers only the image threshold to the real test, which ships its own masks).
    has_masks = anomalies["mask_path"].notna().any()
    dataset.samples.attrs["task"] = "segmentation" if has_masks else "classification"
    return dataset


def build_datamodule(job: JobConfig) -> object:
    """Construct the base datamodule (all train normals; real test as val and test).

    The model trains on all official train normals and validates on the full official
    test set (SAME_AS_TEST), so the fit-time threshold is the oracle F1-max. Synthetic
    calibration sets for arms B/C are injected later in :func:`run_job`.

    Datamodules that ship their own validation split (MVTec AD 2) do not accept
    ``val_split_mode``. For those the native split is stashed on
    ``calibration_normals`` as a held-out, test-disjoint pool before ``val_data`` is
    repointed at the real test to recover the oracle threshold.
    """
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

    # Freeze the splits so the trainer does not re-run setup and discard our injection.
    datamodule._is_setup = True  # noqa: SLF001
    return datamodule


def _set_test_data(datamodule: object, dataset: object) -> None:
    """Point the datamodule's test dataloader at ``dataset``.

    ``MVTecAD2.test_dataloader`` dispatches on ``test_type`` and builds its loader from
    ``test_public_data`` rather than ``test_data``, so assigning only ``test_data``
    would be silently ignored and arm C would duplicate arm B. Setting both keeps the
    swap effective without replacing the method (which would make the datamodule
    unpicklable for dataloader workers).
    """
    datamodule.test_data = dataset
    if hasattr(datamodule, "test_public_data"):
        datamodule.test_public_data = dataset


def _calibration_negatives(job: JobConfig, datamodule: object, test_normals: object) -> object:
    """Pick the normal samples used as negatives in the arm-B/C calibration set.

    ``"heldout"`` requires a pool disjoint from the test set; it is the leakage-free
    option and is currently only available where the dataset ships one.
    """
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
    """Reset fitted threshold/normalization buffers so each arm refits from scratch.

    ``PostProcessor.on_validation_epoch_end`` only overwrites a buffer when its metric
    was updated during *that* validation pass. A maskless calibration set (AnomalyAny,
    P7) never updates the pixel-threshold metric, so without this reset the buffer
    silently retains the value fitted during ``engine.fit`` -- the *oracle* threshold,
    fitted on the real test -- inflating arm-B pixel metrics. Resetting to NaN forces
    either a clean refit (masked pipelines) or the well-defined midpoint-normalization
    fallback (maskless pipelines, ``_normalize`` uses ``(max + min) / 2``).
    """
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return
    for name in ("_image_threshold", "_pixel_threshold", "image_min", "image_max", "pixel_min", "pixel_max"):
        buffer = getattr(post_processor, name, None)
        if isinstance(buffer, torch.Tensor):
            buffer.fill_(float("nan"))


@torch.no_grad()
def collect_raw_scores(model: object, dataset: object, batch_size: int = 8) -> dict[str, list]:
    """Return per-image *raw* (un-normalised) anomaly scores and labels for a dataset.

    Aggregate metrics cannot answer whether the synthetic anomaly score distribution
    matches the real one, so the per-image scores are recorded separately. The model is
    invoked directly rather than through ``Engine.test`` so the scores bypass the
    post-processor: normalisation is pivoted on the fitted threshold, which differs
    between arms and would make the distributions incomparable.

    Args:
        model (object): A trained anomaly model exposing ``forward``.
        dataset (object): Dataset to score.
        batch_size (int): Batch size for the scoring pass.

    Returns:
        dict[str, list]: ``{"scores": [...], "labels": [...]}`` with one entry per image.
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
    arm: str,
    pipeline: str,
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
        "phase": job.phase,
        "dataset": job.dataset,
        "category": job.category,
        "model": job.model,
        "backbone": job.backbone,
        "pipeline": pipeline,
        "arm": arm,
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
    """Train one model once and evaluate every arm from that identical model.

    Returns one row for arm A (oracle threshold), one row for arm B per pipeline
    (synthetic threshold transferred to the real test) and, when ``job.include_c``
    is set, one row for arm C per pipeline (synthetic threshold on the synthetic set).

    Returns:
        tuple: ``(rows, scores)`` where ``scores`` maps ``"real_test"`` and each
        pipeline id to raw per-image ``{"scores": [...], "labels": [...]}``.
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

        # Train once; fit-time validation on the real test sets the oracle threshold.
        start = time.time()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = round(time.time() - start, 2)

        rows: list[dict] = []

        # Arm A: oracle threshold (from the fit-time real-test validation).
        _reset_metrics(model)
        start = time.time()
        results_a = engine.test(model=model, datamodule=datamodule, verbose=False)
        test_seconds = round(time.time() - start, 2)
        rows.append(
            _row(job, "A", "-", _metrics(results_a), (n_train, n_real_test, n_real_test),
                 _thresholds(model), (fit_seconds, test_seconds), effective_size),
        )

        test_normals, test_anomalies = split_by_label(real_test)
        n_anomalies = len(test_anomalies)
        negatives = _calibration_negatives(job, datamodule, test_normals)

        # Raw per-image scores on the real test, for score-distribution comparisons.
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

            # Refit the threshold on the synthetic calibration set (arm B/C threshold).
            datamodule.val_data = calibration
            _set_test_data(datamodule, real_test)
            _reset_metrics(model)
            _reset_post_processor(model)
            engine.validate(model=model, datamodule=datamodule, verbose=False)

            # Arm B: synthetic-derived threshold transferred to the real test.
            _reset_metrics(model)
            start = time.time()
            results_b = engine.test(model=model, datamodule=datamodule, verbose=False)
            test_seconds = round(time.time() - start, 2)
            rows.append(
                _row(job, "B", pipeline, _metrics(results_b), (n_train, n_calib, n_real_test),
                     _thresholds(model), (fit_seconds, test_seconds), effective_size),
            )

            if job.include_c:
                # Arm C: same synthetic threshold, evaluated on the synthetic set itself.
                _set_test_data(datamodule, calibration)
                _reset_metrics(model)
                start = time.time()
                results_c = engine.test(model=model, datamodule=datamodule, verbose=False)
                test_seconds = round(time.time() - start, 2)
                rows.append(
                    _row(job, "C", pipeline, _metrics(results_c), (n_train, n_calib, n_calib),
                         _thresholds(model), (fit_seconds, test_seconds), effective_size),
                )
                _set_test_data(datamodule, real_test)

    return rows, scores
