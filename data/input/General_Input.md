# FLASH Pipeline: General Input Instruction

This document defines the common external inputs used across all three FLASH modules. Each
module runs as a standalone Kaggle notebook; both inputs below must be attached before the
notebook is run. Module-specific inputs and outputs are defined separately.

### Dataset

**MVTec AD 2** is the common dataset input across **Module 1, Module 2, and Module 3**. The
pipeline operates on normal images from the dataset; module-specific processing and outputs are
defined separately.

- **Kaggle:** https://www.kaggle.com/datasets/zaidenthiha/mvtec-ad-2-dataset
- **Official MVTec AD 2:** https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2

Attach as a Kaggle **Dataset** mount. All eight categories are used: `can`, `fabric`,
`fruit_jelly`, `rice`, `sheet_metal`, `vial`, `wallplugs`, `walnuts`. Normal images are read
from `<category>/train/good`; held-out normals for threshold calibration come from
`<category>/validation/good`.

MVTec AD 2 is the benchmark used in the FLASH paper for evaluating synthetic anomaly generation
and calibration transfer.

### VLM

**Module 1 and Module 2** use **Qwen2.5-VL-7B-Instruct** as the baseline Vision-Language Model
through the Hugging Face Transformers stack.

- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Hugging Face:** https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

Attach as a Kaggle **Model** mount. The kernel runs without internet, so the weights must be
mounted rather than downloaded. Recommended accelerator is **T4 × 2**, with the model loaded in
`float16` and split across both devices. **Module 3 requires no VLM.**

In FLASH, VLM-1 provides global semantic reasoning for anomaly specification, while VLM-2
validates extracted defect crops before they are added to the semantic defect bank. The paper
uses Qwen2.5-VL-7B-Instruct for both VLM stages.

**Model substitution:** Qwen3-VL with a larger parameter count may be used for Module 1 or
Module 2 when supported by the available GPU/VRAM and runtime. On T4 × 2 (32 GB total), the 4B
and 8B variants fit in `float16`; larger variants require quantisation or higher-memory GPUs.
The exact model should be selected according to hardware compatibility.

> **Baseline:** MVTec AD 2 is fixed as the common dataset input across all three FLASH modules;
> Qwen2.5-VL-7B-Instruct is the reference VLM configuration for Modules 1 and 2.
