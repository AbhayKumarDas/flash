# Module 1: Anomaly prompt construction

Produces one anomaly prompt per defect-free image. No image is generated here. The prompts are
handed to an image generation model in the next step, and the resulting anomaly images are the
input to Module 2.

## Requirements

Kaggle notebook, accelerator set to GPU T4 x2, internet off.

Attach two mounts:

1. `Qwen/Qwen2.5-VL-7B-Instruct`, as a Model.
2. MVTec AD 2, as a Dataset.

## Steps

1. Open section 2 and set `N_IMAGES`, the categories you want, and the output directory. If the
   mounts are named differently on your kernel, set `MODEL_ID` and `DATA_ROOT_ID` as well.
2. Run all cells in order.
3. Read the prompts in section 9 before spending any generation budget. Discard any that describe
   a defect the image cannot support.
4. Pass each prompt together with its normal image to the image generation model, and save the
   returned anomaly image for Module 2.

## Output

```
<output dir>/
  <category>/Normal_<category>_<id>.png    the sampled normal images
  prompts.csv                              one row per image: tag, family, defect, prompt, seconds
  manifest.json                            the same content as JSON
```

Each image gets one prompt. Prompts within a category are forced apart: every image is assigned a
different defect family from a fixed vocabulary before the model sees it, the size word is sampled
per call, and a repeated defect name is retried.
