# Module 1: Semantic-guided anomaly generation

Paper Stage 1. Notebook [`notebooks/01_semantic_generation.ipynb`](../../notebooks/01_semantic_generation.ipynb).

## What this stage is for

Before anything can be generated, something has to decide *what defect* to introduce. Writing
that by hand does not scale past a few categories and biases the corpus towards whatever the
author happened to think of. Stage 1 hands the decision to a vision language model that looks at
the actual image.

Given a normal image `I_n`, VLM-1 reasons over it to identify a plausible defect type and its
material context, and emits an anomaly prompt. A config file supplies the surrounding context:
what the dataset is, which defect types the category admits, and the hard generation constraints.
The prompt, the config and `I_n` go to the image-generation model, which returns the anomaly image
`I_a`.

The division of labour is deliberate. VLM-1 supplies the semantic specification of the anomaly.
The config supplies dataset-specific context and the constraints that keep `I_a` comparable with
`I_n`.

Because the generator repaints the entire frame rather than filling a hole, `I_a` can differ from
`I_n` in scale, offset and pixel correspondence. That is expected, and it is exactly what Module 2
is built to undo.

## Scope of the notebook

The notebook produces **prompts only**. It does not call an image generator, because the
generation model used for the reported results (ChatGPT) is not reachable from an offline Kaggle
kernel. The notebook samples normal images, writes one prompt per image, and stops. You take the
prompt and its normal image to the generator yourself and save the returned anomaly image for
Module 2.

## How the prompt is built

Three things go into every prompt, and only the first involves a model.

**1. The VLM reply.** VLM-1 is asked for four fields as JSON:

| field | example | used for |
|:--|:--|:--|
| `defect_name` | `shell crack` | the bank key and the filename |
| `target` | `the walnut left of centre` | which item in the frame carries it |
| `defect` | `crack splitting the shell open` | the defect itself, as a noun phrase |
| `where` | `near its upper seam` | position on that item |

They are read straight into one sentence, so they have to fit it grammatically. That constraint is
stated in the ask, which is why the model is told the sentence it is filling.

**2. The template.** Pure string substitution, no model involved, so one reply always yields one
prompt:

```
Generate a natural-looking anomaly image of this scene: {target} has a {size} {defect}, {where}.
The defect is part of the material, not laid on top, its edges, depth and shadow follow the
surface. Small, clearly visible, photorealistic.
```

The wording matters more than it looks. Two earlier framings failed in ways worth recording:

* *"Generate the same image ... Keep everything else unchanged."* The preservation clause
  dominated and the generator returned the input untouched.
* *"Generate the same image in which X shows a defect."* "The same image" frames the task as
  copy-then-overlay, and the generator composited a sprite on top of the picture: a rock lying on
  the rice, an insect sitting on the jelly. Nothing was damaged, something was pasted.

The current wording asks for an anomaly image rather than an edit, and then states outright that
the defect belongs to the material. Preservation is enforced separately, by the hard rules, where
it no longer competes with the defect instruction.

**3. The hard rules.** Appended verbatim to every prompt:

1. **Exact image preservation.** Geometry, composition, background, texture, lighting, shadows,
   reflections, perspective and pixel correspondence, everywhere except the requested defect.
2. **Photometric and dimensional invariance.** Colour, saturation, exposure, contrast,
   illumination, material appearance, dimensions, aspect ratio, crop, framing and resolution.
   No resizing, cropping, padding, zooming or global enhancement.
3. **Single localized defect.** Exactly one small, category-valid, physically realistic anomaly.
   No reconstruction, no secondary defects, no unrequested modification.

Rule 3 is what makes `ONE_DEFECT_PER_IMAGE` a safe assumption in Module 2: when several regions
come back from one donor, at most one of them is the defect.

## Diversity is enforced, not hoped for

Ask a VLM for a defect eight times and you get roughly the same defect eight times. So the defect
family is assigned **before** the model sees the image, round-robin from a shuffled list, and each
image in a category draws a different one. The size word is sampled per call, and a repeated
`defect_name` inside a category triggers a retry.

The family vocabulary follows MVTec AD 2 Table 3, the dataset's own description of occurring
defects: print defects, scratches, cuts, holes, colour inconsistency, loose threads and extra
fabric, foreign object contamination of varying texture and size, semi-transparent plastic
contamination, cracks, missing parts, broken pieces, damaged or missing QR codes.

The family hints name concrete industrial materials on purpose. Left open, a VLM proposes leaves
and twigs, which never reach a sealed inspection rig. What actually contaminates these lines is
glass, plastic, metal swarf, grit and fibre, plus insects on food and liquid products.

## Reply validation

The reply is checked with plain word lists rather than regexes, so they can be edited without
knowing regex syntax.

| list | rejects | reason |
|:--|:--|:--|
| `VAGUE_WORDS` | spot, mark, blemish, anomaly, imperfection, irregularity, defect, damage, flaw | These name a shape or a judgement, not a physical fault. A generator given "a small dark spot" has nothing to render. Given "a bored hole" it does. |
| `STAIN_ONLY_WORDS` | discolouration, patch, area, region | Vague in general, legitimate for the stain family, so they are allowed only there. |
| `NORMAL_WORDS` | bubble, reflection, highlight, glare, shadow, natural grain, pattern variation | Normal variation in this data. Bubbles in a vial, specular bands on dark-field metal and walnut shell grain are the trap MVTec AD 2 is built around. |

A rejected reply is retried, up to `TRIES = 4`. The first attempt is greedy (`TEMP_GREEDY = 0.0`)
so it reproduces; retries sample at `TEMP_SAMPLE = 0.8` so they actually diverge instead of
repeating the rejected answer.

## Per-category configuration

`CONFIG[cat]` carries four fields, all optional except `object`:

| field | purpose |
|:--|:--|
| `object` | one phrase naming what is in frame, used so the model refers to it correctly |
| `families` | which defect families this category admits |
| `defects` | seed vocabulary, binding only when `DEFECT_SOURCE = "config"` |
| `notes` | category hazards, always passed as a hint |

`notes` is the field that earns its keep. Telling the model that air bubbles are normal in a vial
stops it proposing bubbles as an anomaly, and the same for specular bands on sheet metal and shell
grain on walnuts. These are hints, not rules. The model still looks at the image.

`DEFECT_SOURCE` switches between `"config"`, where the model must pick from the listed defects,
and `"derive"`, where it may name its own and the list is only an example set.

To extend to a new dataset, add an entry to `CONFIG`, list it in `CATS`, and rewrite
`DATASET_CONTEXT`. Nothing else in the notebook names a dataset.

## Parameters

| name | value | what it controls |
|:--|:--|:--|
| `SEED` | 0 | image sampling, family shuffling |
| `N_IMAGES` | 3 | normal images sampled per category, one prompt each |
| `DEFECT_SOURCE` | `config` | whether the model must pick from the listed defects |
| `MAX_PIXELS` | 1280 x 28 x 28 | visual-token cap. AD 2 natives are far larger than the model needs |
| `MIN_PIXELS` | 256 x 28 x 28 | floor on the same |
| `MAX_NEW_TOKENS` | 200 | the JSON reply is short, this is headroom |
| `TRIES` | 4 | attempts per image before the fallback fires |
| `TEMP_GREEDY` / `TEMP_SAMPLE` | 0.0 / 0.8 | first attempt reproduces, retries diverge |
| `LOAD_4BIT` | False | fp16 across both T4s, which avoids a `bitsandbytes` install |

`sheet_metal` overrides `max_pixels` to 2560 x 28 x 28. The frame is a wide strip and the default
cap leaves too little resolution to see anything on it.

## Input and output

**Input.** MVTec AD 2 normal images, and `Qwen/Qwen2.5-VL-7B-Instruct` as VLM-1. Both attached as
Kaggle mounts, since the kernel runs with internet off.

**Output.**

```
<output dir>/
  <category>/Normal_<category>_<id>.png    the sampled normal images
  prompts.csv                              tag, family, defect, prompt, seconds
  manifest.json                            the same content as JSON
```

## Running it

1. Set `N_IMAGES`, the categories, and the output directory in section 2. If the mounts are named
   differently on your kernel, set `MODEL_ID` and `DATA_ROOT_ID` too.
2. Run all cells in order.
3. Read the prompts in section 9 before spending any generation budget. Discard any that describe
   a defect the image cannot support.
4. Pass each prompt with its normal image to the image generator and save the returned anomaly
   image for Module 2, as `<category>/<id>_anomaly.png` beside `<category>/<id>_regular.png`.

## What to check before moving on

* The prompts inside one category name different defects. If they collapse onto one, the family
  assignment is not reaching the model.
* No prompt describes something the image cannot support, for example a label defect on a frame
  where no label is visible.
* Every prompt reads as one sentence. A field that does not fit the sentence grammatically is a
  sign the model ignored the format instruction, and the generator will read it badly.

## Cost

Roughly 15 s per anomaly image on the generation side. This is the only generative cost in FLASH,
it is paid once per category, and it is the number that Module 3's timing is compared against.
