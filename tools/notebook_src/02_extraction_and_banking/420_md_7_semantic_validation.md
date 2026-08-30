## 7. Semantic validation

Each candidate crop is shown to the model, which decides whether the outlined region is a genuine
defect or an artifact of extraction.

The model sees three panels: the same view before the edit, after it, and with the candidate
region outlined. It is asked what changed inside the outline first, and only then whether that
change is a defect. Comparing against a before image is what makes the question answerable; from a
single image, normal texture and a real defect look alike.

It returns JSON: whether anything changed, what changed, whether it is a defect, a name for it, a
confidence, a reason, and an artifact class. The verdict gates the crop; the name becomes the
`defect_type` column; the artifact class and reason explain rejections.

When several regions come from one image, the model is told the count and the rank, because
Module 1 introduces exactly one defect per image and the rest are extraction artifacts.

Set `VLM_MIN_CONF` and `USE_VLM_CONFIDENCE` in section 1 if you want confidence to gate as well as
the verdict.
