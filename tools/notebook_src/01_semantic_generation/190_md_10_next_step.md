## 10. Next step

Each normal image now has one anomaly prompt in `prompts.csv` and `manifest.json`.

Pass each prompt with its normal image to an image generation model and save the returned anomaly
image alongside the normal one, named so the two pair up:

```
<category>/<id>_regular.png     the normal image
<category>/<id>_anomaly.png     the generated anomaly
```

That layout is what Module 2 expects. Generation currently costs roughly 50 to 60 seconds per
image when done by hand; an API call is faster and records the model version against each image.

This cost is paid once per category. Module 3 replays the banked defects onto new hosts without
calling a generator again.
