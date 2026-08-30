## 5. Load the model

Loads Qwen2.5-VL in fp16 with `device_map="auto"`, which splits the layers across both T4s. The
device map is printed so you can confirm the split.

`min_pixels` and `max_pixels` cap the visual token count. MVTec AD 2 images are around 2448x2048,
and at native resolution generation is very slow. Categories with extreme aspect ratios can
override the cap in their config entry.

If the processor fails to load, the cell falls through several routes and finally builds it from
its parts. This handles mounts that are missing `preprocessor_config.json`.
