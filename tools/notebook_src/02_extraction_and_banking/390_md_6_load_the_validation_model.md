## 6. Load the validation model

Loads the same model as Module 1, in fp16 across both T4s.

If the processor fails to load, the cell falls through several routes and finally builds it from
its parts, then recovers the chat template. This handles mounts missing
`preprocessor_config.json`.
