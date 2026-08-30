## 8. Timing

Reports how long the model takes per image. The clock starts once the encoded image reaches the
model, so disk read, preprocessing and tokenisation are excluded.

Model load is reported separately because it is paid once per session rather than per image.
