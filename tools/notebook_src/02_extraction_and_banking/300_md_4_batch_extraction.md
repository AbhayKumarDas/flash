## 4. Batch extraction

Recovers a mask for every pair and writes it beside the anomaly frame it came from:

```
Defect Masks and Anomalies/<category>/<id>_mask.png
Defect Masks and Anomalies/<category>/<id>_anomaly.png
```

Each pair prints its runtime and the recovered mask size. Watch for two markers. `EMPTY` means
nothing was recovered from that pair, which is a result rather than an error. `FAIL` means the
call raised; the batch continues so one bad pair does not cost the rest.

Debug intermediates are written to `_work/`. One of them, the registered normal image, is used in
section 7, so this is not optional.
