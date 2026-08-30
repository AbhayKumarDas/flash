## 2. Pair discovery

Finds the image pairs and reports what it found.

Expected layout, one directory per category:

```
<root>/<category>/<id>_regular.png     the normal image
<root>/<category>/<id>_anomaly.png     the generated anomaly
```

A pair is used only when both files exist. An anomaly with no matching normal is listed separately
so a failed generation is visible rather than silently absent.

The two images do not need to be aligned, the same size, or the same exposure. Registration is
handled in section 3. Do not resize them to match.

If the configured path is not found, the notebook searches for the directory holding the most
complete pairs and prints the one it chose. Check that line.
