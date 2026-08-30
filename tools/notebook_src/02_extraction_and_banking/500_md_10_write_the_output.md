## 10. Write the output

Writes the two folders described at the top of this notebook, plus a rejection log beside them
recording why each discarded crop was dropped.

The bank holds accepted crops only. `manifest.csv` carries the geometry Module 3 needs, including
`defect_frac`, which cannot be recovered from the PNG and without which defects replay at the
wrong size.

Any pair whose mask came back empty is removed from the second folder at this point.
