## 9. Agreement with manual labels

Optional. Compares the model's rejections against a list of regions you have rejected by hand.

Populate `HUMAN_DROP` with the keys you rejected, in the `<category>-<id>_<x>_<y>` format printed
in section 5. The cell prints a confusion table and, once enough labels exist, agreement, recall
on known artifacts and the false reject rate.

Below the minimum label count it prints counts only and refuses to quote a rate, because a single
label swings a recall by 100 points. The disagreement gallery is the place to start labelling.
