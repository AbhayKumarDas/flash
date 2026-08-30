## 5. Crop extraction

Turns each recovered region into a bank candidate: a square crop centred on the defect, at
`CTX_EXPAND` times its bounding box, so the crop carries its own surrounding substrate.

The substrate is required. Module 3 matches the crop's own background against the host's before
compositing, and a cut-out with no background cannot be matched. For the same reason the defect
mask is stored as a separate file rather than as an alpha channel.

The overlays below draw every candidate region on the frame it came from. Look at them. Anything
here that is not a defect has to be rejected by the gates in sections 7 and 8.
