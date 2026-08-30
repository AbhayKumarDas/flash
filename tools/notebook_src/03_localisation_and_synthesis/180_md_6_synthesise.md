## 6. Synthesise

Generates the corpus. Progress prints every ten images with an elapsed time and an estimate.

Hosts are shared across seeds, so host cost does not grow with the number of seeds. The seed varies
the placement blob, the bank entry and the rotation.

The loop walks the host pool until `N_SYNTH` images are made rather than taking the first
`N_SYNTH` hosts, because a host occasionally yields no usable placement.
