## 1. Environment

Loads the libraries and prints the versions actually in use.

Weights and data come from the attached mounts, so nothing downloads and the kernel can run with
internet off. The model is loaded in fp16 rather than 4-bit, which avoids a `bitsandbytes`
install. T4 is Turing, so bf16 and flash-attention are unavailable and the dtype and attention
implementation are set explicitly.
