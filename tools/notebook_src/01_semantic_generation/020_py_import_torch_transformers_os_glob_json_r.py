import torch, transformers, os, glob, json, re, time, random, shutil, csv

print("torch       ", torch.__version__)
print("transformers", transformers.__version__)

n_gpu = torch.cuda.device_count()
print("cuda        ", n_gpu, "x", torch.cuda.get_device_name(0) if n_gpu else "cpu")
if n_gpu:
    tot = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)) / 1e9
    print(f"vram         {tot:.0f} GB total across {n_gpu} device(s)")
# torch may report True on T4 (emulated), but Turing has no NATIVE bf16 -- emulated
# bf16 is slower than fp16 here, so the model is loaded in float16 regardless.
print("bf16 (may be emulated):", torch.cuda.is_bf16_supported() if n_gpu else False)
