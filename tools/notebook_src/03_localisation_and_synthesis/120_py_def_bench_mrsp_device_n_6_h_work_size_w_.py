def _bench_mrsp(device, n=6, h=WORK_SIZE, w=WORK_SIZE):
    try:
        generate_mrsp(h, w, scale=AD2_NOISE_SCALE, device=device, seed=0,
                      alpha=MRSP_ALPHA)                     # warmup / autotune
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n):
            generate_mrsp(h, w, scale=AD2_NOISE_SCALE, device=device,
                          seed=i, alpha=MRSP_ALPHA).cpu()
        if device != "cpu":
            torch.cuda.synchronize()
        return (time.time() - t0) / n
    except Exception as e:
        print(f"  {device}: unavailable ({type(e).__name__})")
        return float("inf")

# Synthesis runs SERIALLY, exactly as flash_part1 does it. An earlier version fanned out over
# ProcessPoolExecutor and deadlocked: the pool forks, a CUDA context does not survive fork(),
# and every worker hung re-initialising CUDA on its first job. Serial removes that whole class
# of bug and lets MRSP use the GPU safely, which is the better trade at this corpus size.
_cpu = _bench_mrsp("cpu")
print(f"  cpu   {_cpu*1000:7.1f} ms/field")
if torch.cuda.is_available():
    _gpu = _bench_mrsp("cuda:0")
    print(f"  cuda  {_gpu*1000:7.1f} ms/field")
    MRSP_DEVICE = "cuda:0" if _gpu < _cpu else "cpu"
    print(f"\nMRSP -> {MRSP_DEVICE} ({max(_cpu,_gpu)/max(min(_cpu,_gpu),1e-9):.1f}x faster)")
else:
    MRSP_DEVICE = "cpu"
    print("\nMRSP -> cpu (no cuda)")

print("compositing -> serial; cv2.seamlessClone has no GPU path and forking a CUDA "
      "context deadlocks")
