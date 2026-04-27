"""Model profiling: parameters, size, inference time, training memory."""
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn


def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def profile(model_module, datamodule) -> dict:
    """
    Profile model_module.model and return a dict of key metrics.
    datamodule must have batch_size=1 for single-sample inference measurement.
    """
    if datamodule.batch_size != 1:
        raise ValueError("Set batch_size=1 for profiling so inference time is per-sample.")

    model = model_module.model
    results = {}

    # --- parameter count ---
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(p.nonzero().shape[0] for p in model.parameters())
    results["total_params"] = total_params
    results["nonzero_params"] = nonzero_params
    results["sparsity"] = 1.0 - nonzero_params / total_params

    # --- file size (FP32 weights only) ---
    tmp = "/tmp/_profile_model.pth"
    torch.save(model.state_dict(), tmp)
    results["size_mb"] = os.path.getsize(tmp) / (1024 ** 2)
    os.remove(tmp)

    print(f"\n{'─'*40}")
    print(f"  Total params   : {total_params:,}")
    print(f"  Non-zero params: {nonzero_params:,}  (sparsity={results['sparsity']:.1%})")
    print(f"  Checkpoint size: {results['size_mb']:.2f} MB")

    # --- prepare a single test batch ---
    datamodule.setup(stage="test")
    batch = next(iter(datamodule.test_dataloader()))
    x_cpu, _ = batch

    def _measure(device, x):
        m = deepcopy(model).to(device).eval()
        xi = x.to(device)
        with torch.no_grad():
            for _ in range(20):          # warm-up
                _ = m(xi)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = m(xi)
        return (time.perf_counter() - t0) / 100 * 1000  # ms

    # CPU inference
    cpu_ms = _measure(torch.device("cpu"), x_cpu)
    results["inference_cpu_ms"] = cpu_ms
    print(f"  Inference CPU  : {cpu_ms:.2f} ms/sample")

    # GPU inference (optional)
    accel = _best_device()
    if accel is not None and str(accel) != "cpu":
        gpu_ms = _measure(accel, x_cpu)
        results["inference_gpu_ms"] = gpu_ms
        print(f"  Inference {str(accel).upper():<4} : {gpu_ms:.2f} ms/sample")
    else:
        results["inference_gpu_ms"] = None
        print("  Inference GPU  : N/A (no CUDA/MPS)")

    # --- training memory (GPU only) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        m = deepcopy(model).cuda().train()
        opt = torch.optim.Adam(m.parameters())
        xb = x_cpu.cuda()
        yb = torch.zeros(xb.shape[0], dtype=torch.long).cuda()
        out = m(xb)
        nn.functional.cross_entropy(out, yb).backward()
        opt.step()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results["peak_memory_mb"] = peak_mb
        print(f"  Peak VRAM      : {peak_mb:.1f} MB")
    else:
        results["peak_memory_mb"] = None
        print("  Peak VRAM      : N/A")

    print(f"{'─'*40}\n")
    return results
