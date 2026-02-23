import os
import time
import zipfile
from copy import deepcopy

import torch
import torch.nn as nn


def _best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def _safe_nonzero(p: torch.Tensor) -> int:
    try:
        return p.nonzero().shape[0]
    except Exception:
        return p.numel()


def profile(model_module, datamodule) -> dict:
    if datamodule.batch_size != 1:
        raise ValueError(
            "Set batch_size=1 for profiling so inference time is per-sample."
        )

    model = model_module.model
    results = {}

    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(_safe_nonzero(p) for p in model.parameters())
    results["total_params"] = total_params
    results["nonzero_params"] = nonzero_params
    results["sparsity"] = 1.0 - nonzero_params / total_params if total_params else 0.0

    tmp = "/tmp/_profile_model.pth"
    tmp_zip = "/tmp/_profile_model.zip"
    try:
        torch.save(model.state_dict(), tmp)
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(tmp, arcname="model.pth")
        results["size_mb"] = os.path.getsize(tmp) / 1024**2
        results["size_zip_mb"] = os.path.getsize(tmp_zip) / 1024**2
        print(
            f"  Checkpoint size: {results['size_mb']:.2f} MB "
            f"(Zipped: {results['size_zip_mb']:.2f} MB)"
        )
    except Exception as e:
        results["size_mb"] = None
        results["size_zip_mb"] = None
        print(f"  Checkpoint size: N/A ({e})")
    finally:
        for p in (tmp, tmp_zip):
            try:
                os.remove(p)
            except OSError:
                pass

    datamodule.setup(stage="test")
    batch = next(iter(datamodule.test_dataloader()))
    x_cpu, _ = batch

    def _measure(device, x):
        m = None
        try:
            m = deepcopy(model).to(device).eval()
        except Exception:
            if str(device) != "cpu":
                return None
            m = model
            m.eval()
        xi = x.to(device)
        try:
            with torch.no_grad():
                for _ in range(20):
                    _ = m(xi)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(100):
                    _ = m(xi)
            return (time.perf_counter() - t0) / 100 * 1000  # ms
        except Exception:
            return None

    cpu_ms = _measure(torch.device("cpu"), x_cpu)
    results["inference_cpu_ms"] = cpu_ms
    if cpu_ms is not None:
        print(f"  Inference CPU  : {cpu_ms:.2f} ms/sample")
    else:
        print("  Inference CPU  : N/A (error during measurement)")

    accel = _best_device()
    if accel is not None and str(accel) != "cpu":
        gpu_ms = _measure(accel, x_cpu)
        if gpu_ms is not None:
            results["inference_gpu_ms"] = gpu_ms
            print(f"  Inference {str(accel).upper():<4} : {gpu_ms:.2f} ms/sample")
        else:
            results["inference_gpu_ms"] = None
            print(
                f"  Inference {str(accel).upper():<4} : N/A (not available for this model)"
            )
    else:
        results["inference_gpu_ms"] = None
        print("  Inference GPU  : N/A (no CUDA/MPS)")

    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            m = deepcopy(model).cuda().train()
            opt = torch.optim.Adam(m.parameters())
            xb = x_cpu.cuda()
            yb = torch.zeros(xb.shape[0], dtype=torch.long).cuda()
            out = m(xb)
            nn.functional.cross_entropy(out, yb).backward()
            opt.step()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            results["peak_memory_mb"] = peak_mb
            print(f"  Peak VRAM      : {peak_mb:.1f} MB")
        except Exception:
            results["peak_memory_mb"] = None
            print("  Peak VRAM      : N/A (quantized model skipped)")
    else:
        results["peak_memory_mb"] = None
        print("  Peak VRAM      : N/A")

    print(f"{'─' * 40}\n")
    return results
