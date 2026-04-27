"""
Etap 6 – Post-Training Quantization (PTQ) dynamiczna INT8.
Kwantyzuje warstwy Linear do INT8 bez ponownego treningu.
Conv2d nie jest obsługiwane przez quantize_dynamic.
"""
import glob
import os
import time
import zipfile
import warnings
import torch
import pytorch_lightning as pl

from osiosn import WasteSortingModule, WasteSortingDataModule
from osiosn import save_results

warnings.filterwarnings("ignore", category=DeprecationWarning)

BATCH = 64
SEED  = 42

NOISE_VARIANTS = [
    ("noisy",),
    ("clean",),
]

for (noise_tag,) in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline – pomijam.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"quant_ptq_dynamic_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    model = WasteSortingModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # Record original (float) param counts before quantization
    orig_params = sum(p.numel() for p in model.model.parameters())

    # Apply dynamic PTQ to the inner model only (not the Lightning wrapper).
    # quantize_dynamic on the whole LightningModule causes packed-param errors
    # because the wrapper's forward calls into the quantized inner model with
    # a packed-param type mismatch.
    torch.backends.quantized.engine = "fbgemm"
    inner_q = torch.ao.quantization.quantize_dynamic(
        model.model.cpu(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    model.model = inner_q  # swap the quantized inner model back in

    # --- test accuracy ---
    dm = WasteSortingDataModule(batch_size=BATCH, seed=SEED)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
    results = trainer.test(model, datamodule=dm, verbose=False)
    metrics = results[0] if results else {}

    # --- model size ---
    tmp_path = "/tmp/model_ptq.pt"
    tmp_zip  = "/tmp/model_ptq.zip"
    torch.save(inner_q.state_dict(), tmp_path)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, arcname="model.pth")
    metrics["size_mb"]     = os.path.getsize(tmp_path) / 1024 ** 2
    metrics["size_zip_mb"] = os.path.getsize(tmp_zip)  / 1024 ** 2
    os.remove(tmp_path)
    os.remove(tmp_zip)
    print(f"  Rozmiar: {metrics['size_mb']:.2f} MB  (zip: {metrics['size_zip_mb']:.2f} MB)")

    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    dm_prof.setup(stage="test")
    x_sample, _ = next(iter(dm_prof.test_dataloader()))
    x_sample = x_sample.cpu()
    inner_q.eval()
    with torch.no_grad():
        for _ in range(20): inner_q(x_sample)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100): inner_q(x_sample)
    metrics["inference_cpu_ms"] = (time.perf_counter() - t0) / 100 * 1000
    metrics["inference_gpu_ms"] = None   # dynamic quant is CPU-only
    print(f"  Inference CPU  : {metrics['inference_cpu_ms']:.2f} ms/sample")

    metrics["total_params"]   = orig_params
    metrics["nonzero_params"] = orig_params
    metrics["sparsity"]       = 0.0

    save_results(exp_id, metrics, config={
        "method": "ptq_dynamic",
        "dtype": "qint8",
        "quantized_layers": ["Linear"],
        "based_on": ckpt_path,
        "noise_tag": noise_tag,
    })
