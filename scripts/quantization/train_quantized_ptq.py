
import glob
import os
import zipfile
import torch
import pytorch_lightning as pl

from osiosn import WasteSortingModule, WasteSortingDataModule, profile
from osiosn import save_results

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
        print(f"[{noise_tag}] Brak checkpointu baseline - pomijam.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"quant_ptq_dynamic_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    model = WasteSortingModule.load_from_checkpoint(ckpt_path)
    model.eval()

    torch.backends.quantized.engine = "fbgemm"
    model_q = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Evaluate on test set (CPU; quantized ops don't support CUDA)
    dm = WasteSortingDataModule(batch_size=BATCH, seed=SEED)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
    results = trainer.test(model_q, datamodule=dm, verbose=False)
    metrics = results[0] if results else {}

    tmp_path = "/tmp/model_ptq.pt"
    tmp_zip  = "/tmp/model_ptq.zip"
    torch.save(model_q.model.state_dict(), tmp_path)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, arcname="model.pth")
    metrics["size_mb"]     = os.path.getsize(tmp_path) / 1024 ** 2
    metrics["size_zip_mb"] = os.path.getsize(tmp_zip)  / 1024 ** 2
    os.remove(tmp_path)
    os.remove(tmp_zip)
    print(f"  Rozmiar: {metrics['size_mb']:.2f} MB  (zip: {metrics['size_zip_mb']:.2f} MB)")

    # --- inference time & params (profiler handles quantized-on-CPU gracefully) ---
    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    prof = profile(model_q, dm_prof)
    metrics["inference_cpu_ms"] = prof["inference_cpu_ms"]
    metrics["inference_gpu_ms"] = prof.get("inference_gpu_ms")
    metrics["total_params"]     = prof["total_params"]
    metrics["nonzero_params"]   = prof["nonzero_params"]
    metrics["sparsity"]         = prof["sparsity"]

    save_results(exp_id, metrics, config={
        "method": "ptq_dynamic",
        "dtype": "qint8",
        "quantized_layers": ["Linear"],
        "based_on": ckpt_path,
        "noise_tag": noise_tag,
    })
