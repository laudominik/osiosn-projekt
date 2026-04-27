"""
Etap 6 – Post-Training Quantization (PTQ) dynamiczna INT8.

Po wytrenowaniu modelu odniesienia (baseline) stosuje się dynamiczną
kwantyzację INT8 bez potrzeby ponownego treningu ani zestawu kalibracyjnego.
Warstwy Linear są kwantyzowane do INT8 (Conv2d nie jest obsługiwane przez
torch.quantization.quantize_dynamic).

Wymaga wcześniejszego uruchomienia train_baseline.py.
"""
import glob
import os
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
        print(f"[{noise_tag}] No baseline checkpoint found – skipping.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"quant_ptq_dynamic_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    model = WasteSortingModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # Dynamic PTQ: quantize Linear layers to INT8
    # (quantize_dynamic does not support Conv2d – use static quant for that)
    model_q = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    # Evaluate on test set
    dm = WasteSortingDataModule(batch_size=BATCH, seed=SEED)
    dm.setup(stage="test")

    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False)
    results = trainer.test(model_q, datamodule=dm, verbose=False)
    metrics = results[0] if results else {}

    # Model size: save only the inner model weights
    tmp_path = "/tmp/model_ptq.pt"
    torch.save(model_q.model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)
    metrics["size_mb"] = size_mb

    # Inference time via profiler (batch_size=1)
    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    prof = profile(model_q, dm_prof)
    metrics["inference_cpu_ms"] = prof["inference_cpu_ms"]
    if prof.get("inference_gpu_ms"):
        metrics["inference_gpu_ms"] = prof["inference_gpu_ms"]

    print(f"Quantized model size: {size_mb:.2f} MB")

    save_results(exp_id, metrics, config={
        "method": "ptq_dynamic", "dtype": "qint8",
        "quantized_layers": ["Linear"],
        "based_on": ckpt_path,
        "noise_tag": noise_tag,
    })
