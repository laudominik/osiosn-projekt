"""
Etap 6 – Post-Training Quantization (PTQ) dynamiczna INT8.

Po wytrenowaniu modelu odniesienia (baseline) stosuje się dynamiczną
kwantyzację INT8 bez potrzeby ponownego treningu ani zestawu kalibracyjnego.
Warstwy Linear i Conv2d są kwantyzowane do INT8.

Wymaga wcześniejszego uruchomienia train_baseline.py.
"""
import glob
import torch
import os

from osiosn import WasteSortingModule, WasteSortingDataModule, test
from osiosn import save_results, profile

BATCH = 64
SEED  = 42

# find latest baseline checkpoint
ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
if not ckpts:
    raise FileNotFoundError("No baseline checkpoint found - run train_baseline.py first.")

ckpt_path = ckpts[-1]
print(f"Loading baseline from: {ckpt_path}")

model = WasteSortingModule.load_from_checkpoint(ckpt_path)
model.eval()

# ── Dynamic PTQ (quantize Linear and Conv2d layers to INT8) ──────────────────
print("Applying dynamic INT8 quantization...")
model_q = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8,
)

# ── Evaluate on test set ─────────────────────────────────────────────────────
dm = WasteSortingDataModule(batch_size=BATCH, seed=SEED)
dm.setup(stage="test")

import pytorch_lightning as pl
trainer = pl.Trainer(accelerator="cpu", devices=1)
results = trainer.test(model_q, datamodule=dm)
metrics = results[0] if results else {}

# ── Profile quantized model ──────────────────────────────────────────────────
tmp_path = "/tmp/model_ptq.pt"
torch.save(model_q.state_dict(), tmp_path)
size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
os.remove(tmp_path)
metrics["size_mb"] = size_mb
print(f"Quantized model size: {size_mb:.2f} MB")

save_results("quant_ptq_dynamic", metrics, config={
    "method": "ptq_dynamic", "dtype": "qint8",
    "quantized_layers": ["Linear", "Conv2d"],
    "based_on": ckpt_path,
})
