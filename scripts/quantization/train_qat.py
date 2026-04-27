"""
Etap 6b – Quantization-Aware Training (QAT) INT8.
Wstawia operacje FakeQuantize do modelu przed doszkalaniem, dzięki czemu
sieć uczy się być odporna na szum kwantyzacji.
Po treningu model jest konwertowany do prawdziwego INT8 (fbgemm / qnnpack).
"""
import glob
import os
import zipfile
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback

from aim.pytorch_lightning import AimLogger

from osiosn import WasteSortingModule, WasteSortingDataModule, profile
from osiosn import save_results

EPOCHS = 20
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]


class HistoryCallback(Callback):
    def __init__(self):
        self.history = {"epochs": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.history["epochs"].append(trainer.current_epoch + 1)
        self.history["train_loss"].append(float(m.get("train_loss", 0)))
        self.history["train_acc"].append(float(m.get("train_acc", 0)))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        self.history["val_loss"].append(float(m.get("val_loss", 0)))
        self.history["val_acc"].append(float(m.get("val_acc", 0)))


def _prepare_qat(inner_model: nn.Module) -> bool:
    torch.backends.quantized.engine = "fbgemm"
    inner_model.train()
    inner_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")
    torch.ao.quantization.prepare_qat(inner_model, inplace=True)
    return True
  

for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline – pomijam.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"quant_qat_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    model = WasteSortingModule.load_from_checkpoint(ckpt_path)

    qat_ok = _prepare_qat(model.model)

    dm = WasteSortingDataModule(
        batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
        augmentation="basic", seed=SEED
    )
    dm.setup("fit")
    steps_per_epoch = len(dm.train_dataloader())

    history_cb = HistoryCallback()
    ckpt_cb = ModelCheckpoint(
        monitor="val_acc", dirpath="checkpoints/",
        filename=exp_id + "_waste-{epoch:02d}-{val_acc:.2f}",
        save_top_k=1, mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="step"), history_cb],
        deterministic=True,
        logger=AimLogger(experiment=exp_id),
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    model.model.eval()
    torch.ao.quantization.convert(model.model, inplace=True)
    
    trainer_test = pl.Trainer(
        accelerator="cpu", devices=1, logger=False, enable_progress_bar=False
    )
    test_results = trainer_test.test(model, datamodule=dm, verbose=False)
    metrics = test_results[0] if test_results else {}
    metrics["training_history"] = history_cb.history
    metrics["qat_applied"] = qat_ok

    tmp_path = "/tmp/model_qat.pt"
    tmp_zip  = "/tmp/model_qat.zip"
    torch.save(model.model.state_dict(), tmp_path)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, arcname="model.pth")
    metrics["size_mb"]     = os.path.getsize(tmp_path) / 1024 ** 2
    metrics["size_zip_mb"] = os.path.getsize(tmp_zip)  / 1024 ** 2
    os.remove(tmp_path)
    os.remove(tmp_zip)
    print(f"  Rozmiar: {metrics['size_mb']:.2f} MB  (zip: {metrics['size_zip_mb']:.2f} MB)")

    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    prof = profile(model, dm_prof)
    metrics["inference_cpu_ms"] = prof["inference_cpu_ms"]
    metrics["inference_gpu_ms"] = prof.get("inference_gpu_ms")
    metrics["total_params"]     = prof["total_params"]
    metrics["nonzero_params"]   = prof["nonzero_params"]
    metrics["sparsity"]         = prof["sparsity"]

    save_results(exp_id, metrics, config={
        "method": "qat",
        "dtype": "qint8",
        "epochs": EPOCHS,
        "based_on": ckpt_path,
        "noise_tag": noise_tag,
        "noise_rate": noise_rate,
        "p_dog": p_dog,
        "qat_applied": qat_ok,
    })
