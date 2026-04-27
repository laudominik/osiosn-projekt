"""
Etap 6b – Quantization-Aware Training (QAT) INT8 za pomocą FX Graph Mode.
Wykorzystuje FX do automatycznego śledzenia i kwantyzacji całej sieci.
Po treningu konwertuje statycznie CAŁY model (Conv2d + Linear) do INT8.
"""
import glob
import os
import time
import zipfile
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch.ao.quantization.quantize_fx as quant_fx
from torch.ao.quantization import get_default_qat_qconfig

from osiosn import WasteSortingModule, WasteSortingDataModule
from osiosn import save_results

EPOCHS = 1
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


def _prepare_qat_fx(pl_module: pl.LightningModule) -> bool:
    torch.backends.quantized.engine = "fbgemm"
    
    qconfig_dict = {"": get_default_qat_qconfig("fbgemm")}
    
    example_inputs = (torch.randn(1, 3, 32, 32),)
    
    pl_module.model.train()
    
    pl_module.model = quant_fx.prepare_qat_fx(
        pl_module.model, 
        qconfig_dict, 
        example_inputs
    )
    
    return True


for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline – pomijam.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"quant_qat_fx_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")

    model = WasteSortingModule.load_from_checkpoint(ckpt_path)
    
    orig_params = sum(p.numel() for p in model.model.parameters())

    qat_ok = _prepare_qat_fx(model)

    dm = WasteSortingDataModule(
        batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
        augmentation="basic", seed=SEED
    )
    dm.setup("fit")

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
        callbacks=[ckpt_cb, history_cb],
        deterministic=False,
        logger=False,
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)

    torch.use_deterministic_algorithms(False)

    torch.backends.quantized.engine = "fbgemm"
    model.model.cpu().eval()
    
    inner_q = quant_fx.convert_fx(model.model)
    model.model = inner_q

    # --- test accuracy ---
    trainer_test = pl.Trainer(
        accelerator="cpu", 
        devices=1, 
        logger=False, 
        enable_progress_bar=False,
        deterministic=False # <--- FIX: Keep it off here too
    )
    test_results = trainer_test.test(model, datamodule=dm, verbose=False)
    metrics = test_results[0] if test_results else {}
    metrics["training_history"] = history_cb.history
    metrics["qat_applied"] = qat_ok

    # --- model size ---
    tmp_path = "/tmp/model_qat.pt"
    tmp_zip  = "/tmp/model_qat.zip"
    torch.save(inner_q.state_dict(), tmp_path)
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_path, arcname="model.pth")
    metrics["size_mb"]     = os.path.getsize(tmp_path) / 1024 ** 2
    metrics["size_zip_mb"] = os.path.getsize(tmp_zip)  / 1024 ** 2
    os.remove(tmp_path)
    os.remove(tmp_zip)
    print(f"  Rozmiar: {metrics['size_mb']:.2f} MB  (zip: {metrics['size_zip_mb']:.2f} MB)")

    # --- inference time ---
    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    dm_prof.setup(stage="test")
    x_sample, _ = next(iter(dm_prof.test_dataloader()))
    x_sample = x_sample.cpu()
    inner_q.eval()
    
    with torch.no_grad():
        for _ in range(20): inner_q(x_sample)   # warm-up
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100): inner_q(x_sample)
    metrics["inference_cpu_ms"] = (time.perf_counter() - t0) / 100 * 1000
    metrics["inference_gpu_ms"] = None
    print(f"  Inference CPU  : {metrics['inference_cpu_ms']:.2f} ms/sample")

    # --- param counts ---
    metrics["total_params"]   = orig_params
    metrics["nonzero_params"] = orig_params
    metrics["sparsity"]       = 0.0

    save_results(exp_id, metrics, config={
        "method": "qat_static_fx",
        "dtype": "qint8",
        "epochs": EPOCHS,
        "based_on": ckpt_path,
        "noise_tag": noise_tag,
        "noise_rate": noise_rate,
        "p_dog": p_dog,
        "qat_applied": qat_ok,
    })