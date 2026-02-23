import glob
import os
import time
import zipfile
import torch
import torch_pruning as tp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch.ao.quantization.quantize_fx as quant_fx
from torch.ao.quantization import get_default_qat_qconfig

from osiosn import WasteSortingModule, WasteSortingDataModule
from osiosn import save_results

PRUNING_RATIO  = 0.15
EPOCHS         = 2
BATCH          = 64
SEED           = 42
EXAMPLE_INPUTS = torch.randn(1, 3, 32, 32)

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]


class HistoryCallback(Callback):
    def __init__(self):
        self.history = {
            "epochs": [], "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

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


def prune_channels(inner_model: torch.nn.Module, pruning_ratio: float) -> int:
    for p in inner_model.parameters():
        p.requires_grad_(True)

    last_linear = None
    for m in inner_model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        inner_model,
        EXAMPLE_INPUTS,
        importance=imp,
        pruning_ratio=pruning_ratio,
        iterative_steps=1,
        ignored_layers=[last_linear],
        round_to=8
    )
    pruner.step()
    return sum(p.numel() for p in inner_model.parameters())


for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline - pomijam.")
        continue
    ckpt_path = ckpts[-1]
    exp_id = f"final_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {ckpt_path}\n{'='*60}")

    pl.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("medium")
    torch.use_deterministic_algorithms(False)

    dm = WasteSortingDataModule(
        batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
        augmentation="standard", seed=SEED,
    )
    dm.setup("fit")
    steps_per_epoch = len(dm.train_dataloader())

    model = WasteSortingModule.load_from_checkpoint(
        ckpt_path,
        learning_rate=1e-4,
        weight_decay=1e-4,
        optimizer_type="adamw",
        scheduler_type="cosine",
        steps_per_epoch=steps_per_epoch,
        max_epochs=EPOCHS,
    )

    orig_params = sum(p.numel() for p in model.model.parameters())
    model.model.cpu().eval()
    pruned_params = prune_channels(model.model, PRUNING_RATIO)
    actual_sparsity = 1.0 - pruned_params / orig_params
    print(f"  Params: {orig_params:,} → {pruned_params:,} "
          f"({pruned_params/orig_params*100:.1f}% pozostało, "
          f"sparsity={actual_sparsity*100:.1f}%)")

    torch.backends.quantized.engine = "fbgemm"
    qconfig_dict = {"": get_default_qat_qconfig("fbgemm")}
    model.model.train()
    model.model = quant_fx.prepare_qat_fx(
        model.model, qconfig_dict, (EXAMPLE_INPUTS,)
    )

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
    print("  Konwersja do INT8 zakończona.")

    trainer_test = pl.Trainer(
        accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, deterministic=False,
    )
    test_results = trainer_test.test(model, datamodule=dm, verbose=False)
    metrics = test_results[0] if test_results else {}
    metrics["training_history"] = history_cb.history

    tmp_path = "/tmp/model_final.pt"
    tmp_zip  = "/tmp/model_final.zip"
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
        for _ in range(20):
            inner_q(x_sample)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            inner_q(x_sample)
    metrics["inference_cpu_ms"] = (time.perf_counter() - t0) / 100 * 1000
    metrics["inference_gpu_ms"] = None
    print(f"  Inference CPU  : {metrics['inference_cpu_ms']:.2f} ms/sample")

    metrics["total_params"]    = pruned_params
    metrics["nonzero_params"]  = pruned_params
    metrics["sparsity"]        = actual_sparsity
    metrics["sparsity_target"] = PRUNING_RATIO

    save_results(exp_id, metrics, config={
        "method": "struct_pruning_15pct + qat_fx_int8 + optimal_hparams",
        "pruning_ratio":   PRUNING_RATIO,
        "actual_sparsity": actual_sparsity,
        "qat_epochs":      EPOCHS,
        "augmentation":    "standard",
        "optimizer":       "adamw",
        "weight_decay":    1e-4,
        "scheduler":       "cosine",
        "dropout_rate":    0.1,
        "based_on":        ckpt_path,
        "noise_tag":       noise_tag,
        "noise_rate":      noise_rate,
        "p_dog":           p_dog,
    })
