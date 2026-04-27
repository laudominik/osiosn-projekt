"""
Etap 6 – Połączenie przerzedzania niestrukturalnego i PTQ INT8.

Stosuje jednocześnie:
  1. L1 Unstructured pruning (50 % rzadkości) podczas fine-tuningu
  2. Dynamic PTQ INT8 po zakończeniu treningu

QuantizationAwareTraining callback został usunięty w PL 2.x; zamiast QAT
stosuje się dynamiczną kwantyzację po treningu (PTQ), co daje porównywalny
efekt kompresji przy mniejszej złożoności.
"""
import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics

SPARSITY = 0.50
EPOCHS   = 10
BATCH    = 64
SEED     = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] No baseline checkpoint found – skipping.")
        continue
    baseline_ckpt = ckpts[-1]
    exp_id = f"prune50_quant_{noise_tag}"
    print(f"\n{'='*60}\n{exp_id}\nLoading: {baseline_ckpt}\n{'='*60}")

    pruning_cb = ModelPruning(
        pruning_fn="l1_unstructured",
        parameter_names=["weight"],
        use_global_unstructured=True,
        amount=SPARSITY,
        make_pruning_permanent=True,
        verbose=1,
        prune_on_train_epoch_end=True,
    )

    dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                   augmentation="basic", seed=SEED)
    model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)

    trainer, model = train(model, dm, ckpt_prefix=exp_id,
                           max_epochs=EPOCHS, extra_callbacks=[pruning_cb])

    metrics = extract_trainer_metrics(trainer)
    metrics["sparsity_target"] = SPARSITY

    total = sum(p.numel() for p in model.model.parameters())
    nz    = sum(p.count_nonzero().item() for p in model.model.parameters())
    metrics["total_params"]    = total
    metrics["nonzero_params"]  = nz
    metrics["sparsity_actual"] = 1.0 - nz / total

    # Apply dynamic PTQ after pruned training
    model.eval()
    model_q = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8,
    )

    # Test the quantized pruned model
    dm_test = WasteSortingDataModule(batch_size=BATCH, seed=SEED)
    dm_test.setup(stage="test")
    tester = pl.Trainer(accelerator="cpu", devices=1, logger=False)
    q_results = tester.test(model_q, datamodule=dm_test, verbose=False)
    if q_results:
        metrics.update({f"quant_{k}": v for k, v in q_results[0].items()})

    # Model size (quantized + pruned)
    tmp_path = "/tmp/model_prune_quant.pt"
    torch.save(model_q.model.state_dict(), tmp_path)
    metrics["size_mb"] = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)

    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    prof = profile(model_q, dm_prof)
    metrics["inference_cpu_ms"] = prof["inference_cpu_ms"]

    save_results(exp_id, metrics, config={
        "epochs": EPOCHS, "batch_size": BATCH,
        "pruning": "l1_unstructured", "sparsity": SPARSITY,
        "quantization": "ptq_dynamic_int8",
        "noise_rate": noise_rate, "p_dog": p_dog,
    })
