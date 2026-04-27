"""
Etap 6 – Połączenie przerzedzania niestrukturalnego i QAT (Quantization-Aware Training).

Stosuje jednocześnie:
  1. L1 Unstructured pruning (50 % rzadkości) – usuwanie wag o małej normie L1
  2. QAT – kwantyzacja INT8 świadoma podczas treningu (lepsza od PTQ dla małych zbiorów)

Wynik to skompresowany model spełniający oba typy optymalizacji równocześnie.
"""
from pytorch_lightning.callbacks import ModelPruning, QuantizationAwareTraining

from osiosn import WasteSortingModule, WasteSortingDataModule, train
from osiosn import save_results, extract_trainer_metrics

SPARSITY = 0.50
EPOCHS   = 10
BATCH    = 64
SEED     = 42

pruning_cb = ModelPruning(
    pruning_fn="l1_unstructured",
    parameter_names=["weight"],
    use_global_unstructured=True,
    amount=SPARSITY,
    make_pruning_permanent=True,
    verbose=1,
    prune_on_train_epoch_end=True,
)

qat_cb = QuantizationAwareTraining(
    observer_type="histogram",
    quant_compatible=True,
)

dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=0.5, p_dog=0.1,
                               augmentation="basic", seed=SEED)
model = WasteSortingModule(learning_rate=1e-4)

trainer, model = train(model, dm, ckpt_prefix="prune50_qat",
                       max_epochs=EPOCHS,
                       extra_callbacks=[pruning_cb, qat_cb])  # both applied

metrics = extract_trainer_metrics(trainer)
metrics["sparsity_target"] = SPARSITY

total = sum(p.numel() for p in model.model.parameters())
nz    = sum(p.count_nonzero().item() for p in model.model.parameters())
metrics["total_params"]    = total
metrics["nonzero_params"]  = nz
metrics["sparsity_actual"] = 1.0 - nz / total

save_results("prune50_qat", metrics, config={
    "epochs": EPOCHS, "batch_size": BATCH,
    "pruning": "l1_unstructured", "sparsity": SPARSITY,
    "quantization": "qat_int8",
})
