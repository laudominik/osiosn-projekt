"""
Etap 5 – Przerzedzanie strukturalne (LN Structured Channel Pruning).

W odróżnieniu od przerzedzania niestrukturalnego, przerzedzanie strukturalne
zeruje całe filtry (kanały) w warstwach konwolucyjnych na podstawie normy L1.
Daje to rzeczywiste zmniejszenie rozmiaru modelu i przyspieszenie wnioskowania,
ponieważ możliwe jest zrekonstruowanie mniejszej sieci.

Poziomy rzadkości: 10 %, 30 %, 50 %, 70 %.
"""
import sys
import torch
import torch.nn.utils.prune as prune
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS = [0.10, 0.30, 0.50, 0.70]
EPOCHS = 10
BATCH  = 64
SEED   = 42

if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]


def make_structured_pruning_cb(amount: float):
    """
    ModelPruning callback using ln_structured on dim=0 (output channels / filters).
    Pruning is applied globally and made permanent.
    """
    return ModelPruning(
        pruning_fn="ln_structured",
        parameter_names=["weight"],
        use_global_unstructured=False,
        amount=amount,
        make_pruning_permanent=True,
        verbose=1,
        pruning_dim=0,
        pruning_norm=1,
        prune_on_train_epoch_end=True,
    )


for sparsity in SPARSITY_LEVELS:
    exp_id = f"prune_struct_{int(sparsity*100):02d}"
    print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

    dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=0.5, p_dog=0.1,
                                   augmentation="basic", seed=SEED)
    model = WasteSortingModule(learning_rate=1e-4)

    trainer, model = train(model, dm, ckpt_prefix=exp_id,
                           max_epochs=EPOCHS,
                           extra_callbacks=[make_structured_pruning_cb(sparsity)])

    metrics = extract_trainer_metrics(trainer)
    metrics["sparsity_target"] = sparsity

    total = sum(p.numel() for p in model.model.parameters())
    nz    = sum(p.count_nonzero().item() for p in model.model.parameters())
    metrics["total_params"]    = total
    metrics["nonzero_params"]  = nz
    metrics["sparsity_actual"] = 1.0 - nz / total

    save_results(exp_id, metrics, config={
        "epochs": EPOCHS, "batch_size": BATCH, "sparsity": sparsity,
        "pruning": "ln_structured", "dim": 0, "norm": 1,
    })
