"""
Etap 5 – Przerzedzanie strukturalne (LN Structured Channel Pruning).

W odróżnieniu od przerzedzania niestrukturalnego, przerzedzanie strukturalne
zeruje całe filtry (kanały) w warstwach konwolucyjnych na podstawie normy L1.
Daje to rzeczywiste zmniejszenie rozmiaru modelu i przyspieszenie wnioskowania,
ponieważ możliwe jest zrekonstruowanie mniejszej sieci.

Inicjalizacja od wytrenowanego checkpointu (baseline), nie od zera.
Poziomy rzadkości: 10 %, 30 %, 50 %, 70 %.
"""
import sys
import glob
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS = [0.10, 0.30, 0.50, 0.70]
EPOCHS = 10
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]


def make_structured_pruning_cb(amount: float):
    class SafeStructuredPruning(ModelPruning):
        def filter_parameters_to_prune(self, parameters_to_prune=None):
            params = super().filter_parameters_to_prune(parameters_to_prune)
            # skip BatchNorm / LayerNorm (1-D weight tensors)
            return [
                (module, name) for module, name in params
                if getattr(module, name).dim() > 1
            ]

    return SafeStructuredPruning(
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


for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] No baseline checkpoint found – run train_baseline.py first. Skipping.")
        continue
    baseline_ckpt = ckpts[-1]
    print(f"\n[{noise_tag}] Loading baseline from: {baseline_ckpt}")

    for sparsity in SPARSITY_LEVELS:
        exp_id = f"prune_struct_{int(sparsity*100):02d}_{noise_tag}"
        print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                       augmentation="basic", seed=SEED)
        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)

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
            "noise_rate": noise_rate, "p_dog": p_dog,
        })
