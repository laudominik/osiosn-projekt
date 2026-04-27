import sys
import glob
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS = [0.10, 0.15, 0.20, 0.30, 0.50]
EPOCHS = 50
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]


def make_structured_pruning_cb(target_sparsity: float):
    class SafeStructuredPruning(ModelPruning):
        def filter_parameters_to_prune(self, parameters_to_prune=None):
            params = super().filter_parameters_to_prune(parameters_to_prune)
            # Pomijamy BatchNorm / LayerNorm (tensory wag 1-wymiarowe)
            return [
                (module, name) for module, name in params
                if getattr(module, name).dim() > 1
            ]

    return SafeStructuredPruning(
        pruning_fn="ln_structured",
        parameter_names=["weight"],
        use_global_unstructured=False,
        # Kluczowa zmiana: tnie mocno RAZ w epoce 0, potem tylko doucza (0.0)
        amount=lambda epoch: target_sparsity if epoch == 0 else 0.0,
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
        exp_id = f"prune_struct_o_{int(sparsity*100):02d}_{noise_tag}"
        print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                       augmentation="basic", seed=SEED)
        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)

        # Odbieramy history, żeby generatory wykresów z LaTeX-a działały poprawnie!
        trainer, model, history = train(model, dm, ckpt_prefix=exp_id,
                                        max_epochs=EPOCHS,
                                        extra_callbacks=[make_structured_pruning_cb(sparsity)])

        metrics = extract_trainer_metrics(trainer)
        metrics["sparsity_target"] = sparsity
        metrics["training_history"] = history

        # Profilowanie - to tutaj zobaczymy zyski w czasie wnioskowania CPU!
        dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
       
        prof = profile(model, dm_prof)
        metrics.update(prof)



        save_results(exp_id, metrics, config={
            "epochs": EPOCHS, "batch_size": BATCH, "sparsity": sparsity,
            "pruning": "ln_structured_oneshot", "dim": 0, "norm": 1,
            "noise_rate": noise_rate, "p_dog": p_dog,
        })