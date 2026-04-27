"""
Etap 5 – Przerzedzanie niestrukturalne (L1 Unstructured Pruning).

Iteracyjne przerzedzanie wag metodą L1 podczas treningu z użyciem callbacku
ModelPruning z PyTorch Lightning.  Eksperyment powtarzany dla różnych poziomów
rzadkości (sparsity): 10 %, 30 %, 50 %, 70 %, 90 %.

Inicjalizacja od wytrenowanego checkpointu (baseline), nie od zera.
"""
import sys
import glob
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.90]
EPOCHS = 10
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

# allow overriding single sparsity from CLI: python train_pruning_unstructured.py 0.5
if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]

for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    # find matching baseline checkpoint
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        # fall back to any baseline checkpoint
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] No baseline checkpoint found – run train_baseline.py first. Skipping.")
        continue
    baseline_ckpt = ckpts[-1]
    print(f"\n[{noise_tag}] Loading baseline from: {baseline_ckpt}")

    for sparsity in SPARSITY_LEVELS:
        exp_id = f"prune_unstruct_{int(sparsity*100):02d}_{noise_tag}"
        print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        pruning_cb = ModelPruning(
            pruning_fn="l1_unstructured",
            parameter_names=["weight"],
            use_global_unstructured=True,
            amount=sparsity,
            make_pruning_permanent=True,
            use_lottery_ticket_hypothesis=False,
            resample_parameters=False,
            verbose=1,
            prune_on_train_epoch_end=True,
        )

        dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                       augmentation="basic", seed=SEED)
        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)

        trainer, model = train(model, dm, ckpt_prefix=exp_id,
                               max_epochs=EPOCHS, extra_callbacks=[pruning_cb])

        metrics = extract_trainer_metrics(trainer)
        metrics["sparsity_target"] = sparsity

        total = sum(p.numel() for p in model.model.parameters())
        nz    = sum(p.count_nonzero().item() for p in model.model.parameters())
        metrics["total_params"]   = total
        metrics["nonzero_params"] = nz
        metrics["sparsity_actual"] = 1.0 - nz / total

        save_results(exp_id, metrics, config={
            "epochs": EPOCHS, "batch_size": BATCH, "sparsity": sparsity,
            "pruning": "l1_unstructured", "global": True,
            "noise_rate": noise_rate, "p_dog": p_dog,
        })
