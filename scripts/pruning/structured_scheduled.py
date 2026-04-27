import sys
import glob
import torch
import torch_pruning as tp

from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS = [0.10, 0.15, 0.20, 0.30, 0.50]
ITERATIVE_STEPS = 5
EPOCHS = 10
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

EXAMPLE_INPUTS = torch.randn(1, 3, 32, 32)

if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]


def prune_channels_iterative(inner_model: torch.nn.Module,
                              pruning_ratio: float,
                              iterative_steps: int) -> None:
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
        global_pruning=False,
        pruning_ratio=pruning_ratio,
        iterative_steps=iterative_steps,
        ignored_layers=[last_linear],
    )
    for _ in range(iterative_steps):
        pruner.step()


for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline – run train_baseline.py first. Skipping.")
        continue
    baseline_ckpt = ckpts[-1]
    print(f"\n[{noise_tag}] Loading baseline from: {baseline_ckpt}")

    for sparsity in SPARSITY_LEVELS:
        exp_id = f"prune_struct_s_{int(sparsity * 100):02d}_{noise_tag}"
        print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%}, "
              f"steps={ITERATIVE_STEPS})\n{'='*60}")

        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt, lr=1e-4)
        model.model.cpu().eval()

        before_params = sum(p.numel() for p in model.model.parameters())
        prune_channels_iterative(model.model, sparsity, ITERATIVE_STEPS)
        after_params  = sum(p.numel() for p in model.model.parameters())
        print(f"  Params: {before_params:,} → {after_params:,}  "
              f"({after_params / before_params * 100:.1f}% remaining)")

        dm = WasteSortingDataModule(
            batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
            augmentation="basic", seed=SEED
        )
        trainer, model, history = train(model, dm, ckpt_prefix=exp_id, max_epochs=EPOCHS)

        metrics = extract_trainer_metrics(trainer)
        metrics["sparsity_target"] = sparsity
        metrics["training_history"] = history

        dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
        prof = profile(model, dm_prof)
        metrics.update(prof)

        save_results(exp_id, metrics, config={
            "epochs": EPOCHS, "batch_size": BATCH, "sparsity": sparsity,
            "iterative_steps": ITERATIVE_STEPS,
            "pruning": "torch_pruning_magnitude_structured_iterative", "global": True,
            "noise_rate": noise_rate, "p_dog": p_dog,
        })
