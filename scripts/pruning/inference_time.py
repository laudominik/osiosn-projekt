import glob
import torch
import torch_pruning as tp
import pytorch_lightning as pl

from osiosn import WasteSortingModule, WasteSortingDataModule, profile
from osiosn import save_results

SPARSITY_LEVELS = [0.10, 0.15, 0.25, 0.30, 0.50, 0.7, 0.9]
BATCH = 64
SEED  = 42
EXAMPLE_INPUTS = torch.randn(1, 3, 32, 32)

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]


def prune_channels(inner_model: torch.nn.Module, pruning_ratio: float) -> None:
    """Physically remove channels; inner_model is modified in-place."""
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
    )
    pruner.step()

for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] Brak checkpointu baseline – pomijam.")
        continue
    baseline_ckpt = ckpts[-1]
    print(f"\n[{noise_tag}] Baseline: {baseline_ckpt}")

    for sparsity in SPARSITY_LEVELS:
        exp_id = f"prune_infer_{int(sparsity * 100):02d}_{noise_tag}"
        print(f"\n{'='*60}\n{exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)
        model.model.cpu().eval()

        before = sum(p.numel() for p in model.model.parameters())
        prune_channels(model.model, sparsity)
        after  = sum(p.numel() for p in model.model.parameters())
        print(f"  Params: {before:,} → {after:,}  ({after / before * 100:.1f}% remaining)")

        model.eval()
        dm = WasteSortingDataModule(
            batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog, seed=SEED
        )
        trainer = pl.Trainer(
            accelerator="auto", devices=1, logger=False, enable_progress_bar=False
        )
        results = trainer.test(model, datamodule=dm, verbose=False)
        metrics = results[0] if results else {}

        metrics["sparsity_target"] = sparsity
        metrics["total_params"]    = after
        metrics["nonzero_params"]  = after
        metrics["sparsity"]        = 1.0 - after / before

        dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
        prof = profile(model, dm_prof)
        metrics.update(prof)

        save_results(exp_id, metrics, config={
            "method": "inference_time_structural_magnitude",
            "sparsity": sparsity,
            "noise_rate": noise_rate,
            "p_dog": p_dog,
            "retrain_epochs": 0,
        })
