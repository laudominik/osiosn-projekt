import glob
from copy import deepcopy
from contextlib import contextmanager

import torch
import torch_pruning as tp
import pytorch_lightning as pl

from osiosn import WasteSortingModule, WasteSortingDataModule, profile
from osiosn import save_results
from osiosn import calculate_pruning_ratio

SPARSITY_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.80, 0.90]
BATCH = 64
SEED  = 42
EXAMPLE_INPUTS = torch.randn(1, 3, 32, 32)

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]


def _prune_channels(inner_model: torch.nn.Module, desired_sparsity: float) -> None:
    pruning_ratio = calculate_pruning_ratio(desired_sparsity)
    for p in inner_model.parameters():
        p.requires_grad_(True)
    last_linear = None
    for m in inner_model.modules():
        if isinstance(m, torch.nn.Linear):
            last_linear = m
    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        inner_model, EXAMPLE_INPUTS, importance=imp,
        pruning_ratio=pruning_ratio,
        iterative_steps=1, ignored_layers=[last_linear],
    )
    pruner.step()


@contextmanager
def ephemeral_structural_prune(lightning_module, sparsity: float):
    """
    Temporarily replace lightning_module.model with a structurally pruned
    deepcopy. Original model is restored on exit.
    """
    original_inner = lightning_module.model
    pruned_inner = deepcopy(original_inner).cpu()
    _prune_channels(pruned_inner, sparsity)
    lightning_module.model = pruned_inner
    try:
        yield lightning_module
    finally:
        lightning_module.model = original_inner


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
        exp_id = f"prune_ephemeral_{int(sparsity * 100):02d}_{noise_tag}"
        print(f"\n{'='*60}\n{exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)
        original_params = sum(p.numel() for p in model.model.parameters())

        with ephemeral_structural_prune(model, sparsity) as pruned_model:
            pruned_params = sum(p.numel() for p in pruned_model.model.parameters())
            print(f"  Pruned params : {original_params:,} → {pruned_params:,}  "
                  f"({pruned_params / original_params * 100:.1f}%)")

            pruned_model.eval()
            dm = WasteSortingDataModule(
                batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog, seed=SEED
            )
            trainer = pl.Trainer(
                accelerator="auto", devices=1, logger=False, enable_progress_bar=False
            )
            results = trainer.test(pruned_model, datamodule=dm, verbose=False)
            metrics = results[0] if results else {}

            metrics["sparsity_target"] = sparsity
            metrics["total_params"]    = pruned_params
            metrics["nonzero_params"]  = pruned_params
            metrics["sparsity"]        = 1.0 - pruned_params / original_params

            dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
            prof = profile(pruned_model, dm_prof)
            metrics.update(prof)

        # Verify restoration
        restored_params = sum(p.numel() for p in model.model.parameters())
        print(f"  Restored params: {restored_params:,} (oczekiwane {original_params:,})")
        assert restored_params == original_params, "Przywracanie parametrów nie powiodło się!"

        save_results(exp_id, metrics, config={
            "method": "ephemeral_structural_magnitude",
            "sparsity": sparsity,
            "noise_rate": noise_rate,
            "p_dog": p_dog,
            "retrain_epochs": 0,
            "ephemeral": True,
        })
