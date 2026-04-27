import sys
import glob
from pytorch_lightning.callbacks import ModelPruning

from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics

SPARSITY_LEVELS=[0.50, 0.70, 0.8, 0.9, 0.95, 0.99]
EPOCHS = 50
PRUNE_EPOCHS = 30 
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

if len(sys.argv) > 1:
    SPARSITY_LEVELS = [float(sys.argv[1])]


def make_gradual_pruning_schedule(target_sparsity: float, prune_epochs: int):
    if target_sparsity >= 1.0:
        target_sparsity = 0.9999

    def schedule(epoch):
        if epoch >= prune_epochs:
            return 0.0
        
        s_prev = target_sparsity * (1.0 - (1.0 - (epoch) / prune_epochs)**3)
        s_curr = target_sparsity * (1.0 - (1.0 - (epoch + 1) / prune_epochs)**3)
        
        if s_prev >= 1.0:
            return 0.0
            
        amount_to_prune = (s_curr - s_prev) / (1.0 - s_prev)
        
        return amount_to_prune

    return schedule


for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    ckpts = sorted(glob.glob(f"checkpoints/baseline_{noise_tag}_waste*.ckpt"))
    if not ckpts:
        ckpts = sorted(glob.glob("checkpoints/baseline_waste*.ckpt"))
    if not ckpts:
        print(f"[{noise_tag}] No baseline checkpoint found - run train_baseline.py first. Skipping.")
        continue
    
    baseline_ckpt = ckpts[-1]
    print(f"\n[{noise_tag}] Loading baseline from: {baseline_ckpt}")

    for sparsity in SPARSITY_LEVELS:
        exp_id = f"prune_unstruct_s_{int(sparsity*100):02d}_{noise_tag}"
        print(f"\n{'='*60}\nRunning {exp_id}  (sparsity={sparsity:.0%})\n{'='*60}")

        pruning_cb = ModelPruning(
            pruning_fn="l1_unstructured",
            parameter_names=["weight"],
            use_global_unstructured=True,
            amount=make_gradual_pruning_schedule(sparsity, PRUNE_EPOCHS), 
            make_pruning_permanent=True,
            use_lottery_ticket_hypothesis=False,
            resample_parameters=False,
            verbose=1,
            prune_on_train_epoch_end=True,
        )

        dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                       augmentation="basic", seed=SEED)
        model = WasteSortingModule.load_from_checkpoint(baseline_ckpt)

        trainer, model, history = train(model, dm, ckpt_prefix=exp_id,
                                        max_epochs=EPOCHS, extra_callbacks=[pruning_cb])

        metrics = extract_trainer_metrics(trainer)
        metrics["sparsity_target"] = sparsity
        metrics["training_history"] = history

        dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
        prof = profile(model, dm_prof)
        metrics.update(prof)

        total = sum(p.numel() for p in model.model.parameters())
        nz    = sum(p.count_nonzero().item() for p in model.model.parameters())
        
        metrics["total_params"] = total
        metrics["nonzero_params"] = nz
        metrics["sparsity_actual"] = 1.0 - (nz / total)

        save_results(exp_id, metrics, config={
            "epochs": EPOCHS, "prune_epochs": PRUNE_EPOCHS, "batch_size": BATCH, 
            "sparsity": sparsity, "pruning": "l1_unstructured_gradual", 
            "global": True, "noise_rate": noise_rate, "p_dog": p_dog,
        })