import sys
import math
from osiosn import WasteSortingModule, WasteSortingDataModule, train
from osiosn import save_results, extract_trainer_metrics

EPOCHS = 10
BATCH  = 64
SEED   = 42


AUGMENTATION_EXPERIMENTS = [
    dict(id="aug_none",       augmentation="none",       optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="aug_basic",      augmentation="basic",      optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="aug_standard",   augmentation="standard",   optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="aug_aggressive", augmentation="aggressive",  optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
]

OPTIMIZER_EXPERIMENTS = [
    dict(id="opt_adam",       augmentation="standard", optimizer_type="adam",  weight_decay=0.0,  scheduler_type="cosine", dropout_rate=0.0),
    dict(id="opt_adamw_1e4",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="opt_adamw_1e3",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-3, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="opt_sgd",        augmentation="standard", optimizer_type="sgd",   weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
]

SCHEDULER_EXPERIMENTS = [
    dict(id="sched_cosine",   augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine",   dropout_rate=0.0),
    dict(id="sched_plateau",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="plateau",  dropout_rate=0.0),
    dict(id="sched_onecycle", augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="onecycle", dropout_rate=0.0),
    dict(id="sched_step",     augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="step",     dropout_rate=0.0),
]

DROPOUT_EXPERIMENTS = [
    dict(id="dropout_00",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.0),
    dict(id="dropout_01",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.1),
    dict(id="dropout_03",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.3),
    dict(id="dropout_05",  augmentation="standard", optimizer_type="adamw", weight_decay=1e-4, scheduler_type="cosine", dropout_rate=0.5),
]

BEST_EXPERIMENT = [
    dict(id="best_etap7", augmentation="standard", optimizer_type="adamw", weight_decay=1e-4,
         scheduler_type="onecycle", dropout_rate=0.1),
]

ALL_GROUPS = {
    "augmentation": AUGMENTATION_EXPERIMENTS,
    "optimizer":    OPTIMIZER_EXPERIMENTS,
    "scheduler":    SCHEDULER_EXPERIMENTS,
    "dropout":      DROPOUT_EXPERIMENTS,
    "best":         BEST_EXPERIMENT,
}

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]


def run_experiment(cfg: dict, noise_tag: str, noise_rate: float, p_dog: float):
    base_id = cfg["id"]
    exp_id  = f"{base_id}_{noise_tag}"
    aug     = cfg["augmentation"]
    print(f"\n{'='*60}\nExperiment: {exp_id}\n{'='*60}")

    dm = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                augmentation=aug, seed=SEED)
    dm.prepare_data()
    dm.setup(stage="fit")
    steps_per_epoch = math.ceil(len(dm.train_dataset) / BATCH)

    model = WasteSortingModule(
        learning_rate   = 1e-4,
        weight_decay    = cfg.get("weight_decay", 0.0),
        optimizer_type  = cfg.get("optimizer_type", "adam"),
        scheduler_type  = cfg.get("scheduler_type", "cosine"),
        dropout_rate    = cfg.get("dropout_rate", 0.0),
        steps_per_epoch = steps_per_epoch,
        max_epochs      = EPOCHS,
    )

    trainer, model, _ = train(model, dm, ckpt_prefix=exp_id, max_epochs=EPOCHS)
    metrics = extract_trainer_metrics(trainer)

    save_results(exp_id, metrics, config={
        **cfg, "epochs": EPOCHS, "batch_size": BATCH,
        "noise_rate": noise_rate, "p_dog": p_dog,
    })


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else "all"

    if group == "all":
        experiments = [e for g in ALL_GROUPS.values() for e in g]
    elif group in ALL_GROUPS:
        experiments = ALL_GROUPS[group]
    else:
        print(f"Unknown group '{group}'. Choose from: {list(ALL_GROUPS)} or 'all'")
        sys.exit(1)

    for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
        for cfg in experiments:
            run_experiment(cfg, noise_tag, noise_rate, p_dog)


if __name__ == "__main__":
    main()
