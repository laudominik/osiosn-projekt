from pytorch_lightning.callbacks import Callback
from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics


EPOCHS = 50
BATCH  = 64
SEED   = 42

NOISE_VARIANTS = [
    ("noisy", 0.5, 0.1),
    ("clean", 0.0, 0.0),
]

for noise_tag, noise_rate, p_dog in NOISE_VARIANTS:
    exp_id = f"baseline_{noise_tag}"
    print(f"\n{'='*60}\nBaseline - {noise_tag} training\n{'='*60}")

    dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=noise_rate, p_dog=p_dog,
                                   augmentation="basic", seed=SEED)
    model = WasteSortingModule(learning_rate=1e-5, optimizer_type="adam",
                                scheduler_type="cosine")

    trainer, model, history = train(model, dm, ckpt_prefix=exp_id, max_epochs=EPOCHS)

    metrics = extract_trainer_metrics(trainer)
    
    metrics["training_history"] = history

    dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
    prof = profile(model, dm_prof)
    metrics.update(prof)

    save_results(exp_id, metrics, config={
        "epochs": EPOCHS, "batch_size": BATCH, "lr": 1e-4,
        "optimizer": "adam", "scheduler": "cosine",
        "augmentation": "basic", "noise_rate": noise_rate, "p_dog": p_dog,
    })