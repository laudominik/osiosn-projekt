"""
Etap 4 – Douczenie modelu odniesienia (baseline).

MobileNetV2 x0.75 (pretrained CIFAR-100) fine-tuned on the 3-class waste
sorting task.  Only the classifier head is trained; all feature layers are
frozen.  Noisy labels and dog-image injection are applied to the training set
as specified in Etap 3.
"""
from osiosn import WasteSortingModule, WasteSortingDataModule, train, profile
from osiosn import save_results, extract_trainer_metrics

EPOCHS = 10
BATCH  = 64
SEED   = 42

dm    = WasteSortingDataModule(batch_size=BATCH, noise_rate=0.5, p_dog=0.1,
                               augmentation="basic", seed=SEED)
model = WasteSortingModule(learning_rate=1e-4, optimizer_type="adam",
                            scheduler_type="cosine")

trainer, model = train(model, dm, ckpt_prefix="baseline", max_epochs=EPOCHS)

# collect metrics
metrics = extract_trainer_metrics(trainer)

# profiling
dm_prof = WasteSortingDataModule(batch_size=1, seed=SEED)
prof = profile(model, dm_prof)
metrics.update(prof)

save_results("baseline", metrics, config={
    "epochs": EPOCHS, "batch_size": BATCH, "lr": 1e-4,
    "optimizer": "adam", "scheduler": "cosine",
    "augmentation": "basic", "noise_rate": 0.5, "p_dog": 0.1,
})
