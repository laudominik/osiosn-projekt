import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelPruning, QuantizationAwareTraining
from osiosn import WasteSortingModule, WasteSortingDataModule, train


pruning_callback = ModelPruning(
    pruning_fn="l1_unstructured",
    parameter_names=["weight"],
    use_global_unstructured=True,
    amount=0.2,
    make_pruning_permanent=True,
    use_lottery_ticket_hypothesis=False,
    resample_parameters=False,
    pruning_dim=None,
    pruning_norm=None,
    verbose=1,
    prune_on_train_epoch_end=True
)
qat_callback = QuantizationAwareTraining(
    observer_type='histogram', 
    quant_compatible=True
)

dm = WasteSortingDataModule(batch_size=64, seed=42,  noise_rate=0.15, p_dog=0.1)
model = WasteSortingModule()

train(
    model, 
    dm, 
    ckpt_prefix='pruning_unstructured', 
    extra_callbacks=[pruning_callback], 
    max_epochs=5
)