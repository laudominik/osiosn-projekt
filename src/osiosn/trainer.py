import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
import torch

from aim.pytorch_lightning import AimLogger


def train(model, dm, ckpt_prefix, max_epochs=10, extra_callbacks=[]):
    class HistoryCallback(Callback):
        def __init__(self):
            self.history = {"epochs": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            self.history["epochs"].append(trainer.current_epoch + 1)
            self.history["train_loss"].append(metrics.get("train_loss", 0).item() if "train_loss" in metrics else 0)
            self.history["train_acc"].append(metrics.get("train_acc", 0).item() if "train_acc" in metrics else 0)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metrics = trainer.callback_metrics
            self.history["val_loss"].append(metrics.get("val_loss", 0).item() if "val_loss" in metrics else 0)
            self.history["val_acc"].append(metrics.get("val_acc", 0).item() if "val_acc" in metrics else 0)


    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename=ckpt_prefix + '_waste-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    history_cb = HistoryCallback()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    aim_logger = AimLogger(
        experiment=ckpt_prefix,
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, history_cb] + extra_callbacks,
        deterministic=True,
        logger=aim_logger,
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5
    )
    

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return trainer, model, history_cb.history
