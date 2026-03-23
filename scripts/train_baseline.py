import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from osiosn import WasteSortingModule, WasteSortingDataModule
from aim.pytorch_lightning import AimLogger


def run_baseline_training():
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    dm = WasteSortingDataModule(batch_size=64, seed=42)
    model = WasteSortingModule(num_classes=3, learning_rate=1e-3)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='waste-baseline-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    aim_logger = AimLogger(
        experiment=f"baseline",
        train_metric_prefix='train_',
        val_metric_prefix='val_',
        test_metric_prefix='test_'
    )


    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
        logger=aim_logger
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return trainer, model

if __name__ == "__main__":
    trainer, model = run_baseline_training()
