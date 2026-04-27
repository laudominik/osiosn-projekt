import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from aim.pytorch_lightning import AimLogger


def train(model, dm, ckpt_prefix, max_epochs=10, extra_callbacks=[]):
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename=ckpt_prefix + '_waste-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
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
        callbacks=[checkpoint_callback, lr_monitor] + extra_callbacks,
        deterministic=True,
        logger=aim_logger,
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return trainer, model
