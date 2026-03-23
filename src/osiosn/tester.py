import pytorch_lightning as pl

def test(model, dm):   
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        deterministic=True,
    )
    trainer.test(model, datamodule=dm)
