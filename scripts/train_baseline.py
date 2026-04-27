import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from osiosn import WasteSortingModule, WasteSortingDataModule, train

dm = WasteSortingDataModule(batch_size=64, seed=42)
model = WasteSortingModule()

train(model, dm, ckpt_prefix='baseline')
