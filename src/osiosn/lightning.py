import time
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, ReduceLROnPlateau, OneCycleLR, StepLR
)


class WasteSortingModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.0,
        optimizer_type: str = "adam",
        scheduler_type: str = "cosine",
        label_smoothing: float = 0.0,
        unfreeze_n_blocks: int = 0,
        steps_per_epoch: int = None,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs

        # load CIFAR-100 pretrained backbone
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar100_mobilenetv2_x0_75",
            pretrained=True,
        )

        # freeze all feature extraction layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # selectively unfreeze the last `unfreeze_n_blocks` feature blocks
        if unfreeze_n_blocks > 0:
            blocks = list(self.model.features.children())
            for block in blocks[-unfreeze_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        # replace classifier head
        in_features = self.model.classifier[1].in_features
        if dropout_rate > 0.0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, num_classes),
            )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._epoch_start_time = 0.0

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1   = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        self.log("epoch_time", time.time() - self._epoch_start_time)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        trainable = filter(lambda p: p.requires_grad, self.parameters())

        if self.optimizer_type == "adam":
            opt = torch.optim.Adam(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            opt = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            opt = torch.optim.SGD(trainable, lr=self.learning_rate, momentum=0.9,
                                  weight_decay=self.weight_decay, nesterov=True)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type!r}")

        if self.scheduler_type == "cosine":
            sched = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=1, eta_min=1e-6)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        elif self.scheduler_type == "plateau":
            sched = ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5, min_lr=1e-6)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_acc"}}

        elif self.scheduler_type == "onecycle":
            steps = (self.steps_per_epoch or 100) * self.max_epochs
            sched = OneCycleLR(opt, max_lr=self.learning_rate * 10, total_steps=steps,
                               pct_start=0.3, anneal_strategy="cos")
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        elif self.scheduler_type == "step":
            sched = StepLR(opt, step_size=5, gamma=0.1)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type!r}")
