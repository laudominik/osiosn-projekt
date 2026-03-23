from osiosn import WasteSortingModule, WasteSortingDataModule, profile, test

model = WasteSortingModule.load_from_checkpoint("checkpoints/waste-baseline-epoch=00-val_acc=0.84.ckpt")
dm = WasteSortingDataModule(batch_size=1, seed=42)

profile(model, dm)
test(model, dm)
