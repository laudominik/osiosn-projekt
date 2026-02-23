import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CLASS_MAPPING = {
    9: 0,
    10: 0,
    16: 0,
    28: 0,
    61: 0,  # bottle, bowl, can, cup, plate -> recyclable
    0: 1,
    51: 1,
    53: 1,
    57: 1,
    83: 1,  # apple, mushroom, orange, pear, sweet_pepper -> bio
    39: 2,
    22: 2,
    86: 2,
    87: 2,  # keyboard, clock, telephone, television -> electrical
}
CLASS_NAMES = ["recyclable", "bio", "electrical_waste"]

DOG_CLASS_IDX = 35


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def build_transforms(strategy: str, image_noise_std: float = 0.01):
    norm = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    noise = AddGaussianNoise(0.0, image_noise_std)

    eval_t = transforms.Compose([norm])

    if strategy == "none":
        train_t = eval_t
    elif strategy == "basic":
        train_t = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                noise,
                norm,
            ]
        )
    elif strategy == "standard":
        train_t = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomGrayscale(p=0.1),
                noise,
                norm,
            ]
        )
    elif strategy == "aggressive":
        train_t = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.15),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                AddGaussianNoise(0.0, image_noise_std * 2),
                norm,
            ]
        )
    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy!r}")

    return train_t, eval_t


class WasteSortingDataModule(pl.LightningDataModule):
    train_dataset: "_TransformDataset"  # pyright: ignore
    val_dataset: "_TransformDataset"  # pyright: ignore
    test_dataset: "_TransformDataset"  # pyright: ignore

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        noise_rate: float = 0.5,
        image_noise_std: float = 0.01,
        p_dog: float = 0.1,
        augmentation: str = "basic",
        seed: int = 42,
        num_workers: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise_rate = noise_rate
        self.p_dog = p_dog
        self.augmentation = augmentation
        self.seed = seed
        self.num_workers = num_workers

        self.transform_train, self.transform_eval = build_transforms(
            augmentation, image_noise_std
        )

    def prepare_data(self):
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def _filter_and_remap(self, dataset):
        indices = [i for i, t in enumerate(dataset.targets) if t in CLASS_MAPPING]
        dataset.data = dataset.data[indices]
        dataset.targets = [CLASS_MAPPING[dataset.targets[i]] for i in indices]
        return dataset

    def _add_label_noise(self, targets: list[int]) -> list[int]:
        rng = np.random.default_rng(self.seed)
        arr = np.array(targets)
        n = len(arr)
        flip_mask = rng.random(n) < self.noise_rate
        num_classes = len(CLASS_NAMES)
        for i in np.where(flip_mask)[0]:
            other = [c for c in range(num_classes) if c != arr[i]]
            arr[i] = rng.choice(other)
        return arr.tolist()

    def setup(self, stage=None):
        if stage in ("fit", None):
            base = datasets.CIFAR100(self.data_dir, train=True)
            dog_idx = np.where(np.array(base.targets) == DOG_CLASS_IDX)[0]
            dog_data = base.data[dog_idx]

            base = self._filter_and_remap(base)

            rng_torch = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(len(base), generator=rng_torch).tolist()
            n_train = int(0.8 * len(base))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]

            rng = np.random.default_rng(self.seed)
            n_dogs = int(self.p_dog * len(train_idx))
            inject_at = rng.choice(train_idx, n_dogs, replace=False)
            src_dogs = rng.choice(len(dog_data), n_dogs, replace=True)
            base.data[inject_at] = dog_data[src_dogs]

            all_t = np.array(base.targets)
            noisy = self._add_label_noise(all_t[train_idx].tolist())
            for i, idx in enumerate(train_idx):
                base.targets[idx] = noisy[i]

            train_sub = torch.utils.data.Subset(base, train_idx)
            val_sub = torch.utils.data.Subset(base, val_idx)
            self.train_dataset = _TransformDataset(train_sub, self.transform_train)
            self.val_dataset = _TransformDataset(val_sub, self.transform_eval)

        if stage in ("test", "fit", None):
            test = datasets.CIFAR100(self.data_dir, train=False)
            test = self._filter_and_remap(test)
            self.test_dataset = _TransformDataset(test, self.transform_eval)

        print(
            "Split SZ:",
            len(self.train_dataset) if hasattr(self, "train_dataset") else "-",
            len(self.val_dataset) if hasattr(self, "val_dataset") else "-",
            len(self.test_dataset) if hasattr(self, "test_dataset") else "-",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def num_train_samples(self):
        return len(self.train_dataset) if hasattr(self, "train_dataset") else None


class _TransformDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, subset, transform):
        self.transform = transform
        to_tensor = transforms.ToTensor()
        self.cache = [
            (to_tensor(x), y) for x, y in (subset[i] for i in range(len(subset)))
        ]

    def __getitem__(self, idx):
        x, y = self.cache[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.cache)
