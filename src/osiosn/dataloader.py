import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import random
import copy


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class WasteSortingDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 64, noise_rate: float = 0.3, image_noise_std=0.03, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.noise_rate = noise_rate
        self.seed = seed
        
        self.class_mapping = {
            # bottle, bowl, can, cup, plate
            9: 0, 10: 0, 16: 0, 28: 0, 61: 0, # recyclable
            # apple, mushroom, orange, pear, sweet_pepper
            0: 1, 51: 1, 53: 1, 57: 1, 83: 1, # bio
            # computer_keyboard, clock, telephone, television
            39: 2, 22: 2, 86: 2, 87: 2 # electrical_waste
        }

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddGaussianNoise(0., image_noise_std),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        self.transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

    def prepare_data(self):
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)

    def _filter_and_remap(self, dataset):
        indices = []
        new_targets = []
        for i, target in enumerate(dataset.targets):
            if target in self.class_mapping:
                indices.append(i)
                new_targets.append(self.class_mapping[target])
        
        dataset.data = dataset.data[indices]
        dataset.targets = new_targets
        return dataset

    def _add_label_noise(self, targets):
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        noisy_targets = np.array(targets)
        n_samples = len(noisy_targets)
        n_noisy = int(self.noise_rate * n_samples)
        
        noise_indices = np.random.choice(n_samples, n_noisy, replace=False)
        unique_classes = list(set(self.class_mapping.values()))
        
        for idx in noise_indices:
            current_label = noisy_targets[idx]
            possible_labels = [c for c in unique_classes if c != current_label]
            noisy_targets[idx] = np.random.choice(possible_labels)
            
        return noisy_targets.tolist()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            base_dataset = datasets.CIFAR100(self.data_dir, train=True)
            base_dataset = self._filter_and_remap(base_dataset)
            
            train_size = int(0.8 * len(base_dataset))
            val_size = len(base_dataset) - train_size
            generator = torch.Generator().manual_seed(self.seed) # Wymóg Etapu 3 [cite: 66]
            
            indices = torch.randperm(len(base_dataset), generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            all_targets = np.array(base_dataset.targets)
            train_targets_subset = all_targets[train_indices].tolist()
            
            noisy_train_targets = self._add_label_noise(train_targets_subset)
            
            for i, idx in enumerate(train_indices):
                base_dataset.targets[idx] = noisy_train_targets[i]

            train_subset = torch.utils.data.Subset(base_dataset, train_indices)
            val_subset = torch.utils.data.Subset(base_dataset, val_indices)

            self.train_dataset = TransformWrapper(train_subset, self.transform_train)
            self.val_dataset = TransformWrapper(val_subset, self.transform_eval)

        if stage == 'test' or stage is None:
            test_dataset = datasets.CIFAR100(self.data_dir, train=False)
            test_dataset = self._filter_and_remap(test_dataset)
            self.test_dataset = TransformWrapper(test_dataset, self.transform_eval)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
