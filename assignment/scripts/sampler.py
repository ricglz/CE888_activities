import random

from torch.utils.data.sampler import Sampler
from torchvision.datasets import ImageFolder

class BalancedBatchSampler(Sampler):
    """
    Inspired by:
    https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
    """
    balanced_max = 0
    current_key = 0
    dataset = dict()
    indices = list()

    def __init__(self, dataset: ImageFolder, shuffle=False):
        super().__init__(None)
        self.shuffle = shuffle

        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = max(len(self.dataset[label]), self.balanced_max)

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))

        self.keys = list(self.dataset.keys())

    def reset_indices(self):
        """Resets the indices for a new iteration"""
        self.indices = [-1] * len(self.keys)

    @staticmethod
    def _get_label(dataset: ImageFolder, idx):
        return dataset.targets[idx]

    def __len__(self):
        return self.balanced_max*len(self.keys)

    def __iter__(self):
        if self.shuffle:
            for key in self.keys:
                random.shuffle(self.dataset[key])
        while self.indices[self.current_key] < self.balanced_max - 1:
            self.indices[self.current_key] += 1
            label = self.keys[self.current_key]
            index_label = self.indices[self.current_key]
            yield self.dataset[label][index_label]
            self.current_key = (self.current_key + 1) % len(self.keys)
        self.reset_indices()
