"""Contains the datamodule class"""
from multiprocessing import cpu_count
from os import path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from timm.data.auto_augment import augment_and_mix_transform

from auto_augment import AutoAugment, parse_hparams
from sampler import BalancedBatchSampler
from utils import get_data_dir, CustomParser as ArgumentParser

class FlameDataModule(LightningDataModule):
    """Datamodule to handle and prepare the Flame dataset"""
    data_dir = ''
    train_ds = None
    test_ds = None
    val_ds = None

    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.minified = not args.no_minified
        resize = T.Resize((224, 224))
        normalize = T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        to_tensor = T.ToTensor()
        augmentations = self.get_augmentations(args)
        self.train_transforms = T.Compose([
            resize,
            *augmentations,
            to_tensor,
            normalize
        ])
        self.transforms = T.Compose([resize, to_tensor, normalize])

    @staticmethod
    def get_augmentations(args) -> list:
        if args.auto_augment:
            return [AutoAugment(args.magnitude, args.amount)]
        if args.augmix:
            return [parse_hparams(args)]
        return [
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.RandomRotation(degrees=45),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]

    def prepare_data(self):
        self.data_dir = get_data_dir(self.minified)

    def create_dataset(self, folder_name, transforms):
        return ImageFolder(path.join(self.data_dir, folder_name), transforms)

    def setup(self, stage=None):
        self.train_ds = self.create_dataset('Training', self.train_transforms)
        self.val_ds = self.create_dataset('Validation', self.transforms)
        self.test_ds = self.create_dataset('Test', self.transforms)

    def _general_dataloader(self, dataset, **kwargs):
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=cpu_count(),
            drop_last=True, pin_memory=True, **kwargs)

    def train_dataloader(self):
        sampler = BalancedBatchSampler(self.train_ds, shuffle=True)
        return self._general_dataloader(self.train_ds, sampler=sampler)
        # print(len(loader))
        # return loader

    def val_dataloader(self):
        return self._general_dataloader(self.test_ds)

    def test_dataloader(self):
        return self._general_dataloader(self.val_ds), self._general_dataloader(self.test_ds)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = AutoAugment.add_argparse_args(parent_parser)

        parser.add_bool_argument('--augmix')
        parser.add_argument('--magnitude', type=int, default=3)
        parser.add_argument('--width', type=int, default=3)
        parser.add_argument('--depth', type=int, default=-1)
        parser.add_argument('--blend', type=int, default=0)
        parser.add_argument('--mstd', type=float, default=0)

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--no_minified', action='store_false')
        return parser

    @staticmethod
    def from_argparse_args(args):
        return FlameDataModule(args)
