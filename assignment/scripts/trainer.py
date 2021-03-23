"""Module containing trainer logic"""
from argparse import ArgumentParser

from pytorch_lightning.utilities import xla_device
from torch import cuda
import pytorch_lightning as pl

from callbacks import get_checkpoint, Freezer, ProgressBar
from model import PretrainedModel

class Trainer():
    def __init__(self, model_name: str, fast_dev_run=False):
        self.fast_dev_run = fast_dev_run
        self.model_name = model_name

    @staticmethod
    def get_callbacks(model_name: str, epochs: int) -> list:
        checkpoint = get_checkpoint(model_name)
        return [checkpoint, Freezer(epochs=epochs), ProgressBar()]

    @staticmethod
    def get_accelerator() -> object:
        tpu_device_exists = xla_device.XLADeviceUtils().tpu_device_exists()
        has_gpu = cuda.is_available()

        return {'tpu_cores': 8} if tpu_device_exists else \
               {'gpus': cuda.device_count()} if has_gpu else {}

    @classmethod
    def create_trainer(cls, model_name, max_epochs=1, **kwargs):
        accelerator = cls.get_accelerator()
        callbacks = cls.get_callbacks(model_name, max_epochs)
        return pl.Trainer(
            max_epochs=max_epochs, deterministic=True, callbacks=callbacks,
            precision=16, stochastic_weight_avg=False, **accelerator, **kwargs)

    def _create_trainer(self, max_epochs: int) -> pl.Trainer:
        return self.create_trainer(
                self.model_name, max_epochs, fast_dev_run=self.fast_dev_run)

    def _fit_cycle(self, model: PretrainedModel, epochs: int, datamodule):
        trainer = self._create_trainer(epochs)
        trainer.fit(model, datamodule=datamodule)
        return trainer

    def train_and_test(self, model: PretrainedModel, epochs: int, datamodule):
        last_trainer = self._fit_cycle(model, epochs, datamodule)
        last_trainer.test(model, datamodule)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--fast_dev_run', action='store_true')
        return parser

    @staticmethod
    def from_argparse_args(args):
        return Trainer(args.model_name, args.fast_dev_run)
