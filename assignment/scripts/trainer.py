"""Module containing trainer logic"""
from dataclasses import dataclass

from pytorch_lightning.utilities import xla_device
from pytorch_lightning.loggers import WandbLogger
from torch import cuda
import pytorch_lightning as pl

from callbacks import get_checkpoint, Freezer, ProgressBar
from model import PretrainedModel
from utils import CustomParser as ArgumentParser

@dataclass
class Trainer():
    fast_dev_run = False
    gradient_clip = 0.0
    model_name: str
    precision = 16
    stages = 2
    train_bn = False
    unfreeze_per_step = 21

    def get_callbacks(self, model_name: str, epochs: int) -> list:
        checkpoint = get_checkpoint(model_name)
        freezer = Freezer(
            epochs,
            self.stages,
            self.unfreeze_per_step,
            self.train_bn
        )
        return [checkpoint, freezer, ProgressBar()]

    @staticmethod
    def get_accelerator() -> object:
        tpu_device_exists = xla_device.XLADeviceUtils().tpu_device_exists()
        has_gpu = cuda.is_available()

        return {'tpu_cores': 1} if tpu_device_exists else \
               {'gpus': cuda.device_count()} if has_gpu else {}

    def create_trainer(self, model_name, max_epochs=1, **kwargs):
        accelerator = self.get_accelerator()
        callbacks = self.get_callbacks(model_name, max_epochs)
        logger = WandbLogger()
        return pl.Trainer(
            max_epochs=max_epochs, deterministic=True, callbacks=callbacks,
            precision=self.precision, stochastic_weight_avg=False, logger=logger,
            gradient_clip_val=self.gradient_clip, **accelerator, **kwargs)

    def _create_trainer(self, max_epochs: int) -> pl.Trainer:
        return self.create_trainer(
                self.model_name, max_epochs, fast_dev_run=self.fast_dev_run)

    def _fit_cycle(self, model: PretrainedModel, epochs: int, datamodule):
        trainer = self._create_trainer(epochs)
        trainer.fit(model, datamodule=datamodule)
        return trainer

    def train_and_test(self, model: PretrainedModel, epochs: int, datamodule):
        last_trainer = self._fit_cycle(model, epochs, datamodule)
        last_trainer.test(datamodule=datamodule)

    def test(self, model: PretrainedModel, datamodule):
        trainer = self.create_trainer(model.hparams.model_name)
        trainer.test(model=model, datamodule=datamodule)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--fast_dev_run', action='store_true')
        parser.add_argument('--precision', type=int, choices=[16, 32], default=16)
        parser.add_argument('--stages', type=int, default=2)
        parser.add_bool_argument('--train_bn')
        parser.add_argument('--unfreeze_per_step', type=int, default=21)
        parser.add_argument('--gradient_clip', type=float, default=0.0)
        return parser

    @staticmethod
    def from_argparse_args(args):
        trainer = Trainer(args.model_name)
        trainer.fast_dev_run = args.fast_dev_run
        trainer.precision = args.precision
        trainer.stages = args.stages
        trainer.train_bn = args.train_bn
        trainer.unfreeze_per_step = args.unfreeze_per_step
        trainer.gradient_clip = args.gradient_clip
        return trainer
