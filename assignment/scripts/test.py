"""Module to train and test a model"""
from argparse import ArgumentParser

from pytorch_lightning import seed_everything
import wandb

from datamodule import FlameDataModule
from model import PretrainedModel
from trainer import Trainer

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='rexnet_200')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tta', type=int, default=0)

    parser = FlameDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args()

def main():
    args = get_args()

    wandb.init(project='repo-assignment_scripts', entity='ricglz')
    seed_everything(args.seed)

    datamodule = FlameDataModule.from_argparse_args(args)
    model = PretrainedModel.load_from_checkpoint(
        args.checkpoint_path,
        **args
    )
    trainer = Trainer.from_argparse_args(args)

    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
