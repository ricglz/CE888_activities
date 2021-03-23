"""Module to train and test a model"""
from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from datamodule import FlameDataModule
from model import PretrainedModel
from trainer import Trainer

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--steps', type=int, default=958)
    parser.add_argument('--seed', type=int, default=42)

    parser = FlameDataModule.add_argparse_args(parser)
    parser = PretrainedModel.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args()

def main():
    args = get_args()

    seed_everything(args.seed)

    datamodule = FlameDataModule.from_argparse_args(args)
    model = PretrainedModel(args)
    trainer = Trainer.from_argparse_args(args)

    trainer.train_and_test(model, args.epochs, datamodule)

if __name__ == "__main__":
    main()
