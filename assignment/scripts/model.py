"""Model module"""
from argparse import ArgumentParser

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, F1, MetricCollection
from timm import create_model
from torch import stack, sigmoid
from torch.nn import BCEWithLogitsLoss, ModuleDict
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T

from callbacks import Freezer

class PretrainedModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.base = create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=1,
            drop_rate=self.hparams.drop_rate
        )

        self.criterion = BCEWithLogitsLoss()
        self.metrics = self.build_metrics()
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def just_train_classifier(self):
        self.freeze()
        Freezer.make_trainable(self.base.get_classifier())

    @staticmethod
    def build_metrics():
        general_metrics = [
            Accuracy(compute_on_step=False),
            F1(num_classes=2, compute_on_step=False),
        ]
        metric = MetricCollection(general_metrics)
        return ModuleDict({
            'test_metrics': metric.clone(),
            'train_metrics': metric.clone(),
            'val_metrics': metric.clone(),
        })

    def forward(self, x):
        return self.base(x).squeeze(-1)

    # Configurations
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = SGD(
            parameters,
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        # optimizer = Lookahead(optimizer)
        scheduler = self._build_scheduler(optimizer)
        scheduler_dict = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler_dict]

    def _build_scheduler(self, optimizer):
        return OneCycleLR(
            optimizer,
            self.hparams.lr,
            self.hparams.epochs * self.hparams.steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy=self.hparams.anneal_strategy,
            base_momentum=self.hparams.base_momentum,
            max_momentum=self.hparams.max_momentum,
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
            three_phase=self.hparams.three_phase
        )

    # Steps
    def _get_dataset_metrics(self, dataset):
        return self.metrics[f'{dataset}_metrics']

    def _update_metrics(self, y_hat, y, dataset):
        proba = sigmoid(y_hat)
        self._get_dataset_metrics(dataset).update(proba, y)

    def _on_step(self, batch, dataset):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())
        self._update_metrics(y_hat, y, dataset)
        self.log(f'{dataset}_loss', loss, prog_bar=True)
        return loss

    def _on_end_epochs(self, dataset):
        metrics = self._get_dataset_metrics(dataset)
        metrics_dict = metrics.compute()
        for key, value in metrics_dict.items():
            self.log(f'{dataset}_{key}', value)
        if dataset != 'train':
            score = stack(list(metrics_dict.values())).mean()
            self.log(f'{dataset}_score', score, prog_bar=True)
        metrics.reset()

    def training_step(self, batch, _batch_idx):
        return self._on_step(batch, 'train')

    def training_epoch_end(self, outputs):
        self._on_end_epochs('train')

    def validation_step(self, batch, _batch_idx):
        return self._on_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        self._on_end_epochs('val')

    def test_step(self, batch, _batch_idx):
        return self._on_step(batch, 'test')

    def test_epoch_end(self, outputs):
        self._on_end_epochs('test')

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, required=True)
        parser.add_argument('--drop_rate', type=float, default=0.4)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--pct_start', type=float, default=0.5)
        parser.add_argument('--base_momentum', type=float, default=0.825)
        parser.add_argument('--max_momentum', type=float, default=0.9)
        parser.add_argument('--div_factor', type=float, default=25)
        parser.add_argument('--final_div_factor', type=float, default=1e4)
        parser.add_argument('--three_phase', action='store_true')
        parser.add_argument(
            '--anneal_strategy',
            type=str,
            default='linear',
            choices=['linear', 'cos']
        )
        return parser
