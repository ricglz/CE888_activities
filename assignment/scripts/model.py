"""Model module"""
from argparse import ArgumentParser

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, F1, MetricCollection

from torch import stack, sigmoid
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, ModuleDict
from torch.nn.functional import softmax
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T

from timm import create_model
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from auto_augment import AutoAugment
from callbacks import Freezer

class PretrainedModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        if hasattr(hparams, 'mixup') and hparams.mixup:
            self.base = create_model(
                self.hparams.model_name,
                pretrained=True,
                num_classes=2,
                drop_rate=self.hparams.drop_rate
            )

            self.mixup = Mixup(hparams.alpha, num_classes=2)
            self.train_criterion = SoftTargetCrossEntropy()
            self.val_criterion = CrossEntropyLoss()
            self.activation = lambda y_val: softmax(y_val, dim=1)
        else:
            self.base = create_model(
                self.hparams.model_name,
                pretrained=True,
                num_classes=1,
                drop_rate=self.hparams.drop_rate
            )

            self.train_criterion = BCEWithLogitsLoss()
            self.val_criterion = self.train_criterion
            self.activation = sigmoid
        self.metrics = self.build_metrics()
        self.transform = self.build_transforms()

    def build_transforms(self):
        hparams = self.hparams
        return AutoAugment(
            hparams.magnitude, hparams.amount, hparams.probability
        ) if hparams.auto_augment else T.Compose([
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=45),
        ])

    def just_train_classifier(self):
        self.freeze()
        Freezer.make_trainable(self.base.get_classifier())

    @property
    def total_steps(self):
        dataset_size = 30689 if self.hparams.no_minified else 61378
        steps_per_epoch = dataset_size // self.hparams.batch_size
        return steps_per_epoch * self.hparams.epochs

    def general_div_factor(self, div_factor):
        epochs = self.hparams.epochs
        value = div_factor * epochs / 5
        return value if epochs <= 5 else value * epochs ** 2

    @property
    def div_factor(self):
        return self.general_div_factor(self.hparams.div_factor)

    @property
    def final_div_factor(self):
        return self.general_div_factor(self.hparams.final_div_factor)

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

    def forward(self, x, tta = 0):
        if tta == 0:
            return self.base(x)
        y_hat_stack = stack([self(self.transform(x)) for _ in range(tta)])
        return y_hat_stack.mean(dim=0)

    # Configurations
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer_class = SGD if self.hparams.optimizer == 'sgd' else \
                          Adam if self.hparams.optimizer == 'adam' else RMSprop
        optimizer = optimizer_class(
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
            self.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy=self.hparams.anneal_strategy,
            base_momentum=self.hparams.base_momentum,
            max_momentum=self.hparams.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            three_phase=self.hparams.three_phase
        )

    # Steps
    def _get_dataset_metrics(self, dataset):
        return self.metrics[f'{dataset}_metrics']

    def _update_metrics(self, y_hat, y, dataset):
        proba = self.activation(y_hat)
        self._get_dataset_metrics(dataset).update(proba, y)

    def _on_step(self, batch, dataset, dataloader_idx=None):
        x, y = batch
        is_train_dataset = dataset == 'train'
        if is_train_dataset and self.hparams.mixup:
            x, y = self.mixup(x, y)
        tta = self.hparams.tta if dataset == 'test' else 0
        y_hat = self(x, tta)
        if not self.hparams.mixup:
            y_hat = y_hat.squeeze(-1)
        criterion = self.train_criterion if is_train_dataset \
                                         else self.val_criterion
        if not isinstance(criterion, CrossEntropyLoss):
            y = y.float()
        loss = criterion(y_hat, y)
        self._update_metrics(y_hat, batch[1], dataset)
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

    def validation_step(self, batch, *extra):
        return self._on_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        self._on_end_epochs('val')

    def test_step(self, batch, _, dataloader_idx: int):
        dataset = 'val' if dataloader_idx == 0 else 'test'
        return self._on_step(batch, dataset)

    def test_epoch_end(self, outputs):
        self._on_end_epochs('val')
        self._on_end_epochs('test')

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', type=float, default=0)
        parser.add_argument('--base_momentum', type=float, default=0.825)
        parser.add_argument('--div_factor', type=float, default=25)
        parser.add_argument('--drop_rate', type=float, default=0.4)
        parser.add_argument('--final_div_factor', type=float, default=1e4)
        parser.add_argument('--lr', type=float, required=True)
        parser.add_argument('--max_momentum', type=float, default=0.9)
        parser.add_argument('--mixup', type=bool, default=False)
        parser.add_argument('--pct_start', type=float, default=0.5)
        parser.add_argument('--three_phase', type=bool, default=False)
        parser.add_argument('--tta', type=int, default=0)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument(
            '--anneal_strategy',
            type=str,
            default='linear',
            choices=['linear', 'cos']
        )
        parser.add_argument(
            '--optimizer',
            type=str,
            default='sgd',
            choices=['sgd', 'adam', 'rmsprop']
        )
        return parser
