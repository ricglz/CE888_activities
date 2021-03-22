from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy, F1, MetricCollection
from timm import create_model
from torch import stack, sigmoid
from torch.nn import BCEWithLogitsLoss, Linear, ModuleDict, ReLU, Sequential
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T

from callbacks import Freezer

class PretrainedModel(LightningModule):
    def __init__(
        self,
        name='rexnet_200',
        epochs=10,
        steps_per_epoch=100,
        lr=1e-3,
        drop_rate=0.4
    ):
        super().__init__()

        self.save_hyperparameters()
        self.base = create_model(name, pretrained=True,
                                 num_classes=1024, drop_rate=drop_rate)
        self.fc = Sequential(ReLU(), Linear(1024, 1))

        self.criterion = BCEWithLogitsLoss()
        self.metrics = self.build_metrics()
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])

    def just_train_classifier(self):
        self.freeze()
        base_classifier = self.base.get_classifier()
        Freezer.make_trainable([base_classifier, self.fc])

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
            output = self.base(x)
            return self.fc(output).squeeze(-1)
        y_hat_stack = stack([self(self.transform(x)) for _ in range(tta)])
        return y_hat_stack.mean(dim=0)

    # Configurations
    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = SGD(parameters, self.hparams.lr, weight_decay=0)
        # optimizer = Lookahead(optimizer)
        scheduler = self._build_scheduler(optimizer)
        scheduler_dict = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler_dict]

    def _build_scheduler(self, optimizer):
        lr, epochs = self.hparams.lr, self.hparams.epochs
        total_steps = epochs * self.hparams.steps_per_epoch
        return OneCycleLR(
            optimizer,
            lr,
            total_steps,
            pct_start=0.5,
            anneal_strategy='linear',
            base_momentum=0.825,
            max_momentum=0.9,
            div_factor=25,
            final_div_factor=1e4,
            three_phase=False
        )

    # Steps
    def _get_dataset_metrics(self, dataset):
        return self.metrics[f'{dataset}_metrics']

    def _update_metrics(self, y_hat, y, dataset):
        proba = sigmoid(y_hat)
        self._get_dataset_metrics(dataset).update(proba, y)

    def _on_step(self, batch, dataset):
        x, y = batch
        tta = 10 if dataset == 'test' else 0
        y_hat = self(x, tta)
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

    def training_step(self, batch, batch_idx):
        return self._on_step(batch, 'train')

    def training_epoch_end(self, outputs):
        self._on_end_epochs('train')

    def validation_step(self, batch, batch_idx):
        return self._on_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        self._on_end_epochs('val')

    def test_step(self, batch, batch_idx):
        return self._on_step(batch, 'test')

    def test_epoch_end(self, outputs):
        self._on_end_epochs('test')
