"""Module to contain callbacks"""
from math import ceil

from torch.nn.modules.batchnorm import _BatchNorm
import pytorch_lightning.callbacks as pl_callbacks

class Freezer(pl_callbacks.BaseFinetuning):
    train_bn = False
    trainable_layers = []

    def __init__(self, epochs=40):
        self.unfreeze_per_step = 27 if epochs >= 60 else \
                                 25 if epochs >= 20 else \
                                 23 if epochs >= 5 else 21
        self.step_size = ceil(epochs / 2)

    @staticmethod
    def flatten_children(module):
        """Flattens the children of the module"""
        vanilla_children = list(module.children())
        if len(vanilla_children) == 0:
            return [module]
        children = []
        for child in vanilla_children:
            child_children = Freezer.flatten_children(child)
            children += child_children
        return children

    @staticmethod
    def filter_non_trainable(children):
        calc_num_params = lambda module: sum(p.numel() for p in module.parameters())
        has_params = lambda module: calc_num_params(module) > 0
        return list(filter(has_params, children))

    @staticmethod
    def filter_non_frozen(children):
        """Filters those children that are already frozen"""
        requires_grad = lambda p: p.requires_grad
        is_frozen_module = \
            lambda module: len(list(filter(requires_grad, module.parameters()))) == 0
        return list(filter(is_frozen_module, children))

    @staticmethod
    def trainable_children(pl_module, train_bn=False, reverse=False):
        """Gets the trainable children of the pl_module"""
        children = Freezer.flatten_children(pl_module)
        children = Freezer.filter_non_trainable(children)
        children = Freezer.filter_non_frozen(children)
        if not train_bn:
            is_not_bn = lambda mod: not isinstance(mod, _BatchNorm)
            children = list(filter(is_not_bn, children))
        if reverse:
            children.reverse()
        return children

    def freeze_before_training(self, pl_module):
        pl_module.just_train_classifier()
        self.trainable_layers = self.trainable_children(
            pl_module, self.train_bn, reverse=True)

    def finetune_function(self, pl_module, epoch, optimizer, _opt_idx):
        trainable_layers_len = len(self.trainable_layers)
        is_empty = trainable_layers_len == 0
        is_finetune_epoch = epoch % self.step_size == 0 and epoch != 0
        if not is_finetune_epoch or is_empty:
            return
        to_be_trained_layers = []
        layers_to_be_freeze = min(self.unfreeze_per_step, trainable_layers_len)
        for _ in range(layers_to_be_freeze):
            to_be_trained_layers.append(self.trainable_layers.pop(0))
        self.unfreeze_and_add_param_group(to_be_trained_layers, optimizer)

    @staticmethod
    def unfreeze_and_add_param_group(
        modules,
        optimizer,
        lr = None,
        initial_denom_lr = 10.,
        train_bn = True,
    ):
        Freezer.make_trainable(modules)
        params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
        denom_lr = initial_denom_lr if lr is None else 1.
        initial_lr = params_lr / denom_lr
        params = Freezer.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = Freezer.filter_on_optimizer(optimizer, params)
        if params:
            param_group = {
                'params': params, 'lr': initial_lr, 'initial_lr': initial_lr,
            }
            extra_data = Freezer.momentum_param_group(optimizer)
            param_group = { **param_group, **extra_data }
            optimizer.add_param_group(param_group)

    @staticmethod
    def momentum_param_group(optimizer):
        """Adds the momentum group for one cycle scheduler"""
        group = optimizer.param_groups[0]
        momentum_group = {
            'base_momentum': group['base_momentum'],
            'max_momentum': group['max_momentum'],
            'max_lr': group['max_lr'],
            'min_lr': group['min_lr'],
        }
        extra_group = {
            'betas': group['betas']
        } if 'betas' in optimizer.defaults else {
            'momentum': group['momentum']
        }
        return { **momentum_group, **extra_group }
