## Hyper-parameters tuning

As we have architectures as backbone that have an inference time small enough to be able to iteratively tune the hyper-parameters of the model, it was decided to do it. This was done by using Weight and Biases (wandb) \cite{wandb2020}, which is a tool that allows to log models, its metrics and more important for our case to create _"sweeps"_. These _"sweeps"_ work by defining a search space it allows to train models with different hyper-parameters with one or many machines, as the wandb backend was the one that decided the hyper-parameters to use from the search space, allowing to parallelize the search between several GPUs more easily, leading to faster results. It also allowed to use either grid, random or bayes search, in our case most of the cases random search was enough, but there were cases where it was used a bayes or grid search instead.

The hyper-parameters to be tuned are the following:

* Amount of augmentations to do for tta
* Drop rate
* Optimizer
* Optimizer's weight decay
* Mixup's alpha
* If BatchNormalizer layers will be trained or not
* AutoAugment hyper-parameters:
    * If will use AutoAugment or pre-defined augmentations
    * Amount of augmentations per policy
    * Magnitude of augmentations per policy
    * Probability of augmentations
* OneCycleLR hyper-parameters:
    * Anneal strategy to use (linear or cos)
    * Min momentum value
    * Max momentum value
    * Max learning rate value
    * Div factor to determine the initial learning rate
    * Div factor to determine the minimum learning rate
    * When the learning rate will start decreasing
* Fine-tuning hyper-parameters:
    * How many layers to unfreeze per epoch
