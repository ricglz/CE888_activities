## Hyper-parameters tuning

As we have architectures as backbone that have an inference time small enough to be able to iteratively tune the hyper-parameters of the model, it was decided to do it. This was done by using Weight and Biases (wandb) \cite{wandb2020}, which is a tool that allows to log models, their metrics and, more importantly for our case, to create _"sweeps"_. These _"sweeps"_ work by defining a search space that allows training models with different hyper-parameters with one or many machines. In these "sweeps" it allows the user to use either grid, random or bayes search to generate the hyper-parameters to experiment with. In our case most of the time random search was enough, but there were occasions where a bayes or grid search were used instead. The hyper-parameters that were tuned were the following:

* Amount of augmentations to do for TTA
* Drop rate
* Optimizer
* Optimizer's weight decay
* Mixup's alpha
* If BatchNormalizer layers will be trained or not
* AutoAugment hyper-parameters:
    * If AutoAugment or pre-defined augmentations will used
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
