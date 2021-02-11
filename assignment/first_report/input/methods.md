# Methods

<!-- Formulate your research question. It should include: -->

<!-- A detailed description of the question. -->
<!-- The methods you used to address the question. -->
<!-- The definitions of any relevant terminology. -->
<!-- Any equations that contributed to your work. -->

<!-- The methods section should be described in enough detail for someone to replicate your work. -->

## Dataset

As mentioned in the introduction \ref{introduction}, the objective of this research is to use both the FLAME and Kaggle datasets

## Model

As mentioned in the cnn architectures section \ref{cnn-arch}, the architecture that the FLAME researchers used was the Xception architecture, which improves over its previous version called Inception, which improves primarily the performance while keeping the same amount of parameters. Nevertheless, the architecture could be consider _old_, being published in 2017 and many other architectures that have improve accuracy, or just other models which already had better performance than this case.

Due to the former, if we want to improve the performance of the classifier it's recommended that we attempt to use other architectures. For this case the models consider to test will be _EfficientNet_ and _ReXNet_. With these models instead of training randomized weights we will do transfer learning on a model trained using the ImageNet dataset.

### EfficientNet

An architecture published in September 2019, that wanted to tackle the problem of scaling up an architecture by having a smaller amount of parameters, but also by keeping/increasing the accuracy of the model, this was able by the use of _compound coefficient_ \cite{Tan2020}.

### ReXNet

An architecture proposed in July 2020, aiming to follow the trend of creating accurate and lightweight architectures, distinguishing themselves by the use of _representational bottlenecks_. The result was quite good improving the accuracy when doing transfer learning from trained models using the COCO dataset, while keeping a small amount of parameters to train \cite{Han2020}.

### Hyperparameters

- Batch-size: 28
- Learning Rate: 2e-3
- Max epochs: 10

### Callbacks

- _EarlyStopping_, a callback to stop the training if the validation loss have increased or decreased by little values in the last three epochs
- _LRScheduler_, this callback will reduce the learning rate every epoch in a ration of 9e-1
