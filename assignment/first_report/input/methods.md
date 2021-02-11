# Methods

## Dataset

As mentioned in the introduction \ref{introduction}, the objective of this research is to use both the FLAME and Kaggle datasets, thus prior to training both datasets will be merged. But also to provide a little more diversity new images will be created, this will help a lot considering that after giving a quick look to the FLAME dataset must of the images are just burst of the same environment with/without fire. Due to this the model could instead of learning to detect fire, could learn to detect the environments.

### Data Augmentation Transformations

> All random transformations have a probability of 50% of happening

- A reduction of the brightness by 50%
- A random rotation of 45º
- A random horizontal flip and a random vertical flip

## Model

As mentioned in the cnn architectures section \ref{cnn-arch}, the architecture that the FLAME researchers used was the Xception architecture, which improves over its previous version called Inception, which improves primarily the performance while keeping the same amount of parameters. Nevertheless, the architecture could be consider _old_, being published in 2017 and many other architectures that have improve accuracy, or just other models which already had better performance than this case.

Due to the former, if we want to improve the performance of the classifier it's recommended that we attempt to use other architectures. For this case the models consider to test will be _EfficientNet_ and _ReXNet_. With these models instead of training randomized weights we will do transfer learning on a model trained using the ImageNet dataset, training only the top layer which is a densely connected layer in both cases.

### EfficientNet

An architecture published in September 2019, that wanted to tackle the problem of scaling up an architecture by having a smaller amount of parameters, but also by keeping/increasing the accuracy of the model, this was able by the use of _compound coefficient_ \cite{Tan2020}.

### ReXNet

An architecture proposed in July 2020, aiming to follow the trend of creating accurate and lightweight architectures, distinguishing themselves by the use of _representational bottlenecks_. The result was quite good improving the accuracy when doing transfer learning from trained models using the COCO dataset, while keeping a small amount of parameters to train \cite{Han2020}.

## Training

For the training must things will be very straightforward, the validation will be checked splitting the training dataset into 80% training and 20% validation, just like in the FLAME paper. In addition this training dataset will also have a set of transformations both to do data augmentation and to normalize the data, this with the intention to not always have the same data in a batch every epoch and give diversity and randomness to the training.

### Optimizer and Criterion/Loss Function

The optimizer chosen for the training is the Adam optimizer. Meanwhile, the criterion used to calculate the loss of the model will be due to being a binary classification problem, the binary crossentropy loss function with logits, meaning that a Sigmoid function will be used as a final activation function and based on this the loss function will calculate the current loss.

### Hyperparameters

- Batch-size: 28
- Learning Rate: 2e-3
- Max epochs: 10

### Callbacks

- _EarlyStopping_, a callback to stop the training if the validation loss have increased or decreased by little values in the last three epochs
- _LRScheduler_, this callback will reduce the learning rate every epoch in a ration of 9e-1

### Transformations

> All random transformations have a probability of 50% of happening. And the random transformations are only for the training dataset

- Resize the image to a size of (254, 254)
- Random horizontal flip
- Random vertical flip
- Normalization
