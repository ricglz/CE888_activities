# Methods

Following it will be described the methods used for each of the sections that we used for the research. If you're actually interested in the code this can be found in the github repo \footnote{\raggedright\url{https://github.com/ricglz/CE888_activities/tree/main/assignment}}

## Dataset

<!-- TODO: Add link to dataset -->

As mentioned in section \ref{introduction}, the objective of this research is to use both the FLAME and Kaggle datasets, thus prior to training both datasets will be merged. As the distribution of the classes end up being skewed, it will be performed as part of the preprocessing data augmentation in the images of the lower class (_No_Fire_), to perfectly balance both classes. If you're interested in the final version of the dataset used in training it's available as a zip in Google Drive \footnote{\raggedright\url{https://drive.google.com/file/d/1t1v4kuBIDk5iwehwyawIUAj9eY2QmY9U/view?usp=sharing}}

### Data Augmentation Transformations

> All random transformations have a probability of 50% of happening

- A reduction or increase of brightness and contrast within a range of 0.75-1.25. The ratio that will be used to transform is defined by a uniform probability
- A random rotation of 5º
- A random horizontal flip and a random vertical flip

## Model

As mentioned in section \ref{cnn-arch}, the architecture that the FLAME researchers used was the Xception architecture, which improves over its previous version called Inception; improving primarily the performance while keeping the same amount of parameters. Nevertheless, the architecture could be consider "_old_", as it was published in 2017 and many other architectures have been published since then, aiming primarily to create lightweight models that have the same or better accuracy than those that have more parameters.

Due to the former, if we want to improve the performance of the classifier it's recommended that we attempt to use other architectures. For this case, the models consider to test will be _EfficientNet_ and _ReXNet_, models that instead of being trained from randomized weights, will be trained by doing transfer learning on the weights of the model trained for the ImageNet dataset \cite{timm}.

## Training

For the training must things will be very straightforward, the validation will be checked splitting the training dataset into 80% training and 20% validation, just like in the FLAME paper. In addition this training dataset will also have a set of transformations both to do data augmentation and to normalize the data, this with the intention to not always have the same data in a batch every epoch and give diversity and randomness to the training.

### Optimizer and Criterion/Loss Function

The optimizer chosen for the training is the Adam optimizer. Meanwhile, the criterion used to calculate the loss of the model will be due to being a binary classification problem, the binary cross-entropy loss function with logits, meaning that a Sigmoid function will be used as a final activation function and based on this the loss function will calculate the current loss.

### Hyperparameters

- Batch-size: 32
- Learning Rate: 5e-5
- Max epochs: 15

### Callbacks

- _LRScheduler_, this callback multiplies the current learning with the value of the function that receives as an argument the current epoch. Said function has the following structure, using as _max_lr_: 5e-3, _min_lr_: 1e-5, _num_warmup_steps_: 6 and _num_training_steps_: 9

```python
if epoch <= num_warmup_steps:
    return log(max_lr / lr) /
           log(num_warmup_steps)
return log(min_lr / max_lr) /
       log(num_training_steps)
```

### Transformations

> All random transformations have a probability of 50% of happening. And the random transformations are only for the training dataset

- Resize the image to a size of (254, 254)
- A reduction or increase of brightness and contrast within a range of 0.75-1.25. The final number is defined by a uniform probability
- A random rotation of 5º
- A random horizontal flip and a random vertical flip
- Normalization using the mean and std of each channel of the ImageNet dataset
