## Training

The model will be trained for 5 epochs using Mixed Precision, in addition of using as learning scheduler the OneCycleLR. Also we used a special data loader which will create batches containing the same amount of random elements of each class. This was possible due to the previous work of \cite{galato2019}, who developed a similar sampler for their use case. Also depending of the model, it will be trained using mixup. And finally, it must be mentioned that the model that will be obtained at the end will be when the model achieves the highest average between accuracy and f1 score in the validation dataset.

<!-- The optimizer for the training will be either SGD, Adam or RMSProp, depending of how the tuning of the hyper-parameters turns out. Meanwhile, the loss function will depend if the model is trained by using mixup or not. When using Mixup it will be trained by using CrossEntropyLoss, meanwhile if it's not trained by it, the model will be trained using BCEWithLogitsLoss. -->

The optimizer for the training will be either SGD, Adam or RMSProp, depending of how the tuning of the hyper-parameters turns out. Meanwhile, the loss function will be BCEWithLogitsLoss.

In addition the prior the complete code where it shows can be found on Github \footnote{\url{https://github.com/ricglz/CE888_activities/tree/main/assignment/scripts}}

### TTA

For the test dataset we will use tta. The merge function that will be used is the mean function. Meanwhile, the amount of augmentations to perform will depend on the backbone of the model. The augmentations to perform will be determined AutoAugment policies.

### Data Augmentation

For training the model will either custom AutoAugment policies or will be random vertical and horizontal flips, rotations of 45º and modifications in the brightness and contrast of the photos.

### Fine-tuning

As part of the training the model is not completely trainable since the start. At the start only the first N layers will be trainable while the others will be frozen. At the end of each epoch the next N layers will be unfrozen, until either all the layers are now trainable or the training has been completed. N will depend of the model, as well as if the BatchNormalizer layers will also be trainable or not.
