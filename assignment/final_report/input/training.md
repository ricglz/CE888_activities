\begin{table*}[t]
\centering
\begin{tabular}{|l|c|c|c|c|c|c|}
\toprule
& \multicolumn{2}{|c|}{Training} & \multicolumn{2}{|c|}{Validation} & \multicolumn{2}{|c|}{Test}\\
\cmidrule{2-3} \cmidrule{4-5} \cmidrule{6-7}
Model & Accuracy (\%) & Loss & Accuracy (\%) & Loss & Accuracy (\%) & Loss \\
\midrule
FLAME & 96.79 & 0.0857 & 94.31 & 0.1506 & 76.23 & 0.7414 \\
GENet & 73.05 & 0.3925 & 99.15 & 0.0749 & 83.17 & 0.3722 \\
EfficientNet & 97.94 & 0.006275 & 99.62 & 0.0293 & 83.18 & 0.4198 \\
RepVGG & 98.58 & 0.2313 & 98.74 & 0.1525 & 0.8463 & 0.5854 \\
ReXNet & 99.23 & 0.0855 & 99.07 & 0.02632 & 90.68 & 0.2259 \\
\bottomrule
\end{tabular}
\caption{Models accuracies and losses in all datasets}
\label{tab:performance}
\end{table*}

## Training

The model will be trained for 5 epochs using Mixed Precision and OneCycleLR as a learning scheduler. In addition to augmenting the data, the images will be resized to a ratio of 224x224 and normalize the values based on the mean and standard deviation of the Imagenet dataset \footnote{$mean = [0.485, 0.456, 0.406]$ and $std=[0.229, 0.224, 0.225]$}. As this was the one used in which the models were pre-trained.

Also we used a special data loader which will create batches containing the same amount of random elements of each class. This was possible due to the previous work of \cite{galato2019}, who developed a similar sampler for their use case. Also depending of the model, it will be trained using mixup.

The optimizer for the training will be either SGD, Adam or RMSProp, depending of how the tuning of the hyper-parameters turns out. Meanwhile, the loss function will depend if the model is trained by using mixup or not. When using Mixup it will be trained by using CrossEntropyLoss, meanwhile if it's not trained by it, the model will be trained using BCEWithLogitsLoss.

When the training has ended we will restore the weights of the model to the epoch in which it had the best score, which we consider to be the average between the accuracy and F1 score in the test dataset.

In addition the complete code can be found on Github \footnote{\url{https://github.com/ricglz/CE888_activities/tree/main/assignment/scripts}}, where it can be found the scripts used to perform the experiments.

### TTA

For the test dataset we will use TTA. The merge function that will be used is the mean function. Meanwhile, the amount of augmentations to perform will depend on the backbone of the model. The augmentations to perform will be determined by AutoAugment policies.

### Data Augmentation

For training the model either custom AutoAugment policies will be used or random vertical and horizontal flips, rotations of 45º and modifications in the brightness and contrast of the photos.

### Fine-tuning

As part of the training the model is not completely trainable from the start. At the start, only the first N layers will be trainable while the others will be frozen. At the end of each epoch the next N layers will be unfrozen, until either all the layers are now trainable or the training has been completed. N will depend of the model, as well as if the BatchNormalizer layers will also be trainable or not.
