# Results

<!-- TODO: Update validation with new stuff -->

The following results were obtained by using NVidia RTX 2080 gpu for the ReXnet model and a GTX 1080 TI for the other.

## Results compared to baseline

\begin{figure}[h]
\centering
\subcaptionbox{Training loss through steps \label{fig:train-loss}}{\includegraphics[width=0.24\textwidth, height=1.5in]{img/training_loss}}%
\hfill
\subcaptionbox{Validation loss through steps \label{fig:val-loss}}{\includegraphics[width=0.24\textwidth, height=1.5in]{img/validation_loss}}%
\caption{Progression of the loss in training and validation datasets}
\label{fig:dataset-losses}
\end{figure}

As we can see in figure \ref{fig:train-loss}, the reduction of the training loss seems normal among most of the models, the GENet model being the exception. This is because the model was the only one that was trained by using mixup. Thus, even though it has overall a high loss through all the training, the loss is also very stable without a lot of peaks, just like the ones that ReXNet and EfficientNet have at the start of the training.

We will consider the model done in \cite{shamsoshoara2020} as our baseline to determine if the tuning and new techniques, actually improve the model performance over what is the current state-of-the-art. As it can be seen in table \ref{tab:performance}, all the models perform better than the FLAME model in the test dataset, as both the accuracy and the loss in the dataset improve. Nevertheless, we have a special case that is the GENet model, in which the performance in the training dataset is considerably lower than the other models, even though it achieved a higher accuracy than the FLAME model in the test dataset. This is because GENet is the only model that used mixup as part of the training, as a result of the tuning of the hyper-parameters.

\begin{table}[h]
\centering
\begin{tabular}{|l|l|}
\toprule
Model & Time Taken (minutes) \\
\midrule
GENet & 9.8 \\
EfficientNet & 26.1 \\
RepVGG & 9.89 \\
ReXNet & 8.22 \\
\bottomrule
\end{tabular}
\caption{Time that each model takes to train}
\label{tab:time}
\end{table}

In addition another metric that we consider worthy to analyze is the amount of time it took to train the model. And as it can be seen in table \ref{tab:time}, it is very impressive that most of the models used can be trained in less than 10 minutes. A comparison with the FLAME model is not possible as \cite{shamsoshoara2020} only shared the amount of epochs needed to train, which were 40. But taking into consideration that in average we took around 3 minutes per epoch to train, then we can assume that probably their model was trained in around 120 minutes, but the real number may be higher.

## Hyper-Parameter tuning insights

After tuning the hyper-parameters an insight concluded is that even though a lot of papers decide to use SGD as their optimizer, in our case none of the models ended up using it. Instead, the most popular optimizer for our models was Adam, which was the one used for most of the models, with the exception of RepVGG which used RMSProp.

In addition to the prior, it was interesting to see that mixup was almost non-effective, as it was only used in the GENet. The reason could be because there were only two classes; thus, reducing the effectiveness of the technique. Also it can be concluded based on the models, that if the OneCycleLR is used, it's possible that the best anneal strategy to use is a linear behavior instead of a cosine. Additionally, also for the OneCycleLR, the minimum value of the momentum should be between 0.75 and 0.82, meanwhile the max value should be 0.86 and 0.95

Finally, for this dataset the models require a drop rate between 0.45 and 0.55, as all the models have the models their best drop rate in this range.
