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

## Efficient training results

\begin{table}[h]
\centering
\begin{tabular}{llp{2cm}}
\toprule
Model & Time Taken (minutes) & Maximum GPU Memory Allocated (\%) \\
\midrule
GENet & 9.8 & 16.92 \\
EfficientNet & 26.1 & 18.98 \\
RepVGG & 9.89 & 15.81 \\
ReXNet & 8.22 & 62.92 \\
\bottomrule
\end{tabular}
\caption{Time and memory metrics of each model}
\label{tab:efficiency}
\end{table}

In addition to the performance metrics, it will be analyzed the computation time to train the model and the maximum percentage of memory allocated. And as it can be seen in table \ref{tab:efficiency}, it is very impressive that most of the models used can be trained in less than 10 minutes. A comparison with the FLAME model is not possible as \cite{shamsoshoara2020} only shared the amount of epochs needed to train, which were 40. But taking into consideration that in average we took around 3 minutes per epoch to train, then we can assume that probably their model was trained in around 120 minutes, but the real number may be higher.

Regarding the memory consumption, there are 2 factors to consider prior to analyzing the results. First the GPUs memories, the GTX 1080 Ti has 11GB of memory, meanwhile the RTX 2080 that was the one used for the ReXNet training has only 8GB. Secondly, the ReXNet model was trained in a multi-node GPU which later on was discovered had a problem of allocating the same GPU to different process. Taking that in consideration is very interesting how all of the models are using less than 5GB, as even though the ReXNet model is reporting a use of 62.92% of the memory it can be assume that part of it is used for another process as the percentage is significantly higher than the other models. Also it means, that we could have experimented with higher batch sizes and analyze if it would improve or decrease the performance of the model.

## Hyper-Parameter tuning insights

After tuning the hyper-parameters an insight concluded is that even though a lot of papers decide to use SGD as their optimizer, in our case none of the models ended up using it. Instead, the most popular optimizer for our models was Adam, which was the one used for most of the models, with the exception of RepVGG which used RMSProp.

In addition to the prior, it was interesting to see that mixup was almost non-effective, as it was only used in the GENet. The reason could be because there were only two classes; thus, reducing the effectiveness of the technique. Also it can be concluded based on the models, that if the OneCycleLR is used, it's possible that the best anneal strategy to use is a linear behavior instead of a cosine. Additionally, also for the OneCycleLR, the minimum value of the momentum should be between 0.75 and 0.82, meanwhile the max value should be 0.86 and 0.95

Finally, for this dataset the models require a drop rate between 0.45 and 0.55, as all the models have the models their best drop rate in this range.
