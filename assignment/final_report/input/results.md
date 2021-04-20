# Results

<!-- TODO: Update validation with new stuff -->

\begin{figure}[h]
\centering
\subcaptionbox{Training loss through steps \label{fig:train-loss}}{\includegraphics[width=0.49\textwidth]{img/training_loss}}%
\hfill
\subcaptionbox{Validation loss through steps \label{fig:val-loss}}{\includegraphics[width=0.49\textwidth]{img/validation_loss}}%
\caption{Progression of the loss in training and validation datasets}
\label{fig:dataset-losses}
\end{figure}

The following results were obtained by using NVidia RTX 2080 gpu for the ReXnet model and a GTX 1080 TI for the other.

As we can see in figure \ref{fig:train-loss}, the reduction of the training loss seems normal among must of the models, being the exception the GENet model. This is because the model was the only one that was trained by using mixup. Thus, even though it has overall a high loss through all the training, the loss is also very stable without a lot of peaks, just like the ones that ReXNet and EfficientNet have at the start of the training.

We will consider the model done in \cite{shamsoshoara2020} as our baseline to determine if the tuning and new techniques, actually improves the model performance over what is the current state of the art. As it can be seen in table \ref{tab:performance}, all the models performs better than the FLAME model in the test dataset, being this that both the accuracy and the loss in the dataset improve. Nevertheless, we have a special case that is the GENet model, in which the performance in the training dataset is considerably lower than the other models, even though it achieved a higher accuracy than the FLAME model in the test dataset. This is because GENet is the only model that it was decided based on the tuning of hyper-parameters that it will use Mixup as part of the training.

In addition another metric that we consider worthy to analyse is the amount of time it took to train the model. And as it can be seen at table \ref{tab:time} is very impressive that must of the models used can be trained in less than 10 minutes. A comparison with the FLAME model is not possible as \cite{shamsoshoara2020} only shared the amount of epochs, which were 40, needed to trained the epochs. But taking in consideration than in average we took around 3 minutes per epoch to be trained, then we can assume that probably their model was trained in around 120 minutes, but the real number may be higher.

\begin{table}[b]
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
