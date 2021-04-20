# Results

<!-- TODO: Update validation with new stuff -->

\begin{figure}
\centering
\subcaptionbox{Train loss \label{fig:train-loss}}{\includegraphics[width=0.49\textwidth]{img/training_loss}}%
\hfill
\subcaptionbox{Val loss \label{fig:val-loss}}{\includegraphics[width=0.49\textwidth]{img/validation_loss}}%
\caption{Losses progression in training and validation datasets}
\label{fig:dataset-losses}
\end{figure}

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
\label{tab:3}
\end{table*}

The following results were obtained by using NVidia RTX 2080 gpu for the Rexnet model and a GTX 1080 TI for the other.

As we can see in figure \ref{fig:train-loss}, the reduction of the training loss seems normal, except on the GENet model, this is because this model is the only one that uses Mixup. Thus, it allows the model to always have a loss in which it will be learning, stabilizing the training.

In this case we will consider the model done in \cite{shamsoshoara2020} as out baseline to determine if the tuning and new techniques, actually improves the model accuracy. As it can be seen in table \ref{tab:3}, all the models improves over the FLAME model in the test dataset, improving in term of accuracy and loss of the test. Nevertheless, we have a special case that is the GENet model, in which the performance in the training dataset is considerably lower than the other models, even though it achieved a higher accuracy than the FLAME model in the test dataset. This is because GENet is the only model that it was decided based on the tuning of hyper-parameters that it will use Mixup as part of the training.
