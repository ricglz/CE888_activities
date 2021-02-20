# Results

\begin{table}[H]
\centering
\begin{tabular}{|l|l|l|}
\toprule
Model & Accuracy & Loss \\
\midrule
ReXNet & 99.62 & 0.0293 \\
EfficientNet & 98.28 & 0.0582 \\
FLAME & 96.79 & 0.0857 \\
\bottomrule
\end{tabular}
\caption{Model's accuracies and losses in training dataset}
\label{tab-1}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{|l|l|l|}
\toprule
Model & Accuracy & Loss \\
\midrule
ReXNet & 99.62 & 0.0136 \\
EfficientNet & 98.32 & 0.053 \\
FLAME & 94.31 & 0.1506 \\
\bottomrule
\end{tabular}
\caption{Model's accuracies and losses in validation dataset}
\label{tab-2}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{|l|l|}
\toprule
Model & Accuracy \\
\midrule
ReXNet & 71.46 \\
EfficientNet & 61.46 \\
FLAME & 76 \\
\bottomrule
\end{tabular}
\caption{Model's accuracies in test dataset}
\label{tab-3}
\end{table}

To determined which of the models trained is the best, it will be used the information displayed in table \ref{tab-3} showing the results of the models in the testing dataset, showing that the ReXNet is definitely the best model of the ones which were trained. Nevertheless the results doesn't seem as favorable, having an accuracy of 71%, which means a 7% decrease in the accuracy compared to the model done by the FLAME team. All of this seems to be a case of over-fitting as the current model outperforms FLAME's, with a significantly decrease in both the validation and training loss, showing also an increase in the accuracy in both datasets as shown in table \ref{tab-1} and \ref{tab-2}.

Some other things it should be consider is that due to hardware constraints and to be 100% to transfer learning, it was only trained the top layer of the CNN. This brought improvements, such that it was necessary to train the model in less epochs and probably in less time in the same hardware not only due to the use of transfer learning but also due to the nature of the architecture itself, nevertheless it could also be the reason that it's difficult to tune for a better generalization more specific to our current situation. This shows the advantages and disadvantages that can bring transfer learning.
