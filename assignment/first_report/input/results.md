# Results

<!-- Show the results that you achieved in your work and offer an interpretation of those results. Acknowledge any limitations of your work and avoid exaggerating the importance of the results. -->

## ReXNet results

<!-- TODO: Add accuracy chart and confusion matrix -->

## EfficientNet results

<!-- TODO: Add accuracy chart and confusion matrix -->

## Best Model vs FLAME's model

The results overall doesn't seem as favorable, the new model shows a current accuracy of 71%, being that a 7% decrease in the accuracy compared to the model done by the FLAME team. Nevertheless this seems to be a case of over-fitting as the current model outperforms FLAME's, with a significantly decrease in both the validation and training loss, showing also an increase in the accuracy all of this shown in table 1.

<!-- TODO: Verify these are from the best model -->

| Metric     | Current Loss | Current Accuracy | FLAME's Loss | FLAME's Accuracy |
| -          | -            | -                | -            | -                |
| Training   | 0.0539       | -                | 0.0857       | 96.79            |
| Validation | 0.0239       | 99.35            | 0.1506       | 94.31            |
| Testing    | -            | 71.19            | 0.7414       | 76.23            |
Table: Comparison between the current model and FLAME's \cite{IEEE-Flame}

Some other things it should be consider is that due to hardware constraints and to be 100% to transfer learning, it was only trained the top layer of the CNN. This brought improvements, such that it was necessary to train the model in less epochs and probably in less time in the same hardware not only due to the use of transfer learning but also due to the nature of the architecture itself, nevertheless it could also be the reason that it's difficult to tune for a better generalization more specific to our current situation. This shows the advantages and disadvantages that can bring transfer learning.
