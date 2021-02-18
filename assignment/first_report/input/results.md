# Results

## ReXNet results

<!-- ![ReXNet training model progress](images/rexnet_training.png) -->

| Dataset    | (Loss) Accuracy |
| -          | -               |
| Training   | (0.0293) 99.62  |
| Validation | (0.0153) 99.62  |
| Testing    | (-)      71.46  |
Table: Final ReXNet model metrics

<!-- ![ReXNet confusion matrix](images/rexnet_cm.png) -->

## EfficientNet results

<!-- ![EfficientNet training model progress](images/efficientnet_training.png) -->

| Dataset    | (Loss) Accuracy |
| -          | -               |
| Training   | (0.0582) 98.28  |
| Validation | (0.0530) 98.32  |
| Testing    | (-)      61.46  |
Table: Final EfficientNet model metrics

<!-- ![EfficientNet confusion matrix](images/efficientnet_cm.png) -->

## Best Model vs FLAME's model

As shown with the tables 1 and 2, it's evident that the best model that was trained is the ReXNet nevertheless the results overall doesn't seem as favorable, the new model shows a current accuracy of 71%, being that a 7% decrease in the accuracy compared to the model done by the FLAME team. Nevertheless this seems to be a case of over-fitting as the current model outperforms FLAME's, with a significantly decrease in both the validation and training loss, showing also an increase in the accuracy.

Some other things it should be consider is that due to hardware constraints and to be 100% to transfer learning, it was only trained the top layer of the CNN. This brought improvements, such that it was necessary to train the model in less epochs and probably in less time in the same hardware not only due to the use of transfer learning but also due to the nature of the architecture itself, nevertheless it could also be the reason that it's difficult to tune for a better generalization more specific to our current situation. This shows the advantages and disadvantages that can bring transfer learning.
