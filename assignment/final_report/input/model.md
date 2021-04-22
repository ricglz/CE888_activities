<!-- TODO: Add about the normalization -->

## Model

For this paper we will experiment with different architectures as the backbone of our model. Taking in consideration that \cite{shamsoshoara2020} used the Xception \cite{chollet2017} architecture as the backbone of their model, which could be considered an old architecture. Therefore it is important to look for newer models which are more efficient in terms of time of inference, training and amount of parameters.

These conditions reduced the experimentation to the next architectures: EfficientNet \cite{tan2020}, ReXNet \cite{han2020}, GENet \cite{lin2020} and RepVGG \cite{ding2021}.

With these architectures we will perform transfer learning to be able to learn from the current task. It must be mentioned that all of these architectures have already been trained and have been shared by \cite{timm2019} who have also shared the code to use the models.
