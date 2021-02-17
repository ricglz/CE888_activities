# Background / Literature Review

<!-- Background information expands upon the key points stated in your introduction but is not the main focus of the paper. Sufficient background information helps your reader determine if you have a basic understanding of the research problem being investigated and promotes confidence in the overall quality of your analysis and findings. -->

<!-- Background information can also include summaries of important, relevant research studies. The key is to summarize for the reader what is known about the specific research problem before you conducted your analysis. This is accomplished with a general review of the foundational research literature (with citations) that report findings that inform your study's aims and objectives. -->

## Convolutional Neural Network

<!-- TODO: Miss citing -->

Are a type of Neural Networks commonly used for the areas of Image Recognition and Computer Vision, this type of networks have the main feature of being divided in several deeply connected _convolutional_ and _pooling_ layers, to finally end with a _classifier_ which is a fully connected layer comparable to a layer of traditional Multi-Layer Perceptron (MLP).

### CNN Architectures \label{cnn-arch}

<!-- TODO: Miss citing -->

As CNN's are one of the state of the art topics in research nowadays, there have been many popular architectures that either have very high performance, or have very little parameters with the objective of having a lightweight model with good performance. For developing actual applications for a problem researchers actually use this predefined architectures to solve the problem they're facing as this provides the advantages of being a proven useful architecture.

In addition that due to their popularity people train these models with popular datasets so researchers can use these pre-trained weights enabling to do _transfer learning_, which is when using an already trained model, retrained the top layer to be able to have a good performance in the new problem.

Actually in the case of the FLAME research done, it's mentioned in the paper that the model that achieved a 76% of accuracy was using an architecture called Xception \cite{Shamsoshoara2020}.

Another example of an architecture is EfficientNet published in September 2019, that wanted to tackle the problem of scaling up an architecture by having a smaller amount of parameters, but also by keeping/increasing the accuracy of the model, this was able by the use of _compound coefficient_ \cite{Tan2020}.

Also it exists the architecture of ReXNet which was proposed in July 2020, aiming to follow the trend of creating accurate and lightweight architectures, distinguishing themselves by the use of _representational bottlenecks_. The result was quite good improving the accuracy when doing transfer learning from trained models using the COCO dataset, while keeping a small amount of parameters to train \cite{Han2020}.


## Data augmentation

Is a method that allows to increase the amount of data that we have in a dataset, allowing to be persistent being that you generate new data based on the previous one and save it on the same dataset. Or add randomness prior to training a batch, making that every epoch and batch should not consist of the same data in the same order \cite{Van2001}.

For images datasets, normally the transformation/augmentations done to increase the dataset is by performing modifications in the same image being by simple image transformations such as shearing, rotation and translation. But it also may be the modification of the brightness, flipping the image vertically or horizontally, or by cropping it \cite{Perez2017}.
