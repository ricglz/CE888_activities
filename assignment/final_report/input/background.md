# Background / Literature Review

<!-- Background information expands upon the key points stated in your introduction but is not the main focus of the paper. Sufficient background information helps your reader determine if you have a basic understanding of the research problem being investigated and promotes confidence in the overall quality of your analysis and findings. -->

<!-- Background information can also include summaries of important, relevant research studies. The key is to summarize for the reader what is known about the specific research problem before you conducted your analysis. This is accomplished with a general review of the foundational research literature (with citations) that report findings that inform your study's aims and objectives. -->

<!-- TODO: Probably add information related to CNN and/or weight decays -->

## Mixed-precision training

One of the current problems that people are facing nowadays with Deep Learning (DL) is the amount of resources that it takes to train a model, either because the architecture has a lot of parameters and takes a lot of memory of the GPU. Or because it takes a lot of time to be trained due to the computational power it needs. To solve this problem, \cite{micikevicius2018} proposed what is known as mixed-precision training, where instead of using the full-precision number of 8 bytes, it would use the 4 byte format. As showed in \cite{he2019} using this technique led to a reduction of the amount of memory it took to train the model, in addition to a speedup in the time that the model took to be trained.

## Data Augmentation

Data Augmentation is a technique that improves the generalization of the network. This is done by performing manipulation of the data, in the case of images, this mean applying techniques such as: shearing, rotation, saturation and others \cite{van2001}. This technique is effective because it generates new data each time a new epoch is run; thus, resulting quite effective when learning from an overall small dataset. In addition to the prior, it also has been proven to be effective as it adds randomness to the training and it reduces the changes that the same batch have the same data each epoch as was shown in \cite{perez2017}.

### AutoAugment

Nevertheless, one limitation this technique has is that its effectiveness depends on the augmentations done over the data, as well as choosing the correct parameters, mainly their magnitude and probability. A solution of this limitation is proposed by \cite{cubuk2019} who created a procedure called _AutoAugment_ that created a search space consisting of a policy which consisted of sub-policies that decide which augmentation to do and which are their parameters. This resulted in an improvement of the previous state-of-the-art models.

### Test Time Augmentation

Even though, data augmentation is commonly used only for the training phase it also has a purpose during the testing phase. This technique is called Test Time Augmentation (TTA), in which the input is augmented and passed as an input for the model n times to result in a total of n outputs. With these outputs, a merge operation is then performed, which normally is a mean function. This merge result is then used to obtain the desired test metrics \cite{kim2020}.

## Mixup

<!-- TODO: Add information related to the alpha value -->

\begin{figure}[h]
\centering
\subcaptionbox{Before Mixup}{\includegraphics[width=0.2\textwidth]{img/before_mixup}}%
\hfill
\subcaptionbox{After Mixup}{\includegraphics[width=0.2\textwidth]{img/after_mixup}}%
\caption{Example of transformation of an image when using mixup}
\label{fig:mixup}
\end{figure}

Mixup is a technique that was proposed by \cite{zhang2018}, that aims to help in the stabilization of adversarial networks in generative models; nevertheless, it has also found success in classification tasks as was demostrated in \cite{gong2020}. The technique consists of mixing both the data and labels of elements in the batch, resulting in an overall generalization of how the distribution of the data would look for two different elements of the same or different class. An example of the result of mixup can be seen in figure \ref{fig:mixup}.

## Transfer learning

As said by \cite{torrey2010} "Transfer learning is the improvement of learning in a new task through the transfer of knowledge from a related task that has already been learned". This technique helps in reducing the amount of computational time to achieve the same or better accuracy. By using the weights of a model with the same architecture trained for another task, the model will use that prior knowledge to learn this new task faster, proven by the works of \cite{shao2018}, \cite{zoph2016} and \cite{chiang2020}.

## OneCycleLR

\begin{figure}
\centering
\subcaptionbox{Learning rate behavior}{\includegraphics[width=0.24\textwidth]{img/one_cycle_lr}}%
\hfill
\subcaptionbox{Momentum behavior}{\includegraphics[width=0.24\textwidth]{img/one_cycle_momentum}}%
\caption{Behavior of momentum and LR when using OneCycleLR. Plots done by \cite{gugger2018}}
\label{fig:one_cycle}
\end{figure}

Proposed by \cite{smith2018}, OneCycleLR is a scheduler that was created to achieve what \cite{smith2018} called as "Super-Convergence", where the model achieves an improvement of the accuracy in less epochs.

The scheduler, as can be seen at figure \ref{fig:one_cycle}, doesn't only consist of modifying the learning rate as most schedulers do, but it also modifies the momentum. In the case of the learning rate, it starts with an initial value, which will increase by each step until it reaches a max value; conventionally this is at the middle of the training, but can be at any other point. Then, the learning rate will decrease until reaching a value lower than the initial value. In the case of the momentum, the behavior is the opposite of the learning rate, starting with a very high value, then descending and finally going back again to the max value.

The scheduler has proven its usefulness in reducing the amount of iterations need, while keeping or increasing the accuracy in the works of \cite{kuo2020} and \cite{kim2020Pynet}. In the case of the former it was possible to reduce the amount of epochs needed by a 68% compared to their baseline, while also improving the model's accuracy.
