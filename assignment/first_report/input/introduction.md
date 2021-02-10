\IEEEraisesectionheading{\section{Introduction}\label{sec:introduction}}

\IEEEPARstart{C}{onvolutional} neural networks (CNN's) currently are one of the must explored topics due to their high efficiency in several areas of ML, in particular Computer Vision and Image Recognition. This led to the creation of popular datasets such as MNIST and ImageNet, which are used nowadays to introduce the topic or to pre-train the CNN's to be able to perform _Transfer Learning_ for more specific tasks. This had led to the researchers to not only improve the performance of the CNN's but also found a use for them, one of these cases have been for the detection of Wildfires.

Wildfires currently are on of the must dangerous natural disasters that are threating the world. According to an investigation done by the NOAA (National Oceanic and Atmospheric Administration), the emissions produced by the wildfires often lead to harmful pollution not only in the area near the wildfire but also can extent even farther from the area, harming humans and other specials alike \cite{NOAA}. In addition to that, there are ecosystems which fauna and flora are being threatened due to the fires and to human intervention. Being that the case of the Amazon Rainforest, where the mentioned causes alongside droughts, could led to a possible "tipping point" where the Reinforces would be unsustainable in the case that there's no effective intervention of the matter \cite{Malhi}.

Due to the formerly mentioned there have been attempts by the scientific community to develop models that can detect based on the image if there is fire or not. Leading to the creation of datasets such as FLAME, a dataset created and provided by the IEEE \cite{IEEE-Flame}, or less academic but still important FIRE dataset hosted on Kaggle which is also home of several images showing environment with and without fire \cite{Saied}.

Nevertheless, the models are not very accurate, as an example the model trained with the FLAME dataset had only a 76% accuracy \cite{IEEE-Flame}, showing that there are still are of improvement in the regard. And with the knowledge that the FLAME model was trained only with the FLAME dataset, it's possible that using both the FLAME and Kaggle dataset in addition to some data augmentation and a new model it will be possible to improve the score previously obtained solely by the FLAME dataset.