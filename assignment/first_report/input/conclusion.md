# Conclusion

<!-- Summarize your key findings. Include important conclusions that can be drawn and further implications for the field. Discuss the benefits or shortcomings of your work and suggest future areas for research. -->

At the end we were not able to actually outperform in the testing dataset, even though we augment the dataset and trained a model using transfer learning of the ImageNet dataset. Nevertheless, the model and training overall seems to have potential, as it was able to have a decent performance with fewer epochs and with an architecture that requires less resources. Being an advantage specially if the model ends up being used in a IoT environment where efficient models are required due to the hardware that is being used.

Also even though the performance on the test dataset is not the best, it doesn't mean that there's not hope for actually outperforming what we have as the best model yet, as there are some areas which could help to improve the overall performance of the models. Such as training the inner layer of the CNN, not only the top layer, this is known as Fine-tuning and it seems like it will be the next step to take to train and improve the performance of the models.

In addition to that, it's possible that new architectures can bring even more potential results, being the one that caught my attention was the recently published RepVGG. A new approach to the traditional VGG architecture which aims to simplify the architecture improve the speed of preprocessing and inference, while also keeping a good accuracy, a model that attempts all of this while avoiding the know complexity that the current state-of-the-art models have \cite{ding2021}.

It could also be consider to improve the performance of the model, an increase of the dataset, as the Kaggle dataset seems to not be able to have a big difference in the accuracy. This could be by either finding another dataset for fire detection training or scalping videos with fire in it and extracting random frames from it.
