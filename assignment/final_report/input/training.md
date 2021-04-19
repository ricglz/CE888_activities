## Training

The model will be trained using Mixed Precision, in addition of using as learning scheduler the OneCycleLR. Also we used a special data loader which will create batches containing the same amount of random elements of each class. This was possible due to the previous work of \cite{galato2019}, who developed a similar sampler for their use case.
