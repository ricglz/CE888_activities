# Methodology

## Dataset

The dataset used was a merge between the datasets done by \cite{Flame2020}, \cite{saied2020} and \cite{dunnings18}.

### FLAME dataset

The FLAME dataset consists of 47,992 images that are labeled as having fire or not. 39,375 of the total amount of images are for training/validation. As can be seen at \ref{tab:1} the training/validation set, the labels are skewed towards the class with fire. These images were obtained by the researchers by extracting the frames of videos recorded by drones of forest areas \cite{Flame2020}.

\begin{table}
\centering
\begin{tabular}{|l|l|l|}
\toprule
Dataset & Fire & No Fire & Total \\
\midrule
Train/Val & 25018 (63.54\%) & 14357 (36.46\%) & 39375 (100.00\%) \\
Test & 5137 (59.61\%) & 3480 (40.39\%) & 8617 (100.00\%) \\
\bottomrule
\end{tabular}
\caption{FLAME dataset distribution}
\label{tab:1}
\end{table}

### Kaggle's dataset

This dataset was created for a NASA challenge in 2018, the authors collected a total of 1,000 images all labeled for training data. These images contrary to the previous dataset are from a wide range of environments, from urban to rural areas. Nevertheless, the dataset is skewed, containing 755 images labeled as fire and the rest as no-fire \cite{saied2020}.

### Dunning's dataset

The dataset was created by Dunning et al. consisting of 23,408 images for training. This dataset was created by merging other datasets and material from public videos \cite{dunnings18}. This dataset also has a skew over the fire images.

### Merging datasets

All the images of the Kaggle's and Dunning's dataset were merged into the training/validation dataset of flame.

### Balancing the datasets

After merging the datasets, the next part of the preprocessing was to balance the dataset. Because as mentioned in the prior sections all the datasets are skewed towards the label with fire. To balance the dataset, we over-sample the no fire class label by performing Data Augmentation over random samples of the label. The augmentations done to the dataset were brightness, contrast, rotation, horizontal and vertical flip. This resulted in a dataset \footnote{https://drive.google.com/file/d/1uv9vAl55IinuEMXHocnJQUhPbMikuSIX} containing 61,378 images with a perfect balance between the 2 classes.
