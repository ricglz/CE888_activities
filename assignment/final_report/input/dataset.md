# Methodology

## Dataset

The dataset used was a merge between the datasets done by \cite{Flame2020}, \cite{saied2020} and \cite{dunnings18}. The code to perform the same preprocessing is fully available on Github \footnote{\url{https://github.com/ricglz/CE888_activities/blob/main/assignment/scripts/data_preprocessing.py}}.

### FLAME dataset

The FLAME dataset consists of 47,992 images that are labeled as having fire or not. 39,375 of the total amount of images are for training/validation. As can be seen at figure \ref{tab:1}, the training/validation set, the labels are skewed towards the class with fire. These images were obtained by the researchers by extracting the frames of videos recorded by drones of forest areas \cite{Flame2020}.

\begin{table}[b]
\centering
\begin{tabular}{|l|c|c|r|}
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

### FIRE's dataset

This dataset was created for a NASA challenge in 2018, the authors collected a total of 1,000 images, all labeled for training data. These images, contrary to the previous dataset are from a wide range of environments, from urban to rural areas. Nevertheless, the dataset is skewed, containing 755 images labeled as fire and the rest as no-fire \cite{saied2020}.

### Dunning's dataset

The dataset was created by Dunning et al. consisting of 23,408 images for training. This dataset was created by merging other datasets and material from public videos \cite{dunnings18}. This dataset also has a skew over the fire images.

### Merging datasets

All the images of the FIRE's and Dunning's dataset were merged into the training/validation dataset of flame.

### Balancing the datasets

After merging the datasets, the next part of the preprocessing was to balance the dataset; because, as mentioned in the prior sections, all the datasets are skewed towards the label with fire. To balance the dataset, we over-sample the no fire class label by performing Data Augmentation over random samples of the label. The augmentations done to the dataset were brightness, contrast, rotation, horizontal and vertical flip. This resulted in a dataset containing 76,726 images with a perfect balance between the 2 classes.

### Dividing Training/Validation

The next step would be to split the training/validation dataset into their own predefined folders, this will help by always using the same images for training and validation, instead of random ones. Therefore the dataset \footnote{Dataset without halving training: \url{https://drive.google.com/file/d/1uv9vAl55IinuEMXHocnJQUhPbMikuSIX}} was split into 80% training and 20% validation, keeping the balanced ratios between the labels.

### Reducing the amount of data in training

With a total of 61,378 images, there was a lot of data to process. If we want that the training to be as efficient as possible we need to reduce the amount of data used for training. As there are a lot of images that are very similar between each other, due to being frames extracted from videos. Then it was decided to cut the amount of training data into half, while keeping the ratio of classes as before. This was the last step for the creation of the dataset \footnote{Dataset after halving training: \url{https://drive.google.com/file/d/1RrO4boe9jHUsCY1l9Z55iG1sfydJzubs/view}} and resulted in a distribution as it shows in table \ref{tab:2}

\begin{table}
\centering
\begin{tabular}{|l|c|c|r|}
\toprule
Dataset & Fire & No Fire & Total \\
\midrule
Train & 15341 (50\%) & 15341 (50\%) & 30682 (100.00\%) \\
Validation & 7671 (50\%) & 7671 (50\%) & 15342 (100.00\%) \\
Test & 5137 (59.61\%) & 3480 (40.39\%) & 8617 (100.00\%) \\
\bottomrule
\end{tabular}
\caption{Dataset distribution after preprocessing}
\label{tab:2}
\end{table}
