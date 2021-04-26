# FLAME model

The objective of this project is to predict fire detection using as a base the [FLAME dataset provided by the IEEE](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

> Disclaimer. This projects requires the use of a GPU

# Pre-requisites

- Have the dataset in a Google Drive folder. We have two datasets available for the training: [With all the training images](https://drive.google.com/file/d/1uv9vAl55IinuEMXHocnJQUhPbMikuSIX/view?usp=sharing) and [with a reduce amount of training images](https://drive.google.com/file/d/1RrO4boe9jHUsCY1l9Z55iG1sfydJzubs/view?usp=sharing)
- (_Optional_) If you wish to use the preprocessing functions you should have a Kaggle account and know how to use the Kaggle API

# Running

## Notebooks

To run the project you must run the notebooks in the following order

1. (_Optional_) Run the `Data_augmentation.ipynb`
    - _This should only be done if you have only the_ [IEEE FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) _and want to do data augmentation on it, in addition to add the kaggle dataset to it_
    - Make sure that the `training_path` path is correct
    - Upload your kaggle.json for the kaggle API downloading
2. Run `training.ipynb`
