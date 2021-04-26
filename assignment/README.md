# FLAME model

The objective of this project is to predict fire detection using as a base the [FLAME dataset provided by the IEEE](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

> Disclaimer. This projects requires the use of a GPU

# Pre-requisites

- Have the dataset in a Google Drive folder. We have two datasets available for the training: [With all the training images](https://drive.google.com/file/d/1uv9vAl55IinuEMXHocnJQUhPbMikuSIX/view?usp=sharing) and [with a reduce amount of training images](https://drive.google.com/file/d/1RrO4boe9jHUsCY1l9Z55iG1sfydJzubs/view?usp=sharing)
- (_Optional_) If you wish to use the preprocessing functions you should have a Kaggle account and know how to use the Kaggle API

# Running

## Notebooks

To run the project you must run the notebooks in the following order

1. (_Optional_) Run `notebooks/preprocessing.ipynb`
    - _This should only be done if you have only the_ [IEEE FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) _and want to do data augmentation on it, in addition to add the kaggle dataset to it_
    - It is expected that the dataset folder is named **FLAME** to be able to work
    - Upload your kaggle.json for the kaggle API downloading
2. Run `notebooks/training.ipynb`
    - It is expected that the dataset folder is named **FLAME** to be able to work
    - If you already have the FLAME folder you can skip the command last command of the first cell

## Scripts

To run the project you must run the scripts in the following order while being in the scripts folder

1. (_Optional_) Run `data_preprocessing.py`
    - _This should only be done if you have only the_ [IEEE FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) _and want to do data augmentation on it, in addition to add the kaggle dataset to it_
    - It is expected that the dataset folder is named **FLAME** to be able to work
    - Upload your kaggle.json for the kaggle API downloading
2. Run `train.py`
    - It is expected that the dataset folder is named **FLAME** to be able to work
    - (_Optional_) If you want to see the arguments for the hyper-parameters and others you can do `train.py --help`
3. Run `test.py`
    - It is expected that the dataset folder is named **FLAME** to be able to work
    - (_Optional_) If you want to see the arguments for the hyper-parameters and others you can do `test.py --help`
