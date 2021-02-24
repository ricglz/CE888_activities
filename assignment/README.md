# FLAME model

The objective of this project is to predict fire detection using as a base the [FLAME dataset provided by the IEEE](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

> Disclaimer. This projects requires the use of Google Colab

# Pre-requisites

- Have the dataset in a Google Drive folder. The dataset used for training can be found [here](https://drive.google.com/file/d/1t1v4kuBIDk5iwehwyawIUAj9eY2QmY9U/view?usp=sharing)
- Have a Kaggle account and know how to use the Kaggle API

# Running

To run the project you must run the notebooks in the following order

1. (_Optional_) Run the `Data_augmentation.ipynb`
    - _This should only be done if you have only the_ [IEEE FLAME dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) _and want to do data augmentation on it, in addition to add the kaggle dataset to it_
    - Make sure that the `training_path` path is correct
    - Upload your kaggle.json for the kaggle API downloading
2. Run the `Initial_Training.ipynb`
    - Make sure that the `data_dir` path is correct
