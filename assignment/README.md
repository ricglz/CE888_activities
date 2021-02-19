# FLAME model

The objective of this project is to predict fire detection using as a base the [FLAME dataset provided by the IEEE](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

> Disclaimer. This projects requires the use of Google Colab

# Pre-requisites

- Have the dataset in a Google Drive folder
- Have a Kaggle account

# Running

To run the project you must run the notebooks in the following order

1. Run the `Data_augmentation.ipynb`
    - Make sure that the `training_path` path is correct
    - Upload your kaggle.json for the kaggle API downloading
2. Run the `Initial_Training.ipynb`
    - Make sure that the `data_dir` path is correct
