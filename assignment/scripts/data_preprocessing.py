"""Module for preprocess the dataset"""
from glob import glob
from os import path, mkdir, remove
from pathlib import Path
from shutil import move
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torchvision.utils as t_utils

data_path = './Flame'

classes = ['Fire', 'No_Fire']
training_path = path.join(data_path, 'Training')
test_path = path.join(data_path, 'Test')
resize = T.Resize((224, 224))
validation_path = path.join(data_path, 'Validation')

def save_image(img, label: str, index: int, prefix: str):
    klass = classes[label]
    img_path = path.join(training_path, f'{klass}/{prefix}_{index}.png')
    t_utils.save_image(img, img_path)

def resize_dataset(ds_path: str):
    dataset = ImageFolder(ds_path, T.Compose([resize, T.ToTensor()]))
    for index, (img_path, _) in enumerate(tqdm(dataset.imgs)):
        t_utils.save_image(dataset[index][0], img_path)

def clear_balanced():
    balance_imgs = glob(
        f'{training_path}/**/balance*.png', recursive=True)
    for balance_img in tqdm(balance_imgs):
        remove(balance_img)

def get_minor_klass(train_ds: ImageFolder):
    targets = np.array(train_ds.targets)
    fire_data_count = np.count_nonzero(targets == 0)
    non_fire_data_count = np.count_nonzero(targets == 1)
    klass_counts = [fire_data_count, non_fire_data_count]
    minor_klass = np.argmin(klass_counts)
    minor_count, max_count = min(klass_counts), max(klass_counts)
    images_to_save = min(max_count - minor_count, minor_count)
    return minor_klass, images_to_save

def balance_dataset():
    transforms = T.Compose([
      resize,
      T.ColorJitter(brightness=0.25, contrast=0.25),
      T.RandomRotation(degrees=5),
      T.RandomHorizontalFlip(),
      T.RandomVerticalFlip(),
      T.ToTensor(),
    ])
    train_ds = ImageFolder(training_path, transforms)
    minor_klass, images_to_save = get_minor_klass(train_ds)
    indexes_to_enhance = np.where(train_ds.targets == minor_klass)[0]
    assert train_ds.targets[indexes_to_enhance[0]] == minor_klass
    indexes_to_enhance = np.random.choice(indexes_to_enhance, images_to_save, replace=False)
    assert len(indexes_to_enhance) == images_to_save
    for save_img_index, index in enumerate(tqdm(indexes_to_enhance)):
        img, label = train_ds[index]
        save_image(img, label, save_img_index, 'balance')

def half_the_data():
    transforms = T.Compose([resize, T.ToTensor()])
    train_ds = ImageFolder(training_path, transforms)
    files = list(map(lambda a: a[0], train_ds.samples))
    _, erase_files = train_test_split(
        files, test_size=0.5, shuffle=True, stratify=train_ds.targets)
    for file_to_erase in tqdm(erase_files):
        remove(file_to_erase)

def split_training_dataset():
    if not path.exists(validation_path):
        mkdir(validation_path)
        for klass in classes:
            mkdir(path.join(validation_path, klass))
    train_ds = ImageFolder(training_path)
    targets = train_ds.targets
    _, valid_idx= train_test_split(
        range(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    for idx in tqdm(valid_idx):
        img_path, label = train_ds.imgs[idx]
        filename = Path(img_path).name
        klass = classes[label]
        new_path = path.join(validation_path, klass, filename)
        move(img_path, new_path)

def count_training_dataset():
    train_ds = ImageFolder(training_path)
    targets = np.array(train_ds.targets)
    fire_data_count = np.count_nonzero(targets == 0)
    non_fire_data_count = np.count_nonzero(targets == 1)
    tqdm.write(f'Fire data: {fire_data_count}')
    tqdm.write(f'Non-fire data: {non_fire_data_count}')
    tqdm.write(f'Total: {fire_data_count + non_fire_data_count}')

def main():
    tqdm.write('Resizing training dataset')
    resize_dataset(training_path)
    tqdm.write('Resizing test dataset')
    resize_dataset(test_path)
    tqdm.write('Balancing training dataset')
    balance_dataset()
    tqdm.write('Splitting dataset')
    split_training_dataset()
    count_training_dataset()

if __name__ == "__main__":
    main()
