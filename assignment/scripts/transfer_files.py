from os import listdir, path, rename
from tqdm import tqdm

classes = ['Fire', 'No_Fire']

def join_into_training(dir_path: str):
    for klass in classes:
        val_folder = path.join(dir_path, 'Validation', klass)
        training_folder = path.join(dir_path, 'Training', klass)
        description = f'Moving {klass} from validation to training: '
        for filename in tqdm(listdir(val_folder), desc=description):
            old_filename = path.join(val_folder, filename)
            new_filename = path.join(training_folder, filename)
            rename(old_filename, new_filename)

def move_into_new(original_path: str, new_path: str):
    for klass in classes:
        original_folder = path.join(original_path, 'Training', klass)
        new_folder = path.join(new_path, 'Training', klass)
        description = f'Moving {klass} from validation to training: '
        for filename in tqdm(listdir(original_folder), desc=description):
            if filename.startswith('balance'):
                continue
            old_filename = path.join(original_folder, filename)
            new_filename = path.join(new_folder, filename)
            if path.exists(new_filename):
                continue
            rename(old_filename, new_filename)

def main():
    original_path = './Flame-2'
    new_path = './Flame'

    join_into_training(original_path)
    join_into_training(new_path)

    move_into_new(original_path, new_path)

if __name__ == "__main__":
    main()
