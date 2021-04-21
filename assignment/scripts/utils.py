"""Module containing utility functions"""
from argparse import ArgumentParser
from os import path
from zipfile import ZipFile
from distutils.util import strtobool as fake_strtobool

from tqdm import tqdm

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

strtobool = lambda x: bool(fake_strtobool(x))

class CustomParser(ArgumentParser):
    def add_bool_argument(self, flag: str):
        self.add_argument(flag, type=strtobool, nargs='?', const=True, default=False)

def unzip_file(zip_path, dest_path):
    with ZipFile(zip_path, 'r') as zip_file:
        for member in tqdm(zip_file.infolist(), desc='Extracting '):
            zip_file.extract(member, dest_path)

def get_data_dir(minified=True):
    """Gets the directory where the data is contained"""
    general_dir = '.'
    data_dir = general_dir + '/Flame'
    if IN_COLAB and not path.exists(data_dir):
        drive_path = '/content/gdrive'
        drive.mount(drive_path, force_remount=False)
        zip_file = 'Minified-Flame.zip' if minified else 'Flame.zip'
        zip_path = path.join(drive_path, 'MyDrive/Essex/Datasets/zipped', zip_file)
        unzip_file(zip_path, general_dir)
    return data_dir

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--no_minified', action='store_false')
    args = parser.parse_args()

    get_data_dir(args.no_minified)
