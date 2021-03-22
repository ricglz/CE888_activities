"""Functions to create new callbacks without creating a new class"""
from os import path
from pytorch_lightning.callbacks import ModelCheckpoint

get_model_dir = lambda: './Models'

def get_checkpoint(model_name: str):
    """Gets the checkpoint callback for the function"""
    return ModelCheckpoint(
        dirpath=path.join(get_model_dir(), model_name),
        monitor='val_score',
        mode='max',
        filename='{epoch:02d}-{val_score:.4f}'
    )
