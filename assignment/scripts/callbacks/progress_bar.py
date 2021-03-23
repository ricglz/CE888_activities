"""ProgressBar callback"""
from time import time

from tqdm import tqdm
import pytorch_lightning.callbacks as pl_callbacks

class ProgressBar(pl_callbacks.ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.epoch_time = 0
        self.stage_time = 0

    def disable(self):
        pass

    def enable(self):
        pass

    @staticmethod
    def format_num(number) -> str:
        """Add additional padding to the formatted numbers"""
        should_be_padded = isinstance(number, (float, str))
        if not isinstance(number, str):
            number = tqdm.format_num(number)
        if should_be_padded and 'e' not in number:
            if '.' not in number and len(number) < 5:
                try:
                    _ = float(number)
                except ValueError:
                    return number
                number += '.'
            number += "0" * (5 - len(number))
        return number

    def get_formatted_duration(self, prev_time):
        """Formats the duration in a human format"""
        duration = time() - prev_time
        if duration < 60:
            unit = 's'
        elif duration < 3600:
            duration /= 60
            unit = 'm'
        else:
            duration /= 3600
            unit = 'h'
        return self.format_num(duration) + unit

    def on_train_start(self, trainer, pl_module):
        self.stage_time = time()
        print('Start training')

    def on_train_end(self, trainer, pl_module):
        print(f'Total duration: {self.get_formatted_duration(self.stage_time)}')

    def on_test_start(self, trainer, pl_module):
        self.stage_time = time()
        print('Start test')

    def on_test_end(self, trainer, pl_module):
        print(f'Total duration: {self.get_formatted_duration(self.stage_time)}')

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_time = time()

    def on_validation_epoch_end(self, trainer, pl_module):
        values = [
            f'Epoch: {trainer.current_epoch}',
            f'Time: {self.get_formatted_duration(self.epoch_time)}'
        ]
        values += [
            f'{key}: {self.format_num(value)}'
            for key, value in trainer.progress_bar_dict.items()
        ]
        print(' - '.join(values))
