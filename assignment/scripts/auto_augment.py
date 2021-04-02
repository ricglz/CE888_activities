from torch.nn import Module
from torchvision import transforms as T

class AutoAugment(T.AutoAugment):
    def __init__(self):
        super().__init__()
        self.transforms = [
            (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
            (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
            (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
            (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
            (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
            (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
            (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
            (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
            (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
            (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
            (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
            (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
            (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
            (('Rotate', 0.6, 9), ('Invert', 0.6, None)),
        ]
