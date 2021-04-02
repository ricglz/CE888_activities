"""Customized AutoAugment module"""
from random import sample

from torchvision import transforms as T

class AutoAugment(T.AutoAugment):
    def __init__(self, magnitude=0, amount=2):
        super().__init__()
        self.actual_transforms = [
            ('Invert', 0.5, None),
            ('Rotate', 0.5, magnitude),
            ('ShearX', 0.5, magnitude),
            ('ShearY', 0.5, magnitude),
            ('TranslateX', 0.5, magnitude),
            ('TranslateY', 0.5, magnitude),
        ]
        self.amount = amount

    def forward(self, img):
        self.transforms = [sample(self.actual_transforms, self.amount)]
        return super().forward(img)
