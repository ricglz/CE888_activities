"""Customized AutoAugment module"""
from random import sample

import torch
from torchvision import transforms as T

class AutoAugment(T.AutoAugment):
    def __init__(self, magnitude=0, amount=2):
        super().__init__()
        self.actual_transforms = (
            ('Invert', 0.5, None),
            ('Rotate', 0.5, magnitude),
            ('ShearX', 0.5, magnitude),
            ('ShearY', 0.5, magnitude),
            ('TranslateX', 0.5, magnitude),
            ('TranslateY', 0.5, magnitude),
        )
        self.amount = amount

    @staticmethod
    def get_params(transform_num):
        """Get parameters for autoaugment transformation
        Returns:
            params required by the autoaugment transformation
        """
        policy_id = torch.randint(transform_num, (1,)).item()
        probs = torch.rand((transform_num,))
        signs = torch.randint(2, (transform_num,))

        return policy_id, probs, signs

    def forward(self, img):
        self.transforms = [sample(self.actual_transforms, self.amount)]
        transform_id, probs, signs = self.get_params(len(self.transforms))
        print(transform_id, probs.shape, signs.shape)
        return super().forward(img)
