"""Customized AutoAugment module"""
import math
from random import sample

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

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
        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(self.amount)
        transforms = sample(self.actual_transforms, self.amount)

        for i, (op_name, p, magnitude_id) in enumerate(transforms[transform_id]):
            if probs[i] <= p:
                magnitudes, signed = self._get_op_meta(op_name)
                magnitude = float(magnitudes[magnitude_id].item()) \
                    if magnitudes is not None and magnitude_id is not None else 0.0
                if signed is not None and signed and signs[i] == 0:
                    magnitude *= -1.0

                if op_name == "ShearX":
                    img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                                   interpolation=self.interpolation, fill=fill)
                elif op_name == "ShearY":
                    img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                                   interpolation=self.interpolation, fill=fill)
                elif op_name == "TranslateX":
                    img = F.affine(img, angle=0.0, translate=[int(F._get_image_size(img)[0] * magnitude), 0], scale=1.0,
                                   interpolation=self.interpolation, shear=[0.0, 0.0], fill=fill)
                elif op_name == "TranslateY":
                    img = F.affine(img, angle=0.0, translate=[0, int(F._get_image_size(img)[1] * magnitude)], scale=1.0,
                                   interpolation=self.interpolation, shear=[0.0, 0.0], fill=fill)
                elif op_name == "Rotate":
                    img = F.rotate(img, magnitude, interpolation=self.interpolation, fill=fill)
                elif op_name == "Invert":
                    img = F.invert(img)
                else:
                    raise ValueError("The provided operator {} is not recognized.".format(op_name))
        return img
