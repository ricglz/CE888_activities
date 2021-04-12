from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch

__all__ = ['mixup']

def partial_mixup(tensor: torch.Tensor, gamma: float, indices):
    if tensor.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    return tensor.mul(gamma).add(tensor[indices], alpha=1-gamma)

def mixup(x, y, gamma: float):
    indices = torch.randperm(x.size(0), device=x.device, dtype=torch.long)
    return partial_mixup(x, gamma, indices), partial_mixup(y, gamma, indices)

def get_image(path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    return torch.Tensor(arr)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()

    img_1 = get_image('./Flame/Training/Fire/Two0.png')
    img_2 = get_image('./Flame/Training/Fire/CarOne0.png')
    img_3 = get_image('./Flame/Training/No_Fire/Six7.png')
    img_4 = get_image('./Flame/Training/No_Fire/rover0.png')

    x = torch.stack([img_1, img_2, img_3, img_4])
    y = torch.Tensor([0, 0, 1, 1])
    gamma = np.random.beta(args.alpha, args.alpha)

    x, y = mixup(x, y, gamma)
    test_img = x[0].byte().numpy()
    Image.fromarray(test_img).show()
    print(y)
