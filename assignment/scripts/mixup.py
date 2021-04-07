from PIL.Image import fromarray, open, Image
import numpy as np
import torch

__all__ = ['mixup']

def partial_mixup(tensor: torch.Tensor, gamma: float, indices):
    if tensor.size(0) != indices.size(0):
        raise RuntimeError("Size mismatch!")
    perm_input = tensor[indices]
    return tensor.mul(gamma).add(perm_input, alpha=1-gamma)

def mixup(x, y, gamma: float):
    indices = torch.randperm(x.size(0), device=x.device, dtype=torch.long)
    return partial_mixup(x, gamma, indices), partial_mixup(y, gamma, indices)

def get_image(path) -> np.ndarray:
    with open(path) as img:
        arr = np.array(img)
    return torch.Tensor(arr)

if __name__ == "__main__":
    img_1 = get_image('./Flame/Training/Fire/Two0.png')
    img_2 = get_image('./Flame/Training/Fire/CarOne0.png')

    x = torch.stack([img_1, img_2])
    y = torch.randint(1, (2,))
    gamma = np.random.beta(4, 4)

    x, _ = mixup(x, y, gamma)
    test_img = x[0].byte().numpy()
    fromarray(test_img).show()
