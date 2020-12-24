import os
import numpy as np
import random
import string
from typing import Callable, List, Optional, Sequence, Union
from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    pass


def random_string(length):
    pool = string.ascii_lowercase + string.digits
    return ''.join(random.choice(pool) for i in range(length))


def print_params_number(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{type(model).__name__} of {total_params:,} parameters')


def get_image_files(path: str) -> List[str]:
    return [
        os.path.join(r, fyle)
        for r, d, f in os.walk(path)
        for fyle in f
        if ".jpg" in fyle
    ]


def get_random_examples(model, test_dataset, device):
    model.eval()
    model.to(device)

    rand_idx = random.randint(0, len(test_dataset) - 1)
    file = test_dataset._files[rand_idx]
    data, target = test_dataset[rand_idx].source.to(device), test_dataset[rand_idx].target.to(device)
    output = model(data.unsqueeze(0)).squeeze()

    image = Image.open(file)
    w, h = image.size

    image_from = image.crop((w // 2, 0, w, h))
    image_to = image.crop((0, 0, w // 2, h))
    
    to_pil = transforms.ToPILImage()
    generated = to_pil(output.cpu().detach())
    return image_from, image_to, generated


def show_random_examples(model, test_dataset, device, path):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    image_from, image_to, generated = get_random_examples(model, test_dataset, device)
    
    ax0.imshow(image_from)
    ax1.imshow(image_to)
    ax2.imshow(generated)
    plt.savefig(path, dpi=100)
    plt.show()

    
def SSIM(im1, im2):
    mean1 = torch.mean(im1, dim=[-2, -1], keepdim=True)
    mean2 = torch.mean(im2, dim=[-2, -1], keepdim=True)
    
    var1 = torch.sqrt(torch.var(im1, dim=(-2, -1)))
    var2 = torch.sqrt(torch.var(im2, dim=(-2, -1)))
    
    cov = torch.mean((im1 - mean1) * (im2 - mean2), dim=(-2, -1))
    
    mean1 = mean1.squeeze()
    mean2 = mean2.squeeze()
    
    L = 1
    k1 = 0.01
    k2 = 0.03
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    numerator = (2 * mean1 * mean2 + c1) * (2 * cov + c2)
    denumerator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 * var1 + var2 * var2 + c2)
    
    ssim = torch.mean(numerator / denumerator)
    return ssim
