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


@torch.no_grad()
def show_random_examples(model, device, path):
    fig, axs = plt.subplots(nrows=1, ncols=10, sharey=True, figsize=(15, 5))

    generated = model.generate_samples(10)
    
    to_pil = transforms.ToPILImage()    
    for ax, g in zip(axs, generated):
        g_prepared = to_pil(g.squeeze().cpu().detach())
        ax.imshow(g_prepared)
    plt.savefig(path, dpi=200)
    plt.show()
