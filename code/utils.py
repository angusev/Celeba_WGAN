import os
import numpy as np
import random
import string
from typing import Callable, List, Optional, Sequence, Union
from pathlib import Path

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms

import matplotlib.pyplot as plt


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
    file = test_dataset.dataset._files[rand_idx]
    image, condition = test_dataset[rand_idx].image.to(device), test_dataset[rand_idx].condition.to(device)
#     output = model(condition.unsqueeze(0).unsqueeze(2).unsqueeze(3)).squeeze()
#     conditions = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 80))))
    output = model(condition.unsqueeze(0)).squeeze()
    
    
    image = Image.open(file)
    
    to_pil = transforms.ToPILImage()
    generated = to_pil(output.cpu().detach())
    return image, generated


@torch.no_grad()
def show_random_examples(model, test_dataset, device, path):
    model.eval()
    fig, axs = plt.subplots(nrows=10, ncols=2, sharex=True, sharey=True, figsize=(2, 10))
    
    for ax0, ax1 in axs:
        image, generated = get_random_examples(model, test_dataset, device)

        ax0.imshow(image)
        ax1.imshow(generated)
    plt.savefig(path, dpi=500)
    plt.show()
