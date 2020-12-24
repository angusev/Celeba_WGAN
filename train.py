import random
import os
import numpy as np
import sys
from PIL import Image

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torchvision import models

from code.dataclasses import get_dataloaders
from code.model import WGAN
from code.utils import (get_image_files,
                        get_random_examples,
                        show_random_examples,
                        SSIM)
from code.trainer import train, test
from code.configs import Parser

import wandb


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


DEVICE = torch.device("cuda")


def main(args):
    set_seed(179)
    
    parser = Parser()
    paths, training_configs, launch_configs = parser.parse_args(args)
    training_configs.device = DEVICE

    if launch_configs.wandb:
        wandb.init(project="celeba_wgan", name=launch_configs.wandb)
    
    train_loader, valid_loader = get_dataloaders(paths.dataset, training_configs, launch_configs)
    
    model = WGAN(training_configs).to(DEVICE)

    try:
        for epoch in range(training_configs.epochs):
            train_losses = train(model, DEVICE, train_loader)
            test_losses = test(model, DEVICE, valid_loader)
            
            print(f'Epoch {epoch + 1} / {training_configs.epochs} \t  [loss_D, loss_G]')
            print(f'\tTrain losses: {[np.round(l.item(), 4) for l in train_losses[:-1]]}')
            print(f'\tValid loss:   {[np.round(l.item(), 4) for l in test_losses[:-1]]}')
            
#             show_random_examples(model.model_G, valid_dataset, DEVICE, paths.examples / f'epoch{epoch + 1}.jpg')
#             if launch_configs.wandb:
#                 source, target, generated = get_random_examples(model.model_G, valid_dataset, DEVICE)
                
#                 train_loss_D, train_loss_G_GAN, train_loss_G_L1, train_ssim = train_losses
#                 test_loss_D, test_loss_G_GAN, test_loss_G_L1, test_ssim = test_losses
#                 wandb.log({"train_loss_D": train_loss_D,
#                            "train_loss_G": train_loss_G,
#                            "test_loss_D": test_loss_D,
#                            "test_loss_G": test_loss_G,
#                            "generated": [wandb.Image(generated)],
#                 })
    except KeyboardInterrupt:
        print('\nBye')


if __name__ == "__main__":
    main(sys.argv)